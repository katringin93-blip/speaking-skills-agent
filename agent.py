import sys
import time
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import yaml
import requests

# Shared HTTP client for OpenAI calls (prevents hangs on Windows proxy)
HTTP = requests.Session()
HTTP.trust_env = False
from pydub import AudioSegment

# ---------- Константы ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 300 * 1000  # 5 минут
TELEGRAM_SAFE_LIMIT = 3500  # запас к лимиту Telegram ~4096

# ---------- Логирование ----------

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

# ---------- Помощники ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Нажмите ENTER, чтобы закрыть окно...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)

def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent

def load_config() -> dict:
    base_path = app_base_dir()
    cfg_path = base_path / "config.local.yaml"
    if not cfg_path.exists():
        cfg_path = Path(os.getcwd()) / "config.local.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def is_file_stable(path: Path, stable_seconds: int = 10, poll_interval: int = 1) -> bool:
    if not path.exists():
        return False
    last_size = path.stat().st_size
    stable_for = 0
    while stable_for < stable_seconds:
        time.sleep(poll_interval)
        if not path.exists():
            return False
        size = path.stat().st_size
        if size != last_size:
            last_size = size
            stable_for = 0
        else:
            stable_for += poll_interval
    return True

def _run_ffmpeg(cmd: List[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip())

def _yaml_frontmatter(meta: Dict[str, Any]) -> str:
    def _dump(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v).replace('"', '\\"')
        return f"\"{s}\""
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {_dump(v)}")
    lines.append("---\n")
    return "\n".join(lines)

def save_to_obsidian(
    obsidian_sessions_dir: Path,
    session_folder_name: str,
    session_dt: str,
    source_recording: str,
    transcript: str,
    full_report: str,
):
    """
    Создаёт папку сессии в Obsidian и сохраняет 2 Markdown-файла:
      - transcript.md
      - analysis.md (расширенный)
    """
    target_dir = obsidian_sessions_dir / session_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    transcript_md = target_dir / "transcript.md"
    analysis_md = target_dir / "analysis.md"

    transcript_meta = {
        "type": "speaking_session_transcript",
        "date": session_dt,
        "source_recording": source_recording,
        "session_folder": session_folder_name,
    }
    analysis_meta = {
        "type": "speaking_session_analysis_full",
        "date": session_dt,
        "source_recording": source_recording,
        "session_folder": session_folder_name,
    }

    transcript_content = (
        _yaml_frontmatter(transcript_meta)
        + "# Transcript\n\n"
        + (transcript.strip() if transcript else "")
        + "\n"
    )
    analysis_content = (
        _yaml_frontmatter(analysis_meta)
        + "# Analysis (Full)\n\n"
        + (full_report.strip() if full_report else "")
        + "\n"
    )

    transcript_md.write_text(transcript_content, encoding="utf-8")
    analysis_md.write_text(analysis_content, encoding="utf-8")

# ---------- Аудио ----------

def extract_me_and_others(ffmpeg: Path, input_video: Path, out_dir: Path) -> Tuple[Path, Path]:
    me_wav = out_dir / "me.wav"
    others_wav = out_dir / "others.wav"
    _run_ffmpeg([str(ffmpeg), "-y", "-i", str(input_video), "-map", "0:a:1", str(me_wav)])
    _run_ffmpeg([str(ffmpeg), "-y", "-i", str(input_video), "-map", "0:a:0", str(others_wav)])
    return me_wav, others_wav

def clean_me_audio(ffmpeg: Path, me_wav: Path, out_dir: Path) -> Path:
    out = out_dir / "me_clean.wav"
    _run_ffmpeg([
        str(ffmpeg), "-y", "-i", str(me_wav),
        "-af", "highpass=f=120, lowpass=f=6000, afftdn",
        str(out)
    ])
    return out

def normalize_for_api(ffmpeg: Path, input_wav: Path, out_dir: Path) -> Path:
    out = out_dir / "me_clean_16k.wav"
    _run_ffmpeg([
        str(ffmpeg), "-y", "-i", str(input_wav),
        "-ac", "1", "-ar", "16000",
        str(out)
    ])
    return out

def slice_audio(audio_path: Path) -> List[Path]:
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i, chunk in enumerate(audio[::CHUNK_LENGTH_MS]):
        p = audio_path.parent / f"chunk_{i}.mp3"
        chunk.export(p, format="mp3")
        chunks.append(p)
    return chunks

# ---------- Telegram ----------

def trim_for_telegram(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_nl = cut.rfind("\n")
    if last_nl > 300:
        cut = cut[:last_nl]
    return cut.rstrip() + "\n…"

def telegram_send_message(token: str, chat_id: str, text: str, use_markdown: bool):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }
    if use_markdown:
        payload["parse_mode"] = "Markdown"
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error: {r.status_code} {r.text[:300]}")

def get_telegram_settings(config: dict):
    tg = config.get("telegram") or {}
    if not tg.get("enabled"):
        return None
    token = str(tg.get("bot_token", "")).strip()
    chat_id = str(tg.get("chat_id", "")).strip()
    send_md = bool(tg.get("send_analysis_md", True))
    if not token or not chat_id:
        return None
    return {"token": token, "chat_id": chat_id, "send_md": send_md}

# ---------- OpenAI ----------

def transcribe_chunk(api_key: str, chunk_path: Path) -> str:
    with chunk_path.open("rb") as f:
        r = HTTP.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (chunk_path.name, f, "audio/mpeg")},
            data={"model": "gpt-4o-transcribe", "response_format": "text"},
            timeout=(15, 900)
        )
    if r.status_code != 200:
        raise RuntimeError(f"Transcription failed: {r.status_code} {r.text[:300]}")
    return r.text.strip()


def analyze_short(api_key: str, transcript: str, full_report: str) -> str:
    """
    Generates a short Telegram-friendly summary of the FULL analysis.
    Pronunciation is intentionally excluded from the main report; it is handled by a separate module.
    """
    prompt = f"""
You are an English speaking teacher.
You are given:
1) The learner transcript (single speaker).
2) A detailed analysis report (FULL).

Task:
Write a SHORT Telegram-friendly session summary.

Rules:
- Respond only in English.
- Do NOT include any pronunciation evaluation.
- Keep it concise: target 1200–1800 characters, hard max 2200 characters.
- Focus on the most important actionable points.
- Use this exact header line first:
🎯 IELTS Speaking — Session Summary

Output structure:
🎯 IELTS Speaking — Session Summary
• Overall: 1 line
• Fluency: 1–2 bullets
• Lexis: 1–2 bullets
• Grammar: 1–2 bullets
• Next session focus: 2 bullets (very concrete)

FULL REPORT:
{full_report}

TRANSCRIPT:
{transcript}
"""
    log("   -> OpenAI: requesting SHORT analysis...")
    r = HTTP.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=(15, 180)
    )
    log(f"   -> OpenAI: SHORT analysis response {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"AI short analysis failed: {r.status_code} {r.text[:300]}")
    out = r.json()["choices"][0]["message"]["content"]
    return (out or "").strip()


def analyze_full(api_key: str, transcript: str) -> str:
    # UPDATED PROMPT ONLY
    prompt = f"""
You are an English speaking teacher and IELTS-style examiner.
You are given the speech of one speaker only (no dialogue context).
Your task is to evaluate the speaking performance and provide learning-focused feedback for improvement.

Rules:
- Respond only in English.
- Use the IELTS 0–9 band scale (.5 allowed).
- Be concise and Telegram-friendly.
- No theory, no motivation, no generic advice.
- All feedback must be example-based.
- When selecting examples, focus on high-impact weaknesses, not the most frequent ones.

For EACH criterion, do the following:
1) Assign a Band Score (precise IELTS band).
2) Give THREE example-based observations.
   Select three non-trivial, high-impact issues that would noticeably lower the speaking score, even if they occur rarely.

For each example include:
- How the speaker sounded (typical phrasing or pattern)
- Why this significantly reduces the speaking score (clarity, naturalness, control, or level ceiling)
- A better version (how the same idea could sound more natural, precise, or advanced)

Avoid:
- basic learner mistakes
- obvious or beginner-level errors
- surface issues that do not affect the band

Criteria (mandatory)

Fluency and Coherence
Focus on:
- breakdowns in extended turns
- unnatural pacing caused by over-planning
- loss of discourse control (topic drift, weak framing)

Lexical Resource
Focus on:
- imprecise abstraction
- weak collocational control
- vocabulary choices that cap the level at mid-C1

Grammatical Range and Accuracy
Focus on:
- syntactic simplicity that limits expressiveness
- failed or avoided complex structures
- errors that undermine perceived control


Final section (mandatory)

Overall Speaking Band: X.X

Vocabulary Micro-Exercises for the Next Session (max 5 min each):

Exercise 1 (≤5 min):
A short, concrete vocabulary task directly targeting the most damaging issues above.

Exercise 2 (≤5 min):
A different short vocabulary task that prevents recurrence of those high-impact issues.

Exercises must:
- take no more than 5 minutes each
- focus only on vocabulary expansion
- be immediately usable in the next speaking session

Output format (strict)

Fluency & Coherence — Band X.X
1) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
2) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
3) Sounded like: "..."
   Why it hurts: ...
   Better: "..."

Lexical Resource — Band X.X
1) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
2) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
3) Sounded like: "..."
   Why it hurts: ...
   Better: "..."

Grammar — Band X.X
1) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
2) Sounded like: "..."
   Why it hurts: ...
   Better: "..."
3) Sounded like: "..."
   Why it hurts: ...
   Better: "..."

Overall Speaking Band: X.X

Vocabulary Micro-Exercises:
Exercise 1 (≤5 min): ...
Exercise 2 (≤5 min): ...

TRANSCRIPT:
{transcript}
"""
    log("   -> OpenAI: requesting FULL analysis...")
    r = HTTP.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        },
        timeout=(15, 180)
    )
    log(f"   -> OpenAI: FULL analysis response {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"AI full analysis failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]



# ---------- Pronunciation (Acoustic, separate module) ----------

def _read_wav_mono_pcm(path: Path) -> Tuple[int, int, bytes]:
    """Reads WAV and returns (sample_rate, sample_width_bytes, raw_pcm_bytes).

    Supports PCM 8/16/32-bit WAV. If multi-channel, it averages channels to mono.
    """
    import wave
    import audioop

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if n_channels > 1:
        raw = audioop.tomono(raw, sampwidth, 0.5, 0.5)
    return sr, sampwidth, raw


def _db(x: float) -> float:
    import math
    return 20.0 * math.log10(max(x, 1e-12))


def _frame_rms_list(sr: int, sampwidth: int, raw: bytes, frame_ms: int = 30, hop_ms: int = 10) -> List[float]:
    """Computes frame RMS values over raw PCM using audioop."""
    import audioop

    frame_bytes = int(sr * frame_ms / 1000) * sampwidth
    hop_bytes = int(sr * hop_ms / 1000) * sampwidth
    if frame_bytes <= 0 or hop_bytes <= 0 or len(raw) < frame_bytes:
        return []

    rms = []
    for i in range(0, len(raw) - frame_bytes, hop_bytes):
        seg = raw[i:i + frame_bytes]
        r = audioop.rms(seg, sampwidth)  # integer RMS in sample units
        rms.append(float(r) + 1e-12)
    return rms


def _simple_vad_energy(sr: int, sampwidth: int, raw: bytes) -> Tuple[float, float, float]:
    """Returns (speech_seconds, silence_seconds, speech_ratio) using a simple energy-based VAD."""
    import statistics

    rms = _frame_rms_list(sr, sampwidth, raw)
    if not rms:
        total_s = (len(raw) / (sr * sampwidth)) if sr and sampwidth else 0.0
        return 0.0, total_s, 0.0

    rms_db = [_db(v) for v in rms]
    med = statistics.median(rms_db)
    thr = med + 6.0  # adaptive threshold

    speech_frames = sum(1 for v in rms_db if v >= thr)
    total_frames = len(rms_db)
    speech_ratio = speech_frames / total_frames if total_frames else 0.0

    total_s = (len(raw) / (sr * sampwidth)) if sr and sampwidth else 0.0
    speech_s = total_s * speech_ratio
    silence_s = max(total_s - speech_s, 0.0)
    return speech_s, silence_s, speech_ratio


def measure_audio_quality_for_pronunciation(wav_path: Path) -> Dict[str, Any]:
    """Computes objective signal metrics for a pronunciation (acoustic) module.

    No numpy dependency: uses standard library + audioop.
    """
    import audioop
    import statistics

    sr, sampwidth, raw = _read_wav_mono_pcm(wav_path)
    duration_s = (len(raw) / (sr * sampwidth)) if sr and sampwidth else 0.0

    peak = float(audioop.max(raw, sampwidth) or 0.0)
    peak_db = _db(peak)

    rms_int = float(audioop.rms(raw, sampwidth) or 0.0)
    rms_db = _db(rms_int)

    # Clipping estimate (fast sampling): fraction of sampled points near full-scale.
    full_scale = float(2 ** (8 * sampwidth - 1) - 1) if sampwidth else 1.0
    clip_thr = 0.98 * full_scale

    n_samples = len(raw) // sampwidth if sampwidth else 0
    step = max(1, n_samples // 20000)  # ~20k checks max
    clip_hits = 0
    checked = 0
    if n_samples and sampwidth in (1, 2, 4):
        for i in range(0, n_samples, step):
            try:
                s = audioop.getsample(raw, sampwidth, i)
            except Exception:
                break
            checked += 1
            if abs(float(s)) >= clip_thr:
                clip_hits += 1
    clipping_ratio = (clip_hits / checked) if checked else 0.0

    speech_s, silence_s, speech_ratio = _simple_vad_energy(sr, sampwidth, raw)

    frame_rms = _frame_rms_list(sr, sampwidth, raw)
    if frame_rms:
        frame_db = [_db(v) for v in frame_rms]
        loudness_std_db = float(statistics.pstdev(frame_db)) if len(frame_db) > 1 else 0.0

        frame_sorted = sorted(frame_rms)
        n = max(int(0.2 * len(frame_sorted)), 1)
        noise_rms = sum(frame_sorted[:n]) / n
        speech_rms = sum(frame_sorted[-n:]) / n
        snr_db = _db((speech_rms + 1e-12) / (noise_rms + 1e-12))
    else:
        loudness_std_db = 0.0
        snr_db = 0.0

    return {
        "sample_rate_hz": sr,
        "duration_seconds": round(duration_s, 2),
        "peak_dbfs": round(peak_db, 1),
        "rms_dbfs": round(rms_db, 1),
        "clipping_ratio_est": round(float(clipping_ratio), 4),
        "speech_seconds_est": round(float(speech_s), 1),
        "speech_ratio_est": round(float(speech_ratio), 3),
        "snr_db_est": round(float(snr_db), 1),
        "loudness_std_db": round(float(loudness_std_db), 1),
    }


def pronunciation_quality_gate(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Decides whether acoustic feedback is reliable enough."""
    reasons = []
    snr = float(metrics.get("snr_db_est", 0))
    clip = float(metrics.get("clipping_ratio_est", 0))
    speech_s = float(metrics.get("speech_seconds_est", 0))

    if snr < 15:
        reasons.append(f"low SNR ({snr} dB)")
    if clip > 0.01:
        reasons.append(f"clipping too high (~{clip*100:.2f}%)")
    if speech_s < 60:
        reasons.append(f"too little active speech (~{speech_s:.0f}s)")

    return (len(reasons) == 0), reasons


def analyze_pronunciation_acoustic(api_key: str, wav_path: Path, transcript: str) -> str:
    """Generates a separate pronunciation report using acoustic metrics + transcript.

    Standalone module: not part of the IELTS banded report.
    """
    metrics = measure_audio_quality_for_pronunciation(wav_path)
    ok, reasons = pronunciation_quality_gate(metrics)
    gate_line = "PASS" if ok else ("FAIL: " + ", ".join(reasons))

    prompt = f"""
You are an expert English pronunciation coach.

You are given:
1) Objective acoustic quality metrics computed from the learner's voice track.
2) A transcript of what the learner said (approximate; do not treat ASR errors as pronunciation errors).

Your task: produce a short, practical 'Pronunciation (Acoustic) module' report.

Critical constraints:
- Do NOT assign an IELTS pronunciation band.
- Do NOT claim phoneme-by-phoneme correctness. This module is acoustic-light.
- Focus only on delivery factors that are actually observable with audio metrics + transcript: pace, pausing, stability, intelligibility risks, and recording constraints.
- If the quality gate FAILS, say the acoustic diagnosis is not reliable and provide only recording-fix guidance.
- Keep it Telegram-ready and under 1800 characters.

Write in English.

Include exactly these sections:

🔈 Pronunciation (Acoustic)
🧪 Quality gate: {gate_line}
📊 Key metrics: SNR {metrics.get('snr_db_est')} dB | speech {metrics.get('speech_seconds_est')}s/{metrics.get('duration_seconds')}s | clip {metrics.get('clipping_ratio_est')}

If gate PASS:
- 3 bullet points: what to fix next (prosody/pace/pauses), concrete and measurable
- 2 micro-drills (≤3 minutes each)

If gate FAIL:
- 3 bullet points: how to fix the recording to enable diagnosis

Transcript (reference):
{transcript}
"""

    log("   -> OpenAI: requesting FULL analysis...")
    r = HTTP.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=(15, 180)
    )
    log(f"   -> OpenAI: FULL analysis response {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"AI pronunciation module failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def save_pronunciation_to_obsidian(
    obsidian_sessions_dir: Path,
    session_folder_name: str,
    session_dt: str,
    source_recording: str,
    pronunciation_report: str,
):
    """Saves a separate pronunciation module markdown file."""
    target_dir = obsidian_sessions_dir / session_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    p_md = target_dir / "pronunciation_acoustic.md"
    meta = {
        "type": "speaking_session_pronunciation_acoustic",
        "date": session_dt,
        "source_recording": source_recording,
        "session_folder": session_folder_name,
    }
    content = _yaml_frontmatter(meta) + "# Pronunciation (Acoustic)\n\n" + (pronunciation_report.strip() if pronunciation_report else "") + "\n"
    p_md.write_text(content, encoding="utf-8")


def generate_vocab_drills(api_key: str, transcript: str) -> str:
    prompt = f"""
You are a high-end IELTS Speaking coach specializing in lexical development.
You are given a recording or transcript of a real speaking session (the learner speaking with peers).
Your task is to design short, targeted vocabulary drills based strictly on what the learner actually said.

You must identify lexical gaps that occur frequently enough to cap the level, not rare or cosmetic issues.

Core Principles (mandatory)

Work only from the learner’s speech data.
Focus on high-frequency lexical weaknesses, not isolated mistakes.
Target vocabulary that would make the learner sound:
- more precise
- more natural
- more advanced (C1–C2-leaning)

No grammar drills. No pronunciation drills.
All exercises must be ≤5 minutes each.
Respond only in English.
Keep output Telegram-friendly.

TELEGRAM LENGTH (mandatory):
- Entire output MUST be under 2800 characters.
- Prefer 2 drills if 3 would exceed the limit.

5-Minute Lexical Drills 
Create 2–3 micro-drills, each ≤5 minutes, directly tied to the learner’s real speech.

For EACH drill include:
🎯 Drill name

🟦 Input examples (use or lightly paraphrase learner’s wording) -> Target Output



Avoid:
- fill-in-the-blank grammar tasks
- memorisation without context
- textbook-style exercises

TRANSCRIPT:
{transcript}
"""
    log("   -> OpenAI: requesting FULL analysis...")
    r = HTTP.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=(15, 180)
    )
    log(f"   -> OpenAI: FULL analysis response {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"AI vocab drills failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


# ---------- Main ----------

def extract_topics(api_key: str, transcript: str) -> List[str]:
    """
    Extract 3–6 short topics discussed in the session, based strictly on the transcript.
    Returns a list of short noun phrases.
    """
    prompt = f"""
You are given a transcript of an English speaking practice session.

Task:
Extract the main topics that were discussed.

Rules (mandatory):
- Work strictly from the transcript (no guesses).
- Return 3 to 6 topics.
- Each topic must be a short noun phrase (2–6 words).
- No sentences. No punctuation at the end.
- Respond only in English.
- Output format: one topic per line (no numbering, no bullets).

TRANSCRIPT:
{transcript}
"""
    log("   -> OpenAI: requesting FULL analysis...")
    r = HTTP.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=(15, 120)
    )
    if r.status_code != 200:
        raise RuntimeError(f"AI topics extraction failed: {r.status_code} {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"].strip()
    topics = []
    for line in text.splitlines():
        t = line.strip().lstrip("-•* 	").strip()
        if not t:
            continue
        # keep it short/safe
        if len(t) > 60:
            t = t[:60].rstrip()
        topics.append(t)
    # de-dup while preserving order
    seen = set()
    out = []
    for t in topics:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:6]


def main():
    log("Agent started")
    config = load_config()

    obs_dir = Path(config["paths"]["obs_recordings_dir"])
    sess_root = Path(config["paths"]["sessions_dir"])
    ffmpeg = Path(config["paths"]["ffmpeg_path"])
    api_key = config["whisper_api"]["api_key"]

    obsidian_sessions_dir_raw = (config.get("paths") or {}).get("obsidian_sessions_dir")
    obsidian_sessions_dir = Path(obsidian_sessions_dir_raw) if obsidian_sessions_dir_raw else None

    tg = get_telegram_settings(config)

    # Run mode: when built as .exe, default to processing ONE session and exiting (prevents endless looping).
    # You can override via config.local.yaml: run_once: false
    run_once_cfg = (config.get("run_once") if isinstance(config, dict) else None)
    run_once = bool(run_once_cfg) if run_once_cfg is not None else bool(getattr(sys, "frozen", False))
    AudioSegment.converter = str(ffmpeg)

    seen = set()

    log(f"Watching OBS dir: {obs_dir}")
    log(f"Sessions dir: {sess_root}")
    log(f"Obsidian sessions dir: {obsidian_sessions_dir if obsidian_sessions_dir else '(not set)'}")
    log(f"Telegram enabled: {bool(tg)}")

    while True:
        processed_any_in_loop = False
        try:
            files = list(obs_dir.iterdir())
        except Exception as e:
            log(f"ERROR: cannot read OBS dir: {e}")
            pause_and_exit(1)

        for f in files:
            if f.suffix.lower() not in (".mkv", ".mp4") or f in seen:
                continue

            if not is_file_stable(f):
                continue

            # Mark as seen immediately to avoid repeated processing loops
            seen.add(f)
            processed_any_in_loop = True

            # Use the recording file timestamp (mtime) as the factual session date/time
            rec_ts = f.stat().st_mtime
            rec_tm = time.localtime(rec_ts)
            session_folder_name = time.strftime("%Y-%m-%d_%H-%M-%S", rec_tm)
            session_dt = time.strftime("%Y-%m-%d %H:%M:%S", rec_tm)

            seen.add(f)
            log(f"New recording detected: {f.name}")
            s_dir = sess_root / session_folder_name
            s_dir.mkdir(parents=True, exist_ok=True)
            log(f"Session folder created: {s_dir}")

            try:
                log("Step 1/7: extracting me.wav and others.wav")
                me_wav, _others_wav = extract_me_and_others(ffmpeg, f, s_dir)

                log("Step 2/7: cleaning me.wav -> me_clean.wav")
                clean = clean_me_audio(ffmpeg, me_wav, s_dir)

                log("Step 3/7: normalizing for API (mono 16k)")
                norm = normalize_for_api(ffmpeg, clean, s_dir)

                log("Step 4/7: chunking audio")
                chunks = slice_audio(norm)
                log(f"Chunks created: {len(chunks)}")

                log("Step 5/7: transcribing chunks")
                texts: List[str] = []
                for i, c in enumerate(chunks, start=1):
                    log(f"  Transcribe chunk {i}/{len(chunks)}: {c.name}")
                    texts.append(transcribe_chunk(api_key, c))
                    try:
                        c.unlink(missing_ok=True)
                    except Exception:
                        pass

                transcript = "\n\n".join(texts).strip()
                transcript_path = s_dir / "transcript_me.txt"
                transcript_path.write_text(transcript, encoding="utf-8")
                log(f"Transcript saved: {transcript_path.name} ({len(transcript)} chars)")

                log("Step 6/7: generating FULL analysis (file)")
                full_report = analyze_full(api_key, transcript)
                # Add factual session date (from recording timestamp) into the report text
                full_report = f"Session date: {session_dt}\n\n" + full_report.strip() + "\n"
                full_path = s_dir / "ai_analysis_report_full.txt"
                full_path.write_text(full_report, encoding="utf-8")
                log(f"Full analysis saved: {full_path.name} ({len(full_report)} chars)")

                # ---- сохранение в Obsidian ----
                if obsidian_sessions_dir:
                    try:
                        log("Saving transcript + full analysis to Obsidian (Markdown)")
                        save_to_obsidian(
                            obsidian_sessions_dir=obsidian_sessions_dir,
                            session_folder_name=session_folder_name,
                            session_dt=session_dt,
                            source_recording=f.name,
                            transcript=transcript,
                            full_report=full_report,
                        )
                        log("Saved to Obsidian")
                    except Exception as e:
                        log(f"ERROR: Obsidian save failed: {e}")
                else:
                    log("Obsidian sessions dir not set; skipping Obsidian save")
                # -------------------------------

                log("Step 7/7: generating SHORT analysis (Telegram)")
                short_report = analyze_short(api_key, transcript, full_report)
                # Add factual session date into the SHORT report (Telegram)
                try:
                    short_report = short_report.replace(
                        "🎯 IELTS Speaking — Session Summary\n",
                        f"🎯 IELTS Speaking — Session Summary\n📅 Session date: {session_dt}\n",
                        1,
                    )
                except Exception:
                    pass

                # Extract session topics and append to the end of the SHORT report
                topics_block = ""
                try:
                    topics = extract_topics(api_key, transcript)
                    if topics:
                        topics_lines = "\n".join([f"• {t}" for t in topics])
                        topics_block = "\n\n🗂 Topics discussed:\n" + topics_lines
                except Exception as e:
                    log(f"WARNING: topics extraction failed: {e}")

                if topics_block:
                    short_report = short_report.rstrip() + topics_block + "\n"

                if tg:
                    msg = trim_for_telegram(short_report, TELEGRAM_SAFE_LIMIT)
                    log(f"Sending Telegram message (analysis) ({len(msg)} chars)")
                    telegram_send_message(
                        tg["token"],
                        tg["chat_id"],
                        msg,
                        use_markdown=tg["send_md"]
                    )
                    log("Telegram analysis message sent")
                else:
                    log("Telegram disabled or not configured; skipping analysis send")

                # ---- separate pronunciation (AUDIO) module ----
                try:
                    log("Generating Pronunciation (Audio) module")
                    pron_report = analyze_pronunciation_acoustic(api_key, norm, transcript)

                    # Save to Obsidian (separate file)
                    if obsidian_sessions_dir:
                        try:
                            save_pronunciation_to_obsidian(
                                obsidian_sessions_dir=obsidian_sessions_dir,
                                session_folder_name=session_folder_name,
                                session_dt=session_dt,
                                source_recording=f.name,
                                pronunciation_report=pron_report,
                            )
                            log("Pronunciation module saved to Obsidian")
                        except Exception as e:
                            log(f"WARNING: pronunciation Obsidian save failed: {e}")

                    # Send as a separate Telegram message
                    if tg:
                        pmsg = trim_for_telegram(pron_report, TELEGRAM_SAFE_LIMIT)
                        log(f"Sending Telegram message (pronunciation) ({len(pmsg)} chars)")
                        telegram_send_message(
                            tg["token"],
                            tg["chat_id"],
                            pmsg,
                            use_markdown=tg["send_md"]
                        )
                        log("Telegram pronunciation message sent")
                except Exception as e:
                    log(f"WARNING: pronunciation module failed: {e}")
# ---- добавлено: упражнения в Telegram ----
                if tg:
                    try:
                        log("Generating vocabulary drills (Telegram)")
                        drills = generate_vocab_drills(api_key, transcript)
                        drills_msg = trim_for_telegram(drills, TELEGRAM_SAFE_LIMIT)
                        log(f"Sending Telegram message (drills) ({len(drills_msg)} chars)")
                        telegram_send_message(
                            tg["token"],
                            tg["chat_id"],
                            drills_msg,
                            use_markdown=tg["send_md"]
                        )
                        log("Telegram drills message sent")
                    except Exception as e:
                        log(f"ERROR: drills generation/sending failed: {e}")
                # -----------------------------------------

                log("Session completed")



            except Exception as e:
                log(f"ERROR during processing {f.name}: {e}")
                if run_once:
                    log("Run-once mode: exiting after error")
                    pause_and_exit(1)

        if run_once and not processed_any_in_loop:
            log("Run-once mode: no more new stable recordings; exiting")
            pause_and_exit(0)

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
