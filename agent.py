import sys
import time
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import yaml
import requests
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
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (chunk_path.name, f, "audio/mpeg")},
            data={"model": "gpt-4o-transcribe", "response_format": "text"},
            timeout=900
        )
    if r.status_code != 200:
        raise RuntimeError(f"Transcription failed: {r.status_code} {r.text[:300]}")
    return r.text.strip()

def analyze_full(api_key: str, transcript: str) -> str:
    prompt = f"""
You are an English speaking teacher and IELTS-style evaluator.
You are given the speech of one speaker only (no dialogue context).
Your task is to assess the speaking performance and give actionable learning feedback.

Rules:

Respond only in English.

Use the IELTS 0–9 band scale (.5 allowed).

Be concise and Telegram-friendly.

Focus on practical improvement, not theory.

Show examples of mistakes AND better versions.

For EACH criterion, provide:

Band score

Main issue to work on (1 line, learning-focused)

Example from the speech (typical error or weakness)

Improved version (how it could sound better)

Criteria (mandatory)

Fluency and Coherence
Focus on:

unnecessary pauses or fillers

speaking too cautiously or too fast

weak logical flow in longer answers

Lexical Resource
Focus on:

repetitive or “safe” vocabulary

vague wording

incorrect or unnatural collocations

Grammatical Range and Accuracy
Focus on:

overuse of simple sentence patterns

frequent small errors (tenses, articles, word order)

missed chances to use more complex structures

Pronunciation
Focus on:

sounds or stress that reduce clarity

flat or unnatural intonation

clarity at normal speaking speed

Final section (mandatory)

Overall Speaking Band: X.X
Main focus for next sessions (1–2 points):

…

…

Output format (strict, compact)

Fluency & Coherence — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Lexical Resource — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Grammar — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Pronunciation — Band X.X
Main issue: …
Example: …
Better target: …

Overall Band: X.X
Next focus: …

Transcript:
{transcript}
"""
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        },
        timeout=180
    )
    if r.status_code != 200:
        raise RuntimeError(f"AI full analysis failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

def analyze_short(api_key: str, transcript: str, full_report: str) -> str:
    prompt = f"""
You are an English speaking teacher and IELTS-style evaluator.
You are given the speech of one speaker only (no dialogue context).
Your task is to assess the speaking performance and give actionable learning feedback.

Rules:

Respond only in English.

Use the IELTS 0–9 band scale (.5 allowed).

Be concise and Telegram-friendly.

Focus on practical improvement, not theory.

Show examples of mistakes AND better versions.

TELEGRAM CONSTRAINTS (mandatory):
- The entire response MUST be under 2800 characters.
- Keep each "Example" short (<= 12 words) and each "Better" short (<= 18 words).
- Prefer one strong example per criterion.

CONSISTENCY CONSTRAINT (mandatory):
- Your band scores, overall band, and "Next focus" MUST be consistent with the FULL REPORT below.
- Do not contradict the FULL REPORT.

Output format (strict, compact)

Fluency & Coherence — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Lexical Resource — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Grammar — Band X.X
Main issue: …
Example: “… ”
Better: “… ”

Pronunciation — Band X.X
Main issue: …
Example: …
Better target: …

Overall Band: X.X
Next focus: …

FULL REPORT:
{full_report}

Transcript (for reference):
{transcript}
"""
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=180
    )
    if r.status_code != 200:
        raise RuntimeError(f"AI short analysis failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

# ---------- Main ----------

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
    AudioSegment.converter = str(ffmpeg)

    seen = set()

    log(f"Watching OBS dir: {obs_dir}")
    log(f"Sessions dir: {sess_root}")
    log(f"Obsidian sessions dir: {obsidian_sessions_dir if obsidian_sessions_dir else '(not set)'}")
    log(f"Telegram enabled: {bool(tg)}")

    while True:
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

            session_folder_name = time.strftime("%Y-%m-%d_%H-%M-%S")
            session_dt = time.strftime("%Y-%m-%d %H:%M:%S")

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
                full_path = s_dir / "ai_analysis_report_full.txt"
                full_path.write_text(full_report, encoding="utf-8")
                log(f"Full analysis saved: {full_path.name} ({len(full_report)} chars)")

                # ---- Добавлено: сохранение в Obsidian ----
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
                # ----------------------------------------

                log("Step 7/7: generating SHORT analysis (Telegram)")
                short_report = analyze_short(api_key, transcript, full_report)

                if tg:
                    msg = trim_for_telegram(short_report, TELEGRAM_SAFE_LIMIT)
                    log(f"Sending Telegram message ({len(msg)} chars)")
                    telegram_send_message(
                        tg["token"],
                        tg["chat_id"],
                        msg,
                        use_markdown=tg["send_md"]
                    )
                    log("Telegram message sent")
                else:
                    log("Telegram disabled or not configured; skipping send")

                seen.add(f)
                log("Session completed")

            except Exception as e:
                log(f"ERROR during processing {f.name}: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
