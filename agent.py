import sys
import time
import subprocess
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yaml
import requests
from pydub import AudioSegment


# ---------- Константы ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 300 * 1000  # 5 минут
TELEGRAM_SAFE_LIMIT = 3500  # запас к лимиту Telegram ~4096

# Weekly scheduler
LISBON_TZ = ZoneInfo("Europe/Lisbon")
WEEKLY_RUN_HOUR = 10
WEEKLY_RUN_MINUTE = 0
WEEKLY_POLL_SECONDS = 60
WEEKLY_TELEGRAM_LIMIT = 2800

# Persistent processed index filename (stored in sessions_dir root)
PROCESSED_INDEX_FILENAME = ".processed_recordings.json"

# Telegram inbound polling (for check-in replies)
TG_UPDATES_POLL_SECONDS = 5
TG_UPDATES_TIMEOUT_SECONDS = 25
TG_UPDATES_STATE_FILENAME = ".telegram_updates_state.json"
TG_CHECKIN_STATE_FILENAME = ".telegram_checkins.json"


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

def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        try:
            return p.read_text(encoding="utf-8-sig")
        except Exception:
            return ""

def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------- Persistent processed index (prevents re-processing on restart) ----------

def _processed_index_path(sess_root: Path) -> Path:
    return sess_root / PROCESSED_INDEX_FILENAME

def load_processed_index(sess_root: Path) -> Dict[str, Any]:
    p = _processed_index_path(sess_root)
    if not p.exists():
        return {"version": 1, "items": {}}
    try:
        data = json.loads(_read_text_safe(p) or "{}")
        if not isinstance(data, dict):
            return {"version": 1, "items": {}}
        if "items" not in data or not isinstance(data["items"], dict):
            data["items"] = {}
        if "version" not in data:
            data["version"] = 1
        return data
    except Exception:
        return {"version": 1, "items": {}}

def save_processed_index(sess_root: Path, data: Dict[str, Any]) -> None:
    _atomic_write_json(_processed_index_path(sess_root), data)

def make_recording_key(f: Path) -> str:
    """
    Stable key for a recording file to avoid re-processing across restarts.
    Uses absolute path + size + mtime.
    """
    try:
        st = f.stat()
        return f"{str(f.resolve())}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        return str(f)

def is_recording_already_processed(
    f: Path,
    sess_root: Path,
    processed_index: Dict[str, Any],
    session_folder_name: str,
) -> bool:
    key = make_recording_key(f)
    if key in (processed_index.get("items") or {}):
        return True

    # Fallback safety: if session folder already contains full report, consider processed
    s_dir = sess_root / session_folder_name
    if (s_dir / "ai_analysis_report_full.txt").exists():
        return True

    return False

def mark_recording_processed(
    f: Path,
    processed_index: Dict[str, Any],
    session_folder_name: str,
) -> None:
    key = make_recording_key(f)
    items = processed_index.setdefault("items", {})
    items[key] = {
        "session_folder": session_folder_name,
        "marked_at": datetime.now(tz=LISBON_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------- Obsidian save ----------

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
      - analysis.md (full)
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

def save_topics_to_obsidian(
    obsidian_sessions_dir: Path,
    session_folder_name: str,
    session_dt: str,
    source_recording: str,
    topics: List[str],
):
    """
    Сохраняет topics в Obsidian в отдельном файле в папке сессии.
    """
    target_dir = obsidian_sessions_dir / session_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    topics_md = target_dir / "topics.md"
    meta = {
        "type": "speaking_session_topics",
        "date": session_dt,
        "source_recording": source_recording,
        "session_folder": session_folder_name,
    }
    lines = ["# Topics discussed", ""]
    for t in topics:
        lines.append(f"- {t}")
    content = _yaml_frontmatter(meta) + "\n".join(lines).strip() + "\n"
    topics_md.write_text(content, encoding="utf-8")

def save_checkin_to_obsidian(
    obsidian_sessions_dir: Path,
    session_folder_name: str,
    session_dt: str,
    source_recording: str,
    prompt_message_id: int,
    reply_message_id: int,
    reply_text: str,
):
    """
    Сохраняет ответ на чек-ин (recovery/mood) в Obsidian в отдельный файл checkin.md.
    ВАЖНО: оригинальный voice НЕ сохраняется, только транскрипт/текст.
    """
    target_dir = obsidian_sessions_dir / session_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    p = target_dir / "checkin.md"
    meta = {
        "type": "speaking_session_checkin",
        "date": session_dt,
        "source_recording": source_recording,
        "session_folder": session_folder_name,
        "telegram_prompt_message_id": prompt_message_id,
        "telegram_reply_message_id": reply_message_id,
        "saved_at": datetime.now(tz=LISBON_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }

    body_lines = [
        "# Post-session check-in",
        "",
        "## Response",
        (reply_text.strip() if reply_text else "").strip(),
        "",
    ]
    content = _yaml_frontmatter(meta) + "\n".join(body_lines).strip() + "\n"
    p.write_text(content, encoding="utf-8")


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

def telegram_send_message(token: str, chat_id: str, text: str, use_markdown: bool) -> int:
    """
    Returns message_id (callers may ignore).
    """
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
    data = r.json()
    return int(data["result"]["message_id"])

def telegram_get_updates(token: str, offset: int) -> Tuple[List[dict], int]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {
        "timeout": TG_UPDATES_TIMEOUT_SECONDS,
        "offset": offset,
        "allowed_updates": ["message"],
    }
    r = requests.get(url, params=params, timeout=TG_UPDATES_TIMEOUT_SECONDS + 10)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram getUpdates error: {r.status_code} {r.text[:300]}")
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getUpdates not ok: {str(data)[:300]}")
    updates = data.get("result") or []
    if updates:
        next_offset = max(int(u["update_id"]) for u in updates) + 1
    else:
        next_offset = offset
    return updates, next_offset

def telegram_get_file_path(token: str, file_id: str) -> str:
    url = f"https://api.telegram.org/bot{token}/getFile"
    r = requests.get(url, params={"file_id": file_id}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram getFile error: {r.status_code} {r.text[:300]}")
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getFile not ok: {str(data)[:300]}")
    return str(data["result"]["file_path"])

def telegram_download_file(token: str, file_path: str, dest: Path) -> None:
    url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Telegram download file error: {r.status_code} {r.text[:200]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)

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


# ---------- Telegram inbound state (check-in) ----------

def _tg_updates_state_path(sess_root: Path) -> Path:
    return sess_root / TG_UPDATES_STATE_FILENAME

def load_tg_updates_state(sess_root: Path) -> Dict[str, Any]:
    p = _tg_updates_state_path(sess_root)
    if not p.exists():
        return {"version": 1, "offset": 0}
    try:
        data = json.loads(_read_text_safe(p) or "{}")
        if not isinstance(data, dict):
            return {"version": 1, "offset": 0}
        if "offset" not in data:
            data["offset"] = 0
        return data
    except Exception:
        return {"version": 1, "offset": 0}

def save_tg_updates_state(sess_root: Path, state: Dict[str, Any]) -> None:
    _atomic_write_json(_tg_updates_state_path(sess_root), state)

def _tg_checkins_state_path(sess_root: Path) -> Path:
    return sess_root / TG_CHECKIN_STATE_FILENAME

def load_tg_checkins(sess_root: Path) -> Dict[str, Any]:
    p = _tg_checkins_state_path(sess_root)
    if not p.exists():
        return {"version": 1, "pending": {}}
    try:
        data = json.loads(_read_text_safe(p) or "{}")
        if not isinstance(data, dict):
            return {"version": 1, "pending": {}}
        if "pending" not in data or not isinstance(data["pending"], dict):
            data["pending"] = {}
        return data
    except Exception:
        return {"version": 1, "pending": {}}

def save_tg_checkins(sess_root: Path, data: Dict[str, Any]) -> None:
    _atomic_write_json(_tg_checkins_state_path(sess_root), data)

def register_pending_checkin(
    sess_root: Path,
    checkins_state: Dict[str, Any],
    prompt_message_id: int,
    session_folder_name: str,
    session_dt: str,
    source_recording: str,
) -> None:
    pending = checkins_state.setdefault("pending", {})
    pending[str(prompt_message_id)] = {
        "session_folder": session_folder_name,
        "session_dt": session_dt,
        "source_recording": source_recording,
        "created_at": datetime.now(tz=LISBON_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_tg_checkins(sess_root, checkins_state)

def pop_pending_checkin(
    sess_root: Path,
    checkins_state: Dict[str, Any],
    prompt_message_id: int,
) -> Optional[Dict[str, Any]]:
    pending = checkins_state.get("pending") or {}
    item = pending.pop(str(prompt_message_id), None)
    if item is not None:
        save_tg_checkins(sess_root, checkins_state)
    return item


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

def transcribe_audio_file(api_key: str, audio_path: Path, mime: str) -> str:
    """
    For Telegram voice (.ogg) etc. Returns plain text.
    """
    with audio_path.open("rb") as f:
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, f, mime)},
            data={"model": "gpt-4o-transcribe", "response_format": "text"},
            timeout=900
        )
    if r.status_code != 200:
        raise RuntimeError(f"Voice transcription failed: {r.status_code} {r.text[:300]}")
    return r.text.strip()

def analyze_full(api_key: str, transcript: str) -> str:
    # Pronunciation REMOVED per requirement
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
    # Pronunciation REMOVED per requirement (no pronunciation section)
    prompt = f"""
You are an English speaking teacher and IELTS-style examiner.

You must produce a SHORT Telegram-ready report that is a condensed, fully consistent version of the FULL REPORT below.

Rules (mandatory):
- Respond only in English.
- Use the SAME band scores and overall band as in the FULL REPORT.
- Emoji-structured and easy to scan.
- No theory, no motivation, no generic advice.
- Example-based only.
- Do NOT invent new issues. Do NOT contradict the FULL REPORT.
- TELEGRAM LENGTH: the entire response MUST be under 2800 characters.

Selection rules:
- Pick only high-impact weaknesses (not the most frequent).
- Keep examples short (<= 12 words) and improved versions short (<= 18 words).

Special requirements:
- Lexical Resource MUST include 3 incorrect/weak examples and 3 better versions.
- Grammar MUST include 3 incorrect/weak examples and 3 better versions.
- Fluency: 1 example + 1 better version.

Format strictly as follows:

🎯 IELTS Speaking — Session Summary

🗣 Fluency & Coherence — Band X.X
Issue: …
Example: “...”
Better: “...”

📚 Lexical Resource — Band X.X
Issue: …
Examples:
1) “...” → “...”
2) “...” → “...”
3) “...” → “...”

🧠 Grammar — Band X.X
Issue: …
Examples:
1) “...” → “...”
2) “...” → “...”
3) “...” → “...”

⭐ Overall Band: X.X

🧩 Vocabulary Focus for Next Session:
• …
• …

FULL REPORT:
{full_report}

TRANSCRIPT (reference only):
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
        raise RuntimeError(f"AI vocab drills failed: {r.status_code} {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

def extract_topics(api_key: str, transcript: str) -> List[str]:
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
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        },
        timeout=120
    )
    if r.status_code != 200:
        raise RuntimeError(f"AI topics extraction failed: {r.status_code} {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"].strip()
    topics = []
    for line in text.splitlines():
        t = line.strip().lstrip("-•* \t").strip()
        if not t:
            continue
        if len(t) > 60:
            t = t[:60].rstrip()
        topics.append(t)
    seen = set()
    out = []
    for t in topics:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:6]


# ---------- Weekly ----------

_SESSION_DATE_RE = re.compile(r"^Session date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*$", re.MULTILINE)

def _parse_session_dt_from_full_report(text: str) -> Optional[datetime]:
    m = _SESSION_DATE_RE.search(text or "")
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=LISBON_TZ)
    except Exception:
        return None

def _week_id(dt: datetime) -> str:
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"

def _week_bounds_for_anchor(dt: datetime) -> Tuple[datetime, datetime]:
    d = dt.astimezone(LISBON_TZ).date()
    monday = d - timedelta(days=d.weekday())
    week_start = datetime(monday.year, monday.month, monday.day, 0, 0, 0, tzinfo=LISBON_TZ)
    sunday = monday + timedelta(days=6)
    week_end = datetime(sunday.year, sunday.month, sunday.day, 23, 59, 59, tzinfo=LISBON_TZ)
    return week_start, week_end

def _last_due_week_anchor(now: datetime) -> Optional[datetime]:
    now = now.astimezone(LISBON_TZ)
    d = now.date()
    sunday = d + timedelta(days=(6 - d.weekday()))
    candidate = datetime(sunday.year, sunday.month, sunday.day, WEEKLY_RUN_HOUR, WEEKLY_RUN_MINUTE, 0, tzinfo=LISBON_TZ)
    if candidate > now:
        prev_sunday = sunday - timedelta(days=7)
        candidate = datetime(prev_sunday.year, prev_sunday.month, prev_sunday.day, WEEKLY_RUN_HOUR, WEEKLY_RUN_MINUTE, 0, tzinfo=LISBON_TZ)
    return candidate

def _weekly_obsidian_paths(obsidian_sessions_dir: Path, week_id: str) -> Tuple[Path, Path]:
    weekly_dir = obsidian_sessions_dir / "Weekly" / week_id
    weekly_dir.mkdir(parents=True, exist_ok=True)
    return weekly_dir / "weekly.md", weekly_dir / "weekly_telegram.txt"

def _collect_full_reports_for_range(sess_root: Path, start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, Path, str]]:
    out: List[Tuple[datetime, Path, str]] = []
    try:
        for sdir in sorted(sess_root.iterdir()):
            if not sdir.is_dir():
                continue
            rp = sdir / "ai_analysis_report_full.txt"
            if not rp.exists():
                continue
            txt = _read_text_safe(rp)
            sdt = _parse_session_dt_from_full_report(txt)
            if not sdt:
                continue
            if start_dt <= sdt <= end_dt:
                out.append((sdt, rp, txt))
    except Exception:
        return []
    out.sort(key=lambda x: x[0])
    return out

def _build_weekly_prompt_variant2_vertical(
    daily_this_week: List[Tuple[datetime, Path, str]],
    daily_prev_week: List[Tuple[datetime, Path, str]],
    week_id: str,
    week_start: datetime,
    week_end: datetime,
) -> str:
    """
    Variant 2: no table. Daily bands must be listed as bullet lines.
    Weekly prompt enriched with emojis (requirement).
    """
    def _pack(items: List[Tuple[datetime, Path, str]]) -> str:
        blocks = []
        for dt, _path, txt in items:
            day = dt.strftime("%Y-%m-%d")
            blocks.append(f"DAY: {day}\nREPORT:\n{txt.strip()}\n")
        return "\n\n".join(blocks).strip()

    this_block = _pack(daily_this_week) if daily_this_week else ""
    prev_block = _pack(daily_prev_week) if daily_prev_week else ""

    prompt = f"""
PROMPT — Weekly Speaking Trend Report

You are an IELTS speaking analyst.
You are given daily speaking session reports for one learner over one week.
Each daily report includes band scores and identified lexical issues.

Your task is to produce a weekly trend report that shows direction of progress, not coaching.

Rules
Respond only in English
No teaching, no advice, no exercises
Only trend analysis
Be factual and concise

EXTRA REQUIREMENT (mandatory)
- Enrich the Weekly report with relevant emojis to improve scanability.
- Emojis must be consistent and not excessive.
- Use emojis in BOTH outputs (Telegram + Detailed).

Step 1: Aggregate Weekly Metrics
For each criterion:
Fluency & Coherence
Lexical Resource
Grammatical Range & Accuracy
Calculate:
weekly average band
comparison vs previous week (↑ / → / ↓)
If there is NO previous-week data, use "n/a" instead of arrows.

Step 2: Lexical Trend Analysis
Identify:
Top recurring lexical gaps of the week (based on frequency across daily sessions)
Persistent lexical blockers (patterns repeating across multiple weeks)
Resolved or reduced lexical issues (previously frequent, now rare or absent)
Focus only on high-impact lexical patterns that affect speaking level.

Step 3: Produce Two Outputs
A. Telegram Summary (ultra-short)
Include:
weekly average band per criterion
trend direction (↑ → ↓ or n/a)
2–3 key lexical trend bullets
No explanations.

B. Detailed Weekly Report (for Obsidian)
Include:
- Daily bands (NO TABLES). Use bullet lines exactly like:
  - YYYY-MM-DD — F&C X.X | LR X.X | GRA X.X | Overall X.X
- Weekly averages
- Week-to-week comparison
- Lexical trends with brief explanations
- Short historical notes if patterns persist

OUTPUT FORMAT (STRICT)

===TELEGRAM===
<telegram summary, MUST be <= {WEEKLY_TELEGRAM_LIMIT} characters>

===DETAILED===
<detailed markdown for Obsidian>

WEEK META:
week_id: {week_id}
week_start: {week_start.strftime("%Y-%m-%d")}
week_end: {week_end.strftime("%Y-%m-%d")}

THIS WEEK DAILY REPORTS:
{this_block}

PREVIOUS WEEK DAILY REPORTS (for comparison, may be empty):
{prev_block if prev_block else "(no previous-week baseline)"}
"""
    return prompt.strip()

def _weekly_generate(
    api_key: str,
    daily_this_week: List[Tuple[datetime, Path, str]],
    daily_prev_week: List[Tuple[datetime, Path, str]],
    week_id: str,
    week_start: datetime,
    week_end: datetime,
) -> Tuple[str, str]:
    prompt = _build_weekly_prompt_variant2_vertical(daily_this_week, daily_prev_week, week_id, week_start, week_end)
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
        raise RuntimeError(f"AI weekly analysis failed: {r.status_code} {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"]

    def _extract(section: str) -> str:
        m = re.search(rf"^===\s*{re.escape(section)}\s*===\s*$", text, flags=re.MULTILINE)
        if not m:
            return ""
        start = m.end()
        m2 = re.search(r"^===\s*(TELEGRAM|DETAILED)\s*===\s*$", text[start:], flags=re.MULTILINE)
        if m2:
            return text[start:start + m2.start()].strip()
        return text[start:].strip()

    telegram = _extract("TELEGRAM")
    detailed = _extract("DETAILED")

    if not telegram or not detailed:
        detailed = text.strip()
        telegram = ""

    return telegram, detailed

def _update_frontmatter_flag(md_path: Path, key: str, value: str) -> None:
    raw = _read_text_safe(md_path)
    if not raw.startswith("---"):
        raw = "---\n" + f"{key}: {value}\n" + "---\n\n" + raw
        md_path.write_text(raw, encoding="utf-8")
        return

    end = raw.find("\n---", 3)
    if end == -1:
        raw = "---\n" + f"{key}: {value}\n" + "---\n\n" + raw
        md_path.write_text(raw, encoding="utf-8")
        return

    fm = raw[0:end+4]
    body = raw[end+4:]
    if re.search(rf"(?m)^{re.escape(key)}\s*:\s*.*$", fm):
        fm2 = re.sub(rf"(?m)^{re.escape(key)}\s*:\s*.*$", f"{key}: {value}", fm)
    else:
        fm_lines = fm.splitlines()
        fm_lines.insert(-1, f"{key}: {value}")
        fm2 = "\n".join(fm_lines)
    md_path.write_text(fm2 + body, encoding="utf-8")

def _weekly_save_to_obsidian(
    obsidian_sessions_dir: Path,
    week_id: str,
    week_start: datetime,
    week_end: datetime,
    detailed_md: str,
    telegram_text: str,
    telegram_sent: bool,
    source_sessions_count: int,
) -> Tuple[Path, Path]:
    weekly_md_path, weekly_tg_path = _weekly_obsidian_paths(obsidian_sessions_dir, week_id)

    meta = {
        "type": "weekly_speaking_trend_report",
        "week": week_id,
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end": week_end.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(tz=LISBON_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "telegram_sent": telegram_sent,
        "source_sessions": source_sessions_count,
    }
    content = _yaml_frontmatter(meta) + detailed_md.strip() + "\n"
    weekly_md_path.write_text(content, encoding="utf-8")

    if telegram_text:
        weekly_tg_path.write_text(telegram_text.strip() + "\n", encoding="utf-8")

    return weekly_md_path, weekly_tg_path

def _weekly_is_done(obsidian_sessions_dir: Path, week_id: str) -> Tuple[bool, bool]:
    weekly_md_path, _weekly_tg_path = _weekly_obsidian_paths(obsidian_sessions_dir, week_id)
    if not weekly_md_path.exists():
        return (False, False)
    raw = _read_text_safe(weekly_md_path)
    m = re.search(r"(?m)^telegram_sent\s*:\s*(true|false)\s*$", raw)
    if not m:
        return (True, False)
    return (True, m.group(1) == "true")

def weekly_tick(
    *,
    now: datetime,
    api_key: str,
    sess_root: Path,
    obsidian_sessions_dir: Optional[Path],
    tg: Optional[Dict[str, Any]],
) -> None:
    """
    - Runs due weeks (Sunday 10:00 Lisbon) that were missed.
    - Uses Obsidian weekly.md as source-of-truth.
    - Does NOT generate weekly if there are 0 daily sessions for that week.
    - If weekly exists but telegram_sent=false, retries Telegram send (no regeneration).
    """
    if not obsidian_sessions_dir:
        return

    now = now.astimezone(LISBON_TZ)
    last_due = _last_due_week_anchor(now)
    if not last_due:
        return

    # look back max 12 weeks
    start_anchor = last_due - timedelta(weeks=12)

    anchors: List[datetime] = []
    a = start_anchor
    d = a.date()
    sunday = d + timedelta(days=(6 - d.weekday()))
    a = datetime(sunday.year, sunday.month, sunday.day, WEEKLY_RUN_HOUR, WEEKLY_RUN_MINUTE, 0, tzinfo=LISBON_TZ)
    if a > start_anchor:
        a -= timedelta(weeks=1)
    while a <= last_due:
        anchors.append(a)
        a += timedelta(weeks=1)

    for anchor in anchors:
        week_start, week_end = _week_bounds_for_anchor(anchor)
        week_id = _week_id(anchor)

        exists, tg_sent = _weekly_is_done(obsidian_sessions_dir, week_id)
        if exists and tg_sent:
            continue

        # Retry Telegram only
        if exists and (not tg_sent) and tg:
            weekly_md_path, weekly_tg_path = _weekly_obsidian_paths(obsidian_sessions_dir, week_id)
            msg = _read_text_safe(weekly_tg_path).strip()
            if msg:
                try:
                    log(f"Weekly {week_id}: retrying Telegram send")
                    telegram_send_message(
                        tg["token"], tg["chat_id"],
                        trim_for_telegram(msg, TELEGRAM_SAFE_LIMIT),
                        use_markdown=tg["send_md"]
                    )
                    _update_frontmatter_flag(weekly_md_path, "telegram_sent", "true")
                    log(f"Weekly {week_id}: Telegram sent (retry) and marked telegram_sent=true")
                except Exception as e:
                    log(f"ERROR: Weekly {week_id}: Telegram retry failed: {e}")
            continue

        daily_this = _collect_full_reports_for_range(sess_root, week_start, week_end)
        if len(daily_this) == 0:
            continue

        try:
            prev_start = week_start - timedelta(days=7)
            prev_end = week_end - timedelta(days=7)
            daily_prev = _collect_full_reports_for_range(sess_root, prev_start, prev_end)

            log(f"Weekly {week_id}: generating (sessions this week: {len(daily_this)})")
            telegram_text, detailed_md = _weekly_generate(api_key, daily_this, daily_prev, week_id, week_start, week_end)

            telegram_text = telegram_text.strip()
            if telegram_text:
                telegram_text = trim_for_telegram(telegram_text, WEEKLY_TELEGRAM_LIMIT)

            weekly_md_path, _weekly_tg_path = _weekly_save_to_obsidian(
                obsidian_sessions_dir=obsidian_sessions_dir,
                week_id=week_id,
                week_start=week_start,
                week_end=week_end,
                detailed_md=detailed_md,
                telegram_text=telegram_text,
                telegram_sent=False,
                source_sessions_count=len(daily_this),
            )
            log(f"Weekly {week_id}: saved to Obsidian: {weekly_md_path}")

            if tg and telegram_text:
                try:
                    log(f"Weekly {week_id}: sending Telegram summary")
                    telegram_send_message(
                        tg["token"], tg["chat_id"],
                        trim_for_telegram(telegram_text, TELEGRAM_SAFE_LIMIT),
                        use_markdown=tg["send_md"]
                    )
                    _update_frontmatter_flag(weekly_md_path, "telegram_sent", "true")
                    log(f"Weekly {week_id}: Telegram sent and marked telegram_sent=true")
                except Exception as e:
                    log(f"ERROR: Weekly {week_id}: Telegram send failed (will retry next run): {e}")

        except Exception as e:
            log(f"ERROR: Weekly {week_id}: generation failed: {e}")


# ---------- Post-session check-in (Recovery + Mood) ----------

def send_post_session_checkin(tg: Dict[str, Any], session_dt: str) -> int:
    text = (
        "🧘 Post-session check-in\n\n"
        "1) Recovery (0–10)?\n"
        "2) Mood (0–10 or 1–3 words)?\n\n"
        "Reply to this message (text or voice)."
    )
    return telegram_send_message(tg["token"], tg["chat_id"], text, use_markdown=tg["send_md"])

def telegram_poll_and_process_checkins(
    *,
    tg: Optional[Dict[str, Any]],
    api_key: str,
    sess_root: Path,
    obsidian_sessions_dir: Optional[Path],
    checkins_state: Dict[str, Any],
    updates_state: Dict[str, Any],
) -> None:
    """
    Polls Telegram updates and saves replies (text or voice) to the correct session folder in Obsidian.

    Matching rule:
      - user reply must be a reply_to_message of our check-in prompt message_id

    IMPORTANT:
      - original voice is NOT saved
      - voice is downloaded to a temp file, transcribed, and then deleted
      - only the transcript is saved to Obsidian
    """
    if not tg or not obsidian_sessions_dir:
        return

    offset = int(updates_state.get("offset", 0))

    try:
        updates, next_offset = telegram_get_updates(tg["token"], offset)
    except Exception as e:
        log(f"WARNING: Telegram polling failed: {e}")
        return

    if next_offset != offset:
        updates_state["offset"] = next_offset
        save_tg_updates_state(sess_root, updates_state)

    if not updates:
        return

    pending = checkins_state.get("pending") or {}

    for upd in updates:
        msg = (upd.get("message") or {})
        if not msg:
            continue

        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != str(tg["chat_id"]):
            continue

        reply_to = msg.get("reply_to_message") or {}
        if not reply_to:
            continue

        prompt_mid = reply_to.get("message_id")
        if not prompt_mid:
            continue

        if str(prompt_mid) not in pending:
            continue

        item = pop_pending_checkin(sess_root, checkins_state, int(prompt_mid))
        if not item:
            continue

        session_folder = item["session_folder"]
        session_dt = item["session_dt"]
        source_recording = item["source_recording"]

        reply_mid = int(msg.get("message_id", 0))
        reply_text = str(msg.get("text") or "").strip()

        voice_transcript = ""

        voice = msg.get("voice")
        if voice and isinstance(voice, dict) and voice.get("file_id"):
            try:
                file_id = str(voice["file_id"])
                file_path = telegram_get_file_path(tg["token"], file_id)

                tmp_dir = sess_root / ".tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_voice = tmp_dir / f"checkin_{prompt_mid}_{reply_mid}.ogg"

                telegram_download_file(tg["token"], file_path, tmp_voice)

                try:
                    voice_transcript = transcribe_audio_file(api_key, tmp_voice, mime="audio/ogg")
                finally:
                    try:
                        tmp_voice.unlink(missing_ok=True)
                    except Exception:
                        pass

            except Exception as e:
                log(f"WARNING: failed to transcribe voice check-in: {e}")
                voice_transcript = ""

        final_text = reply_text
        if voice_transcript:
            if final_text:
                final_text = final_text.rstrip() + "\n\n---\n\n📝 Voice transcript:\n" + voice_transcript
            else:
                final_text = "📝 Voice transcript:\n" + voice_transcript

        try:
            save_checkin_to_obsidian(
                obsidian_sessions_dir=obsidian_sessions_dir,
                session_folder_name=session_folder,
                session_dt=session_dt,
                source_recording=source_recording,
                prompt_message_id=int(prompt_mid),
                reply_message_id=reply_mid,
                reply_text=final_text,
            )
            log(f"Check-in saved to Obsidian for session {session_folder}")
        except Exception as e:
            log(f"ERROR: failed to save check-in to Obsidian: {e}")


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

    log(f"Watching OBS dir: {obs_dir}")
    log(f"Sessions dir: {sess_root}")
    log(f"Obsidian sessions dir: {obsidian_sessions_dir if obsidian_sessions_dir else '(not set)'}")
    log(f"Telegram enabled: {bool(tg)}")

    processed_index = load_processed_index(sess_root)

    last_weekly_check = 0.0

    tg_updates_state = load_tg_updates_state(sess_root)
    tg_checkins_state = load_tg_checkins(sess_root)
    last_tg_poll = 0.0

    while True:
        processed_any_session_this_loop = False

        try:
            files = list(obs_dir.iterdir())
        except Exception as e:
            log(f"ERROR: cannot read OBS dir: {e}")
            pause_and_exit(1)

        for f in files:
            if f.suffix.lower() not in (".mkv", ".mp4"):
                continue

            if not is_file_stable(f):
                continue

            rec_ts = f.stat().st_mtime
            rec_tm = time.localtime(rec_ts)
            session_folder_name = time.strftime("%Y-%m-%d_%H-%M-%S", rec_tm)
            session_dt = time.strftime("%Y-%m-%d %H:%M:%S", rec_tm)

            if is_recording_already_processed(f, sess_root, processed_index, session_folder_name):
                continue

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
                full_report = f"Session date: {session_dt}\n\n" + full_report.strip() + "\n"
                full_path = s_dir / "ai_analysis_report_full.txt"
                full_path.write_text(full_report, encoding="utf-8")
                log(f"Full analysis saved: {full_path.name} ({len(full_report)} chars)")

                topics: List[str] = []
                try:
                    topics = extract_topics(api_key, transcript)
                except Exception as e:
                    log(f"WARNING: topics extraction failed: {e}")
                    topics = []

                if topics:
                    try:
                        (s_dir / "topics.txt").write_text("\n".join(topics).strip() + "\n", encoding="utf-8")
                    except Exception:
                        pass

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
                        if topics:
                            log("Saving topics to Obsidian (separate file)")
                            save_topics_to_obsidian(
                                obsidian_sessions_dir=obsidian_sessions_dir,
                                session_folder_name=session_folder_name,
                                session_dt=session_dt,
                                source_recording=f.name,
                                topics=topics,
                            )
                        log("Saved to Obsidian")
                    except Exception as e:
                        log(f"ERROR: Obsidian save failed: {e}")
                else:
                    log("Obsidian sessions dir not set; skipping Obsidian save")

                log("Step 7/7: generating SHORT analysis (Telegram)")
                short_report = analyze_short(api_key, transcript, full_report)
                try:
                    short_report = short_report.replace(
                        "🎯 IELTS Speaking — Session Summary\n",
                        f"🎯 IELTS Speaking — Session Summary\n📅 Session date: {session_dt}\n",
                        1,
                    )
                except Exception:
                    pass

                if topics:
                    topics_lines = "\n".join([f"• {t}" for t in topics])
                    topics_block = "\n\n🗂 Topics discussed:\n" + topics_lines
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

                # NEW: post-session check-in prompt
                if tg:
                    try:
                        log("Sending post-session check-in (Recovery + Mood)")
                        prompt_mid = send_post_session_checkin(tg, session_dt=session_dt)
                        register_pending_checkin(
                            sess_root=sess_root,
                            checkins_state=tg_checkins_state,
                            prompt_message_id=prompt_mid,
                            session_folder_name=session_folder_name,
                            session_dt=session_dt,
                            source_recording=f.name,
                        )
                        log(f"Check-in prompt sent, message_id={prompt_mid}")
                    except Exception as e:
                        log(f"ERROR: failed to send check-in prompt: {e}")

                mark_recording_processed(f, processed_index, session_folder_name)
                save_processed_index(sess_root, processed_index)

                processed_any_session_this_loop = True
                log("Session completed")

            except Exception as e:
                log(f"ERROR during processing {f.name}: {e}")

        # ---- Weekly tick AFTER processing sessions ----
        try:
            now_ts = time.time()
            if now_ts - last_weekly_check >= WEEKLY_POLL_SECONDS:
                last_weekly_check = now_ts
                weekly_tick(
                    now=datetime.now(tz=LISBON_TZ),
                    api_key=api_key,
                    sess_root=sess_root,
                    obsidian_sessions_dir=obsidian_sessions_dir,
                    tg=tg,
                )
        except Exception as e:
            log(f"ERROR: weekly_tick failed unexpectedly: {e}")
        # ------------------------------------------------

        # ---- Telegram inbound polling for check-ins ----
        try:
            now_ts = time.time()
            if now_ts - last_tg_poll >= TG_UPDATES_POLL_SECONDS:
                last_tg_poll = now_ts
                telegram_poll_and_process_checkins(
                    tg=tg,
                    api_key=api_key,
                    sess_root=sess_root,
                    obsidian_sessions_dir=obsidian_sessions_dir,
                    checkins_state=tg_checkins_state,
                    updates_state=tg_updates_state,
                )
        except Exception as e:
            log(f"WARNING: telegram inbound polling failed: {e}")
        # ------------------------------------------------

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
