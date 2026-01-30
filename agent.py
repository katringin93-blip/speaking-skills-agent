import sys
import time
import shutil
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import yaml
import requests
import httpx

CHECK_INTERVAL_SECONDS = 2
STABLE_SECONDS_DEFAULT = 10
STABLE_POLL_INTERVAL = 1

PRIMARY_CONFIDENCE_MIN_SHARE = 0.15
MAX_TRANSCRIPT_CHARS_FOR_ANALYSIS = 14000
TELEGRAM_MAX_MESSAGE_LEN = 3900  # safer than 4096


# ---------- helpers ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Press ENTER to close this window...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)


def is_file_stable(path: Path, stable_seconds: int) -> bool:
    if not path.exists():
        return False
    last_size = path.stat().st_size
    stable_for = 0
    while stable_for < stable_seconds:
        time.sleep(STABLE_POLL_INTERVAL)
        if not path.exists():
            return False
        size = path.stat().st_size
        if size != last_size:
            last_size = size
            stable_for = 0
        else:
            stable_for += STABLE_POLL_INTERVAL
    return True


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def load_config() -> dict:
    cfg_path = app_base_dir() / "config.local.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def get_required_str(data: dict, dotted_key: str) -> str:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise ValueError(f"Missing required config key: {dotted_key}")
        cur = cur[p]
    if not isinstance(cur, str) or not cur.strip():
        raise ValueError(f"Config key must be non-empty string: {dotted_key}")
    return cur.strip()


def get_optional_str(data: dict, dotted_key: str, default: str) -> str:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if not isinstance(cur, str) or not cur.strip():
        return default
    return cur.strip()


def get_optional_int(data: dict, dotted_key: str, default: int) -> int:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    try:
        return int(cur)
    except Exception:
        return default


def get_optional_bool(data: dict, dotted_key: str, default: bool) -> bool:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if isinstance(cur, bool):
        return cur
    if isinstance(cur, str):
        return cur.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def resolve_ffmpeg(config: dict) -> Path:
    try:
        p = Path(get_required_str(config, "paths.ffmpeg_path"))
        if p.exists():
            return p
    except Exception:
        pass

    local = app_base_dir() / "ffmpeg.exe"
    if local.exists():
        return local

    which = shutil.which("ffmpeg")
    if which:
        return Path(which)

    raise FileNotFoundError("ffmpeg not found. Put ffmpeg.exe next to the exe or set paths.ffmpeg_path.")


def make_session_dir(root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_audio(ffmpeg: Path, input_video: Path, output_audio: Path):
    """Извлекает звук в MP3. Битрейт 48k позволяет вместить 1 час в 25 МБ."""
    output_path = output_audio.with_suffix('.mp3')
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(input_video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "48k",
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")
    return output_path


def read_text_safe(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def clip_text(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return "…(clipped)…\n" + s[-max_chars:]


# ---------- OpenAI diarized transcription ----------

def openai_transcribe_diarized(api_key: str, audio_path: Path) -> dict:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"[TRANSCRIBE] -> POST {audio_path.name} ({size_mb:.1f} MB)")

    # Исправлено на server_vad согласно требованиям API для длинных файлов
    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": json.dumps({"type": "server_vad"})
    }

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "audio/mpeg")}
        timeout = httpx.Timeout(connect=30.0, read=900.0, write=900.0, pool=30.0)
        
        with httpx.Client(timeout=timeout) as client:
            t_req = time.time()
            r = client.post(url, headers=headers, files=files, data=data)

    print(f"[TRANSCRIBE] <- status={r.status_code} in {time.time()-t_req:.1f}s")
    
    if r.status_code == 200:
        return r.json()
    else:
        raise RuntimeError(f"Transcription error {r.status_code}: {r.text}")


def _seg_speaker(seg: dict) -> str:
    return seg.get("speaker") or seg.get("speaker_label") or seg.get("speaker_id") or "UNKNOWN"


def _seg_times(seg: dict) -> Tuple[float, float]:
    try:
        start = float(seg.get("start", 0.0))
    except Exception:
        start = 0.0
    try:
        end = float(seg.get("end", start))
    except Exception:
        end = start
    if end < start:
        end = start
    return start, end


def diarized_json_to_text(diarized: dict) -> str:
    segments = diarized.get("segments") or []
    lines = []
    for s in segments:
        start, end = _seg_times(s)
        speaker = _seg_speaker(s)
        text = (s.get("text") or "").strip()
        lines.append(f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}")
    return "\n".join(lines).strip()


# ---------- primary speaker ----------

def compute_speaker_stats(diarized: dict) -> dict:
    segments = diarized.get("segments") or []
    stats = {}
    for s in segments:
        spk = _seg_speaker(s)
        start, end = _seg_times(s)
        dur = max(0.0, end - start)
        if spk not in stats:
            stats[spk] = {"duration_seconds": 0.0, "segments": 0}
        stats[spk]["duration_seconds"] += dur
        stats[spk]["segments"] += 1
    total = sum(v["duration_seconds"] for v in stats.values()) or 0.0
    return {"by_speaker": stats, "total_duration_seconds": total}


def choose_primary_speaker(stats: dict) -> dict:
    by = stats.get("by_speaker") or {}
    items = [(spk, float(v.get("duration_seconds", 0.0)), int(v.get("segments", 0))) for spk, v in by.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        return {"primary_speaker": "UNKNOWN", "confidence": "none", "reason": "no_segments"}

    top = items[0]
    second = items[1] if len(items) > 1 else ("NONE", 0.0, 0)

    total = float(stats.get("total_duration_seconds", 0.0)) or 0.0
    diff = top[1] - second[1]
    diff_share = (diff / total) if total > 0 else 0.0

    confidence = "high" if diff_share >= PRIMARY_CONFIDENCE_MIN_SHARE else "low"

    return {
        "primary_speaker": top[0],
        "confidence": confidence,
        "top_duration_seconds": round(top[1], 3),
        "second_duration_seconds": round(second[1], 3),
        "total_duration_seconds": round(total, 3),
        "diff_share": round(diff_share, 4),
        "ranking": [{"speaker": spk, "duration_seconds": round(dur, 3), "segments": segs} for spk, dur, segs in items],
    }


def write_primary_and_context(diarized: dict, primary_label: str, out_primary: Path, out_context: Path):
    segments = diarized.get("segments") or []
    primary_lines = []
    context_lines = []
    for s in segments:
        start, end = _seg_times(s)
        spk = _seg_speaker(s)
        text = (s.get("text") or "").strip()
        line = f"[{start:0.2f}-{end:0.2f}] {spk}: {text}"
        if spk == primary_label:
            primary_lines.append(line)
        else:
            context_lines.append(line)

    out_primary.write_text("\n".join(primary_lines).strip(), encoding="utf-8")
    out_context.write_text("\n".join(context_lines).strip(), encoding="utf-8")


# ---------- OpenAI text analysis ----------

def openai_responses_analyze(api_key: str, model: str, diarized_text: str, primary_text: str) -> dict:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"""
You are an English-speaking coach. Analyze the conversation transcript.
Focus ONLY on PRIMARY for mistakes. Use DIARIZED for context.
Output MUST be valid JSON only.

JSON schema:
{{
  "session_summary": "...",
  "topics": ["..."],
  "primary_speaker_coaching": {{
     "strengths": ["..."],
     "top_issues": [ {{"category": "...", "example": "...", "better_version": "...", "why": "..."}} ],
     "next_session_focus": ["..."]
  }},
  "drills": [ {{"name": "...", "goal": "...", "instructions": ["..."], "examples": ["..."]}} ]
}}

DIARIZED:
{diarized_text}

PRIMARY:
{primary_text}
""".strip()

    payload = {"model": model, "input": prompt}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Analysis error {r.status_code}: {r.text}")

    data = r.json()
    out_text = ""
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and c.get("text"):
                out_text += c.get("text")

    try:
        return json.loads(out_text.strip())
    except Exception:
        return {"_raw_output_text": out_text}


def analysis_json_to_md(analysis: dict) -> str:
    if "_raw_output_text" in analysis:
        return "# Analysis (raw)\n\n" + analysis["_raw_output_text"]
    lines = ["# Session analysis\n", "## Summary", analysis.get("session_summary", ""), "\n## Topics"]
    for t in (analysis.get("topics") or []): lines.append(f"- {t}")
    coach = analysis.get("primary_speaker_coaching") or {}
    lines.append("\n## Strengths")
    for s in (coach.get("strengths") or []): lines.append(f"- {s}")
    lines.append("\n## Top issues")
    for it in (coach.get("top_issues") or []):
        lines.append(f"- **{it.get('category','')}**\n  - Ex: {it.get('example','')}\n  - Better: {it.get('better_version','')}")
    return "\n".join(lines).strip()


def build_telegram_message(session_dir: Path, analysis: dict) -> str:
    summary = analysis.get("session_summary", "") if isinstance(analysis, dict) else ""
    return f"AI Analysis ready\n\nSummary: {summary}\n\nFolder: {session_dir}"


# ---------- Telegram ----------

def telegram_send_message(bot_token: str, chat_id: int, text: str):
    requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=30)

def telegram_send_document(bot_token: str, chat_id: int, file_path: Path, caption: str = ""):
    with file_path.open("rb") as f:
        requests.post(f"https://api.telegram.org/bot{bot_token}/sendDocument", data={"chat_id": chat_id, "caption": caption}, files={"document": f}, timeout=60)


# ---------- main ----------

def main():
    print("SpeakingSkillsAgent: started")
    config = load_config()
    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    api_key = get_required_str(config, "whisper_api.api_key")
    ffmpeg = resolve_ffmpeg(config)

    seen = set()
    while True:
        if obs_dir.exists():
            for f in obs_dir.iterdir():
                if f.suffix.lower() not in (".mkv", ".mp4") or f in seen: continue
                if not is_file_stable(f, 10): continue

                session_dir = make_session_dir(sessions_root)
                input_copy = session_dir / f"input{f.suffix}"
                shutil.copy2(f, input_copy)

                print(f"[AUDIO] extracting MP3...")
                mp3_path = extract_audio(ffmpeg, input_copy, session_dir / "audio")

                print("[TRANSCRIBE] diarized transcription...")
                diarized = openai_transcribe_diarized(api_key, mp3_path)
                
                (session_dir / "transcript_diarized.json").write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")
                (session_dir / "transcript_diarized.txt").write_text(diarized_json_to_text(diarized), encoding="utf-8")

                stats = compute_speaker_stats(diarized)
                choice = choose_primary_speaker(stats)
                primary_label = choice.get("primary_speaker", "UNKNOWN")
                write_primary_and_context(diarized, primary_label, session_dir / "transcript_primary.txt", session_dir / "transcript_context.txt")

                if get_optional_bool(config, "analysis.enabled", True):
                    analysis = openai_responses_analyze(api_key, get_optional_str(config, "analysis.model", "gpt-4o-mini"), 
                                                       clip_text(read_text_safe(session_dir / "transcript_diarized.txt"), 14000),
                                                       clip_text(read_text_safe(session_dir / "transcript_primary.txt"), 14000))
                    (session_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
                    (session_dir / "analysis.md").write_text(analysis_json_to_md(analysis), encoding="utf-8")

                    if get_optional_bool(config, "telegram.enabled", False):
                        token = get_optional_str(config, "telegram.bot_token", "")
                        cid = get_optional_int(config, "telegram.chat_id", 0)
                        telegram_send_message(token, cid, build_telegram_message(session_dir, analysis))
                        telegram_send_document(token, cid, session_dir / "analysis.md")

                print(f"[DONE] session: {session_dir}")
                seen.add(f)
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        pause_and_exit(1)
