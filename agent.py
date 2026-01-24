import sys
import time
import shutil
import subprocess
import json
from pathlib import Path

import yaml
import requests


CHECK_INTERVAL_SECONDS = 2
STABLE_SECONDS_DEFAULT = 10
STABLE_POLL_INTERVAL = 1


# ---------- helpers ----------

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


def extract_audio(ffmpeg: Path, input_video: Path, output_wav: Path):
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(input_video),
        "-ac", "1",
        "-ar", "16000",
        str(output_wav)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr}")


# ---------- Whisper API ----------

def whisper_transcribe_verbose_json(api_key: str, wav_path: Path, language: str) -> dict:
    """
    OpenAI Speech-to-Text (Whisper) API with segment timestamps.
    We request verbose_json + timestamp_granularities=segment.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        "file": (wav_path.name, wav_path.open("rb"), "audio/wav"),
    }

    # timestamp_granularities requires verbose_json :contentReference[oaicite:1]{index=1}
    data = {
        "model": "whisper-1",
        "language": language,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Whisper API error {r.status_code}: {r.text}")

    return r.json()


def format_segments_to_text(verbose: dict) -> str:
    """
    Make a readable transcript with timestamps from segments.
    Example line:
    [00:12.34-00:18.90] text...
    """
    segments = verbose.get("segments") or []
    lines = []
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = (s.get("text") or "").strip()
        lines.append(f"[{start:0.2f}-{end:0.2f}] {text}")
    return "\n".join(lines).strip()


# ---------- main ----------

def main():
    config = load_config()

    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    stable_seconds = get_optional_int(config, "processing.stable_seconds", STABLE_SECONDS_DEFAULT)

    api_key = get_required_str(config, "whisper_api.api_key")
    language = get_required_str(config, "whisper_api.language")

    ffmpeg = resolve_ffmpeg(config)

    print("SpeakingSkillsAgent: started")
    print(f"Base dir: {app_base_dir()}")
    print(f"Watching OBS: {obs_dir}")
    print(f"Sessions: {sessions_root}")
    print(f"Stable seconds: {stable_seconds}")
    print("-----")

    seen = set()

    while True:
        if obs_dir.exists():
            for f in obs_dir.iterdir():
                if f.suffix.lower() != ".mkv" or f in seen:
                    continue

                print(f"[FOUND] {f.name} -> waiting to stabilize...")
                if not is_file_stable(f, stable_seconds):
                    print(f"[SKIP] {f.name} not stable")
                    seen.add(f)
                    continue

                session_dir = make_session_dir(sessions_root)
                input_copy = session_dir / "input.mkv"
                wav_path = session_dir / "audio.wav"
                transcript_txt_path = session_dir / "transcript_raw.txt"
                transcript_json_path = session_dir / "transcript_raw.json"

                print(f"[COPY] -> {input_copy}")
                shutil.copy2(f, input_copy)

                print(f"[AUDIO] extracting -> {wav_path}")
                extract_audio(ffmpeg, input_copy, wav_path)

                print("[WHISPER] transcribing via API (segments + timestamps)...")
                verbose = whisper_transcribe_verbose_json(api_key, wav_path, language)

                # Save raw verbose json (source of truth)
                transcript_json_path.write_text(
                    json.dumps(verbose, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                # Save readable text with timestamps
                transcript_txt_path.write_text(format_segments_to_text(verbose), encoding="utf-8")

                print(f"[DONE] saved -> {transcript_txt_path.name} and {transcript_json_path.name}")
                print("-----")

                seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
