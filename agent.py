import sys
import time
import shutil
import subprocess
import json
import hashlib
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- OpenAI Audio Transcriptions API ----------

def transcribe_verbose_json_whisper1(api_key: str, wav_path: Path, language: str) -> dict:
    """
    Whisper-1 with segment timestamps.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        "file": (wav_path.name, wav_path.open("rb"), "audio/wav"),
    }

    data = {
        "model": "whisper-1",
        "language": language,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

    return r.json()


def transcribe_diarized_json(api_key: str, wav_path: Path, language: str) -> dict:
    """
    gpt-4o-transcribe-diarize with diarized_json (speaker labels + segments).
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        "file": (wav_path.name, wav_path.open("rb"), "audio/wav"),
    }

    data = {
        "model": "gpt-4o-transcribe-diarize",
        "language": language,
        "response_format": "diarized_json",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

    return r.json()


def format_segments_verbose_to_text(verbose: dict) -> str:
    """
    [00.00-03.21] text
    """
    segments = verbose.get("segments") or []
    lines = []
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = (s.get("text") or "").strip()
        if text:
            lines.append(f"[{start:0.2f}-{end:0.2f}] {text}")
    return "\n".join(lines).strip()


def format_segments_diarized_to_text(diarized: dict) -> str:
    """
    [00.00-03.21] Speaker A: text
    """
    segments = diarized.get("segments") or []
    lines = []
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        speaker = (s.get("speaker") or "Speaker ?").strip()
        text = (s.get("text") or "").strip()
        if text:
            lines.append(f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}")
    return "\n".join(lines).strip()


# ---------- main ----------

def main():
    config = load_config()

    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    stable_seconds = get_optional_int(config, "processing.stable_seconds", STABLE_SECONDS_DEFAULT)

    # Один ключ на оба запроса (raw + diarized)
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

                raw_txt_path = session_dir / "transcript_raw.txt"
                raw_json_path = session_dir / "transcript_raw.json"

                diar_txt_path = session_dir / "transcript_diarized.txt"
                diar_json_path = session_dir / "transcript_diarized.json"

                manifest_path = session_dir / "session_manifest.json"

                print(f"[COPY] -> {input_copy}")
                shutil.copy2(f, input_copy)

                print(f"[AUDIO] extracting -> {wav_path}")
                extract_audio(ffmpeg, input_copy, wav_path)

                # hashes for serialization / traceability
                input_sha = sha256_file(input_copy)
                wav_sha = sha256_file(wav_path)

                # --- Raw transcript (immutable source-of-truth) ---
                print("[ASR:RAW] whisper-1 verbose_json (segments + timestamps)...")
                raw_verbose = transcribe_verbose_json_whisper1(api_key, wav_path, language)

                raw_json_path.write_text(
                    json.dumps(raw_verbose, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                raw_txt_path.write_text(format_segments_verbose_to_text(raw_verbose), encoding="utf-8")

                # --- Diarized transcript (speaker labels) ---
                print("[ASR:DIARIZED] gpt-4o-transcribe-diarize diarized_json (speaker labels)...")
                diarized = transcribe_diarized_json(api_key, wav_path, language)

                diar_json_path.write_text(
                    json.dumps(diarized, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                diar_txt_path.write_text(format_segments_diarized_to_text(diarized), encoding="utf-8")

                # --- Session manifest (serialization) ---
                manifest = {
                    "session_id": session_dir.name,
                    "created_at_local": session_dir.name,  # same as folder timestamp
                    "source": {
                        "obs_filename": f.name,
                        "obs_full_path": str(f),
                        "input_mkv": str(input_copy),
                        "audio_wav": str(wav_path),
                        "sha256": {
                            "input_mkv": input_sha,
                            "audio_wav": wav_sha
                        }
                    },
                    "config": {
                        "language": language
                    },
                    "asr": {
                        "raw": {
                            "model": "whisper-1",
                            "response_format": "verbose_json",
                            "files": {
                                "json": raw_json_path.name,
                                "txt": raw_txt_path.name
                            }
                        },
                        "diarized": {
                            "model": "gpt-4o-transcribe-diarize",
                            "response_format": "diarized_json",
                            "files": {
                                "json": diar_json_path.name,
                                "txt": diar_txt_path.name
                            }
                        }
                    },
                    "artifacts": [
                        input_copy.name,
                        wav_path.name,
                        raw_json_path.name,
                        raw_txt_path.name,
                        diar_json_path.name,
                        diar_txt_path.name
                    ]
                }

                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                print(f"[DONE] session -> {session_dir}")
                print(f"       saved -> {raw_txt_path.name}, {raw_json_path.name}")
                print(f"       saved -> {diar_txt_path.name}, {diar_json_path.name}")
                print(f"       saved -> {manifest_path.name}")
                print("-----")

                seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
