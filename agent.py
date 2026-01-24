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
PRIMARY_CONFIDENCE_MIN_SHARE = 0.15  # if top speaker is not ahead by this share, mark low confidence


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


# ---------- OpenAI Speech-to-Text (diarized) ----------

def openai_transcribe_diarized(api_key: str, wav_path: Path) -> dict:
    """
    Uses OpenAI Audio Transcriptions endpoint with diarization-capable model.
    Returns diarized_json (segments with speaker labels).
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    files = {
        "file": (wav_path.name, wav_path.open("rb"), "audio/wav"),
    }

    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Transcribe (diarize) API error {r.status_code}: {r.text}")
    return r.json()


def _segment_speaker_label(seg: dict) -> str:
    return seg.get("speaker") or seg.get("speaker_label") or seg.get("speaker_id") or "UNKNOWN"


def _segment_times(seg: dict) -> tuple[float, float]:
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
        start, end = _segment_times(s)
        speaker = _segment_speaker_label(s)
        text = (s.get("text") or "").strip()
        lines.append(f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}")
    return "\n".join(lines).strip()


# ---------- Primary speaker selection ----------

def compute_speaker_stats(diarized: dict) -> dict:
    segments = diarized.get("segments") or []
    stats = {}  # speaker -> {"duration": float, "segments": int}
    for s in segments:
        speaker = _segment_speaker_label(s)
        start, end = _segment_times(s)
        dur = max(0.0, end - start)
        if speaker not in stats:
            stats[speaker] = {"duration_seconds": 0.0, "segments": 0}
        stats[speaker]["duration_seconds"] += dur
        stats[speaker]["segments"] += 1
    total = sum(v["duration_seconds"] for v in stats.values()) or 0.0
    return {"by_speaker": stats, "total_duration_seconds": total}


def choose_primary_speaker(stats: dict) -> dict:
    by_speaker = stats.get("by_speaker") or {}
    items = []
    for spk, v in by_speaker.items():
        items.append((spk, float(v.get("duration_seconds", 0.0)), int(v.get("segments", 0))))
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        return {"primary_speaker": "UNKNOWN", "confidence": "none", "reason": "no_segments"}

    top = items[0]
    second = items[1] if len(items) > 1 else ("NONE", 0.0, 0)

    top_dur = top[1]
    second_dur = second[1]
    total = float(stats.get("total_duration_seconds", 0.0)) or 0.0

    # Simple confidence: top speaker should be clearly ahead of second speaker by share of total
    diff = top_dur - second_dur
    diff_share = (diff / total) if total > 0 else 0.0

    if diff_share >= PRIMARY_CONFIDENCE_MIN_SHARE:
        conf = "high"
    else:
        conf = "low"

    return {
        "primary_speaker": top[0],
        "confidence": conf,
        "top_duration_seconds": round(top_dur, 3),
        "second_duration_seconds": round(second_dur, 3),
        "total_duration_seconds": round(total, 3),
        "diff_share": round(diff_share, 4),
        "ranking": [{"speaker": spk, "duration_seconds": round(dur, 3), "segments": segs} for spk, dur, segs in items],
    }


def write_primary_and_context(diarized: dict, primary_label: str, out_primary: Path, out_context: Path):
    segments = diarized.get("segments") or []
    primary_lines = []
    context_lines = []
    for s in segments:
        start, end = _segment_times(s)
        speaker = _segment_speaker_label(s)
        text = (s.get("text") or "").strip()
        line = f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}"
        if speaker == primary_label:
            primary_lines.append(line)
        else:
            context_lines.append(line)

    out_primary.write_text("\n".join(primary_lines).strip(), encoding="utf-8")
    out_context.write_text("\n".join(context_lines).strip(), encoding="utf-8")


# ---------- main ----------

def main():
    config = load_config()

    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    stable_seconds = get_optional_int(config, "processing.stable_seconds", STABLE_SECONDS_DEFAULT)

    api_key = get_required_str(config, "whisper_api.api_key")
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

                diar_json_path = session_dir / "transcript_diarized.json"
                diar_txt_path = session_dir / "transcript_diarized.txt"

                stats_path = session_dir / "speaker_stats.json"
                primary_txt = session_dir / "transcript_primary.txt"
                context_txt = session_dir / "transcript_context.txt"

                print(f"[COPY] -> {input_copy}")
                shutil.copy2(f, input_copy)

                print(f"[AUDIO] extracting -> {wav_path}")
                extract_audio(ffmpeg, input_copy, wav_path)

                print("[TRANSCRIBE] diarized transcription via API...")
                diarized = openai_transcribe_diarized(api_key, wav_path)

                diar_json_path.write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")
                diar_txt_path.write_text(diarized_json_to_text(diarized), encoding="utf-8")

                stats = compute_speaker_stats(diarized)
                choice = choose_primary_speaker(stats)

                stats_out = {
                    "primary_selection": choice,
                    "stats": stats,
                }
                stats_path.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")

                primary_label = choice.get("primary_speaker", "UNKNOWN")
                write_primary_and_context(diarized, primary_label, primary_txt, context_txt)

                print(f"[PRIMARY] {primary_label} (confidence: {choice.get('confidence')}, diff_share: {choice.get('diff_share')})")
                print(f"[DONE] saved -> {diar_txt_path.name}, {stats_path.name}, {primary_txt.name}, {context_txt.name}")
                print("-----")

                seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
