import sys
import time
import shutil
import subprocess
from pathlib import Path

import yaml


CHECK_INTERVAL_SECONDS = 2
STABLE_SECONDS_DEFAULT = 8
STABLE_POLL_INTERVAL = 1


def is_file_stable(path: Path, stable_seconds: int, poll_interval: int = STABLE_POLL_INTERVAL) -> bool:
    """
    File is considered stable if its size doesn't change for stable_seconds.
    This helps avoid processing while OBS is still finalizing the recording.
    """
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


def app_base_dir() -> Path:
    # For PyInstaller-built exe: use the folder where the exe lives.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def load_config_path() -> Path:
    return app_base_dir() / "config.local.yaml"


def read_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return data


def get_required_str(data: dict, dotted_key: str) -> str:
    parts = dotted_key.split(".")
    cur = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise ValueError(f"Missing required config key: {dotted_key}")
        cur = cur[p]
    if not isinstance(cur, str) or not cur.strip():
        raise ValueError(f"Config key must be a non-empty string: {dotted_key}")
    return cur.strip()


def get_optional_int(data: dict, dotted_key: str, default: int) -> int:
    parts = dotted_key.split(".")
    cur = data
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    try:
        return int(cur)
    except Exception:
        return default


def resolve_ffmpeg_path(config: dict) -> Path:
    # 1) config.paths.ffmpeg_path (optional)
    ffmpeg_path_str = None
    try:
        ffmpeg_path_str = get_required_str(config, "paths.ffmpeg_path")
    except Exception:
        ffmpeg_path_str = None

    if ffmpeg_path_str:
        p = Path(ffmpeg_path_str)
        if p.exists():
            return p

    # 2) ffmpeg.exe next to exe
    local = app_base_dir() / "ffmpeg.exe"
    if local.exists():
        return local

    # 3) ffmpeg in PATH
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)

    raise FileNotFoundError(
        "ffmpeg not found. Put ffmpeg.exe next to SpeakingSkillsAgent.exe "
        "OR set paths.ffmpeg_path in config.local.yaml."
    )


def make_session_dir(sessions_root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = sessions_root / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def extract_audio(ffmpeg: Path, input_video: Path, output_wav: Path) -> None:
    # 16kHz mono WAV is a good default for ASR
    cmd = [
        str(ffmpeg),
        "-y",
        "-i", str(input_video),
        "-ac", "1",
        "-ar", "16000",
        str(output_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{r.stderr}"
        )


def main():
    cfg_path = load_config_path()
    config = read_config(cfg_path)

    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    stable_seconds = get_optional_int(config, "processing.stable_seconds", STABLE_SECONDS_DEFAULT)

    ffmpeg = resolve_ffmpeg_path(config)

    print("SpeakingSkillsAgent: started")
    print(f"Base dir: {app_base_dir()}")
    print(f"Config: {cfg_path}")
    print(f"Watching OBS folder: {obs_dir}")
    print(f"Sessions dir: {sessions_root}")
    print(f"Stable seconds: {stable_seconds}")
    print(f"ffmpeg: {ffmpeg}")
    print("-----")

    seen = set()

    while True:
        if obs_dir.exists():
            for f in obs_dir.iterdir():
                if f.suffix.lower() != ".mkv":
                    continue
                if f in seen:
                    continue

                print(f"[FOUND] {f.name} -> waiting for file to stabilize...")
                ok = is_file_stable(f, stable_seconds=stable_seconds)
                if not ok:
                    print(f"[SKIP] {f.name} disappeared before it stabilized")
                    seen.add(f)
                    continue

                print(f"[READY] {f.name} stabilized. Creating session folder...")
                session_dir = make_session_dir(sessions_root)

                input_copy = session_dir / "input.mkv"
                wav_path = session_dir / "audio.wav"

                print(f"[COPY] {f.name} -> {input_copy}")
                shutil.copy2(f, input_copy)

                print(f"[AUDIO] extracting -> {wav_path}")
                extract_audio(ffmpeg, input_copy, wav_path)

                print(f"[DONE] session: {session_dir}")
                print("-----")

                seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
