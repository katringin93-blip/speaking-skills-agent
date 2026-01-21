import time
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None


CHECK_INTERVAL_SECONDS = 5


def load_config_path() -> Path:
    # config лежит рядом с exe или рядом со скриптом (в репо)
    base = Path(__file__).resolve().parent
    return base / "config.local.yaml"


def read_obs_dir_from_config(cfg_path: Path) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is missing. Build configuration error.")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    obs_dir = (((data.get("paths") or {}).get("obs_recordings_dir")) or "").strip()
    if not obs_dir:
        raise ValueError("paths.obs_recordings_dir is missing in config.local.yaml")

    return Path(obs_dir)


def main():
    cfg_path = load_config_path()
    obs_dir = read_obs_dir_from_config(cfg_path)

    print("SpeakingSkillsAgent: started")
    print(f"Config: {cfg_path}")
    print(f"Watching OBS folder: {obs_dir}")

    seen_files = set()

    while True:
        if obs_dir.exists():
            for file in obs_dir.iterdir():
                if file.suffix.lower() == ".mkv" and file not in seen_files:
                    seen_files.add(file)
                    print(f"[NEW RECORDING DETECTED] {file.name}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()


