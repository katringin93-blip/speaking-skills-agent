import time
from pathlib import Path

OBS_RECORDINGS_DIR = Path(r"C:\Users\User\Videos\Episoden_Raw_Recordings\01_mkv_in")
CHECK_INTERVAL_SECONDS = 5


def main():
    print("SpeakingSkillsAgent: started")
    print(f"Watching OBS folder: {OBS_RECORDINGS_DIR}")

    seen_files = set()

    while True:
        if OBS_RECORDINGS_DIR.exists():
            for file in OBS_RECORDINGS_DIR.iterdir():
                if file.suffix.lower() == ".mkv" and file not in seen_files:
                    seen_files.add(file)
                    print(f"[NEW RECORDING DETECTED] {file.name}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

