import sys
import time
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Optional

import yaml
import requests
from pydub import AudioSegment

# ---------- Константы ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 300 * 1000  # 5 минут
TELEGRAM_SAFE_LIMIT = 3500  # короткий вариант для Telegram

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

def telegram_send_message(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error: {r.text[:300]}")

def get_telegram_settings(config: dict):
    tg = config.get("telegram") or {}
    if not tg.get("enabled"):
        return None
    return str(tg["bot_token"]), str(tg["chat_id"])

# ---------- OpenAI ----------

def transcribe_chunk(api_key: str, chunk_path: Path) -> str:
    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (chunk_path.name, chunk_path.open("rb"), "audio/mpeg")},
        data={"model": "gpt-4o-transcribe", "response_format": "text"},
        timeout=900
    )
    if r.status_code != 200:
        raise RuntimeError("Transcription failed")
    return r.text.strip()

def analyze_full(api_key: str, transcript: str) -> str:
    prompt = f"""You are an IELTS Speaking examiner.
Respond only in English.
Provide a FULL detailed evaluation.

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
        raise RuntimeError("AI full analysis failed")
    return r.json()["choices"][0]["message"]["content"]

def analyze_short(api_key: str, transcript: str) -> str:
    prompt = f"""
You are an IELTS Speaking examiner.
Respond only in English.
Keep under 3000 characters.

Output strictly in this format:

🎯 *IELTS Speaking – Short Report*

🗣 Fluency & Coherence: Band X.X – short explanation and main errors 
📚 Lexical Resource: Band X.X – short explanation and main errors 
🧠 Grammar: Band X.X – short explanation and main errors 
🔊 Pronunciation: Band X.X – short explanation 

⭐ Overall Band: X.X  
📈 CEFR: B1 / B2 / C1 / C2

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
        raise RuntimeError("AI short analysis failed")
    return r.json()["choices"][0]["message"]["content"]

# ---------- Main ----------

def main():
    config = load_config()

    obs_dir = Path(config["paths"]["obs_recordings_dir"])
    sess_root = Path(config["paths"]["sessions_dir"])
    ffmpeg = Path(config["paths"]["ffmpeg_path"])
    api_key = config["whisper_api"]["api_key"]

    tg = get_telegram_settings(config)
    AudioSegment.converter = str(ffmpeg)

    seen = set()

    while True:
        for f in obs_dir.iterdir():
            if f.suffix.lower() not in (".mkv", ".mp4") or f in seen:
                continue
            if not is_file_stable(f):
                continue

            s_dir = sess_root / time.strftime("%Y-%m-%d_%H-%M-%S")
            s_dir.mkdir(parents=True, exist_ok=True)

            me_wav, _ = extract_me_and_others(ffmpeg, f, s_dir)
            clean = clean_me_audio(ffmpeg, me_wav, s_dir)
            norm = normalize_for_api(ffmpeg, clean, s_dir)

            texts = []
            for c in slice_audio(norm):
                texts.append(transcribe_chunk(api_key, c))
                c.unlink(missing_ok=True)

            transcript = "\n\n".join(texts).strip()
            (s_dir / "transcript_me.txt").write_text(transcript, encoding="utf-8")

            full_report = analyze_full(api_key, transcript)
            (s_dir / "ai_analysis_report_full.txt").write_text(full_report, encoding="utf-8")

            short_report = analyze_short(api_key, transcript)

            if tg:
                telegram_send_message(
                    tg[0],
                    tg[1],
                    trim_for_telegram(short_report, TELEGRAM_SAFE_LIMIT)
                )

            seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
