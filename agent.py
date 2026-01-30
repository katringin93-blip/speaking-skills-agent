import sys
import time
import shutil
import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
import requests
from pydub import AudioSegment

# ---------- Константы ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 300 * 1000  # 5 минут (стабильно для API)

# ---------- Помощники ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Нажмите ENTER, чтобы закрыть окно...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)

def load_config() -> dict:
    """Находит конфиг рядом с .exe или .py файлом"""
    if getattr(sys, 'frozen', False):
        base_path = Path(sys.executable).parent
    else:
        base_path = Path(__file__).parent
        
    cfg_path = base_path / "config.local.yaml"
    
    if not cfg_path.exists():
        cfg_path = Path(os.getcwd()) / "config.local.yaml"
        
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {cfg_path}")
        
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def extract_audio(ffmpeg: Path, input_video: Path, output_audio: Path):
    """Извлекает звук в MP3"""
    output_path = output_audio.with_suffix('.mp3')
    cmd = [
        str(ffmpeg), "-y", "-i", str(input_video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path

def slice_audio(audio_path: Path) -> List[Path]:
    """Режет файл на куски по 5 минут"""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i, chunk in enumerate(audio[::CHUNK_LENGTH_MS]):
        p = audio_path.parent / f"chunk_{i}.mp3"
        chunk.export(p, format="mp3")
        chunks.append(p)
    return chunks

# ---------- Работа с OpenAI ----------

def transcribe_chunk(api_key: str, chunk_path: Path) -> dict:
    """Транскрибация с защитой от таймаута"""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": json.dumps({"type": "server_vad"})
    }
    
    # Пытаемся отправить файл до 3 раз в случае сбоя сети
    for attempt in range(3):
        try:
            with chunk_path.open("rb") as f:
                files = {"file": (chunk_path.name, f, "audio/mpeg")}
                r = requests.post(url, headers=headers, files=files, data=data, timeout=900)
            
            if r.status_code == 200:
                return r.json()
            else:
                print(f"  (!) Попытка {attempt+1} неудачна: {r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  (!) Ошибка сети на попытке {attempt+1}: {e}")
        
        time.sleep(10) # Пауза перед повтором
        
    raise RuntimeError(f"Не удалось отправить файл {chunk_path.name} после 3 попыток.")

def analyze_speech_contextual(api_key: str, full_transcript_path: Path, me_id: str) -> str:
    """Контекстный анализ вашей речи"""
    full_text = full_transcript_path.read_text(encoding="utf-8")
    clipped_text = full_text[-25000:] if len(full_text) > 25000 else full_text
    
    current_date = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== SESSION ANALYSIS REPORT ===\nDate: {current_date}\nUser ID: {me_id}\n\n"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"""
    Analyze the speech of the PRIMARY SPEAKER (marked as 'YOU') in English.
    TRANSCRIPT: {clipped_text}
    REPORT (In Russian):
    1. Fluency & Coherence
    2. Grammatical Range & Accuracy
    3. Lexical Resource
    4. Pronunciation & Dynamic
    5. Discourse (Interaction with others)
    SCORE: CEFR Level and Advice.
    """

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an English Examiner. Analyze 'YOU' in the dialog. Answer in Russian."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    return header + r.json()['choices'][0]['message']['content'] if r.status_code == 200 else f"Error: {r.status_code}"

# ---------- Основная логика ----------

def main():
    print(">>> Agent started. Monitoring for new recordings...")
    config = load_config()
    
    obs_dir = Path(config["paths"]["obs_recordings_dir"])
    sess_root = Path(config["paths"]["sessions_dir"])
    api_key = config["whisper_api"]["api_key"]
    ffmpeg = Path(config["paths"]["ffmpeg_path"])
    
    # Настройка pydub для работы с локальным ffmpeg
    AudioSegment.converter = str(ffmpeg)

    seen_files = set()

    while True:
        for file in obs_dir.iterdir():
            if file.suffix.lower() in (".mp4", ".mkv") and file not in seen_files:
                time.sleep(5)
                print(f"\n[!] Обработка: {file.name}")
                s_dir = sess_root / time.strftime("%Y-%m-%d_%H-%M-%S")
                s_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    full_mp3 = extract_audio(ffmpeg, file, s_dir / "full_audio")
                    chunks = slice_audio(full_mp3)
                    
                    all_segments = []
                    offset = 0.0
                    for i, c_path in enumerate(chunks):
                        print(f"    Часть {i+1} из {len(chunks)}...")
                        res = transcribe_chunk(api_key, c_path)
                        for seg in res.get("segments", []):
                            seg["start"] += offset
                            seg["end"] += offset
                            seg["global_id"] = f"P{i}_{seg.get('speaker', 'UNK')}"
                            all_segments.append(seg)
                        offset += (len(AudioSegment.from_file(c_path)) / 1000.0)
                        c_path.unlink()

                    # Определение пользователя по длительности речи
                    stats = {}
                    for s in all_segments:
                        gid = s["global_id"]
                        stats[gid] = stats.get(gid, 0) + (s["end"] - s["start"])
                    
                    me_id = max(stats, key=stats.get) if stats else "UNK"
                    
                    final_lines = []
                    for s in all_segments:
                        label = "YOU" if s["global_id"] == me_id else s["global_id"]
                        final_lines.append(f"[{s['start']:.1f}-{s['end']:.1f}] {label}: {s.get('text','')}")
                    
                    t_file = s_dir / "transcript_full.txt"
                    t_file.write_text("\n".join(final_lines), encoding="utf-8")
                    
                    report = analyze_speech_contextual(api_key, t_file, me_id)
                    (s_dir / "ai_analysis_report.txt").write_text(report, encoding="utf-8")
                    
                    print(f"\n✅ Готово! Результаты в папке: {s_dir.name}")
                    seen_files.add(file)
                except Exception as e:
                    print(f"Ошибка при обработке {file.name}: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
