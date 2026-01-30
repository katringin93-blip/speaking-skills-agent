import sys
import time
import shutil
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
import requests
from pydub import AudioSegment

# ---------- Настройки ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 1200 * 1000  # 20 минут (безопасный порог для лимита OpenAI в 1400 сек)

# ---------- Вспомогательные функции ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Нажмите ENTER, чтобы закрыть окно...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.local.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

def extract_audio(ffmpeg: Path, input_video: Path, output_audio: Path):
    """Извлекает звук в MP3 с оптимальным сжатием для транскрибации"""
    output_path = output_audio.with_suffix('.mp3')
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(input_video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path

def slice_audio(audio_path: Path) -> List[Path]:
    """Разрезает аудио на сегменты по 20 минут, чтобы обойти ограничения API"""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i, chunk in enumerate(audio[::CHUNK_LENGTH_MS]):
        chunk_p = audio_path.parent / f"chunk_{i}.mp3"
        chunk.export(chunk_p, format="mp3")
        chunks.append(chunk_p)
    return chunks

# ---------- Работа с OpenAI API ----------

def transcribe_chunk(api_key: str, chunk_path: Path) -> dict:
    """Отправляет аудио-фрагмент на транскрибацию и диаризацию"""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": json.dumps({"type": "server_vad"})
    }
    
    with chunk_path.open("rb") as f:
        files = {"file": (chunk_path.name, f, "audio/mpeg")}
        # Длительный таймаут для стабильной обработки тяжелых запросов
        r = requests.post(url, headers=headers, files=files, data=data, timeout=900)
    
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI Transcription Error {r.status_code}: {r.text}")
    return r.json()

def analyze_speech_contextual(api_key: str, full_transcript_path: Path, me_id: str) -> str:
    """Глубокий анализ английской речи с учетом взаимодействия с собеседниками"""
    if not full_transcript_path.exists():
        return "Ошибка: файл транскрипта не найден."
    
    full_text = full_transcript_path.read_text(encoding="utf-8")
    # Ограничиваем объем текста для GPT (около 25к символов для контекста)
    clipped_text = full_text[-25000:] if len(full_text) > 25000 else full_text
    
    current_date = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== SESSION ANALYSIS REPORT ===\nDate: {current_date}\nUser ID: {me_id}\n\n"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"""
    You are a professional English Language Examiner (IELTS/Cambridge expert). 
    Analyze the speech of the PRIMARY SPEAKER (marked as 'YOU') in the context of the entire conversation.
    
    FULL TRANSCRIPT FOR CONTEXT:
    {clipped_text}
    
    REPORT STRUCTURE (Please provide responses in Russian):
    1. **Fluency & Coherence**: (Темп речи, плавность, наличие пауз и самоисправлений)
    2. **Grammatical Range & Accuracy**: (Разнообразие грамматических конструкций и точность их использования)
    3. **Lexical Resource**: (Словарный запас, использование идиом, точность подбора слов)
    4. **Pronunciation & Dynamic**: (Ритм, интонация и динамика речи на основе текстовых маркеров)
    5. **Discourse & Interactional Competence**: (Оцени, насколько точно и корректно спикер реагирует на реплики и вопросы собеседников. Насколько эффективно ведется диалог.)
    
    FINAL ASSESSMENT:
    - Estimated CEFR Level (A1-C2):
    - Key advice for the next session: (Конкретный совет по улучшению взаимодействия)
    """

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert linguist analyzing conversational interaction in English."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 200:
            return header + r.json()['choices'][0]['message']['content']
        else:
            return f"Ошибка анализа GPT: {r.status_code}\n{r.text}"
    except Exception as e:
        return f"Критическая ошибка при анализе: {str(e)}"

# ---------- Главный цикл программы ----------

def main():
    print(">>> Agent started. Monitoring for new recordings...")
    config = load_config()
    
    obs_dir = Path(config["paths"]["obs_recordings_dir"])
    sess_root = Path(config["paths"]["sessions_dir"])
    api_key = config["whisper_api"]["api_key"]
    ffmpeg_exe = Path(config["paths"]["ffmpeg_path"])

    processed_files = set()

    while True:
        if not obs_dir.exists():
            time.sleep(5)
            continue

        for video_file in obs_dir.iterdir():
            if video_file.suffix.lower() in (".mp4", ".mkv") and video_file not in processed_files:
                # Даем время файлу "дозаписаться" и закрыться системой
                time.sleep(5)
                
                print(f"\n[NEW] Найдена запись: {video_file.name}")
                session_id = time.strftime("%Y-%m-%d_%H-%M-%S")
                session_dir = sess_root / session_id
                session_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # 1. Извлечение звука
                    print("[1/4] Извлечение аудиодорожки...")
                    full_audio_mp3 = extract_audio(ffmpeg_exe, video_file, session_dir / "full_audio")
                    
                    # 2. Нарезка на части
                    print("[2/4] Разделение на сегменты для обхода лимитов...")
                    audio_chunks = slice_audio(full_audio_mp3)
                    
                    all_segments = []
                    current_time_offset = 0.0
                    
                    # 3. Транскрибация по частям
                    for i, chunk_p in enumerate(audio_chunks):
                        print(f"    Обработка сегмента {i+1} из {len(audio_chunks)}...")
                        chunk_data = transcribe_chunk(api_key, chunk_p)
                        
                        for seg in chunk_data.get("segments", []):
                            seg["start"] += current_time_offset
                            seg["end"] += current_time_offset
                            # Присваиваем временный глобальный ID спикеру
                            seg["global_id"] = f"P{i}_{seg.get('speaker', 'UNK')}"
                            all_segments.append(seg)
                        
                        current_time_offset += (len(AudioSegment.from_file(chunk_p)) / 1000.0)
                        chunk_p.unlink() # Удаляем временный файл

                    # --- ОПРЕДЕЛЕНИЕ ОСНОВНОГО СПИКЕРА (ВАС) ---
                    speaker_durations = {}
                    for s in all_segments:
                        gid = s["global_id"]
                        dur = s["end"] - s["start"]
                        speaker_durations[gid] = speaker_durations.get(gid, 0) + dur
                    
                    if not speaker_durations:
                        print("!!! Речь не обнаружена.")
                        continue

                    # Вы — тот, кто наговорил больше всего секунд за всю суммарную сессию
                    main_speaker_id = max(speaker_durations, key=speaker_durations.get)
                    print(f"[ID] Основной спикер идентифицирован как: {main_speaker_id}")

                    # 4. Сохранение и Анализ
                    print("[4/4] Формирование отчета и лингвистический анализ...")
                    final_lines = []
                    for s in all_segments:
                        label = "YOU" if s["global_id"] == main_speaker_id else s["global_id"]
                        final_lines.append(f"[{s['start']:.1f}-{s['end']:.1f}] {label}: {s.get('text','')}")
                    
                    transcript_file = session_dir / "transcript_full.txt"
                    transcript_file.write_text("\n".join(final_lines), encoding="utf-8")
                    
                    # Запуск контекстного AI-анализа
                    analysis_report = analyze_speech_contextual(api_key, transcript_file, main_speaker_id)
                    (session_dir / "ai_analysis_report.txt").write_text(analysis_report, encoding="utf-8")
                    
                    print("\n" + "="*50)
                    print(analysis_report)
                    print("="*50)
                    print(f"[SUCCESS] Отчет сохранен в папку: {session_id}")
                    
                    # Отправка в Telegram (если настроено в конфиге)
                    tg_conf = config.get("telegram", {})
                    if tg_conf.get("enabled"):
                        msg = f"📊 *New English Session Analysis*\n\n{analysis_report[:3800]}"
                        requests.post(f"https://api.telegram.org/bot{tg_conf['bot_token']}/sendMessage", 
                                      json={"chat_id": tg_conf['chat_id'], "text": msg, "parse_mode": "Markdown"})

                    processed_files.add(video_file)

                except Exception as e:
                    print(f"!!! Ошибка при обработке файла: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        pause_and_exit(1)
