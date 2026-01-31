import sys
import time
import subprocess
import os
from pathlib import Path
from typing import List, Tuple

import yaml
import requests
from pydub import AudioSegment

# ---------- Константы ----------
CHECK_INTERVAL_SECONDS = 2
CHUNK_LENGTH_MS = 300 * 1000  # 5 минут

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
        err = (r.stderr or "").strip()
        out = (r.stdout or "").strip()
        msg = err if err else out
        raise RuntimeError(f"ffmpeg error (code {r.returncode}): {msg}")

def extract_me_and_others(ffmpeg: Path, input_video: Path, out_dir: Path) -> Tuple[Path, Path]:
    me_wav = out_dir / "me.wav"
    others_wav = out_dir / "others.wav"

    cmd_me = [str(ffmpeg), "-y", "-i", str(input_video), "-map", "0:a:1", str(me_wav)]
    cmd_others = [str(ffmpeg), "-y", "-i", str(input_video), "-map", "0:a:0", str(others_wav)]

    _run_ffmpeg(cmd_me)
    _run_ffmpeg(cmd_others)

    if not me_wav.exists() or me_wav.stat().st_size == 0:
        raise RuntimeError("me.wav не создан или пустой (проверьте, что в файле есть 0:a:1)")
    if not others_wav.exists() or others_wav.stat().st_size == 0:
        raise RuntimeError("others.wav не создан или пустой (проверьте, что в файле есть 0:a:0)")

    return me_wav, others_wav

def clean_me_audio(ffmpeg: Path, me_wav: Path, out_dir: Path) -> Path:
    me_clean = out_dir / "me_clean.wav"
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(me_wav),
        "-af", "highpass=f=120, lowpass=f=6000, afftdn",
        str(me_clean)
    ]
    _run_ffmpeg(cmd)

    if not me_clean.exists() or me_clean.stat().st_size == 0:
        raise RuntimeError("me_clean.wav не создан или пустой")

    return me_clean

def normalize_for_api(ffmpeg: Path, input_wav: Path, out_dir: Path) -> Path:
    out = out_dir / "me_clean_16k.wav"
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(input_wav),
        "-ac", "1", "-ar", "16000",
        str(out)
    ]
    _run_ffmpeg(cmd)
    return out

def slice_audio(audio_path: Path) -> List[Path]:
    audio = AudioSegment.from_file(audio_path)
    chunks: List[Path] = []
    for i, chunk in enumerate(audio[::CHUNK_LENGTH_MS]):
        p = audio_path.parent / f"chunk_{i}.mp3"
        chunk.export(p, format="mp3")
        chunks.append(p)
    return chunks

# ---------- Работа с OpenAI ----------

def transcribe_chunk(api_key: str, chunk_path: Path) -> str:
    """
    Транскрибация (без диаризации).
    Используется response_format, совместимый с текущим transcribe-моделями.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-transcribe",
        "response_format": "text"
    }

    for attempt in range(3):
        try:
            with chunk_path.open("rb") as f:
                files = {"file": (chunk_path.name, f, "audio/mpeg")}
                r = requests.post(url, headers=headers, files=files, data=data, timeout=900)

            if r.status_code == 200:
                return (r.text or "").strip()

            print(f"  (!) Попытка {attempt+1} неудачна: {r.status_code} {r.text[:300]}")
        except requests.exceptions.RequestException as e:
            print(f"  (!) Ошибка сети на попытке {attempt+1}: {e}")

        time.sleep(10)

    raise RuntimeError(f"Не удалось отправить файл {chunk_path.name} после 3 попыток.")

def analyze_speech_contextual(api_key: str, full_transcript_path: Path, me_id: str) -> str:
    full_text = full_transcript_path.read_text(encoding="utf-8")
    clipped_text = full_text[-25000:] if len(full_text) > 25000 else full_text

    current_date = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== SESSION ANALYSIS REPORT ===\nDate: {current_date}\nUser ID: {me_id}\n\n"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"""
You are an IELTS Speaking examiner.
You will be given a transcript and/or audio-based description of one speaker only.
There is no dialogue context.
Evaluate the speaker strictly according to IELTS Speaking Band Descriptors.

Important rules:

Assess only what is present in the speaker’s speech.

Do not assume missing abilities.

Use the official 0–9 IELTS band scale (allow .5 scores).

Respond only in English.

Be concise, precise, and analytical.

Evaluation Criteria (all are mandatory)
1. Fluency and Coherence

Assign a band score (0–9 or .5).

Briefly explain the score.

Provide specific examples of issues or strengths, such as:

long or frequent pauses

hesitation due to word search

repetition or self-correction

weak or effective logical progression

overuse or lack of linking devices

2. Lexical Resource

Assign a band score (0–9 or .5).

Briefly explain the score.

Provide examples from the speech, including:

limited vocabulary or excessive repetition

incorrect word choice or collocation errors

successful or failed paraphrasing

inappropriate use of idiomatic language

precision vs vagueness

3. Grammatical Range and Accuracy

Assign a band score (0–9 or .5).

Briefly explain the score.

Give concrete examples, such as:

frequent basic sentence structures only

errors in tense, agreement, word order, articles, prepositions

attempts at complex structures (relative clauses, conditionals, subordination)

whether errors are systematic or occasional

4. Pronunciation

Assign a band score (0–9 or .5).

Briefly explain the score.

Provide examples or observations, including:

mispronounced sounds that affect understanding

word stress errors

sentence stress and intonation issues

rhythm and connected speech

degree to which accent interferes with intelligibility

Final Results

Overall IELTS Speaking Band Score

Calculate the average of the four criteria.

Round to the nearest 0.5 as per IELTS rules.

Estimated CEFR Level

Map the final band score to CEFR (B1 / B2 / C1 / C2).

Output Format (strict)

Fluency and Coherence: Band X.X
Explanation: …
Error / example notes: …

Lexical Resource: Band X.X
Explanation: …
Error / example notes: …

Grammatical Range and Accuracy: Band X.X
Explanation: …
Error / example notes: …

Pronunciation: Band X.X
Explanation: …
Error / example notes: …

Overall IELTS Speaking Band: X.X
Estimated CEFR Level: …

TRANSCRIPT:
{clipped_text}
"""

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an IELTS Speaking examiner. Respond only in English."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code == 200:
        return header + r.json()["choices"][0]["message"]["content"]
    return header + f"Error: {r.status_code}\n{r.text[:2000]}"

# ---------- Основная логика ----------

def main():
    print(">>> Agent started. Monitoring for new recordings...")
    config = load_config()

    obs_dir = Path(config["paths"]["obs_recordings_dir"])
    sess_root = Path(config["paths"]["sessions_dir"])
    api_key = config["whisper_api"]["api_key"]
    ffmpeg = Path(config["paths"]["ffmpeg_path"])
    me_id = str(config.get("me_id", "me"))

    AudioSegment.converter = str(ffmpeg)

    seen_files = set()

    while True:
        try:
            candidates = list(obs_dir.iterdir())
        except FileNotFoundError:
            print(f"Папка OBS не найдена: {obs_dir}")
            pause_and_exit(1)

        for file in candidates:
            if file.suffix.lower() not in (".mp4", ".mkv"):
                continue
            if file in seen_files:
                continue
            if not is_file_stable(file, stable_seconds=10, poll_interval=1):
                continue

            print(f"\n[!] Обработка: {file.name}")
            s_dir = sess_root / time.strftime("%Y-%m-%d_%H-%M-%S")
            s_dir.mkdir(parents=True, exist_ok=True)

            try:
                me_wav, _others_wav = extract_me_and_others(ffmpeg, file, s_dir)
                me_clean = clean_me_audio(ffmpeg, me_wav, s_dir)
                me_clean_16k = normalize_for_api(ffmpeg, me_clean, s_dir)

                chunks = slice_audio(me_clean_16k)

                all_lines: List[str] = []
                for i, c_path in enumerate(chunks):
                    print(f"    Часть {i+1} из {len(chunks)}...")
                    text = transcribe_chunk(api_key, c_path)
                    if text:
                        all_lines.append(text)

                    try:
                        c_path.unlink()
                    except Exception:
                        pass

                t_file = s_dir / "transcript_me.txt"
                t_file.write_text("\n\n".join(all_lines).strip(), encoding="utf-8")

                report = analyze_speech_contextual(api_key, t_file, me_id)
                (s_dir / "ai_analysis_report.txt").write_text(report, encoding="utf-8")

                print(f"\n✅ Готово! Результаты в папке: {s_dir.name}")
                seen_files.add(file)

            except Exception as e:
                print(f"Ошибка при обработке {file.name}: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
