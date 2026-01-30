import sys
import time
import shutil
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any

import yaml
import requests

CHECK_INTERVAL_SECONDS = 2
STABLE_SECONDS_DEFAULT = 10
STABLE_POLL_INTERVAL = 1

PRIMARY_CONFIDENCE_MIN_SHARE = 0.15
MAX_TRANSCRIPT_CHARS_FOR_ANALYSIS = 14000
TELEGRAM_MAX_MESSAGE_LEN = 3900 


# ---------- helpers ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Нажмите ENTER, чтобы закрыть окно...")
    try:
        input()
    except Exception:
        pass
    sys.exit(code)


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
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def get_required_str(data: dict, dotted_key: str) -> str:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise ValueError(f"Отсутствует ключ в конфиге: {dotted_key}")
        cur = cur[p]
    if not isinstance(cur, str) or not cur.strip():
        raise ValueError(f"Ключ должен быть непустой строкой: {dotted_key}")
    return cur.strip()


def get_optional_str(data: dict, dotted_key: str, default: str) -> str:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur: return default
        cur = cur[p]
    return cur.strip() if (isinstance(cur, str) and cur.strip()) else default


def get_optional_int(data: dict, dotted_key: str, default: int) -> int:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur: return default
        cur = cur[p]
    try: return int(cur)
    except: return default


def get_optional_bool(data: dict, dotted_key: str, default: bool) -> bool:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur: return default
        cur = cur[p]
    if isinstance(cur, bool): return cur
    if isinstance(cur, str): return cur.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def resolve_ffmpeg(config: dict) -> Path:
    try:
        p = Path(get_required_str(config, "paths.ffmpeg_path"))
        if p.exists(): return p
    except: pass
    local = app_base_dir() / "ffmpeg.exe"
    if local.exists(): return local
    which = shutil.which("ffmpeg")
    if which: return Path(which)
    raise FileNotFoundError("ffmpeg не найден. Установите ffmpeg.exe в папку с ботом.")


def make_session_dir(root: Path) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    d = root / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_audio(ffmpeg: Path, input_video: Path, output_audio: Path):
    """Извлекает звук в MP3 (48k для 1 часа в 25МБ)"""
    output_path = output_audio.with_suffix('.mp3')
    cmd = [
        str(ffmpeg), "-y",
        "-i", str(input_video),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        str(output_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{r.stderr}")
    return output_path


def read_text_safe(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def clip_text(s: str, max_chars: int) -> str:
    return s if len(s or "") <= max_chars else "…(clipped)…\n" + s[-max_chars:]


# ---------- OpenAI API ----------

def openai_transcribe_diarized(api_key: str, audio_path: Path) -> dict:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": json.dumps({"type": "server_vad"})
    }

    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"[TRANSCRIBE] Отправка файла: {audio_path.name} ({size_mb:.2f} MB)...")
    print("[TRANSCRIBE] Ожидайте, это может занять несколько минут для длинных записей...")
    
    try:
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/mpeg")}
            # Используем requests с большим таймаутом (15 минут)
            response = requests.post(url, headers=headers, files=files, data=data, timeout=900)
        
        if response.status_code == 200:
            print("[TRANSCRIBE] Успешно получено!")
            return response.json()
        else:
            print(f"[ERROR] Код: {response.status_code}. Ответ: {response.text}")
            raise RuntimeError(f"OpenAI API Error {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Ошибка при связи с OpenAI: {str(e)}")


def _seg_speaker(seg: dict) -> str:
    return seg.get("speaker") or seg.get("speaker_label") or "UNKNOWN"


def _seg_times(seg: dict) -> Tuple[float, float]:
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", s))
    return s, (e if e >= s else s)


def diarized_json_to_text(diarized: dict) -> str:
    lines = []
    for s in (diarized.get("segments") or []):
        start, end = _seg_times(s)
        speaker = _seg_speaker(s)
        text = (s.get("text") or "").strip()
        lines.append(f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}")
    return "\n".join(lines)


# ---------- Logic ----------

def compute_speaker_stats(diarized: dict) -> dict:
    stats = {}
    for s in (diarized.get("segments") or []):
        spk = _seg_speaker(s)
        start, end = _seg_times(s)
        dur = max(0.0, end - start)
        if spk not in stats: stats[spk] = {"duration_seconds": 0.0, "segments": 0}
        stats[spk]["duration_seconds"] += dur
        stats[spk]["segments"] += 1
    total = sum(v["duration_seconds"] for v in stats.values()) or 0.1
    return {"by_speaker": stats, "total_duration_seconds": total}


def choose_primary_speaker(stats: dict) -> dict:
    items = sorted([(k, v["duration_seconds"]) for k, v in stats["by_speaker"].items()], key=lambda x: x[1], reverse=True)
    if not items: return {"primary_speaker": "UNKNOWN", "confidence": "none"}
    
    top_spk, top_dur = items[0]
    sec_dur = items[1][1] if len(items) > 1 else 0.0
    diff_share = (top_dur - sec_dur) / stats["total_duration_seconds"]
    
    return {
        "primary_speaker": top_spk,
        "confidence": "high" if diff_share >= PRIMARY_CONFIDENCE_MIN_SHARE else "low",
        "diff_share": round(diff_share, 4)
    }


def write_primary_and_context(diarized: dict, primary_label: str, out_p: Path, out_c: Path):
    p_lines, c_lines = [], []
    for s in (diarized.get("segments") or []):
        line = f"[{s.get('start',0)}-{s.get('end',0)}] {_seg_speaker(s)}: {s.get('text','')}"
        if _seg_speaker(s) == primary_label: p_lines.append(line)
        else: c_lines.append(line)
    out_p.write_text("\n".join(p_lines), encoding="utf-8")
    out_c.write_text("\n".join(c_lines), encoding="utf-8")


def openai_responses_analyze(api_key: str, model: str, diar_txt: str, prim_txt: str) -> dict:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"English Coach Analysis.\n\nDIARIZED:\n{diar_txt}\n\nPRIMARY:\n{prim_txt}\n\nReturn JSON only."
    
    payload = {"model": model, "input": prompt}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=180)
    
    data = r.json()
    out_text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text": out_text += c.get("text")
    
    try: return json.loads(out_text.strip())
    except: return {"_raw_output_text": out_text}


def build_tg_msg(session_dir: Path, analysis: dict) -> str:
    summary = analysis.get("session_summary", "Анализ завершен") if isinstance(analysis, dict) else "Готово"
    return f"✅ Отчет готов!\n\n{summary}\n\nПапка: {session_dir.name}"


# ---------- Main ----------

def main():
    print("Agent started...")
    config = load_config()
    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sess_root = Path(get_required_str(config, "paths.sessions_dir"))
    api_key = get_required_str(config, "whisper_api.api_key")
    ffmpeg = resolve_ffmpeg(config)

    seen = set()
    while True:
        if obs_dir.exists():
            for f in obs_dir.iterdir():
                if f.suffix.lower() not in (".mkv", ".mp4") or f in seen: continue
                if not is_file_stable(f, 10): continue

                print(f"\n[NEW FILE] {f.name}")
                s_dir = make_session_dir(sess_root)
                input_copy = s_dir / f"input{f.suffix}"
                shutil.copy2(f, input_copy)

                print("[STEP 1] Извлечение аудио...")
                mp3_path = extract_audio(ffmpeg, input_copy, s_dir / "audio")

                print("[STEP 2] Транскрибация...")
                diarized = openai_transcribe_diarized(api_key, mp3_path)
                
                (s_dir / "transcript_diarized.json").write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")
                (s_dir / "transcript_diarized.txt").write_text(diarized_json_to_text(diarized), encoding="utf-8")

                stats = compute_speaker_stats(diarized)
                choice = choose_primary_speaker(stats)
                write_primary_and_context(diarized, choice["primary_speaker"], s_dir / "transcript_primary.txt", s_dir / "transcript_context.txt")

                print("[STEP 3] Анализ речи...")
                if get_optional_bool(config, "analysis.enabled", True):
                    analysis = openai_responses_analyze(api_key, get_optional_str(config, "analysis.model", "gpt-4o-mini"), 
                                                       clip_text(read_text_safe(s_dir / "transcript_diarized.txt"), 14000),
                                                       clip_text(read_text_safe(s_dir / "transcript_primary.txt"), 14000))
                    
                    if get_optional_bool(config, "telegram.enabled", False):
                        tok = get_optional_str(config, "telegram.bot_token", "")
                        cid = get_optional_int(config, "telegram.chat_id", 0)
                        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", json={"chat_id": cid, "text": build_tg_msg(s_dir, analysis)})

                print(f"[DONE] Сессия сохранена: {s_dir.name}")
                seen.add(f)
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: {e}")
        pause_and_exit(1)
