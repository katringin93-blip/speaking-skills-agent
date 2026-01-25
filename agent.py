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

PRIMARY_CONFIDENCE_MIN_SHARE = 0.15  # если top не впереди хотя бы на 15% от total -> low confidence
MAX_TRANSCRIPT_CHARS_FOR_ANALYSIS = 14000  # чтобы не упираться в лимиты


# ---------- helpers ----------

def pause_and_exit(code: int = 0):
    print("\n---")
    print("Press ENTER to close this window...")
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


def get_optional_str(data: dict, dotted_key: str, default: str) -> str:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if not isinstance(cur, str) or not cur.strip():
        return default
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


def get_optional_bool(data: dict, dotted_key: str, default: bool) -> bool:
    cur = data
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if isinstance(cur, bool):
        return cur
    if isinstance(cur, str):
        return cur.strip().lower() in ("1", "true", "yes", "y", "on")
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


def read_text_safe(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def clip_text(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    # берём хвост, т.к. часто начало — приветствия, а суть позже
    return "…(clipped)…\n" + s[-max_chars:]


# ---------- OpenAI diarized transcription ----------

def openai_transcribe_diarized(api_key: str, wav_path: Path) -> dict:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (wav_path.name, wav_path.open("rb"), "audio/wav")}
    data = {
        "model": "gpt-4o-transcribe-diarize",
        "response_format": "diarized_json",
        "chunking_strategy": "auto",
    }
    r = requests.post(url, headers=headers, files=files, data=data, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Transcription error {r.status_code}: {r.text}")
    return r.json()


def _seg_speaker(seg: dict) -> str:
    return seg.get("speaker") or seg.get("speaker_label") or seg.get("speaker_id") or "UNKNOWN"


def _seg_times(seg: dict) -> tuple[float, float]:
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
        start, end = _seg_times(s)
        speaker = _seg_speaker(s)
        text = (s.get("text") or "").strip()
        lines.append(f"[{start:0.2f}-{end:0.2f}] {speaker}: {text}")
    return "\n".join(lines).strip()


# ---------- primary speaker ----------

def compute_speaker_stats(diarized: dict) -> dict:
    segments = diarized.get("segments") or []
    stats = {}
    for s in segments:
        spk = _seg_speaker(s)
        start, end = _seg_times(s)
        dur = max(0.0, end - start)
        if spk not in stats:
            stats[spk] = {"duration_seconds": 0.0, "segments": 0}
        stats[spk]["duration_seconds"] += dur
        stats[spk]["segments"] += 1
    total = sum(v["duration_seconds"] for v in stats.values()) or 0.0
    return {"by_speaker": stats, "total_duration_seconds": total}


def choose_primary_speaker(stats: dict) -> dict:
    by = stats.get("by_speaker") or {}
    items = [(spk, float(v.get("duration_seconds", 0.0)), int(v.get("segments", 0))) for spk, v in by.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    if not items:
        return {"primary_speaker": "UNKNOWN", "confidence": "none", "reason": "no_segments"}

    top = items[0]
    second = items[1] if len(items) > 1 else ("NONE", 0.0, 0)

    total = float(stats.get("total_duration_seconds", 0.0)) or 0.0
    diff = top[1] - second[1]
    diff_share = (diff / total) if total > 0 else 0.0

    confidence = "high" if diff_share >= PRIMARY_CONFIDENCE_MIN_SHARE else "low"

    return {
        "primary_speaker": top[0],
        "confidence": confidence,
        "top_duration_seconds": round(top[1], 3),
        "second_duration_seconds": round(second[1], 3),
        "total_duration_seconds": round(total, 3),
        "diff_share": round(diff_share, 4),
        "ranking": [{"speaker": spk, "duration_seconds": round(dur, 3), "segments": segs} for spk, dur, segs in items],
    }


def write_primary_and_context(diarized: dict, primary_label: str, out_primary: Path, out_context: Path):
    segments = diarized.get("segments") or []
    primary_lines = []
    context_lines = []
    for s in segments:
        start, end = _seg_times(s)
        spk = _seg_speaker(s)
        text = (s.get("text") or "").strip()
        line = f"[{start:0.2f}-{end:0.2f}] {spk}: {text}"
        if spk == primary_label:
            primary_lines.append(line)
        else:
            context_lines.append(line)

    out_primary.write_text("\n".join(primary_lines).strip(), encoding="utf-8")
    out_context.write_text("\n".join(context_lines).strip(), encoding="utf-8")


# ---------- OpenAI text analysis (Responses API) ----------

def openai_responses_analyze(api_key: str, model: str, diarized_text: str, primary_text: str) -> dict:
    """
    Uses Responses API (recommended for new projects) to generate a structured JSON analysis.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = f"""
You are an English-speaking coach. Analyze the conversation transcript.

Rules:
- Focus ONLY on the user's speech ("PRIMARY") for mistakes and improvements.
- Use the full diarized transcript only for context.
- Output MUST be valid JSON only. No markdown.

Return JSON with this schema:
{{
  "session_summary": "1-3 sentences",
  "topics": ["..."],
  "primary_speaker_coaching": {{
     "strengths": ["..."],
     "top_issues": [
        {{
          "category": "grammar|vocabulary|clarity|fluency|pronunciation_proxy",
          "example": "short quote from PRIMARY",
          "better_version": "corrected sentence",
          "why": "1 sentence"
        }}
     ],
     "next_session_focus": ["... (max 3)"]
  }},
  "drills": [
     {{
       "name": "drill title",
       "goal": "1 sentence",
       "instructions": ["step 1", "step 2", "step 3"],
       "examples": ["example 1", "example 2"]
     }}
  ]
}}

DIARIZED (for context):
{diarized_text}

PRIMARY (user only):
{primary_text}
""".strip()

    payload = {
        "model": model,
        "input": prompt,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Responses API error {r.status_code}: {r.text}")

    data = r.json()

    # Responses API returns structured output blocks; easiest is output_text when present.
    # But since we demanded JSON-only, we read `output_text` if available, otherwise fallback.
    out_text = data.get("output_text")
    if not out_text:
        # fallback: try to extract from output items
        out_text = ""
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    out_text += c.get("text")

    out_text = (out_text or "").strip()
    if not out_text:
        raise RuntimeError("Responses API returned no output_text")

    try:
        return json.loads(out_text)
    except Exception:
        # If model accidentally returned non-JSON, store raw for debugging
        return {"_raw_output_text": out_text}


def analysis_json_to_md(analysis: dict) -> str:
    if "_raw_output_text" in analysis:
        return "# Analysis (raw)\n\n" + analysis["_raw_output_text"]

    lines = []
    lines.append("# Session analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append(analysis.get("session_summary", ""))
    lines.append("")
    lines.append("## Topics")
    topics = analysis.get("topics") or []
    for t in topics:
        lines.append(f"- {t}")
    lines.append("")
    coach = analysis.get("primary_speaker_coaching") or {}

    lines.append("## Strengths")
    for s in (coach.get("strengths") or []):
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## Top issues (PRIMARY only)")
    for it in (coach.get("top_issues") or []):
        lines.append(f"- **{it.get('category','')}**")
        lines.append(f"  - Example: {it.get('example','')}")
        lines.append(f"  - Better: {it.get('better_version','')}")
        lines.append(f"  - Why: {it.get('why','')}")
    lines.append("")

    lines.append("## Next session focus")
    for f in (coach.get("next_session_focus") or []):
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## Drills")
    for d in (analysis.get("drills") or []):
        lines.append(f"### {d.get('name','')}")
        lines.append(d.get("goal", ""))
        lines.append("")
        lines.append("Instructions:")
        for st in (d.get("instructions") or []):
            lines.append(f"- {st}")
        ex = d.get("examples") or []
        if ex:
            lines.append("")
            lines.append("Examples:")
            for e in ex:
                lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------- main ----------

def main():
    print("SpeakingSkillsAgent: started")
    config = load_config()

    obs_dir = Path(get_required_str(config, "paths.obs_recordings_dir"))
    sessions_root = Path(get_required_str(config, "paths.sessions_dir"))
    stable_seconds = get_optional_int(config, "processing.stable_seconds", STABLE_SECONDS_DEFAULT)

    api_key = get_required_str(config, "whisper_api.api_key")
    ffmpeg = resolve_ffmpeg(config)

    analysis_enabled = get_optional_bool(config, "analysis.enabled", True)
    analysis_model = get_optional_str(config, "analysis.model", "gpt-4o-mini")

    print(f"Base dir: {app_base_dir()}")
    print(f"Watching OBS: {obs_dir}")
    print(f"Sessions: {sessions_root}")
    print(f"Stable seconds: {stable_seconds}")
    print(f"Analysis enabled: {analysis_enabled} (model: {analysis_model})")
    print("-----")
    print("Waiting for recordings...")

    seen = set()

    while True:
        if obs_dir.exists():
            for f in obs_dir.iterdir():
                if f.suffix.lower() != ".mkv" or f in seen:
                    continue

                print(f"\n[FOUND] {f.name} -> waiting to stabilize...")
                if not is_file_stable(f, stable_seconds):
                    print("[SKIP] file not stable")
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

                analysis_json_path = session_dir / "analysis.json"
                analysis_md_path = session_dir / "analysis.md"

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

                stats_out = {"primary_selection": choice, "stats": stats}
                stats_path.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")

                primary_label = choice.get("primary_speaker", "UNKNOWN")
                write_primary_and_context(diarized, primary_label, primary_txt, context_txt)

                print(f"[PRIMARY] {primary_label} (confidence: {choice.get('confidence')}, diff_share: {choice.get('diff_share')})")

                if analysis_enabled:
                    print("[ANALYSIS] generating session report via Responses API...")
                    diarized_text = clip_text(read_text_safe(diar_txt_path), MAX_TRANSCRIPT_CHARS_FOR_ANALYSIS)
                    primary_text = clip_text(read_text_safe(primary_txt), MAX_TRANSCRIPT_CHARS_FOR_ANALYSIS)

                    analysis = openai_responses_analyze(api_key, analysis_model, diarized_text, primary_text)

                    analysis_json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
                    analysis_md_path.write_text(analysis_json_to_md(analysis), encoding="utf-8")

                    print(f"[ANALYSIS] saved -> {analysis_md_path.name}, {analysis_json_path.name}")

                print(f"[DONE] session saved -> {session_dir}")
                seen.add(f)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n!!! ERROR OCCURRED !!!")
        print(type(e).__name__ + ":", str(e))
        pause_and_exit(1)
