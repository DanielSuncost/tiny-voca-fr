"""Generate and screen the synthetic French corpus (resumable).

For every voice in voices.json and every sentence in sentences.jsonl:
  1. Qwen3-TTS 1.7B Base reads the sentence with the voice's reference clip as
     the prompt (batched, sentences of similar length together).
  2. Whisper large-v3-turbo transcribes it; digits and symbols in the
     transcript are spelled out in French words before scoring, because the
     sentences are written in words. Clips above --max-cer, or with an
     implausible speaking rate or long silences, are generated again with a new
     seed, up to --attempts times. Clips that never pass are listed in
     review.jsonl and left out.
  3. Manifests <split>.jsonl are written in the format tiny_voca.prepare reads.

    python scripts/french/generate_fr.py --sentences data/french/text/sentences.jsonl \
        --voices data/french/voices.json --output data/french/dataset
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from build_text_fr import number  # noqa: E402

TEACHER = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ASR = "large-v3-turbo"
SAMPLE_RATE = 24000


def spell_digits(text):
    text = text.replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"(\d+),(\d{2})\s?(€|euros?)", lambda m: f"{number(int(m[1]))} euros {number(int(m[2]))}", text)
    text = re.sub(r"(\d+)\s?€", lambda m: f"{number(int(m[1]))} euros", text)
    text = re.sub(r"(\d+)\s?h\s?(\d{2})", lambda m: f"{number(int(m[1]))} heures {number(int(m[2]))}", text)
    text = re.sub(r"(\d+)\s?h\b", lambda m: f"{number(int(m[1]))} heures", text)
    text = re.sub(r"(\d+)\s?°", lambda m: f"{number(int(m[1]))} degrés", text)
    text = re.sub(r"(\d+)\s?%", lambda m: f"{number(int(m[1]))} pour cent", text)
    text = re.sub(r"\b1er\b", "premier", text)
    text = re.sub(r"(?<=\d)[ .](?=\d{3}\b)", "", text)          # 1 789 / 1.789 -> 1789
    text = re.sub(r"\d+", lambda m: number(int(m[0])) if int(m[0]) < 1000000 else m[0], text)
    return text


def normalize(text):
    text = unicodedata.normalize("NFC", spell_digits(text).lower()).replace("’", "'").replace("œ", "oe")
    text = re.sub(r"[^\w' ]", " ", text.replace("-", " "))
    return " ".join(text.split())


def silences(wave, sr):
    """Leading, trailing and longest inner silence in seconds (-45 dB below peak)."""
    hop = sr // 100
    frames = np.sqrt(np.mean(wave[: len(wave) // hop * hop].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(frames / (frames.max() + 1e-12)) > -45
    if not loud.any():
        return 99.0, 99.0, 99.0
    first, last = int(np.argmax(loud)), len(loud) - 1 - int(np.argmax(loud[::-1]))
    inner, run = 0, 0
    for value in loud[first:last + 1]:
        run = 0 if value else run + 1
        inner = max(inner, run)
    return first / 100, (len(loud) - 1 - last) / 100, inner / 100


def load_state(path):
    state = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                state[row["id"]] = row
    return state


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(jobs, voices, sentences, out, batch_size, log):
    if not jobs:
        return
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(TEACHER, device_map="cuda", dtype=torch.bfloat16)
    by_voice = {}
    for job in jobs:
        by_voice.setdefault(job["voice_id"], []).append(job)
    started, done = time.time(), 0
    for voice_id, items in by_voice.items():
        voice = voices[voice_id]
        prompt = model.create_voice_clone_prompt(ref_audio=voice["reference_audio"], ref_text=voice["reference_text"])
        items.sort(key=lambda j: len(sentences[j["text_id"]]["text"]))
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            seed = 100003 * batch[0]["attempt"] + start
            torch.manual_seed(seed)
            texts = [sentences[j["text_id"]]["text"] for j in batch]
            wavs, sr = model.generate_voice_clone(text=texts, language=["french"] * len(batch),
                                                  voice_clone_prompt=prompt * len(batch), non_streaming_mode=True)
            for job, wave in zip(batch, wavs):
                wave = wave.detach().float().cpu().numpy() if hasattr(wave, "detach") else np.asarray(wave, np.float32)
                peak = float(np.abs(wave).max()) or 1.0
                path = out / "audio" / voice_id / f"{job['text_id']}_a{job['attempt']}.flac"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, wave * (0.95 / peak), sr, subtype="PCM_16")
                append(log, dict(id=job["id"], text_id=job["text_id"], voice_id=voice_id, attempt=job["attempt"],
                                 seed=seed, audio=str(path.relative_to(out)), stage="generated"))
            done += len(batch)
            rate = done / (time.time() - started)
            print(f"[gen] {voice_id} {done}/{len(jobs)} clips, {rate:.2f} clips/s", flush=True)
    del model
    torch.cuda.empty_cache()


def screen(rows, sentences, out, args, log):
    if not rows:
        return
    from faster_whisper import WhisperModel
    import jiwer
    asr = WhisperModel(ASR, device="cuda", compute_type="float16")
    for count, row in enumerate(rows, 1):
        wave, sr = sf.read(out / row["audio"], dtype="float32")
        segments, _ = asr.transcribe(wave if sr == 16000 else _resample(wave, sr), language="fr", beam_size=5,
                                     condition_on_previous_text=False, temperature=0.0)
        transcript = " ".join(s.text.strip() for s in segments).strip()
        text = sentences[row["text_id"]]["text"]
        ref, hyp = normalize(text), normalize(transcript)
        cer = jiwer.cer(ref, hyp) if hyp else 1.0
        seconds = len(wave) / sr
        lead, tail, inner = silences(wave, sr)
        rate = len(ref.replace(" ", "")) / max(seconds - lead - tail, 0.1)
        reasons = []
        if cer > args.max_cer:
            reasons.append(f"cer {cer:.3f}")
        if seconds - lead - tail > 1.5 and not 6 <= rate <= 26:
            reasons.append(f"rate {rate:.1f} chars/s")
        if lead > 1.0 or tail > 1.5 or inner > 1.2:
            reasons.append(f"silence lead {lead:.2f} tail {tail:.2f} inner {inner:.2f}")
        if seconds > 25:
            reasons.append(f"too long {seconds:.1f}s")
        append(log, {**row, "stage": "screened", "asr_transcript": transcript, "normalized_cer": round(cer, 4),
                     "duration_seconds": round(seconds, 3), "chars_per_second": round(rate, 2),
                     "qc_reasons": reasons, "accepted": not reasons})
        if count % 200 == 0:
            print(f"[asr] {count}/{len(rows)}", flush=True)
    del asr
    torch.cuda.empty_cache()


def _resample(wave, sr):
    import librosa
    return librosa.resample(wave, orig_sr=sr, target_sr=16000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-cer", type=float, default=0.10)
    parser.add_argument("--limit", type=int, help="First N sentences only (smoke test)")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    voices = json.loads(args.voices.read_text())
    sentences = {}
    for line in args.sentences.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sentences[row["text_id"]] = row
    if args.limit:
        sentences = dict(list(sentences.items())[:args.limit])
    log = out / "log.jsonl"

    for attempt in range(1, args.attempts + 1):
        state = load_state(log)
        jobs = []
        for text_id in sentences:
            for voice_id in voices:
                clip = f"{text_id}_{voice_id}"
                key = f"{clip}_a{attempt}"
                previous = state.get(f"{clip}_a{attempt - 1}")
                if attempt > 1 and (previous is None or previous.get("accepted", True)):
                    continue
                if key not in state:
                    jobs.append(dict(id=key, text_id=text_id, voice_id=voice_id, attempt=attempt))
        print(f"attempt {attempt}: {len(jobs)} clips to generate", flush=True)
        generate(jobs, voices, sentences, out, args.batch_size, log)
        state = load_state(log)
        pending = [row for row in state.values() if row["stage"] == "generated" and row["attempt"] == attempt]
        print(f"attempt {attempt}: {len(pending)} clips to screen", flush=True)
        screen(pending, sentences, out, args, log)

    state = load_state(log)
    manifests, review = {}, []
    for text_id, sentence in sentences.items():
        for voice_id, voice in voices.items():
            tries = [state.get(f"{text_id}_{voice_id}_a{a}") for a in range(1, args.attempts + 1)]
            tries = [t for t in tries if t and t["stage"] == "screened"]
            accepted = next((t for t in tries if t["accepted"]), None)
            if accepted is None:
                review.append(dict(id=f"{text_id}_{voice_id}", text=sentence["text"], tries=tries))
                continue
            manifests.setdefault(sentence["split"], []).append(dict(
                id=f"{text_id}_{voice_id}", text_id=text_id, voice_id=voice_id, text=sentence["text"],
                split=sentence["split"], category=sentence["category"], origin=sentence["origin"],
                source_license=sentence["source_license"], source_url=sentence["source_url"], audio=accepted["audio"],
                duration_seconds=accepted["duration_seconds"], sample_rate=SAMPLE_RATE,
                asr_transcript=accepted["asr_transcript"], normalized_cer=accepted["normalized_cer"],
                attempt=accepted["attempt"], seed=accepted["seed"], teacher_model=TEACHER,
                reference_audio=voice["reference_audio"]))
    for split, rows in manifests.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    summary = {split: dict(clips=len(rows), hours=round(sum(r["duration_seconds"] for r in rows) / 3600, 2))
               for split, rows in manifests.items()}
    print(json.dumps(dict(summary=summary, left_out=len(review)), indent=1))


if __name__ == "__main__":
    main()
