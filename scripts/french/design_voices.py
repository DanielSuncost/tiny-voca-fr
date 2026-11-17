"""Design candidate French voices and check how the teacher reads French.

Designs candidate voices with Qwen3-TTS VoiceDesign (one reference clip per
description and seed), then has Qwen3-TTS Base read pronunciation test sentences
with each reference clip as the voice prompt, the same path the corpus uses.
Every clip is transcribed with multilingual Whisper and phonemized with eSpeak
NG (fr-fr), and a local review page is written for picking the voices by ear.

    python scripts/french/design_voices.py --output data/french/voices
"""

import argparse
import html
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REFERENCE_TEXT = ("Bonjour et bienvenue. Aujourd'hui, nous allons apprendre quelques phrases simples "
                  "pour se présenter, demander son chemin et commander un café en terrasse.")

VOICES = {
    "F_paris_30": "A native French woman from Paris in her early thirties. Warm, clear, mid-pitched adult voice with a "
                  "standard metropolitan French accent. Calm, articulate and natural, like a radio presenter.",
    "F_mature_45": "A native French woman in her mid forties with a mature, slightly low, warm voice. Standard Parisian "
                   "French accent. Friendly, patient and very clear, like an experienced language teacher.",
    "M_paris_30": "A native French man from Paris in his thirties with a warm baritone voice. Clear standard "
                  "metropolitan French accent. Relaxed, natural and friendly.",
    "M_mature_55": "A native French man in his fifties with a deep, calm, resonant voice. Standard metropolitan French "
                   "accent. Measured and articulate, like a documentary narrator.",
}
SEEDS = (11, 23)

# Each sentence targets a pronunciation learners must hear right.
TRAPS = [
    ("Les amis sont arrivés hier soir.", "liaison [z]: les‿amis"),
    ("Les haricots verts sont dans le panier.", "h aspiré: no liaison"),
    ("Le bateau est beau, le pain est bon, et le banc est blanc.", "oral vs nasal vowels"),
    ("Tu as vu tout le monde au bout de la rue ?", "[y] vs [u]; question"),
    ("Il est deux heures et quart, pas douze heures.", "[ø]; liaison deux‿heures"),
    ("Soixante-dix-sept, quatre-vingt-onze et quatre-vingt-dix-neuf.", "numbers; no liaison before onze"),
    ("Les poules du couvent couvent leurs œufs.", "homograph couvent / couvent"),
    ("Je ne sais pas ce que tu veux dire.", "schwa deletion"),
    ("Mon fils a acheté des fils de laine bleue.", "homograph fils [fis] / [fil]"),
    ("Il y a un grand hôtel juste en face.", "liaison [t]: grand‿hôtel"),
    ("Vingt et un, vingt-deux, vingt-trois, et cent ans plus tard.", "vingt [vɛ̃t]/[vɛ̃]; cent‿ans"),
    ("Est-ce que vous êtes allés à Paris l'été dernier ?", "question; vous‿êtes"),
    ("Quelle belle journée ! On va se promener au bord de la mer ?", "exclamation; question intonation"),
    ("Le médecin m'a dit de prendre ce médicament deux fois par jour.", "everyday sentence"),
    ("Ils parlent, ils parlaient, et demain ils parleront.", "silent -ent"),
    ("Le prix est de vingt-neuf euros quatre-vingt-dix-neuf.", "neuf‿euros [v]? price reading"),
    ("Nous habitons au premier étage, juste à côté de la gare.", "liaison nous‿habitons, premier‿étage"),
    ("Un œuf, des œufs ; un bœuf, des bœufs.", "plural [ø] vs singular [œf]"),
    ("Le chat de Sophie s'appelle Mistigri et il dort toute la journée.", "names; narrative"),
    ("Prenez la deuxième rue à gauche, puis continuez tout droit jusqu'au rond-point.", "directions; long sentence"),
]

SAMPLE_RATE = 24000


def ipa(text):
    out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "fr-fr", text], capture_output=True, text=True, check=True)
    return " ".join(out.stdout.split())


def normalize(text):
    text = unicodedata.normalize("NFC", text.lower()).replace("’", "'")
    text = re.sub(r"[^\w' -]", " ", text).replace("-", " ")
    return " ".join(text.split())


def to_numpy(audio):
    return audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio, dtype=np.float32)


def save(path, audio, sr):
    audio = to_numpy(audio)
    peak = float(np.abs(audio).max()) or 1.0
    sf.write(path, audio * min(1.0, 0.95 / peak), sr, subtype="PCM_16")
    return len(audio) / sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/french/voices"))
    parser.add_argument("--whisper", default="large-v3-turbo")
    args = parser.parse_args()
    out = args.output
    (out / "refs").mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    from qwen_tts import Qwen3TTSModel

    rows = []
    load = dict(device_map="cuda", dtype=torch.bfloat16)
    design = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", **load)
    refs = {}
    for name, description in VOICES.items():
        for seed in SEEDS:
            voice = f"{name}_s{seed}"
            torch.manual_seed(seed)
            wavs, sr = design.generate_voice_design(text=REFERENCE_TEXT, instruct=description, language="french")
            path = out / "refs" / f"{voice}.wav"
            seconds = save(path, wavs[0], sr)
            refs[voice] = path
            print(f"ref {voice}: {seconds:.1f}s", flush=True)
    del design
    torch.cuda.empty_cache()

    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **load)
    for voice, ref in refs.items():
        prompt = base.create_voice_clone_prompt(ref_audio=str(ref), ref_text=REFERENCE_TEXT)
        for index, (text, trap) in enumerate(TRAPS):
            torch.manual_seed(1000 + index)
            wavs, sr = base.generate_voice_clone(text=text, language="french", voice_clone_prompt=prompt,
                                                 non_streaming_mode=True)
            path = out / "clips" / f"{voice}_{index:02d}.wav"
            seconds = save(path, wavs[0], sr)
            rows.append(dict(voice=voice, index=index, text=text, trap=trap, path=str(path.relative_to(out)),
                             seconds=round(seconds, 2)))
        print(f"clips {voice}: done", flush=True)
    del base
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel
    import jiwer
    asr = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    for row in rows:
        segments, _ = asr.transcribe(str(out / row["path"]), language="fr", beam_size=5)
        row["asr"] = " ".join(s.text.strip() for s in segments)
        ref, hyp = normalize(row["text"]), normalize(row["asr"])
        row["wer"] = round(jiwer.wer(ref, hyp), 3)
        row["cer"] = round(jiwer.cer(ref, hyp), 3)
    for row in rows:
        row["ipa"] = ipa(row["text"])
    (out / "results.json").write_text(json.dumps(dict(voices=VOICES, reference_text=REFERENCE_TEXT, rows=rows),
                                                 ensure_ascii=False, indent=1))
    write_page(out, refs, rows)
    for voice in refs:
        mine = [r for r in rows if r["voice"] == voice]
        print(f"{voice}: mean CER {np.mean([r['cer'] for r in mine]):.3f}, mean WER {np.mean([r['wer'] for r in mine]):.3f}")


def write_page(out, refs, rows):
    e = html.escape
    parts = ["""<!doctype html><meta charset="utf-8"><title>French voice check</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--bad:#b3261e;--ok:#1e6b3a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9b9b95;--line:#33332f;--bad:#f2867c;--ok:#7fd19b}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}
.m{color:var(--muted);font-size:13px}.bad{color:var(--bad)}.ok{color:var(--ok)}audio{height:32px;width:240px}
h2{margin-top:36px}textarea{width:100%;min-height:60px}
</style>
<h1>French voice check</h1>
<p>Listen to the references first and pick two voices (ideally one F, one M). Then check the trap sentences.
Mark a clip <b>bad</b> if a native speaker would notice the pronunciation. Your marks stay in this browser; use “Copy
notes” and paste them back into the chat.</p>
<h2>Reference voices</h2><table><tr><th>Voice</th><th>Reference</th><th>Pick</th></tr>"""]
    for voice, path in refs.items():
        parts.append(f'<tr><td>{e(voice)}</td><td><audio controls preload="none" src="refs/{e(path.name)}"></audio></td>'
                     f'<td><input type="checkbox" data-k="pick:{e(voice)}"></td></tr>')
    parts.append("</table>")
    for voice in refs:
        parts.append(f"<h2>{e(voice)}</h2><table><tr><th>#</th><th>Sentence / trap</th><th>Audio</th><th>Whisper</th><th>Bad?</th></tr>")
        for r in (r for r in rows if r["voice"] == voice):
            cls = "bad" if r["cer"] > 0.05 else "ok"
            parts.append(
                f'<tr><td>{r["index"]}</td><td>{e(r["text"])}<div class="m">{e(r["trap"])} · eSpeak: {e(r["ipa"])}</div></td>'
                f'<td><audio controls preload="none" src="{e(r["path"])}"></audio></td>'
                f'<td>{e(r["asr"])}<div class="m {cls}">CER {r["cer"]:.3f}</div></td>'
                f'<td><input type="checkbox" data-k="bad:{e(voice)}:{r["index"]}"></td></tr>')
        parts.append("</table>")
    parts.append("""<h2>Notes</h2><textarea data-k="notes"></textarea><p><button id="copy">Copy notes</button></p>
<script>
const key='fr-voices';let s={};try{s=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
document.querySelectorAll('[data-k]').forEach(el=>{const k=el.dataset.k;
 if(el.type==='checkbox'){el.checked=!!s[k];el.onchange=()=>{s[k]=el.checked;save()}}
 else{el.value=s[k]||'';el.oninput=()=>{s[k]=el.value;save()}}});
function save(){try{localStorage.setItem(key,JSON.stringify(s))}catch(e){}}
document.getElementById('copy').onclick=()=>{const picks=Object.keys(s).filter(k=>s[k]&&k.startsWith('pick:')).map(k=>k.slice(5));
 const bad=Object.keys(s).filter(k=>s[k]&&k.startsWith('bad:')).map(k=>k.slice(4));
 navigator.clipboard.writeText('picks: '+picks.join(', ')+'\\nbad: '+bad.join(', ')+'\\nnotes: '+(s.notes||''))};
</script>""")
    (out / "index.html").write_text("\n".join(parts))


if __name__ == "__main__":
    main()
