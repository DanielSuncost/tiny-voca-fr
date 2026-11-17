# tiny-voca-fr

A small standard French text-to-speech model. It has two voices, 4.5M
parameters, and runs on a CPU or in the browser.

This is a French port of [tiny-voca](https://github.com/n4ze3m/tiny-voca) by
Muhammed Nazeem. The model architecture is unchanged; the front end, voices,
training data and evaluation are new.

[Demo](https://www.suncostlabs.com/tinyvocafr) · [Weights](https://huggingface.co/DanielSuncost/tiny-voca-fr)

## Features

- **Small**: 4.5M parameters, 18 MB, waveform generator included.
- **Two voices**: `FR_F1` (female, mid-forties) and `FR_M1` (male, mid-fifties),
  standard French (France), 24 kHz.
- **French front end**: eSpeak NG `fr-fr` with liaison, nasal vowels and
  spelled-out numbers.
- **Runs anywhere**: PyTorch and ONNX Runtime.
- **Reproducible**: every step from sentence list to evaluation is a script.

## Install

```bash
git clone https://github.com/DanielSuncost/tiny-voca-fr.git
cd tiny-voca-fr
pip install -e .
sudo apt install espeak-ng          # the French front end calls eSpeak NG
hf download DanielSuncost/tiny-voca-fr --local-dir weights
```

## Usage

```bash
export TINY_VOCA_LANGUAGE=fr
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca-fr.pt \
  --voice FR_F1 \
  --text "Bonjour ! Comment allez-vous aujourd'hui ?" \
  --output bonjour.wav
```

`TINY_VOCA_LANGUAGE=fr` switches the front end, the voices and the evaluation
to French. Without it the code behaves exactly like upstream tiny-voca
(English).

| Option | Default | Meaning |
|---|---|---|
| `--voice` | `FR_F1` | `FR_F1` or `FR_M1` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Added to log F0 (0.058 is about one semitone) |
| `--pitch-range` | 1.0 | Scale of pitch movement |

An ONNX export is in the weights repository as `onnx/model.onnx`:

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca-fr.pt --output tiny-voca-fr.onnx
```

## How it was made

The voices and the training audio are synthetic, distilled from a larger
open model.

1. **Voices.** Candidate voices were designed from text descriptions with
   Qwen3-TTS VoiceDesign, then judged by ear on pronunciation test sentences
   (liaison, h aspiré, nasal vowels, *tu/tout*, *œuf/œufs*, numbers).
   Two were kept.
2. **Sentences.** 8,000 sentences from the Common Voice sentence collector
   (CC0), filtered to plain text with no digits, symbols or English words, plus
   1,200 template sentences for prices, times, dates, ages and phone numbers
   with every number written out. 25% are questions and 10% exclamations.
3. **Audio.** Qwen3-TTS 1.7B Base read every sentence in both voices. Each clip
   was transcribed with Whisper large-v3-turbo and generated again when the
   transcript, speaking rate or silences were off (up to two attempts).
   17,913 clips were kept.
4. **Alignment.** Phone durations come from CTC forced alignment with
   wav2vec2 (espeak phone labels). 97.5% of clips passed the alignment checks,
   leaving 14.1 hours of training audio.
5. **Training.** 100,000 steps, adversarial losses from step 8,000, about six
   hours on one GPU.

```bash
export TINY_VOCA_LANGUAGE=fr PYTHONPATH=src
python scripts/french/design_voices.py                      # candidate voices and a review page
python scripts/french/build_text_fr.py \
  --collector data/french/text/sentence-collector.txt \
  --output data/french/text/sentences.jsonl                 # sentence list
bash scripts/french/run_fr.sh data/french/voices.json fr_v1   # generate, align, train, evaluate
```

`voices.json` maps each voice to its reference clip and transcript:

```json
{"FR_F1": {"reference_audio": "path/to/ref.wav", "reference_text": "Bonjour et bienvenue. ..."}}
```

## Results

Whisper large-v3-turbo on the generated audio:

| Set | FR_F1 WER | FR_M1 WER |
|---|---|---|
| Unseen pronunciation test sentences | 0% | 0% |
| Held-out test sentences | 0% | 4.5% (same as the teacher recordings) |

Pitch movement matches the teacher (3.2 vs 3.3 semitones standard deviation on
the test set). Speech is about 8% faster than the teacher and pauses are
somewhat shorter; `--speed 0.93` compensates.

## Limits

- **The male voice is less natural than the female voice.** Its teacher is
  breathy above 1 kHz, and a strictly periodic pulse there was heard as a
  mechanical buzz. The vocoder now replaces half of FR_M1's pulse power above
  1 kHz with noise under the same spectral envelope (`Config.aperiodicity`, no
  extra parameters), which removes most of the buzz; a faint trace can remain.
- eSpeak NG misses some liaisons (*comment‿allez*), adds a few false ones
  (*Dupont‿et*), and reads *fils* (son) as [fil]. Sentences with *fils* were left
  out of the training data.
- Two fixed voices. No voice cloning. No streaming.

## Web demo

The [demo](https://www.suncostlabs.com/tinyvocafr) runs the model in the browser with ONNX Runtime Web and
eSpeak NG compiled to WebAssembly. Its source is in `web/public/`: static files with relative paths, ready to
upload to any static host.

```bash
cd web/public && python3 -m http.server 8078   # then open http://127.0.0.1:8078/
```

## Tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests
```

## License

Apache-2.0 for code and weights, as upstream. The training sentences are CC0
(Common Voice) or written for this project. The teacher model, Qwen3-TTS, is
Apache-2.0. eSpeak NG (GPL-3.0) is called as an external program and is not
included.

## Citation

Please cite the original model:

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
