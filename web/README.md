# tiny-voca-fr demo

Static page that runs [tiny-voca-fr](https://github.com/DanielSuncost/tiny-voca-fr) in the browser.
Upload the files in `public/` to any static host (for example `/tinyvocafr/`); paths are relative.

- `index.html`, `player.js`: page and gapless sentence player (adapted from tiny-voca's demo, Apache-2.0)
- `worker.js`: runs the phonemizer and the model in a module Web Worker
- `frontend-fr.js`: French text to model tokens, the same rules as `tiny_voca.text` with `TINY_VOCA_LANGUAGE=fr`

Loaded at runtime, not hosted here:
- the ONNX model and `model.json` from `huggingface.co/DanielSuncost/tiny-voca-fr`
- ONNX Runtime Web 1.20.1 from jsDelivr
- eSpeak NG (GPL-3.0), compiled to WebAssembly, from the `espeak-ng@1.0.2` npm package on jsDelivr

Local test: `cd public && python3 -m http.server 8078`, then open http://127.0.0.1:8078/.
