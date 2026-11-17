// Runs the phonemizer (eSpeak NG compiled to WebAssembly) and the model (ONNX Runtime Web) off the main
// thread. Messages in: {type: "load"} and {type: "speak", id, text, voice, speed}. Messages out: status, ready,
// sentence (with its audio, as soon as it is made), done (with the whole clip), error.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
import ESpeakNg from "https://cdn.jsdelivr.net/npm/espeak-ng@1.0.2/dist/espeak-ng.js";
import { phonemize, sentences, NothingToPronounce } from "./frontend-fr.js";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const ESPEAK_WASM = "https://cdn.jsdelivr.net/npm/espeak-ng@1.0.2/dist/espeak-ng.wasm";
const WEIGHTS = "https://huggingface.co/DanielSuncost/tiny-voca-fr/resolve/main/";
const VERSION = "2";   // bump when the model changes, so browsers fetch it again

let session, config, espeakModule;
const status = (text, progress) => postMessage({ type: "status", text, progress });

// Downloads are kept in the browser's cache storage: a second visit starts without the network.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    for (const name of await caches.keys())     // drop the copies of earlier model versions
      if (name.startsWith("tiny-voca-fr-") && name !== "tiny-voca-fr-" + VERSION) caches.delete(name);
    store = await caches.open("tiny-voca-fr-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(url, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

const progress = {};
async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  progress[url] = [0, total];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progress[url][0] = received;
    const [got, all] = Object.values(progress).reduce((a, b) => [a[0] + b[0], a[1] + b[1]], [0, 0]);
    status(label, got / all);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  const bytes = await download(ESPEAK_WASM, "Downloading the model and the phonemizer");
  espeakModule = await WebAssembly.compile(bytes);
  await espeak("Bonjour.");
}

// One run of `espeak-ng -q --ipa --sep=_ -v fr-fr TEXT`, the command the Python package uses.
async function espeak(text) {
  let out = "";
  await ESpeakNg({
    arguments: ["-q", "--ipa", "--sep=_", "-v", "fr-fr", text],
    print: (line) => { out += line + "\n"; },
    printErr: () => {},
    instantiateWasm: (imports, ready) => { WebAssembly.instantiate(espeakModule, imports).then(ready); return {}; },
  });
  return out;
}

async function loadModel() {
  config = await (await fetch(WEIGHTS + "onnx/model.json")).json();
  const bytes = await download(WEIGHTS + "onnx/model.onnx", "Downloading the model and the phonemizer");
  status("Starting the model");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

// One utterance holds at most max_tokens phonemes; longer sentences are cut at commas, then by words.
async function fit(sentence) {
  let result;
  try {
    result = await phonemize(sentence, espeak);
  } catch (error) {
    if (error instanceof NothingToPronounce) return [];
    throw error;
  }
  if (result.ids.length <= config.max_tokens - 12) return [result];
  const pieces = sentence.split(/(?<=[,;:—])\s+/);
  const parts = pieces.length > 1 ? pieces : (() => {
    const words = sentence.split(" ");
    if (words.length < 2) return [];
    const half = Math.ceil(words.length / 2);
    return [words.slice(0, half).join(" "), words.slice(half).join(" ")];
  })();
  const out = [];
  for (const part of parts) out.push(...await fit(part));
  return out;
}

async function synthesize(phones, voice, speed) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const output = await session.run({
    tokens: integers(phones.ids), stress: integers(phones.stress),
    voice: new ort.Tensor("int64", BigInt64Array.from([BigInt(voice)]), [1]),
    speed: scalar(speed), pitch_shift: scalar(0), pitch_range: scalar(1),
  });
  return output.audio.data;
}

// The model sometimes peaks above 1.0, which would clip in the 16-bit file. Scale the whole clip down then.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

// A full stop is added when the text ends without a mark, so the last syllable falls instead of breaking off.
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…»"]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : body + ".";
}

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const list = sentences(finished(text));
  const chunks = [];
  let phonemeTime = 0, modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = await fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = await synthesize(phones, voice, speed);
      modelTime += performance.now() - mark;
      chunks.push(audio);
      const heard = audio.slice();
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.symbols.slice(1, -1).join(" "),
                    audio: heard, sampleRate: config.sample_rate }, [heard.buffer]);
    }
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: "Nothing in the text could be read.", plain: true });
    return;
  }
  const audio = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      status("Downloading the model and the phonemizer", 0);
      await Promise.all([loadPhonemizer(), loadModel()]);
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
