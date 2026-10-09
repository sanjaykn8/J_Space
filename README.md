# J-Space Console — Live

A local, self-hosted dashboard that runs **real forward passes and generation
against a Gemma model you already have on your machine**, and visualizes:

- the residual stream at every layer/token (`output_hidden_states`)
- a **logit-lens** style readout of ~20 tracked "concept" words at every
  layer/token, using your model's own unembedding matrix
- a 3D layout of those concepts (real PCA of their unembedding vectors)
- a real **activation-steering** demo (contrastive activation addition)

It is inspired by the "J-lens / J-space" ideas from Anthropic's public
interpretability research on verbalizable, global-workspace-like
representations in language models — but it does **not** reproduce that
paper's method. See "What's real vs. simplified here" below.

Everything runs on `127.0.0.1`. Nothing is uploaded anywhere.

---

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` is the big one. If you already have a working PyTorch install tuned
for your GPU (CUDA/ROCm/MPS), keep it and just `pip install` the rest of
`requirements.txt` without letting it clobber your torch build.

## 2. Run

```bash
python server.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 3. Use it

1. **Model Setup** — enter the path to your local Gemma checkpoint (a local
   folder passed to `from_pretrained`, or a Hugging Face repo id if you'd
   rather download one). Pick a device/dtype, click **Load model**.
2. **Prompt** — edit the prompt if you like, click **Analyze**. This runs one
   real forward pass and fills in every panel.
3. **Layer slider / token clicks** — scrub through the network; every panel
   updates from data already fetched in step 2, except the "Top-8 readout"
   panel, which does one small extra request per position (a real full-vocab
   logit-lens computation).
4. **Activation steering** — edit the two exemplar prompts if you like, pick a
   layer and click **Compute steering vector**, then **Generate** on both
   sides to compare real outputs with and without the intervention.

### Memory / hardware notes

An 8B-parameter model needs real memory:

| dtype       | approx. RAM/VRAM for weights |
|-------------|-------------------------------|
| float32     | ~32 GB                        |
| float16/bf16| ~16 GB                         |
| 4-bit (bnb) | ~5–6 GB (needs `bitsandbytes`, see below) |

- If you have an NVIDIA GPU, pick `device: cuda`, `dtype: bfloat16` (or
  `float16`).
- Apple Silicon: `device: mps` works for many models, but not all ops are
  implemented for `mps` in every `transformers`/`torch` version — if you hit
  an error, fall back to `cpu`.
- CPU-only with an 8B model will work but will be slow, especially
  `Generate`. Everything in the "Analyze" panels only needs one forward
  pass, so it's the generation buttons that will feel slowest.
- For a lower-VRAM GPU, install `bitsandbytes` and load with 4-bit
  quantization — this project doesn't wire that up automatically (to keep
  `jspace_core.py` simple), but it's a small change to `load_model()` in
  `jspace_core.py`: pass `load_in_4bit=True` via a `BitsAndBytesConfig` in
  the `from_pretrained(...)` call.

### Using a .gguf checkpoint

**Current status for Gemma 4 (as of this writing): GGUF loading will fail.**
`transformers`' GGUF loader raises `Architecture gemma4 not supported` /
`GGUF model with architecture gemma4 is not supported yet` for Gemma 4 GGUF
files — this is confirmed with people hitting the same error on very recent
`transformers` releases too, so upgrading `transformers` alone will not fix
it. It's an upstream gap (gguf support has to be added per-architecture),
not a bug in this project or a version mismatch on your end.

For Gemma 4 specifically, use the **safetensors checkpoint** instead — e.g.
`google/gemma-4-E4B-it` on Hugging Face (~16 GB, bf16). This project's
loader now also auto-detects that Gemma 4 is a **multimodal**
("image-text-to-text") checkpoint and falls back to
`AutoModelForImageTextToText` / `AutoProcessor` automatically if the plain
`AutoModelForCausalLM` path fails — you'll see a note about this in the load
status once it's loaded. Everything downstream (forward passes, logit lens,
steering, generation) runs text-only against that checkpoint as normal.

For **other**, non-Gemma-4 GGUF checkpoints, `transformers` GGUF loading may
still work fine (it covers Llama/Gemma-2/Gemma-3/Mistral/Qwen-family models
that were added earlier) — see the details below if you want to try it.

- Set **model path** to the *folder* containing your `.gguf` file (a local
  directory is fine — it does not need to be a full HF repo with
  `config.json`; GGUF embeds its own architecture/tokenizer metadata).
- Set **GGUF filename** to the file itself, e.g. `gemma-9b-it-Q4_K_M.gguf`.
- Click Load model as usual.

Two things to know:
- `transformers` **dequantizes** the GGUF weights into a regular PyTorch
  model in memory on load. So a 4-bit GGUF file still needs roughly the RAM
  of the dequantized dtype (see the memory table above) — it does not stay
  compressed in memory, it's just a smaller *download*. If that's a problem,
  choose a smaller quantized model file, or use a smaller Gemma variant.
- Every number this tool shows will reflect the **quantized** weights in
  that specific `.gguf` file, not an idealized full-precision checkpoint —
  that's still real, honest data, just slightly different from what you'd
  get from an fp16/bf16 checkpoint of the same model.

## If model loading or the readout panels error out

Different Gemma checkpoints (and different `transformers` versions) can
structure the model slightly differently internally. This project tries to
auto-detect two things:

- The **final norm layer** applied right before the output projection. If
  auto-detection fails, you'll see a note in the load status, and the
  logit-lens readouts will be less accurate. Fix: open a Python shell,
  `print(model)` , find the attribute path to the final norm (something like
  `model.norm`), and paste it into the **"final-norm override"** field in
  the Model Setup panel before loading.
- The **decoder layer list**, tried as `model.layers` then `transformer.h`.
  If your checkpoint uses something else, edit `_get_layers()` in
  `jspace_core.py` and add your architecture's attribute path.

## What's real vs. simplified here

This project is built to be honest about the gap between "a fun local
interpretability toy" and the actual published research it's inspired by.

**Real, computed live from your model:**
- residual stream / hidden states at every layer and token
- the "concept score" grid (logit-lens dot products against your model's own
  unembedding matrix)
- the top-8 full-vocabulary readout at a selected layer/token
- the 3D concept layout (PCA of real unembedding row vectors, plus a random
  sample of real vocabulary rows as background context)
- the steering vector (a real mean-difference-of-activations direction) and
  the generations with/without it added to the residual stream

**Deliberate simplifications, not reproductions of the source research:**
- **Logit lens vs. Jacobian lens.** This tool uses "logit lens" (unembed a
  layer's residual stream through the model's own final norm + output
  projection). The Anthropic research this is inspired by describes a more
  sophisticated **Jacobian lens**, fit by linearizing and averaging over a
  large corpus to correct for representational drift across layers. Logit
  lens is a well-known, much cheaper approximation, and can be noisy or
  misleading, especially in early/middle layers.
- **No trained SAE.** The "J-Shield" panel does not use a sparse
  autoencoder. It reuses the same logit-lens concept scores as a stand-in
  for what a trained SAE-based feature detector might report.
- **Steering vector = contrastive activation addition from two exemplar
  prompts**, not a trained latent guardrail. It's a real, well-established
  technique (see "representation engineering" / "activation steering"
  literature), just a simple version of it.
- The tracked "concept" score for a multi-token word only looks at the
  probability the model's own unembedding assigns to that word's **first
  subword token** — a standard shortcut in logit-lens style analyses, not a
  measure of the whole phrase.

None of this is a safety guarantee. Steering effects are empirical, can be
partial, and can have side effects on unrelated behavior.

## Files

```
server.py         Flask app: serves the dashboard + JSON API
jspace_core.py    Model loading, hidden-state capture, logit lens, steering
concepts.py       The list of tracked concept words (edit this freely)
static/index.html The dashboard itself (vanilla HTML/CSS/JS, no build step)
requirements.txt
```

## Customizing

- Add/remove/rename tracked concepts in `concepts.py`. Categories (`task`,
  `risk`, `safety`, `unc`) just control the color used in the dashboard.
- Change the default prompt and steering exemplars, also in `concepts.py`.
- The frontend is a single static HTML file with no build step — open it in
  an editor and change colors, layout, panel copy, etc. directly.
