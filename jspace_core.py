"""
jspace_core.py
---------------
Real interpretability plumbing for the live J-space dashboard.

Everything in this file operates on the actual weights and activations of
the model you load. Nothing here is synthetic or simulated. What IS a
simplification (relative to the Anthropic "Jacobian lens" research this
project is inspired by) is called out explicitly in each docstring — please
read those before treating outputs as a faithful reproduction of that paper's
method. In short:

  * The "concept score" and "top-k readout" below use LOGIT LENS
    (unembed each layer's residual stream through the model's own final
    norm + output projection). This is a well-known, much simpler technique
    than the paper's Jacobian lens, which corrects for representational
    drift across layers using a corpus-averaged linearization. Logit lens
    can be noisy or misleading in early/middle layers for that reason.

  * "Steering" here is a simple contrastive activation difference
    (mean hidden state of a "safe" exemplar minus a "neutral" exemplar,
    added back into the residual stream at generation time). This is a
    real, well-established technique in the activation-steering /
    representation-engineering literature, but it is not the paper's
    trained latent guardrail.

  * There is no trained sparse autoencoder (SAE) here. The dashboard's
    "J-Shield" panel reuses the same logit-lens concept scores as a stand-in
    for what an SAE feature detector might report, and says so on screen.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from concepts import CONCEPTS


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(requested: str, device: str):
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: str
    dtype: str
    norm_path: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    model_name: str
    is_multimodal: bool = False


def _get_by_path(obj, path: str):
    return functools.reduce(getattr, path.split("."), obj)


def _find_final_norm(model, manual_path: Optional[str] = None):
    """
    Locate the normalization module applied to the residual stream right
    before the output projection (lm_head). This is architecture-specific,
    so we try a short list of common attribute paths used by Gemma / Llama
    style models in `transformers` — including the nested `language_model`
    submodule used by multimodal ("image-text-to-text") checkpoints such as
    Gemma 4 / Gemma 3n — and fall back to `manual_path` if given.
    Returns (norm_module_or_None, path_string_or_None).
    """
    candidates = [manual_path] if manual_path else []
    candidates += [
        "model.norm", "model.language_model.norm", "language_model.model.norm",
        "language_model.norm", "model.final_layernorm", "transformer.ln_f", "norm",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            mod = _get_by_path(model, path)
            if mod is not None:
                return mod, path
        except AttributeError:
            continue
    return None, None


def _get_layers(model):
    """Return the list/ModuleList of transformer decoder blocks, including
    the nested `language_model` submodule used by multimodal checkpoints."""
    for path in ["model.layers", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers", "transformer.h"]:
        try:
            return _get_by_path(model, path)
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not find the transformer layer list on this model. See README "
        "to add your architecture's attribute path to jspace_core._get_layers()."
    )


def load_model(model_path: str, device: str = "auto", dtype: str = "auto",
                norm_path: Optional[str] = None, gguf_file: Optional[str] = None) -> LoadedModel:
    """
    model_path: a local folder or HF repo id.
      - normal (safetensors/bin) checkpoint: the folder/repo containing them.
      - GGUF checkpoint: the folder/repo that CONTAINS the .gguf file, with
        `gguf_file` set to that file's name. transformers dequantizes GGUF
        into a regular PyTorch model on load (needs the `gguf` pip package).
        Activations reflect the quantized weights in that file (e.g. a
        Q4_K_M checkpoint), which is still real data — just not identical to
        an fp16 checkpoint of the same model.

    Some Gemma checkpoints (e.g. Gemma 4, Gemma 3n) are multimodal
    ("image-text-to-text") models rather than plain causal LMs. We try the
    plain `AutoModelForCausalLM` / `AutoTokenizer` path first, and fall back
    to `AutoModelForImageTextToText` / `AutoProcessor` if that fails — the
    rest of this file only ever uses `lm.tokenizer` for text, so everything
    downstream keeps working against the multimodal checkpoint's text-only
    forward pass.
    """
    resolved_device = pick_device(device)
    torch_dtype = pick_dtype(dtype, resolved_device)

    common_kwargs = {}
    if gguf_file:
        common_kwargs["gguf_file"] = gguf_file

    def _load_kwargs():
        kwargs = {"torch_dtype": torch_dtype, **common_kwargs}
        if resolved_device == "cuda" and not gguf_file:
            # device_map="auto" is not reliably supported together with GGUF
            # loading across transformers versions; keep GGUF loads simple
            # and move the whole model afterward instead.
            kwargs["device_map"] = "auto"
        return kwargs

    is_multimodal = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **common_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_path, **_load_kwargs())
    except Exception as e_causal:
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e_import:
            raise RuntimeError(
                "This checkpoint didn't load as a plain causal LM, and your "
                "transformers version doesn't have AutoModelForImageTextToText "
                "(needed for multimodal Gemma checkpoints like Gemma 4 / "
                "Gemma 3n). Try `pip install -U transformers`. "
                f"Original error: {e_causal}"
            ) from e_import
        try:
            processor = AutoProcessor.from_pretrained(model_path, **common_kwargs)
            model = AutoModelForImageTextToText.from_pretrained(model_path, **_load_kwargs())
        except Exception as e_multimodal:
            raise RuntimeError(
                f"Failed to load as a causal LM ({e_causal}) and also failed "
                f"to load as a multimodal image-text-to-text model "
                f"({e_multimodal})."
            ) from e_multimodal
        tokenizer = processor.tokenizer
        is_multimodal = True

    if resolved_device != "cuda" or gguf_file:
        model = model.to(resolved_device)
    model.eval()

    norm_module, found_path = _find_final_norm(model, norm_path)
    layers = _get_layers(model)

    cfg = model.config
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        dtype=str(torch_dtype).replace("torch.", ""),
        norm_path=found_path or "(none found — readouts will be un-normalized)",
        num_layers=len(layers),
        hidden_size=getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", -1),
        vocab_size=getattr(cfg, "vocab_size", None) or getattr(getattr(cfg, "text_config", None), "vocab_size", -1),
        model_name=getattr(cfg, "_name_or_path", model_path),
        is_multimodal=is_multimodal,
    )


# ----------------------------------------------------------------------
# Tokens / display helpers
# ----------------------------------------------------------------------

def pretty_token(tok: str) -> str:
    return tok.replace("\u2581", " ").replace("\u0120", " ").strip() or tok


def concept_token_ids(tokenizer, concepts=CONCEPTS):
    ids = []
    for c in concepts:
        word = c["id"]
        enc = tokenizer.encode(" " + word, add_special_tokens=False)
        if not enc:
            enc = tokenizer.encode(word, add_special_tokens=False)
        ids.append(enc[0] if enc else tokenizer.unk_token_id or 0)
    return ids


# ----------------------------------------------------------------------
# Forward pass + logit-lens scoring
# ----------------------------------------------------------------------

def run_forward(lm: LoadedModel, prompt: str):
    """Run one forward pass and return (input_ids, tokens, hidden_states tuple)."""
    inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
    with torch.no_grad():
        out = lm.model(**inputs, output_hidden_states=True, return_dict=True)
    tokens = [pretty_token(t) for t in lm.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])]
    return inputs["input_ids"], tokens, out.hidden_states


def _apply_norm(lm: LoadedModel, h: torch.Tensor) -> torch.Tensor:
    norm_module, _ = _find_final_norm(lm.model, None if lm.norm_path.startswith("(") else lm.norm_path)
    if norm_module is not None:
        return norm_module(h)
    return h


def compute_concept_grid(lm: LoadedModel, hidden_states, concepts=CONCEPTS) -> np.ndarray:
    """
    Real per-layer, per-token, per-concept logit-lens scores.
    Returns a numpy array of shape [num_layers, seq_len, num_concepts].
    Cheap: only `len(concepts)` dot products per layer/token, not a full
    vocab softmax.
    """
    lm_head = lm.model.get_output_embeddings()
    ids = concept_token_ids(lm.tokenizer, concepts)
    ids_t = torch.tensor(ids, device=lm_head.weight.device)
    W = lm_head.weight.index_select(0, ids_t)  # [num_concepts, hidden]
    bias = lm_head.bias
    b = bias.index_select(0, ids_t) if bias is not None else None

    num_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    grid = np.zeros((num_layers, seq_len, len(concepts)), dtype=np.float32)

    for L in range(num_layers):
        h = hidden_states[L + 1][0]  # [seq, hidden], output after block L
        h = _apply_norm(lm, h)
        scores = h.to(W.dtype) @ W.T
        if b is not None:
            scores = scores + b
        grid[L] = scores.detach().float().cpu().numpy()
    return grid


def topk_logit_lens(lm: LoadedModel, hidden_states, layer: int, token: int, k: int = 8):
    """
    Real top-k readout for ONE (layer, token) position, computed against the
    FULL vocabulary (not just the tracked concept list). Cheap enough to run
    on demand for a single position.
    """
    lm_head = lm.model.get_output_embeddings()
    h = hidden_states[layer + 1][0, token].unsqueeze(0)  # [1, hidden] — keep 2D for norm-module safety
    h = _apply_norm(lm, h)
    logits = lm_head(h).squeeze(0)
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    out = []
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        tok_str = lm.tokenizer.convert_ids_to_tokens([idx])[0]
        out.append({"token": pretty_token(tok_str), "prob": float(prob)})
    return out


def hidden_norm_stats(hidden_states, layer: int) -> dict:
    h = hidden_states[layer + 1][0]
    norms = h.norm(dim=-1).detach().float().cpu().numpy()
    return {"mean": float(norms.mean()), "min": float(norms.min()), "max": float(norms.max())}


# ----------------------------------------------------------------------
# J-space 3D layout (real PCA of unembedding directions)
# ----------------------------------------------------------------------

def concept_space_layout(lm: LoadedModel, concepts=CONCEPTS, n_background: int = 260, seed: int = 0):
    """
    Projects the real unembedding-row vectors of the tracked concepts (plus a
    random sample of ordinary vocabulary rows, as background context) down to
    3D with PCA. This is real geometry from the model's own output embedding
    space, not a fabricated layout.
    """
    lm_head = lm.model.get_output_embeddings()
    W = lm_head.weight.detach().float().cpu().numpy()  # [vocab, hidden]

    ids = concept_token_ids(lm.tokenizer, concepts)
    concept_vecs = W[ids]

    rng = np.random.default_rng(seed)
    vocab_size = W.shape[0]
    n_background = min(n_background, vocab_size)
    bg_idx = rng.choice(vocab_size, size=n_background, replace=False)
    bg_vecs = W[bg_idx]

    pca = PCA(n_components=3, random_state=seed)
    pca.fit(np.vstack([concept_vecs, bg_vecs]))
    concept_xyz = pca.transform(concept_vecs)
    bg_xyz = pca.transform(bg_vecs)

    # normalize to roughly [-1, 1] for easy client-side rendering
    all_xyz = np.vstack([concept_xyz, bg_xyz])
    scale = np.abs(all_xyz).max() or 1.0
    concept_xyz = concept_xyz / scale
    bg_xyz = bg_xyz / scale

    return concept_xyz.tolist(), bg_xyz.tolist()


# ----------------------------------------------------------------------
# Activation steering (contrastive activation addition)
# ----------------------------------------------------------------------

class _AddDirectionHook:
    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0] + self.alpha * self.direction.to(output[0].dtype).to(output[0].device)
            return (hs,) + output[1:]
        return output + self.alpha * self.direction.to(output.dtype).to(output.device)


def compute_steering_direction(lm: LoadedModel, pos_prompt: str, neg_prompt: str, layer: int):
    """
    direction = (last-token hidden state of pos_prompt) - (last-token hidden
    state of neg_prompt) at the given layer, unit-normalized. This is the
    standard "contrastive activation addition" recipe used widely in the
    activation-steering literature.
    """
    _, _, hs_pos = run_forward(lm, pos_prompt)
    _, _, hs_neg = run_forward(lm, neg_prompt)
    h_pos = hs_pos[layer + 1][0, -1]
    h_neg = hs_neg[layer + 1][0, -1]
    direction = (h_pos - h_neg).detach()
    direction = direction / (direction.norm() + 1e-6)
    return direction


def generate_text(lm: LoadedModel, prompt: str, max_new_tokens: int = 40,
                   layer: Optional[int] = None, direction: Optional[torch.Tensor] = None,
                   alpha: float = 0.0):
    """Greedy generation, optionally with a steering vector added at `layer`."""
    inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
    handle = None
    if direction is not None and alpha != 0.0 and layer is not None:
        layers = _get_layers(lm.model)
        handle = layers[layer].register_forward_hook(_AddDirectionHook(direction, alpha))
    try:
        with torch.no_grad():
            out = lm.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=lm.tokenizer.eos_token_id or lm.tokenizer.pad_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return lm.tokenizer.decode(new_tokens, skip_special_tokens=True)


def next_token_candidates(lm: LoadedModel, prompt: str, k: int = 4,
                           layer: Optional[int] = None, direction: Optional[torch.Tensor] = None,
                           alpha: float = 0.0):
    """Top-k next-token candidates for `prompt`, optionally under steering."""
    inputs = lm.tokenizer(prompt, return_tensors="pt").to(lm.model.device)
    handle = None
    if direction is not None and alpha != 0.0 and layer is not None:
        layers = _get_layers(lm.model)
        handle = layers[layer].register_forward_hook(_AddDirectionHook(direction, alpha))
    try:
        with torch.no_grad():
            out = lm.model(**inputs, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
    logits = out.logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k)
    return [
        {"token": pretty_token(lm.tokenizer.convert_ids_to_tokens([idx])[0]), "prob": float(p)}
        for p, idx in zip(top.values.tolist(), top.indices.tolist())
    ]
