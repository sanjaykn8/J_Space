"""
server.py
---------
A small local Flask server that:
  1. Serves the dashboard (static/index.html).
  2. Loads YOUR local Gemma model on request.
  3. Exposes a few JSON endpoints the dashboard calls to run real forward
     passes / generation against that model and get back real numbers.

This is meant to run on your own machine, for your own use — there is no
authentication, and it is not hardened for exposure to a network. Run it,
open the printed localhost URL, and it never leaves your machine.

Usage:
    python server.py
    # then open http://127.0.0.1:5000 in your browser
"""

from __future__ import annotations

import traceback

from flask import Flask, jsonify, request, send_from_directory

import jspace_core as core
from concepts import CONCEPTS, DEFAULT_PROMPT, DEFAULT_NEUTRAL_EXEMPLAR, DEFAULT_SAFE_EXEMPLAR

app = Flask(__name__, static_folder="static", static_url_path="")

# Single-process, single-user in-memory state. This tool is meant to be run
# locally by one person at a time; it intentionally avoids the complexity of
# real session/multi-user handling.
STATE = {
    "lm": None,                 # core.LoadedModel
    "hidden_states": None,      # last forward pass's hidden_states tuple
    "tokens": None,             # last forward pass's decoded tokens
    "layout": None,             # cached concept_space_layout() result
    "steer_direction": None,    # torch tensor
    "steer_layer": None,
}


def require_model():
    if STATE["lm"] is None:
        raise RuntimeError("No model is loaded yet. Load one from the sidebar first.")
    return STATE["lm"]


@app.errorhandler(Exception)
def handle_error(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 400


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/defaults")
def defaults():
    return jsonify({
        "prompt": DEFAULT_PROMPT,
        "neutral_exemplar": DEFAULT_NEUTRAL_EXEMPLAR,
        "safe_exemplar": DEFAULT_SAFE_EXEMPLAR,
        "concepts": CONCEPTS,
    })


@app.route("/api/model_info")
def model_info():
    lm = STATE["lm"]
    if lm is None:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded": True,
        "model_name": lm.model_name,
        "num_layers": lm.num_layers,
        "hidden_size": lm.hidden_size,
        "vocab_size": lm.vocab_size,
        "device": lm.device,
        "dtype": lm.dtype,
        "norm_path": lm.norm_path,
        "is_multimodal": lm.is_multimodal,
    })


@app.route("/api/load_model", methods=["POST"])
def load_model_route():
    data = request.get_json(force=True)
    model_path = data.get("model_path", "").strip()
    if not model_path:
        return jsonify({"error": "model_path is required (a local folder or a HF repo id)."}), 400
    device = data.get("device", "auto")
    dtype = data.get("dtype", "auto")
    norm_path = data.get("norm_path") or None
    gguf_file = data.get("gguf_file") or None

    lm = core.load_model(model_path, device=device, dtype=dtype, norm_path=norm_path, gguf_file=gguf_file)
    STATE["lm"] = lm
    STATE["hidden_states"] = None
    STATE["tokens"] = None
    STATE["layout"] = None
    STATE["steer_direction"] = None
    STATE["steer_layer"] = None

    return jsonify({
        "loaded": True,
        "model_name": lm.model_name,
        "num_layers": lm.num_layers,
        "hidden_size": lm.hidden_size,
        "vocab_size": lm.vocab_size,
        "device": lm.device,
        "dtype": lm.dtype,
        "norm_path": lm.norm_path,
        "is_multimodal": lm.is_multimodal,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    lm = require_model()
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt.strip():
        return jsonify({"error": "prompt is empty"}), 400

    input_ids, tokens, hidden_states = core.run_forward(lm, prompt)
    grid = core.compute_concept_grid(lm, hidden_states, CONCEPTS)  # [layers, tokens, concepts]

    STATE["hidden_states"] = hidden_states
    STATE["tokens"] = tokens

    if STATE["layout"] is None:
        concept_xyz, bg_xyz = core.concept_space_layout(lm, CONCEPTS)
        STATE["layout"] = {"concepts": concept_xyz, "background": bg_xyz}

    return jsonify({
        "tokens": tokens,
        "num_layers": lm.num_layers,
        "concepts": CONCEPTS,
        "concept_grid": grid.tolist(),
        "layout": STATE["layout"],
    })


@app.route("/api/readout", methods=["POST"])
def readout():
    lm = require_model()
    if STATE["hidden_states"] is None:
        return jsonify({"error": "Run /api/analyze first."}), 400
    data = request.get_json(force=True)
    layer = int(data.get("layer", 0))
    token = int(data.get("token", 0))
    top = core.topk_logit_lens(lm, STATE["hidden_states"], layer, token, k=8)
    return jsonify({"layer": layer, "token": token, "top": top})


@app.route("/api/steering_vector", methods=["POST"])
def steering_vector():
    lm = require_model()
    data = request.get_json(force=True)
    pos_prompt = data.get("safe_exemplar", DEFAULT_SAFE_EXEMPLAR)
    neg_prompt = data.get("neutral_exemplar", DEFAULT_NEUTRAL_EXEMPLAR)
    layer = int(data.get("layer", lm.num_layers // 2))

    direction = core.compute_steering_direction(lm, pos_prompt, neg_prompt, layer)
    STATE["steer_direction"] = direction
    STATE["steer_layer"] = layer

    norm_info = {}
    if STATE["hidden_states"] is not None:
        norm_info = core.hidden_norm_stats(STATE["hidden_states"], layer)

    return jsonify({"ready": True, "layer": layer, "hidden_norm": norm_info})


@app.route("/api/generate", methods=["POST"])
def generate():
    lm = require_model()
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    steer = bool(data.get("steer", False))
    alpha = float(data.get("alpha", 8.0))
    max_new_tokens = int(data.get("max_new_tokens", 40))

    direction = STATE["steer_direction"] if steer else None
    layer = STATE["steer_layer"] if steer else None
    if steer and direction is None:
        return jsonify({"error": "Compute a steering vector first."}), 400

    text = core.generate_text(lm, prompt, max_new_tokens=max_new_tokens,
                               layer=layer, direction=direction, alpha=alpha)
    candidates = core.next_token_candidates(lm, prompt, k=4,
                                             layer=layer, direction=direction, alpha=alpha)
    return jsonify({"text": text, "candidates": candidates, "steered": steer, "alpha": alpha})


if __name__ == "__main__":
    print("\n  J-Space live dashboard starting…")
    print("  Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
