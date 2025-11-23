import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import types
import torch.nn.functional as F

LAYER_LABELS = {i: f"Layer {i}" for i in range(32)}


def trace_moe(model_name, prompt, output_dir="traces"):
    """
    Trace MoE model layers and gate decisions, saving results to a JSON file.
    """
    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Trace storage
    trace = {
        "prompt": prompt,
        "tokens": [],
        "layers": []
    }

    def get_token_list(input_ids):
        return tokenizer.convert_ids_to_tokens(input_ids[0])

    def hook_layer(layer_index):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms = hidden.norm(dim=-1).mean(dim=0).tolist()

            if len(trace["layers"]) <= layer_index:
                trace["layers"].append({
                    "layer_index": layer_index,
                    "label": LAYER_LABELS.get(layer_index, f"Layer {layer_index}"),
                    "hidden_norms": norms
                })
            else:
                trace["layers"][layer_index]["hidden_norms"] = norms
        return hook_fn

    # Shared buffer for extracted MoE gate info
    moe_gate_trace = {}

    def patch_moe_gate(moe_block, layer_index, top_k: int = 2):
        """
        Patch a Mixtral-style MoE block. Assumes moe_block.gate exists.
        """
        original_forward = moe_block.forward

        def hacked_forward(self, hidden_states):
            try:
                gate_logits = self.gate(hidden_states)
                top_scores, top_indices = gate_logits.topk(top_k, dim=-1)
                gates = F.softmax(top_scores, dim=-1)

                moe_gate_trace[layer_index] = {
                    "gates": gates.detach().cpu(),
                    "indices": top_indices.detach().cpu(),
                }

            except Exception as e:
                print(f"[WARN] Failed to read MoE gate in layer {layer_index}: {e}")

            return original_forward(hidden_states)

        moe_block.forward = types.MethodType(hacked_forward, moe_block)
        print(f"[*] Patched MoE gate in layer {layer_index}")

    for idx, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_layer(idx))

        moe_block = None

        # Mixtral
        if hasattr(layer, "block_sparse_moe"):
            moe_block = layer.block_sparse_moe

        # Fallbacks
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            moe_block = layer.mlp
        elif hasattr(layer, "moe"):
            moe_block = layer.moe

        if moe_block is not None:
            patch_moe_gate(moe_block, idx)
        else:
            print(f"[INFO] No MoE gate found in layer {idx}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    trace["tokens"] = get_token_list(inputs["input_ids"])

    # Run inference
    with torch.no_grad():
        _ = model(**inputs)

    # Inject MoE traces
    for layer_idx, moe_data in moe_gate_trace.items():
        if layer_idx < len(trace["layers"]):
            trace["layers"][layer_idx]["moe"] = {
                "gate_scores": moe_data["gates"].tolist(),
                "selected_experts": moe_data["indices"].tolist()
            }

    # Save JSON trace
    os.makedirs(output_dir, exist_ok=True)

    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt[:40])
    out_file = f"{output_dir}/trace_{model_name.split('/')[-1]}_{safe_prompt}.json"

    with open(out_file, "w") as f:
        json.dump(trace, f, indent=2)

    return out_file
