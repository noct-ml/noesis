import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import types
import torch.nn.functional as F

# Human-readable labels per transformer layer
LAYER_LABELS = {
    0:  "Lexical Primer",
    1:  "Symbol Refiner",
    2:  "Positional Encoder",
    3:  "Local Context Synthesizer",
    4:  "Short-Term Planner",
    5:  "Dependency Tracker",
    6:  "Syntax Harmonizer",
    7:  "Early Meaning Builder",
    8:  "Coherence Assembler",
    9:  "Attention Sculptor",
    10: "Midrange Fusion",
    11: "Comprehension Kernel",
    12: "Semantic Elevator",
    13: "Long-Range Matcher",
    14: "Core Reason Engine",
    15: "Memory Crosslinker",
    16: "Conceptual Integrator",
    17: "Truth Stabilizer",
    18: "Abstraction Synthesizer",
    19: "Narrative Architect",
    20: "Temporal Linker",
    21: "Alignment Amplifier",
    22: "Discourse Calibrator",
    23: "Context Magnetizer",
    24: "Goal Projector",
    25: "Intent Refiner",
    26: "Moral Weight Sorter",
    27: "Language Polisher",
    28: "Stylistic Filter",
    29: "Token Finalizer",
    30: "Prediction Tuner",
    31: "Causal Closure"
}

def trace_moe(model_name, prompt, output_dir="traces"):
    """
    Trace MoE model layers and gate decisions, saving results to a JSON file.
    
    Args:
        model_name (str): Name of the MoE model (e.g., 'deepseek/R-1').
        prompt (str): Input prompt for inference.
        output_dir (str): Directory to save trace JSON file.
    
    Returns:
        str: Path to the saved trace file.
    """
    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
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
            norms = hidden.norm(dim=-1).mean(dim=0).tolist()  # Avg L2 norm per token
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

    def patch_moe_gate(module, layer_index):
        original_forward = module.forward

        def hacked_forward(self, hidden_states):
            try:
                # Generic MoE gate logic: assume gate outputs logits or scores for expert selection
                gate_logits = self.gate(hidden_states) if hasattr(self, 'gate') else hidden_states
                top_k = 2  # Default to top-2 experts, adjustable for other models
                top_scores, top_indices = gate_logits.topk(top_k, dim=-1)
                gates = F.softmax(top_scores, dim=-1)

                # Store trace
                moe_gate_trace[layer_index] = {
                    "gates": gates.detach().cpu(),
                    "indices": top_indices.detach().cpu()
                }

                # Call original forward
                output = original_forward(hidden_states)
                return output
            except Exception as e:
                print(f"[WARN] Failed to patch MoE gate in layer {layer_index}: {str(e)}")
                return original_forward(hidden_states)

        module.forward = types.MethodType(hacked_forward, module)
        print(f"[*] Patched MoE gate in layer {layer_index}")

    # Register hooks and patch MoE gates
    for idx, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_layer(idx))
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            patch_moe_gate(layer.mlp.gate, idx)
        elif hasattr(layer, 'gate'):
            patch_moe_gate(layer.gate, idx)
        elif hasattr(layer, 'moe'):
            patch_moe_gate(layer.moe, idx)
        else:
            print(f"[INFO] No MoE gate found in layer {idx}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    trace["tokens"] = get_token_list(inputs["input_ids"])

    # Run inference
    with torch.no_grad():
        _ = model(**inputs)

    # Inject MoE traces into final JSON
    for layer_idx, moe_data in moe_gate_trace.items():
        if layer_idx < len(trace["layers"]):
            trace["layers"][layer_idx]["moe"] = {
                "gate_scores": moe_data["gates"].tolist(),
                "selected_experts": moe_data["indices"].tolist()
            }

    # Save JSON trace to file
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/trace_{model_name.split('/')[-1]}_{prompt[:20].replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(trace, f, indent=2)

    return out_file

if __name__ == "__main__":
    # Example usage for standalone execution
    model_name = "deepseek/R-1"
    prompt = "Generate a detailed list of all the paradoxes related to quantum logic, causality loops, and metaphysical recursion in simulation theory."
    trace_file = trace_moe(model_name, prompt)
    print(f"Trace saved to: {trace_file}")