# noesis_trace.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import json

# mistralai/Mistral-7B-v0.1 tested and supported

class NoesisTracer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", trace_mode="full", token_wise=False):
        self.model_name = model_name
        self.trace_mode = trace_mode  # 'submodules', 'residual', 'norms', 'full'
        self.token_wise = token_wise  # If True, trace each token separately
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trace_data = defaultdict(list)
        self.last_tokens = []  # store last decoded tokens

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                traced = output.detach().cpu().squeeze()
                if not self.token_wise:
                    traced = traced.mean(dim=0) if traced.ndim > 1 else traced
                self.trace_data[name].append(traced.tolist())
            elif isinstance(output, tuple):
                traced = output[0].detach().cpu().squeeze()
                if not self.token_wise:
                    traced = traced.mean(dim=0) if traced.ndim > 1 else traced
                self.trace_data[name].append(traced.tolist())
        return hook

    def trace(self, prompt):
        self.trace_data.clear()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.last_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if self.trace_mode == "submodules":
                if "mlp" in name or "self_attn" in name:
                    hooks.append(module.register_forward_hook(self._hook_fn(name)))

            elif self.trace_mode == "norms":
                if "input_layernorm" in name or "post_attention_layernorm" in name:
                    hooks.append(module.register_forward_hook(self._hook_fn(name)))

            elif self.trace_mode == "residual":
                if "model.layers." in name and name.count(".") == 2:
                    hooks.append(module.register_forward_hook(self._hook_fn(f"{name}.residual_out")))

            elif self.trace_mode == "full":
                if ("mlp" in name or "self_attn" in name or
                    "input_layernorm" in name or "post_attention_layernorm" in name):
                    hooks.append(module.register_forward_hook(self._hook_fn(name)))
                elif "model.layers." in name and name.count(".") == 2:
                    hooks.append(module.register_forward_hook(self._hook_fn(f"{name}.residual_out")))

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hooks
        for h in hooks:
            h.remove()

        return dict(self.trace_data)

    def save_trace(self, trace, filepath):
        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)