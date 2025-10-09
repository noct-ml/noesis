# noesis/tracing/unet_tracer.py
from __future__ import annotations
import re, time, json, gzip
from pathlib import Path
import torch

DEFAULT_INCLUDE = [
    r"down_blocks\.\d+\.resnets\.\d+",
    r"down_blocks\.\d+\.attentions\.\d+",
    r"mid_block\.attentions\.\d+",
    r"mid_block\.resnets\.\d+",
    r"up_blocks\.\d+\.resnets\.\d+",
    r"up_blocks\.\d+\.attentions\.\d+",
]

class UNetTracer:
    def __init__(
        self,
        unet,
        out_dir,
        include_patterns=None,
        keep_samples=False,
        sample_limit=0,
        meta=None,
        gzip_output=True,
        per_step=True,
    ):
        self.unet = unet
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.include = [re.compile(p) for p in (include_patterns or DEFAULT_INCLUDE)]
        self.keep_samples = bool(keep_samples)
        self.sample_limit = int(sample_limit or 0)
        self.meta = meta or {}
        self.gzip_output = bool(gzip_output)
        self.per_step = bool(per_step)

        self.handles = []
        self.records = []        # used when per_step=False
        self._step_buffer = []   # used when per_step=True
        self.current_step = None
        self.run_id = str(int(time.time()))

    # ---------- internals ----------
    def _match(self, name: str) -> bool:
        return any(p.search(name) for p in self.include)

    @torch.no_grad()
    def _hook(self, name: str):
        def fn(_mod, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            x = t.detach()
            stats = {
                "name": name,
                "shape": list(x.shape),
                "dtype": str(x.dtype).replace("torch.", ""),
                "device": str(x.device),
                "mean": float(x.float().mean().cpu()),
                "std":  float(x.float().std().cpu()),
                "abs_mean": float(x.abs().float().mean().cpu()),
                "max": float(x.float().amax().cpu()),
                "min": float(x.float().amin().cpu()),
                "step": self.current_step,
            }
            if self.keep_samples:
                flat = x.flatten()
                if self.sample_limit > 0:
                    flat = flat[: self.sample_limit]
                stats["sample"] = flat.float().cpu().tolist()

            if self.per_step:
                self._step_buffer.append(stats)
            else:
                self.records.append(stats)
        return fn

    # ---------- lifecycle ----------
    def register(self):
        for name, module in self.unet.named_modules():
            if self._match(name):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def unregister(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()

    # ---------- step control ----------
    def start_step(self, step_idx: int):
        # If we were already recording a step, flush that step first
        if self.per_step and self.current_step is not None and self._step_buffer:
            self._flush_current_step()
        self.current_step = int(step_idx)
        self._step_buffer = []

    def end_step(self):
        if self.per_step and self.current_step is not None and self._step_buffer:
            self._flush_current_step()
            self.current_step = None
            self._step_buffer = []

    def _flush_current_step(self):
        obj = {
            "type": "sdxl_unet",
            "run_id": self.run_id,
            "step": self.current_step,
            "meta": self.meta,
            "records": self._step_buffer,
        }
        name = f"unet_trace_{self.run_id}_step{self.current_step:03d}.json"
        path = self.out_dir / name
        if self.gzip_output:
            if not str(path).endswith(".gz"):
                path = Path(str(path) + ".gz")
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(obj, f)
        else:
            path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # ---------- outputs ----------
    def summary(self) -> dict:
        return {
            "type": "sdxl_unet",
            "run_id": self.run_id,
            "layers_captured": len(self.records) if not self.per_step else None,
            "meta": self.meta,
        }

    def save_json(self, filename: str | None = None) -> str:
        """Single-file mode only (per_step=False): write all collected records."""
        out = {
            "type": "sdxl_unet",
            "run_id": self.run_id,
            "meta": self.meta,
            "records": self.records,
        }
        path = self.out_dir / (filename or f"unet_trace_{self.run_id}.json")
        if self.gzip_output:
            if not str(path).endswith(".gz"):
                path = Path(str(path) + ".gz")
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(out, f)
            return str(path)
        else:
            path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            return str(path)
