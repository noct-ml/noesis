# noesis/unet_soulprint.py
from __future__ import annotations
import json, gzip, math, os, re
from pathlib import Path
from typing import Dict, List, Tuple

def _read_json_any(path: Path):
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(p.read_text())

_STEP_RE = re.compile(r"unet_trace_(?P<run>\d+)_step(?P<step>\d+)\.json(\.gz)?$")

def _collect_trace_files(trace_path: str) -> Tuple[List[Path], Dict]:
    """
    Returns (list of files ordered by step, meta-from-first-file)
    Accepts: a directory with per-step files OR a single trace file path.
    """
    path = Path(trace_path)
    if path.is_file():
        return [path], {}
    if not path.exists():
        raise FileNotFoundError(f"Trace path not found: {trace_path}")

    files = []
    for fp in path.glob("unet_trace_*_step*.json*"):
        m = _STEP_RE.search(fp.name)
        if m:
            files.append((int(m.group("step")), fp))
    if not files:
        # fallback: include any unet_trace*.json in dir
        files = [(0, fp) for fp in path.glob("unet_trace*.json*")]
    files.sort(key=lambda t: t[0])
    return [f for _, f in files], {}

def summarize_unet_trace(trace_path: str) -> dict:
    """
    Build a UNet SoulPrint from per-step UNet traces (or a single trace file).
    Returns a dict with:
      - type: "sdxl_unet_soulprint"
      - run_id, steps, meta
      - layer_index: {layer_name: index_in_vector_block}  (index for mean/std blocks)
      - soulprint: [means..., stds...]
    """
    files, _ = _collect_trace_files(trace_path)
    if not files:
        raise FileNotFoundError(f"No trace files found at: {trace_path}")

    # read first file to establish layer order + meta
    first = _read_json_any(files[0])
    meta = {
        "model": first.get("meta", {}).get("model"),
        "prompt": first.get("meta", {}).get("prompt"),
        "negative_prompt": first.get("meta", {}).get("negative_prompt"),
        "steps_declared": first.get("meta", {}).get("steps"),
        "device": first.get("meta", {}).get("device"),
        "torch_version": first.get("meta", {}).get("torch_version"),
        "run_id": first.get("run_id"),
    }

    records0 = first.get("records", [])
    if not records0:
        raise ValueError("Trace file has no 'records'")

    # Stable layer ordering by 'name' as seen in the first file
    layer_names = [r["name"] for r in records0]
    layer_to_idx = {n: i for i, n in enumerate(layer_names)}
    L = len(layer_names)

    # Accumulators
    sum_mean = [0.0] * L
    sum_std  = [0.0] * L
    step_count = 0

    # Walk all files (steps)
    for p in files:
        j = _read_json_any(p)
        recs = j.get("records", [])
        if len(recs) != L:
            # if lengths differ, ignore mismatched file (or you can handle partial)
            continue
        # assume same order of names per file (our tracer logs in named_modules order)
        for i, r in enumerate(recs):
            sum_mean[i] += float(r.get("mean", 0.0))
            sum_std[i]  += float(r.get("std",  0.0))
        step_count += 1

    if step_count == 0:
        raise ValueError("No valid step files found/matched the first file's layer set")

    # Average across steps, then concatenate [means..., stds...]
    means = [m / step_count for m in sum_mean]
    stds  = [s / step_count for s in sum_std]
    soulprint = means + stds

    out = {
        "type": "sdxl_unet_soulprint",
        "run_id": meta.get("run_id"),
        "meta": meta,
        "steps_aggregated": step_count,
        "layer_index": {name: idx for idx, name in enumerate(layer_names)},
        "vector_layout": "concat(means[0:L], stds[0:L])",
        "soulprint": soulprint,
    }
    return out

def write_unet_soulprint(trace_path: str, out_path: str) -> str:
    sp = summarize_unet_trace(trace_path)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sp, indent=2))
    return str(p)
