# noesis/compare_unet_soulprints.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

def _load_sp(path: str):
    j = json.loads(Path(path).read_text())
    if j.get("type") != "sdxl_unet_soulprint" or "soulprint" not in j:
        raise ValueError(f"Not a UNet soulprint JSON: {path}")
    v = np.array(j["soulprint"], dtype=np.float64)
    layer_index = j.get("layer_index", {})
    return v, layer_index, j

def _cosine(a, b):
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    if an == 0 or bn == 0: return float("nan")
    return float(np.dot(a, b) / (an * bn))

def compare_unet_soulprints(sp1_path: str, sp2_path: str, top_k: int = 25):
    v1, idx1, j1 = _load_sp(sp1_path)
    v2, idx2, j2 = _load_sp(sp2_path)

    if v1.shape[0] != v2.shape[0]:
        n = max(v1.shape[0], v2.shape[0])
        v1 = np.pad(v1, (0, n - v1.shape[0]))
        v2 = np.pad(v2, (0, n - v2.shape[0]))

    cos = _cosine(v1, v2)

    # Vector layout is concat([means],[stds]); split to map back to layers
    L = len(idx1) if idx1 else len(idx2)
    means_1, stds_1 = v1[:L], v1[L:L*2]
    means_2, stds_2 = v2[:L], v2[L:L*2]

    # Build reverse index -> layer name (favor sp1 mapping, fallback to sp2)
    rev = {i: name for name, i in idx1.items()} if idx1 else {}
    if not rev and idx2:
        rev = {i: name for name, i in idx2.items()}

    # Deltas per layer for mean/std separately
    d_mean = means_2 - means_1
    d_std  = stds_2  - stds_1
    abs_d  = np.abs(d_mean) + np.abs(d_std)  # simple salience score

    order = np.argsort(-abs_d)[:top_k]
    rows = []
    for i in order:
        lname = rev.get(i, f"layer_{i}")
        rows.append(dict(
            layer=lname,
            idx=int(i),
            mean_v1=float(means_1[i]),
            mean_v2=float(means_2[i]),
            d_mean=float(d_mean[i]),
            std_v1=float(stds_1[i]),
            std_v2=float(stds_2[i]),
            d_std=float(d_std[i]),
            salience=float(abs_d[i]),
        ))

    df = pd.DataFrame(rows)

    summary = dict(
        cosine_similarity=cos,
        length=int(v1.shape[0]),
        layers=int(L),
        top_k=int(top_k),
        run1=j1.get("meta", {}).get("run_id"),
        run2=j2.get("meta", {}).get("run_id"),
    )
    return summary, df
