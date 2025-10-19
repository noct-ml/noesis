# soulprint_compare.py
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def natural_layer_key(name: str):
    parts = []
    for tok in name.replace('.', ' ').split():
        parts.append(int(tok) if tok.isdigit() else tok)
    return parts



def load_trace(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_soulprints(trace_a, trace_b):
    layers = sorted(set(trace_a.keys()) & set(trace_b.keys()))
    soulprint_deltas = []

    for layer in layers:
        a_layer = np.array(trace_a[layer])  # could be (1, tokens, 4096)
        b_layer = np.array(trace_b[layer])

        print(f" DEBUG: Layer {layer} shapes -> a: {a_layer.shape}, b: {b_layer.shape}")

        # Squeeze batch dim if present
        if len(a_layer.shape) == 3:
            a_layer = a_layer.squeeze(0)  # (tokens, 4096)
        if len(b_layer.shape) == 3:
            b_layer = b_layer.squeeze(0)

        print(f" Squeezed -> a: {a_layer.shape}, b: {b_layer.shape}")

        # Mean pooling
        a = a_layer.mean(axis=0)
        b = b_layer.mean(axis=0)

        print(f" Pooled -> a: {a.shape}, b: {b.shape}")

        # Cosine similarity
        cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        soulprint_deltas.append(1 - cosine_sim)

    return layers, soulprint_deltas


def compare_tokenwise(trace_a, trace_b):
    """Returns dict[layer] -> list[delta_per_token] with robust shapes."""
    def _as_row_matrix(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.empty((0,0), dtype=float)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
    def _row_norm(x):
        if x.size == 0:
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return x / norms
    common = sorted(set(trace_a.keys()).intersection(set(trace_b.keys())), key=natural_layer_key)
    layerwise_token_deltas = {}
    for layer in common:
        a_raw = trace_a[layer][0] if isinstance(trace_a[layer], list) else trace_a[layer]
        b_raw = trace_b[layer][0] if isinstance(trace_b[layer], list) else trace_b[layer]
        A = _as_row_matrix(a_raw)
        B = _as_row_matrix(b_raw)
        if A.size == 0 or B.size == 0:
            layerwise_token_deltas[layer] = []
            continue
        T = min(A.shape[0], B.shape[0])
        D = min(A.shape[1], B.shape[1])
        if T == 0 or D == 0:
            layerwise_token_deltas[layer] = []
            continue
        A = _row_norm(A[:T, :D])
        B = _row_norm(B[:T, :D])
        deltas = 1.0 - np.sum(A*B, axis=1)
        deltas = np.nan_to_num(deltas, nan=0.0, posinf=1.0, neginf=1.0)
        layerwise_token_deltas[layer] = deltas.tolist()
    return layerwise_token_deltas


def plot_comparison(layers, distances, out_path="soulprint_diff.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(layers, distances, marker='o')
    plt.xticks(rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("1 - Cosine Similarity")
    plt.title("Soulprint Delta Across Layers")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_token_layer_heatmap(layerwise_token_deltas, token_labels=None, out_path="token_heatmap.png"):
    layers = list(layerwise_token_deltas.keys())
    tokens = max(len(v) for v in layerwise_token_deltas.values())
    heat_data = np.zeros((len(layers), tokens))

    for i, layer in enumerate(layers):
        for j, delta in enumerate(layerwise_token_deltas[layer]):
            heat_data[i, j] = delta

    plt.figure(figsize=(12, 6))
    plt.imshow(heat_data, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label="1 - Cosine Similarity")
    if token_labels and len(token_labels) >= tokens:
        labels = [t.replace("▁", "_") for t in token_labels[:tokens]]
        plt.xticks(np.arange(tokens), labels, rotation=90, fontsize=8)
    else:
        plt.xticks(np.arange(tokens), [f"T{i}" for i in range(tokens)], rotation=90)
    plt.yticks(np.arange(len(layers)), layers)
    plt.title("Token-wise Divergence per Layer")
    plt.xlabel("Token")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def flag_divergent_tokens(layerwise_token_deltas, token_labels=None, threshold=0.3, top_n=5):
    token_scores = defaultdict(lambda: {"max": 0, "layer": ""})

    for layer, deltas in layerwise_token_deltas.items():
        for i, delta in enumerate(deltas):
            if delta > token_scores[i]["max"]:
                token_scores[i]["max"] = delta
                token_scores[i]["layer"] = layer

    sorted_tokens = sorted(token_scores.items(), key=lambda x: -x[1]["max"])
    flagged = []
    for idx, data in sorted_tokens:
        if data["max"] == 0.0:
            continue  # Skip tokens with no divergence at all
        flagged.append({
            "token_idx": idx,
            "token_str": token_labels[idx] if token_labels and idx < len(token_labels) else f"T{idx}",
            "max_delta": round(data["max"], 4),
            "layer": data["layer"]
        })
        if len(flagged) >= top_n:
            break

    return flagged

def summarize_delta(layers, distances, delta_type="unknown"):
    mean_delta = np.mean(distances)
    max_delta = max(distances)
    sensitive_layers = [l for l, d in zip(layers, distances) if d > 0.3]

    summary = {
        "delta_type": delta_type,
        "vector_delta_mean": round(mean_delta, 4),
        "vector_delta_max": round(max_delta, 4),
        "sensitive_layers": sensitive_layers,
        "summary": ""
    }

    if mean_delta < 0.1:
        summary["summary"] = "Minor behavioral change detected. Model response is mostly stable."
    elif mean_delta < 0.3:
        summary["summary"] = "Moderate behavioral shift detected. Model shows partial adaptation."
    else:
        summary["summary"] = "Significant behavioral divergence. Model response pattern is highly sensitive to this delta."

    if sensitive_layers:
        summary["summary"] += f" Sensitive layers include: {', '.join(sensitive_layers)}."

    return summary

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / (norm_product + 1e-8)

def compute_tokenwise_delta_snapshots(trace_a, trace_b, layer_name="model.layers.31.residual_out"):
    """
    Compare activations for a specific layer and return per-token cosine deltas.
    `trace_a` and `trace_b` are expected to be numpy arrays of shape (tokens, hidden_size).
    """
    a = trace_a[layer_name]  # Shape: (T1, D)
    b = trace_b[layer_name]  # Shape: (T2, D)

    min_len = min(a.shape[0], b.shape[0])
    deltas = []

    for i in range(min_len):
        sim = compute_cosine_similarity(a[i], b[i])
        delta = 1.0 - sim
        deltas.append(delta)

    return np.array(deltas)

def plot_token_delta_snapshots(deltas, token_strings=None, title="Delta Snapshot per Token", max_tokens=50):
    """
    Plot cosine deltas for each token in a specific layer.
    Optionally annotate with token strings.
    """
    num_tokens = min(len(deltas), max_tokens)
    x = list(range(num_tokens))
    y = deltas[:num_tokens]

    plt.figure(figsize=(14, 5))
    bars = plt.bar(x, y, color="slateblue", alpha=0.7)

    if token_strings:
        for i in range(num_tokens):
            plt.text(i, y[i] + 0.01, token_strings[i], ha='center', va='bottom', rotation=90, fontsize=8)

    plt.title(title)
    plt.xlabel("Token Position")
    plt.ylabel("Cosine Distance (Δ)")
    plt.tight_layout()
    plt.show()