# Noesis


<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/Version-0.1-lightgrey.svg">
  <img alt="Status" src="https://img.shields.io/badge/Portfolio%20Project-✓-purple.svg">
</p>

**[noesis.png](./noesis.png)**

 — UNet Tracing & “Soulprints” for SDXL
 
<p align="center">


_Noesis_ is a small research/portfolio project that instruments the **Stable Diffusion XL (SDXL) U‑Net** and extracts lightweight statistics from intermediate activations during image generation. Those statistics are aggregated into a compact vector — a **soulprint** — that summarizes a run. You can:
- **Trace** SDXL U‑Net layers during a generation run.
- **Summarize** per‑layer activation stats into a single fixed‑length vector (“soulprint”).
- **Compare** two runs/snapshots by cosine similarity and per‑layer deltas.
- **Visualize** token × layer deltas as quick heatmaps (optional).

> Portfolio‑oriented: clean structure, clear CLI, minimal dependencies, and practical code that shows systems thinking (hooks, summaries, comparisons) without heavy ceremony.

---

## Why this exists

While Noesis began as a diffusion introspection tool, its tracer also supports language models such as Mistral-7B-v0.1.
This allows you to compare how text and image models think — different architectures, same lens.
One codebase, two modalities, one question: what does the network remember of its own dreaming?

---

## Features

- **UNetTracer**: a minimal hook‑based tracer focused on SDXL U‑Net blocks (down/mid/up).
- **Per‑step or single‑file modes**: store every step separately _or_ aggregate after the run.
- **Configurable include patterns**: regex control over which submodules to watch.
- **Compressed outputs**: optional `.json.gz` for traces.
- **Soulprint writer**: turn traces into a compact vector with a stable layer index.
- **Comparator utilities**: compute cosine similarity and top‑k salient layer differences.
- **Visualization helpers**: quick Matplotlib heatmaps for token × layer deltas.

---

## Project status

> **v0.1 (portfolio)** — functional tracer and summarizer with comparison utilities. API is intentionally small and may evolve. Good for demonstration, reproducible mini‑experiments, and as scaffolding for deeper research.

---
Forensic introspection for generative models
Traces the hidden currents of cognition - from LLM reasoning to diffusion flows - mapping activation fields, MoE routes, and soulprints of thought.
## Repo structure

```
noesis/
  __init__.py
  cli.py                      # CLI entry points (trace, summarize, compare)
  unet_soulprint.py           # summarize traces → soulprint
  compare_unet_soulprints.py  # soulprint vs soulprint (cosine, top‑k deltas)
  soulprint_compare.py        # token×layer delta tools + heatmap
  tracing/
    base_tracer.py            # abstract tracer interface
    unet_tracer.py            # SDXL U‑Net tracer (regex include, gzip, per‑step)
  utils/
    io.py
    tensor_stats.py
docs/
examples/
```

---

## Installation

> Requires **Python 3.10+**. Tracing requires **PyTorch** and **diffusers**; comparison utilities use **NumPy/Pandas/Matplotlib**.

```bash
# minimal stack for tracing SDXL
pip install torch diffusers transformers accelerate safetensors

# analysis/visualization extras
pip install numpy pandas matplotlib

# (optional) install this repo in editable mode
pip install -e .
```

> If you only want to use the _comparison_ tools on existing soulprints, you can skip the heavy deps and just install the analysis extras.

---

## CLI quickstart

The CLI is intentionally small. Run `--help` for authoritative flags and updates:

```bash
python -m noesis.cli --help
python -m noesis.cli trace --help
python -m noesis.cli unet-summarize --help
python -m noesis.cli compare-unet-soulprints --help
python -m noesis.cli soulprint-compare --help
```

### 1) Trace a run (SDXL)

```bash
python -m noesis.cli trace --model "stabilityai/stable-diffusion-xl-base-1.0" --prompt "a glass moth in mauve twilight, macro, film grain" --steps 30 --out "runs/trace_001" --per-step --gzip
```

This writes one JSON (or `.json.gz`) _per step_ under `runs/trace_001/` with light stats (shape, mean, std, abs_mean, min/max, dtype, device) for the selected U‑Net blocks.

### 2) Summarize to a soulprint

```bash
python -m noesis.cli unet-summarize --trace "runs/trace_001" --out "runs/trace_001/soulprint.json"
```

The output is a JSON with fields like:

```json
{
  "type": "sdxl_unet_soulprint",
  "run_id": "...",
  "meta": {...},
  "steps_aggregated": 30,
  "layer_index": {"down_blocks.0.resnets.0": 0, "...": 1},
  "vector_layout": "concat(means[0:L], stds[0:L])",
  "soulprint": [0.0123, 0.0456, ...]
}
```

### 3) Compare two soulprints

```bash
python -m noesis.cli compare-unet-soulprints --a runs/trace_001/soulprint.json --b runs/trace_002/soulprint.json --top-k 10
```

You’ll get a small JSON/CSV summary (cosine similarity, top‑k layer deltas) suitable for dashboards or PR comments.

---

## Python API (lightweight)

You can also use the components directly:

```python
from noesis.tracing.unet_tracer import UNetTracer
from noesis.unet_soulprint import write_unet_soulprint

# assuming you already created a StableDiffusionXLPipeline called pipe
tracer = UNetTracer(
    unet=pipe.unet,
    out_dir="runs/trace_003",
    include_patterns=None,   # or provide regexes
    keep_samples=False,
    sample_limit=0,
    meta={"prompt": "…", "seed": 42},
    gzip_output=True,
    per_step=True,
)

tracer.register()
# ... run your generation steps here ...
tracer.unregister()

write_unet_soulprint("runs/trace_003", "runs/trace_003/soulprint.json")
```

---

## Outputs & formats

- **Trace file(s):** `unet_trace_<run>.json` or `unet_trace_<run>_step<k>.json(.gz)`
- **Soulprint:** JSON with `type=sdxl_unet_soulprint`, `layer_index`, and `soulprint` vector
- **Comparison:** JSON/CSV with cosine similarity and top‑k salient layer deltas

---

## Notes on design

- **Tiny surface area:** hooks in, small stats out — easy to diff, diffable in PRs.
- **Deterministic layout:** `layer_index` is persisted; vectors are concatenated as `[means | stds]`.
- **Fast enough to iterate:** per‑step mode supports temporal analysis; single‑file mode is convenient for quick diffs.
- **Guarded heavy imports:** CLI can still run comparison flows without torch/diffusers installed.

---

## Roadmap

- [ ] More backends (SD 1.5, SD3, Flux)
- [ ] Token‑aware traces for attention blocks
- [ ] Optional layer selection presets (e.g., “attn‑only”)
- [ ] Richer similarity metrics beyond cosine
- [ ] Better docs + screenshots (examples/)

---

## License

Choose your preferred permissive license (e.g., **MIT**) and drop it in `LICENSE`. If you’re evaluating this for work, assume MIT unless noted otherwise.

---

## Author
Built by **James Jones** — self‑taught technologist and symbolic‑systems researcher. I design tools at the seam between computation and meaning.  
Say hi: https://github.com/noct-ml/

