# Noesis — LLM Forensic Tracing & Soulprint Analysis
*A lightweight toolkit for inspecting transformer internals through residual traces, layer-wise drift metrics, and token-level activation deltas.*

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/Version-0.1-lightgrey.svg">
  <img alt="Status" src="https://img.shields.io/badge/Portfolio%20Project-✓-purple.svg">
</p>

## Overview
Noesis is a focused forensic toolkit for LLM internal inspection.  
It provides:

- **Residual stream tracing** for any HuggingFace transformer  
- **Layerwise soulprint comparison** between two model runs  
- **Tokenwise activation delta analysis** for fine-grained anomaly detection  
- **Optional MoE gate tracing** (Mixtral, Phi-MoE, etc.)  
- **CLI-first design** for quick experimentation

The goal is simple:  
**make transformer internal behavior observable, comparable, and explainable.**

## Features
### Residual Trace Capture
Capture per-layer, per-token residual activations from any HF transformer:

- Outputs JSON trace files  
- Per-layer activation structure  
- Grab both tokenwise matrices and pooled representations  

### Soulprint Comparison
Compare the “soulprint” of two model runs:

- **Layerwise cosine drift** (`1 - cos_sim`)  
- Sorted delta table  
- Top-layer divergence ranking  
- Clean JSON + optional CSV export

Useful for detecting:

- behavioral drift  
- prompt-induced shifts  
- instabilities  
- trigger effects  
- subtle distribution changes

### Tokenwise Delta Heatmaps
Compare activations token-by-token across layers:

- Token-by-layer cosine deltas  
- Divergent-token detection  
- Heatmap visualization  
- Highlighting layers with abnormal response

This is the most detailed forensic view Noesis provides.

### MoE Gate Tracing (Optional)
For MoE architectures (Mixtral, Phi-MoE):

- Patch MoE gate modules  
- Capture top-k expert selection  
- Store softmax-normalized gate distributions  
- Attach results to the JSON trace

Dense models gracefully skip MoE logic.

## Installation

```bash
git clone https://github.com/noct-ml/noesis.git
cd noesis
pip install -e .
```

Dependencies:  
- torch  
- transformers  
- huggingface_hub  
- tqdm  
- numpy  
- matplotlib

Optional:  
- hf_transfer (if user has HF fast-transfer enabled)

## HuggingFace Authentication

Some models require authentication (e.g., Mistral, Mixtral).  
If Noesis cannot download a model, you’ll see:

```
RuntimeError: HuggingFace authentication is required...
```

Fix:

```bash
huggingface-cli login
```

Or set:

```bash
export HF_TOKEN=your_api_key
```

## Quickstart

### 1. Trace two prompts
```bash
noesis trace-llm --prompts "hello world" "hello there"
```

Outputs files:

```
traces/trace_prompt_1.json
traces/trace_prompt_2.json
```

### 2. Compare soulprints (layerwise)
```bash
noesis soulprint-compare traces/trace_prompt_1.json traces/trace_prompt_2.json
```

### 3. Compare tokenwise activations (heatmap)
Tracing two prompts via `trace-llm` also performs tokenwise comparison automatically.

### 4. MoE tracing
```bash
noesis trace-moe --model mistralai/Mixtral-8x7B-v0.1 --prompt "Explain entropy"
```

## Command Reference

### trace-llm
Trace two or more prompts and compute soulprint deltas.

Options:
- --model – HF model ID (default: Mistral-7B)  
- --prompts – direct CLI prompts  
- --prompt-file – JSON list of prompts  
- --out-dir – where traces are saved  

### soulprint-compare
Layerwise comparison between two trace JSONs.

Options:
- --csv – export table  
- --json-summary – export stats  

### trace-moe
MoE expert gate tracing.

## Repo Structure

```
noesis/
  cli.py               - CLI entrypoints
  noesis_trace.py      - Residual tracer core
  soulprint_compare.py - Layerwise + tokenwise comparison utilities
  analysis/
    moe_trace.py       - MoE gate tracing logic
  utils/
    io.py
    tensor_stats.py
```

## Why Noesis?
Because LLMs are black boxes — and black boxes hide patterns.  
Noesis gives a clean, practical view into:

- how residual streams evolve  
- how prompts shift internal states  
- how layers respond differently  
- how MoE experts route decisions  

It's built as a developer-facing analysis scaffold.
