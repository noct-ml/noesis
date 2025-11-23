# noesis/cli.py
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# Optional heavy deps guarded
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # keep CLI usable for compare-only workflows
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Local imports
from .soulprint_compare import (
    compare_soulprints_layerwise,
    compare_soulprints_tokenwise,
    load_trace,
    flag_divergent_tokens,
    plot_token_layer_heatmap,
)
from .analysis.moe_trace import trace_moe
from .noesis_trace import NoesisTracer


# ---------- helpers ----------

def _ensure_torch_transformers() -> None:
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "Tracing requires torch and transformers. Install e.g.\n"
            "  pip install 'torch' 'transformers' 'tqdm'\n"
        )


def _ensure_hf_auth() -> None:
    """
    Ensure the user is authenticated with HuggingFace Hub.

    This prevents confusing 403 / auth errors when trying to pull models
    from hub without running `huggingface-cli login` first.
    """
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if not token:
            raise RuntimeError
    except Exception:
        raise RuntimeError(
            "HuggingFace authentication is required to download models from the hub.\n"
            "Run:\n"
            "   huggingface-cli login\n"
            "or set the HF_TOKEN environment variable.\n"
        )


def _default_subparser(parser: argparse.ArgumentParser, default: str) -> None:
    """
    If the user calls `noesis` with only flags (no subcommand),
    insert a default subcommand (e.g., 'trace-llm').
    """
    # If help requested, leave args alone
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help", "help"}:
        return

    # If any non-flag arg exists, assume it's a subcommand already
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            return

    # Only flags provided → inject default subcommand
    sys.argv.insert(1, default)


# ---------- command impls ----------

def cmd_soulprint_compare(args: argparse.Namespace) -> None:
    # Load the trace JSONs first
    trace_a = load_trace(args.file1)
    trace_b = load_trace(args.file2)

    # Now compute layerwise soulprint diffs
    summary, df_top = compare_soulprints_layerwise(trace_a, trace_b)

    print("[noesis] Cosine similarity:", summary["cosine_similarity"])
    print("[noesis] Max |Δ|:", summary["max_abs_delta"], "at index", summary["max_abs_delta_index"])

    try:
        print(df_top.to_string(index=False))
    except Exception:
        print(df_top)

    if args.csv:
        Path(args.csv).write_text(df_top.to_csv(index=False))
        print("[noesis] wrote table:", args.csv)

    if args.json_summary:
        Path(args.json_summary).write_text(json.dumps(summary, indent=2))
        print("[noesis] wrote summary:", args.json_summary)


def cmd_trace_moe(args: argparse.Namespace) -> None:
    _ensure_torch_transformers()
    _ensure_hf_auth()
    print(f"[noesis] Running MoE trace for model: {args.model}")
    trace_file = trace_moe(args.model, args.prompt, args.out_dir)
    print(f"[noesis] Trace saved to: {trace_file}")


# TRACE LLM
def cmd_trace_llm(args: argparse.Namespace) -> None:
    _ensure_torch_transformers()
    _ensure_hf_auth()

    print(f"[noesis] Using model: {args.model}")

    tracer = NoesisTracer(
        model_name=args.model,
        trace_mode="residual",
        token_wise=True
    )

    # Load prompts from file or command-line arguments
    prompts: list[str] = []
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
                raise ValueError("Prompt file must contain a JSON list of strings")
            prompts = data
        except Exception as e:
            raise RuntimeError(f"Failed to load prompts from {args.prompt_file}: {str(e)}") from e
    elif args.prompts:
        prompts = args.prompts
    else:
        raise RuntimeError("At least one prompt must be provided via --prompts or --prompt-file")

    if len(prompts) < 2:
        raise RuntimeError(
            "trace-llm requires at least 2 prompts so it can compare runs.\n"
            "Provide multiple --prompts or a JSON prompt file with at least two entries."
        )

    if len(prompts) > 2:
        print(f"[noesis] Warning: {len(prompts)} prompts provided; only the first two will be compared.")

    print(f"[noesis] Using {len(prompts)} prompt(s):")
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}] {p[:80]!r}")

    os.makedirs(args.out_dir, exist_ok=True)

    trace_a = None
    trace_b = None

    # Trace only first two prompts
    for i, prompt in enumerate(prompts, 1):
        trace = tracer.trace(prompt)
        trace_file = os.path.join(args.out_dir, f"trace_prompt_{i}.json")
        tracer.save_trace(trace, trace_file)
        print(f"[noesis] Trace for prompt {i} ('{prompt[:30]}...') saved to {trace_file}")

        if i == 1:
            trace_a = load_trace(trace_file)
        elif i == 2:
            trace_b = load_trace(trace_file)
            break

    if trace_a is None or trace_b is None:
        raise RuntimeError("Internal error: missing trace data for comparison.")

    token_deltas = compare_soulprints_tokenwise(trace_a, trace_b)

    flags = flag_divergent_tokens(
        token_deltas,
        token_labels=tracer.last_tokens,
        threshold=0.3,
        top_n=5,
    )
    for f in flags:
        print(f" Token {f['token_idx']} ('{f['token_str']}') max Δ={f['max_delta']} at {f['layer']}")

    plot_token_layer_heatmap(token_deltas, token_labels=tracer.last_tokens)


# ---------- parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="noesis",
        description="Noesis: LLM tracing and soulprint forensics",
    )
    subparsers = p.add_subparsers(dest="command")

    # SOULPRINT COMPARE (layerwise pooled)
    f = subparsers.add_parser(
        "soulprint-compare",
        help="Compare two layerwise trace JSONs and summarize cosine deltas per layer",
    )
    f.add_argument("file1", help="First trace JSON")
    f.add_argument("file2", help="Second trace JSON")
    f.add_argument("--csv", type=str, default=None, help="Write top layer deltas table to CSV")
    f.add_argument("--json-summary", type=str, default=None, help="Write summary stats to JSON")

    # TRACE MoE
    mt = subparsers.add_parser(
        "trace-moe",
        help="Trace MoE model layers and gate decisions",
    )
    mt.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model identifier (e.g. mistralai/Mixtral-8x7B-v0.1)",
    )
    mt.add_argument(
        "--prompt",
        type=str,
        default=(
            "Generate a detailed list of all the paradoxes related to quantum logic, "
            "causality loops, and metaphysical recursion in simulation theory."
        ),
        help="Prompt to trace through the MoE model",
    )
    mt.add_argument(
        "--out-dir",
        type=str,
        default="traces",
        help="Directory to save trace JSON file",
    )

    # TRACE LLM
    tl = subparsers.add_parser(
        "trace-llm",
        help="Trace LLM residuals for two or more prompts and compare soulprints",
    )
    tl.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=None,
        help="One or more prompts to trace",
    )
    tl.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model ID to trace (default: mistralai/Mistral-7B-v0.1)",
    )
    tl.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to JSON file with list of prompts (list[str])",
    )
    tl.add_argument(
        "--out-dir",
        type=str,
        default="traces",
        help="Directory to save trace JSON files",
    )

    return p


def main() -> None:
    parser = build_parser()
    _default_subparser(parser, default="trace-llm")
    args = parser.parse_args()

    if args.command == "soulprint-compare":
        return cmd_soulprint_compare(args)
    elif args.command == "trace-moe":
        return cmd_trace_moe(args)
    elif args.command == "trace-llm":
        return cmd_trace_llm(args)

    parser.print_help()


if __name__ == "__main__":
    main()
