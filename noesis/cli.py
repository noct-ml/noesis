# noesis/cli.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Optional heavy deps guarded
try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # keep CLI usable for compare-only workflows
    torch = None
    StableDiffusionXLPipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Local imports
from .tracing.unet_tracer import UNetTracer
from .unet_soulprint import write_unet_soulprint
from .compare_unet_soulprints import compare_unet_soulprints
from .soulprint_compare import compare_soulprints  # flat vector compare
from .analysis.moe_trace import trace_moe

# ---------- helpers ----------

def _ensure_torch():
    if torch is None or StableDiffusionXLPipeline is None:
        raise RuntimeError(
            "Tracing requires torch and diffusers. Install e.g.\n"
            "  pip install 'torch' 'diffusers[torch]' 'transformers' 'accelerate'\n"
        )

def _ensure_torch_transformers():
    if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
        raise RuntimeError(
            "MoE tracing requires torch and transformers. Install e.g.\n"
            "  pip install 'torch' 'transformers' 'tqdm'\n"
        )

def _default_subparser(parser: argparse.ArgumentParser, subparsers, default: str):
    """
    If user didn't provide a subcommand, inject a default (e.g., 'trace').
    Call right before parser.parse_args().
    """
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help", "help"}:
        return
    # If first non-option arg looks like a flag, assume default subcommand
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            return  # user passed a real subcommand
    sys.argv.insert(1, default)

# ---------- command impls ----------

def cmd_trace(args: argparse.Namespace) -> None:
    _ensure_torch()

    # seed
    if args.seed is not None:
        try:
            torch.manual_seed(int(args.seed))
        except Exception:
            pass

    # pipeline
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(args.device)

    # tracer (optional)
    tracer = None
    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    if args.trace_unet:
        tracer = UNetTracer(
            pipe.unet,
            out_dir=args.out_dir,
            include_patterns=args.include,
            keep_samples=args.keep_samples,
            sample_limit=args.sample_limit,
            meta=meta,
            gzip_output=(not args.no_gzip),
            per_step=(not args.no_per_step),
        )
        tracer.register()

    # callback for per-step logging
    def _cb(step, timestep, latents):
        if tracer is not None:
            tracer.start_step(step)

    # run
    with torch.inference_mode():
        _ = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            callback=_cb if tracer is not None else None,
            callback_steps=1 if tracer is not None else None,
        )

    # finalize tracer
    if tracer is not None and not args.no_per_step:
        tracer.end_step()

    if tracer is not None:
        if args.no_per_step:
            path = tracer.save_json()
            print(f"[noesis] UNet trace saved to {path}")
        else:
            out_dir = Path(args.out_dir)
            summary_path = out_dir / f"unet_trace_{tracer.run_id}_summary.json"
            summary_path.write_text(json.dumps(tracer.summary(), indent=2))
            print(f"[noesis] Per-step traces written to {args.out_dir}; summary: {summary_path}")
        tracer.unregister()

        # optional: emit summarized UNet soulprint directly
        if args.emit_unet_soulprint:
            sp_path = Path(args.out_dir) / f"unet_soulprint_{tracer.run_id}.json"
            out = write_unet_soulprint(args.out_dir, sp_path.as_posix())
            print(f"[noesis] UNet soulprint written: {out}")

def cmd_unet_summarize(args: argparse.Namespace) -> None:
    out = write_unet_soulprint(args.trace_path, args.out)
    print("[noesis] wrote UNet soulprint:", out)

def cmd_compare_unet_soulprints(args: argparse.Namespace) -> None:
    summary, df = compare_unet_soulprints(args.sp1, args.sp2, top_k=args.top)
    print("[noesis] Cosine similarity:", summary["cosine_similarity"])
    try:
        print(df.to_string(index=False))
    except Exception:
        print(df)
    if args.csv:
        Path(args.csv).write_text(df.to_csv(index=False))
        print("[noesis] wrote table:", args.csv)
    if args.json_summary:
        Path(args.json_summary).write_text(json.dumps(summary, indent=2))
        print("[noesis] wrote summary:", args.json_summary)

def cmd_soulprint_compare(args: argparse.Namespace) -> None:
    summary, df_top = compare_soulprints(args.file1, args.file2, top_k=args.top)
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
    print(f"[noesis] Running MoE trace for model: {args.model}")
    trace_file = trace_moe(args.model, args.prompt, args.out_dir)
    print(f"[noesis] Trace saved to: {trace_file}")

# ---------- parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="noesis", description="Noesis: tracing and soulprint tools")
    subparsers = p.add_subparsers(dest="command")

    # TRACE (default)
    t = subparsers.add_parser("trace", help="Run SDXL and optionally trace UNet")
    t.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    t.add_argument("--prompt", default="a glass orb on velvet")
    t.add_argument("--negative-prompt", default=None)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--device", default="cuda", help="cuda or cpu")
    t.add_argument("--out-dir", default="noesis_out", help="Outputs/logs directory")
    # UNet tracing flags
    t.add_argument("--trace-unet", action="store_true", help="Enable UNet tracing hooks")
    t.add_argument("--include", nargs="*", default=None, help="Regex patterns for layer names to capture")
    t.add_argument("--keep-samples", action="store_true", help="Include tiny activation samples (bigger logs)")
    t.add_argument("--sample-limit", type=int, default=64, help="Max sample elements per record")
    t.add_argument("--no-gzip", action="store_true", help="Disable gzip compression for logs")
    t.add_argument("--no-per-step", action="store_true", help="Write a single log file instead of per-step files")
    t.add_argument("--emit-unet-soulprint", action="store_true", help="After tracing, auto-summarize per-step logs into a compact UNet soulprint JSON")

    # UNET → SOULPRINT
    sp = subparsers.add_parser("unet-summarize", help="Summarize per-step UNet traces into a compact soulprint JSON")
    sp.add_argument("trace_path", help="Dir with unet_trace_*_stepXXX.json[.gz] OR a single trace file")
    sp.add_argument("--out", required=True, help="Output JSON path for the UNet soulprint")

    # COMPARE UNET SOULPRINTS (layer-aware)
    cu = subparsers.add_parser("compare-unet-soulprints", help="Compare two UNet soulprint JSONs (layer-aware)")
    cu.add_argument("sp1")
    cu.add_argument("sp2")
    cu.add_argument("--top", type=int, default=25)
    cu.add_argument("--csv", type=str, default=None)
    cu.add_argument("--json-summary", type=str, default=None)

    # FLAT SOULPRINT COMPARE (vector vs vector)
    f = subparsers.add_parser("soulprint-compare", help="Compare two flat soulprint JSONs (vector vs vector)")
    f.add_argument("file1")
    f.add_argument("file2")
    f.add_argument("--top", type=int, default=25)
    f.add_argument("--csv", type=str, default=None)
    f.add_argument("--json-summary", type=str, default=None)

    # TRACE MoE
    mt = subparsers.add_parser("trace-moe", help="Trace MoE model layers and gate decisions")
    mt.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Name of the MoE model (e.g., 'mistralai/Mixtral-8x7B-v0.1')"
    )
    mt.add_argument(
        "--prompt",
        type=str,
        default="Generate a detailed list of all the paradoxes related to quantum logic, causality loops, and metaphysical recursion in simulation theory.",
        help="Input prompt for MoE tracing"
    )
    mt.add_argument(
        "--out-dir",
        type=str,
        default="traces",
        help="Directory to save trace JSON file"
    )

    return p

def main():
    parser = build_parser()
    # Make 'trace' the default when no subcommand is provided
    _default_subparser(parser, parser.add_subparsers, default="trace")
    args = parser.parse_args()

    if args.command == "trace":
        return cmd_trace(args)
    elif args.command == "unet-summarize":
        return cmd_unet_summarize(args)
    elif args.command == "compare-unet-soulprints":
        return cmd_compare_unet_soulprints(args)
    elif args.command == "soulprint-compare":
        return cmd_soulprint_compare(args)
    elif args.command == "trace-moe":
        return cmd_trace_moe(args)

    parser.print_help()

if __name__ == "__main__":
    main()
