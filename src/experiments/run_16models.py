#!/usr/bin/env python3
"""
Run AgentPanel + LLM Arena experiments across all 16 models.

Usage:
    python -m src.experiments.run_16models --exp agentpanel   # AgentPanel only
    python -m src.experiments.run_16models --exp arena         # LLM Arena only
    python -m src.experiments.run_16models --exp both          # Both (default)
    python -m src.experiments.run_16models --models 0,1,5      # Run specific models by index
    python -m src.experiments.run_16models --dry-run            # Print commands without executing
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.model_configs import MODELS_16, ModelSpec, get_api_url


RESULTS_DIR = ROOT / "results" / "multimodel_16"
LOG_DIR = RESULTS_DIR / "logs"


def run_agentpanel(spec: ModelSpec, dry_run: bool = False) -> dict:
    """Run AgentPanel experiment for one model."""
    out_dir = RESULTS_DIR / "agentpanel" / spec.name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"agentpanel_{spec.name.replace('/', '_')}.log"

    cmd = [
        sys.executable, "-m", "src.experiments.agentpanel_experiment",
        "--model", spec.model_id,
        "--api-url", get_api_url(),
        "--outdir", str(out_dir),
        "--tag", spec.name,
        "--n-agents", "8",
        "--n-rounds", "8",
        "--seeds", "42,123",
        "--rho0", "0.10,0.50",
    ]

    print(f"  [AgentPanel] {spec.name} ({spec.vendor})")
    if dry_run:
        print(f"    CMD: {' '.join(cmd[:6])} ...")
        return {"model": spec.name, "status": "dry-run"}

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=14400,  # 4 hours max per model
        )
        elapsed = time.time() - t0
        log_file.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)

        if result.returncode == 0:
            print(f"    OK ({elapsed:.0f}s)")
            return {"model": spec.name, "status": "ok", "time": elapsed}
        else:
            print(f"    FAIL (rc={result.returncode}, {elapsed:.0f}s)")
            return {"model": spec.name, "status": "fail", "rc": result.returncode, "time": elapsed}
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (3600s)")
        return {"model": spec.name, "status": "timeout"}
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"model": spec.name, "status": "error", "msg": str(e)}


def run_arena(spec: ModelSpec, dry_run: bool = False) -> dict:
    """Run LLM Arena (PGG) experiment for one model."""
    out_dir = RESULTS_DIR / "arena" / spec.name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"arena_{spec.name.replace('/', '_')}.log"

    # Arena saves results to results/llm_experiment/raw/ by default.
    # We use a dedicated raw dir per model to avoid collisions.
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.experiments.run_llm_experiment",
        "--mode", "full",
        "--provider", "custom",
        "--custom-model", spec.model_id,
        "--custom-api-url", get_api_url().removesuffix("/chat/completions"),
        "--n-agents", "10",
        "--n-rounds", "10",
        "--n-repeats", "2",
        "--output-dir", str(out_dir),
    ]

    print(f"  [Arena] {spec.name} ({spec.vendor})")
    if dry_run:
        print(f"    CMD: {' '.join(cmd[:6])} ...")
        return {"model": spec.name, "status": "dry-run"}

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.time() - t0
        log_file.write_text(result.stdout + "\n---STDERR---\n" + result.stderr)

        if result.returncode == 0:
            print(f"    OK ({elapsed:.0f}s)")
            return {"model": spec.name, "status": "ok", "time": elapsed}
        else:
            print(f"    FAIL (rc={result.returncode}, {elapsed:.0f}s)")
            return {"model": spec.name, "status": "fail", "rc": result.returncode, "time": elapsed}
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (3600s)")
        return {"model": spec.name, "status": "timeout"}
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"model": spec.name, "status": "error", "msg": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run 16-model experiment sweep")
    parser.add_argument("--exp", choices=["agentpanel", "arena", "both"], default="both",
                        help="Which experiment to run")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model indices (0-15) to run. Default: all")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.models:
        indices = [int(i) for i in args.models.split(",")]
        models = [MODELS_16[i] for i in indices]
    else:
        models = MODELS_16

    print(f"=" * 60)
    print(f"16-Model Experiment Sweep")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Models: {len(models)} / {len(MODELS_16)}")
    print(f"Experiments: {args.exp}")
    print(f"=" * 60)

    for i, m in enumerate(models):
        print(f"  [{i:2d}] {m.name:20s} | {m.vendor:10s} | {m.model_id}")
    print()

    all_results = {"agentpanel": [], "arena": []}
    start = time.time()

    for i, spec in enumerate(models):
        print(f"\n{'─' * 50}")
        print(f"Model {i+1}/{len(models)}: {spec.name} ({spec.vendor})")
        print(f"{'─' * 50}")

        if args.exp in ("agentpanel", "both"):
            ap_dir = RESULTS_DIR / "agentpanel" / spec.name.replace("/", "_")
            if args.skip_existing and (ap_dir / "agentpanel_results.json").exists():
                print(f"  [AgentPanel] SKIP (agentpanel_results.json exists)")
                all_results["agentpanel"].append({"model": spec.name, "status": "skipped"})
            else:
                r = run_agentpanel(spec, dry_run=args.dry_run)
                all_results["agentpanel"].append(r)

        if args.exp in ("arena", "both"):
            ar_dir = RESULTS_DIR / "arena" / spec.name.replace("/", "_")
            if args.skip_existing and (ar_dir / "llm_summary.json").exists():
                print(f"  [Arena] SKIP (llm_summary.json exists)")
                all_results["arena"].append({"model": spec.name, "status": "skipped"})
            else:
                r = run_arena(spec, dry_run=args.dry_run)
                all_results["arena"].append(r)

    elapsed_total = time.time() - start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({elapsed_total:.0f}s total)")
    print(f"{'=' * 60}")
    for exp_name, results in all_results.items():
        if results:
            ok = sum(1 for r in results if r["status"] == "ok")
            fail = sum(1 for r in results if r["status"] == "fail")
            err = sum(1 for r in results if r["status"] in ("error", "timeout"))
            print(f"  {exp_name}: {ok} OK / {fail} FAIL / {err} ERR")
            for r in results:
                if r["status"] != "ok":
                    print(f"    {r['model']}: {r['status']}")

    # Save summary
    summary_file = RESULTS_DIR / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
