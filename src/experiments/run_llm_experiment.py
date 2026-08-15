"""
Run the LLM PGG experiment across conditions.

Usage:
  # Pilot: Condition A + C only
  python -m src.experiments.run_llm_experiment --mode pilot

  # Full: all 4 conditions
  python -m src.experiments.run_llm_experiment --mode full

  # Single run for testing
  python -m src.experiments.run_llm_experiment --mode test

  # GPTPlus5 external API
  python -m src.experiments.run_llm_experiment --mode test --provider gptplus5

Environment variables (via .env):
  DEEPSEEK_API_KEY   — for DeepSeek V3
  OPENAI_API_KEY     — for GPT-4o-mini
  ANTHROPIC_API_KEY  — for Claude
  LLM_GATEWAY_API_KEY       — for a configurable OpenAI-compatible endpoint
  LLM_GATEWAY_API_BASE_URL  — base URL for that endpoint
  GPTPLUS5_API_KEY   — for GPTPlus5 external API
  GPTPLUS5_API_BASE  — GPTPlus5 base URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.experiments.llm_arena import (
    GameConfig, LLMConfig, run_game, save_result,
    get_token_usage, reset_token_usage,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = ROOT / "results" / "llm_experiment"


def get_llm_config(provider: str = "deepseek") -> LLMConfig:
    """Get LLM config for specified provider."""
    configs = {
        "deepseek": LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=100,
        ),
        "openai": LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
        ),
        "anthropic": LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
            max_tokens=100,
        ),
        "gptplus5": LLMConfig(
            provider="gptplus5",
            model="gpt-4o-mini",
            temperature=1.0,
            max_tokens=100,
        ),
        "gptplus5_gpt5": LLMConfig(
            provider="gptplus5",
            model="gpt-5.2",
            temperature=1.0,
            max_tokens=100,
        ),
    }
    return configs.get(provider, configs["deepseek"])


async def run_experiment(
    mode: str = "pilot",
    provider: str = "deepseek",
    n_agents: int = 20,
    n_rounds: int = 200,
    n_repeats: int = 10,
    seed_fraction: float = 0.05,
    conditions_override: str = "",
) -> dict:
    """
    Run the experiment.

    Modes:
      test:  1 condition, 1 repeat, 20 rounds (for debugging)
      pilot: A + C, n_repeats each
      full:  A + B + C + D, n_repeats each
    """
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = LLM_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    llm_config = get_llm_config(provider)
    logger.info("Using LLM: %s/%s", llm_config.provider, llm_config.model)

    if mode == "test":
        conditions = [conditions_override] if conditions_override else ["A"]
        n_repeats = 1
        n_rounds = 50
    elif mode == "pilot":
        conditions = ["A", "C"]
    elif mode == "full":
        conditions = ["A", "B", "C", "D"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    all_results = []
    total_calls = 0
    t0 = time.time()

    for cond in conditions:
        logger.info("=== Condition %s (%d repeats) ===", cond, n_repeats)

        for rep in range(n_repeats):
            seed = 1000 * ord(cond) + rep
            existing = raw_dir / f"llm_game_{cond}_seed{seed}.json"
            if existing.exists():
                logger.info("  Run %d/%d (seed=%d) — SKIPPED (already exists)", rep + 1, n_repeats, seed)
                data = json.loads(existing.read_text())
                all_results.append({
                    "condition": cond, "seed": seed,
                    "final_cooperation": data["cooperation_rate"][-1] if data.get("cooperation_rate") else 0,
                    "mean_cooperation": float(np.mean(data.get("cooperation_rate", [0]))),
                    "cooperation_trajectory": data.get("cooperation_rate", []),
                    "mean_contribution_trajectory": data.get("mean_contribution", []),
                    "total_calls": data.get("total_llm_calls", 0),
                    "elapsed": data.get("elapsed_seconds", 0),
                })
                continue
            logger.info("  Run %d/%d (seed=%d)...", rep + 1, n_repeats, seed)

            config = GameConfig(
                condition=cond,
                n_agents=n_agents,
                n_rounds=n_rounds,
                endowment=20,
                multiplier=1.6,
                seed_fraction=seed_fraction,
                seed=seed,
                llm_config=llm_config,
                concurrency=10,
            )

            result = await run_game(config)
            save_result(result, raw_dir)

            all_results.append({
                "condition": result.condition,
                "seed": result.seed,
                "final_cooperation": result.cooperation_rate[-1] if result.cooperation_rate else 0,
                "mean_cooperation": float(np.mean(result.cooperation_rate)),
                "cooperation_trajectory": result.cooperation_rate,
                "mean_contribution_trajectory": result.mean_contribution,
                "total_calls": result.total_llm_calls,
                "elapsed": result.elapsed_seconds,
            })

            total_calls += result.total_llm_calls
            logger.info("    Done: coop=%.2f, calls=%d, time=%.1fs",
                        result.cooperation_rate[-1],
                        result.total_llm_calls,
                        result.elapsed_seconds)

    elapsed = time.time() - t0
    token_usage = get_token_usage()
    logger.info("\nTotal: %d runs, %d LLM calls, %.1fs", len(all_results), total_calls, elapsed)
    logger.info("Token usage: prompt=%d, completion=%d, total=%d",
                token_usage["prompt_tokens"], token_usage["completion_tokens"],
                token_usage["total_tokens"])

    summary = _aggregate(all_results)
    summary["meta"] = {
        "mode": mode,
        "provider": provider,
        "model": llm_config.model,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "n_repeats": n_repeats,
        "total_llm_calls": total_calls,
        "elapsed_seconds": elapsed,
        "token_usage": token_usage,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    summary_path = LLM_DIR / "llm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary: %s", summary_path)

    # Archive a copy with timestamp for safety
    archive_dir = LLM_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"summary_{mode}_{provider}_{ts}.json"
    archive_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Archived: %s", archive_path)

    return summary


def _aggregate(results: list[dict]) -> dict:
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    summary = {}
    for cond, runs in by_cond.items():
        coops = [r["final_cooperation"] for r in runs]
        mean_coops = [r["mean_cooperation"] for r in runs]

        # Pad trajectories to same length before averaging
        max_len = max(len(r["cooperation_trajectory"]) for r in runs)
        coop_trajs = []
        contrib_trajs = []
        for r in runs:
            ct = r["cooperation_trajectory"]
            mt = r["mean_contribution_trajectory"]
            # Pad with last value
            ct = ct + [ct[-1]] * (max_len - len(ct)) if len(ct) < max_len else ct
            mt = mt + [mt[-1]] * (max_len - len(mt)) if len(mt) < max_len else mt
            coop_trajs.append(ct)
            contrib_trajs.append(mt)

        avg_coop = np.mean(coop_trajs, axis=0).tolist()
        std_coop = np.std(coop_trajs, axis=0).tolist()
        avg_contrib = np.mean(contrib_trajs, axis=0).tolist()

        summary[cond] = {
            "n_runs": len(runs),
            "final_cooperation_mean": float(np.mean(coops)),
            "final_cooperation_std": float(np.std(coops)),
            "mean_cooperation_mean": float(np.mean(mean_coops)),
            "avg_cooperation_trajectory": avg_coop,
            "std_cooperation_trajectory": std_coop,
            "avg_contribution_trajectory": avg_contrib,
        }

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="LLM PGG Experiment")
    parser.add_argument("--mode", default="test", choices=["test", "pilot", "full"])
    parser.add_argument("--provider", default="gptplus5",
                        choices=["deepseek", "openai", "anthropic",
                                 "gptplus5", "gptplus5_gpt5", "custom"])
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=200)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--seed-fraction", type=float, default=0.05)
    parser.add_argument("--condition", default="", help="Override condition for test mode (A/B/C/D)")
    parser.add_argument("--conditions", nargs="*", default=None, help="List of conditions to run (A B C D)")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    # Custom model support (for 16-model sweep)
    parser.add_argument("--custom-model", default=None, help="Custom model ID")
    parser.add_argument("--custom-api-url", default=None, help="Custom API base URL")
    parser.add_argument("--custom-api-key", default=None, help="Custom API key")
    args = parser.parse_args()

    # Handle custom provider
    if args.provider == "custom" and args.custom_model:
        from src.experiments.llm_arena import LLMConfig
        custom_cfg = LLMConfig(
            provider="gateway",
            model=args.custom_model,
            api_key=args.custom_api_key or os.environ.get("LLM_GATEWAY_API_KEY", ""),
            base_url=args.custom_api_url or os.environ.get("LLM_GATEWAY_API_BASE_URL", ""),
            temperature=0.7,
            max_tokens=100,
        )
        if not custom_cfg.api_key:
            parser.error(
                "Custom providers require --custom-api-key or LLM_GATEWAY_API_KEY."
            )
        if not custom_cfg.base_url:
            parser.error(
                "Custom providers require --custom-api-url or LLM_GATEWAY_API_BASE_URL."
            )
        # Monkey-patch get_llm_config so run_experiment uses our custom config
        globals()["get_llm_config"] = lambda p="custom": custom_cfg

    # Override output directory if specified
    if args.output_dir:
        globals()["LLM_DIR"] = Path(args.output_dir)
        globals()["LLM_DIR"].mkdir(parents=True, exist_ok=True)

    summary = asyncio.run(run_experiment(
        mode=args.mode,
        provider=args.provider,
        n_agents=args.n_agents,
        conditions_override=args.condition if args.condition else "",
        n_rounds=args.n_rounds,
        n_repeats=args.n_repeats,
        seed_fraction=args.seed_fraction,
    ))

    logger.info("\n=== Results ===")
    for cond in ["A", "B", "C", "D"]:
        if cond in summary:
            s = summary[cond]
            logger.info("  Condition %s: coop=%.3f±%.3f (n=%d)",
                        cond, s["final_cooperation_mean"],
                        s["final_cooperation_std"], s["n_runs"])
