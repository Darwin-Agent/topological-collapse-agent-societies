"""
Run the full ABM PGG experiment: 4 conditions x R repeats x T rounds.

Produces:
  results/abm/raw/  — per-run JSON files
  results/abm/      — aggregate CSV + summary JSON
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.experiments.abm_pgg import Condition, SimulationResult, run_simulation

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ABM_DIR = ROOT / "results" / "abm"
RAW_DIR = ABM_DIR / "raw"


def _run_one(args: tuple) -> dict:
    """Worker function for parallel execution."""
    cond_name, n_agents, n_rounds, seed_fraction, seed, avg_degree, avg_membership = args
    condition = Condition(cond_name)
    result = run_simulation(
        condition=condition,
        n_agents=n_agents,
        n_rounds=n_rounds,
        seed_fraction=seed_fraction,
        seed=seed,
        avg_degree=avg_degree,
        avg_membership=avg_membership,
    )
    return {
        "condition": result.condition,
        "seed": result.seed,
        "n_agents": result.n_agents,
        "n_rounds": result.n_rounds,
        "final_cooperation": result.final_cooperation,
        "final_norm_adoption": result.final_norm_adoption,
        "phase_transition_round": result.phase_transition_round,
        "cooperation_rate": result.cooperation_rate.tolist(),
        "norm_adoption_rate": result.norm_adoption_rate.tolist(),
        "mean_payoff": result.mean_payoff.tolist(),
    }


def run_experiment(
    n_agents: int = 100,
    n_rounds: int = 500,
    n_repeats: int = 30,
    seed_fraction: float = 0.05,
    avg_degree: int = 6,
    avg_membership: int = 4,
    n_workers: int = 4,
    seed_fractions_sweep: list[float] | None = None,
) -> dict:
    """
    Run the full factorial experiment.

    If seed_fractions_sweep is provided, also runs Condition C with
    multiple initial seed fractions to map the critical mass.
    """
    ABM_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    conditions = [Condition.A, Condition.B, Condition.C, Condition.D]
    tasks = []

    for cond in conditions:
        for rep in range(n_repeats):
            seed = 1000 * (cond.name.encode()[0]) + rep
            tasks.append((
                cond.value, n_agents, n_rounds, seed_fraction,
                seed, avg_degree, avg_membership,
            ))

    # critical mass sweep for Condition C
    if seed_fractions_sweep:
        for sf in seed_fractions_sweep:
            for rep in range(min(n_repeats, 20)):
                seed = 90000 + int(sf * 10000) + rep
                tasks.append((
                    Condition.C.value, n_agents, n_rounds, sf,
                    seed, avg_degree, avg_membership,
                ))

    logger.info("Launching %d ABM runs across %d workers...", len(tasks), n_workers)
    t0 = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                res = future.result()
                all_results.append(res)
                if done % 20 == 0 or done == len(tasks):
                    logger.info("  [%d/%d] completed (%.1fs)",
                                done, len(tasks), time.time() - t0)
            except Exception as e:
                logger.error("Task failed: %s", e)

    elapsed = time.time() - t0
    logger.info("All %d runs done in %.1fs", len(all_results), elapsed)

    # save raw
    for r in all_results:
        fname = f"{r['condition']}_seed{r['seed']}.json"
        (RAW_DIR / fname).write_text(json.dumps(r))

    # aggregate summary
    summary = _aggregate(all_results)
    summary["meta"] = {
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "n_repeats": n_repeats,
        "seed_fraction": seed_fraction,
        "elapsed_seconds": elapsed,
        "total_runs": len(all_results),
    }

    summary_path = ABM_DIR / "abm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary saved to %s", summary_path)

    return summary


def _aggregate(results: list[dict]) -> dict:
    """Compute per-condition statistics."""
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    summary = {}
    for cond, runs in by_cond.items():
        coops = [r["final_cooperation"] for r in runs]
        norms = [r["final_norm_adoption"] for r in runs]
        pts = [r["phase_transition_round"] for r in runs if r["phase_transition_round"] is not None]

        # average cooperation trajectory
        n_rounds = len(runs[0]["cooperation_rate"])
        avg_coop = np.mean([r["cooperation_rate"] for r in runs], axis=0).tolist()
        std_coop = np.std([r["cooperation_rate"] for r in runs], axis=0).tolist()
        avg_norm = np.mean([r["norm_adoption_rate"] for r in runs], axis=0).tolist()
        std_norm = np.std([r["norm_adoption_rate"] for r in runs], axis=0).tolist()

        summary[cond] = {
            "n_runs": len(runs),
            "final_cooperation_mean": float(np.mean(coops)),
            "final_cooperation_std": float(np.std(coops)),
            "final_norm_mean": float(np.mean(norms)),
            "final_norm_std": float(np.std(norms)),
            "phase_transition_detected": len(pts),
            "phase_transition_round_mean": float(np.mean(pts)) if pts else None,
            "avg_cooperation_trajectory": avg_coop,
            "std_cooperation_trajectory": std_coop,
            "avg_norm_trajectory": avg_norm,
            "std_norm_trajectory": std_norm,
        }

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== ABM PGG Experiment ===")

    summary = run_experiment(
        n_agents=100,
        n_rounds=500,
        n_repeats=30,
        seed_fraction=0.05,
        n_workers=6,
        seed_fractions_sweep=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    )

    logger.info("\n=== Results Summary ===")
    for cond_name in ["dyadic_baseline", "dyadic_reciprocity", "triad_hyperedge", "pentad_hyperedge"]:
        if cond_name in summary:
            s = summary[cond_name]
            logger.info("  %s: coop=%.3f±%.3f  norm=%.3f±%.3f  PT_detected=%d",
                        cond_name,
                        s["final_cooperation_mean"], s["final_cooperation_std"],
                        s["final_norm_mean"], s["final_norm_std"],
                        s["phase_transition_detected"])

    logger.info("=== Done ===")
