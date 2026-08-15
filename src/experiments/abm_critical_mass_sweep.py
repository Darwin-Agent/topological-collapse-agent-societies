"""
Complete critical mass sweep for all 4 conditions.

Runs each condition across a range of seed fractions ρ₀ to map
the phase transition boundary. This produces the paper's key figure
showing discontinuous vs continuous transitions.

Output:
  results/abm/critical_mass_sweep.json
  results/paper_figures/Fig_ABM_critical_mass_all.png  (all 4 conditions)
  results/paper_figures/Fig_ABM_bistability_map.png    (2D heatmap)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.experiments.abm_pgg import Condition, run_simulation

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ABM_DIR = ROOT / "results" / "abm"
FIG_DIR = ROOT / "results" / "paper_figures"

SEED_FRACTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_REPEATS = 20
N_AGENTS = 100
N_ROUNDS = 500


def _run_one(args):
    cond_val, sf, seed = args
    cond = Condition(cond_val)
    r = run_simulation(cond, n_agents=N_AGENTS, n_rounds=N_ROUNDS,
                       seed_fraction=sf, seed=seed)
    return {
        "condition": cond_val,
        "seed_fraction": sf,
        "seed": seed,
        "final_cooperation": r.final_cooperation,
        "final_norm_adoption": r.final_norm_adoption,
        "phase_transition_round": r.phase_transition_round,
    }


def run_sweep(n_workers: int = 6) -> dict:
    ABM_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cond in [Condition.A, Condition.B, Condition.C, Condition.D]:
        for sf in SEED_FRACTIONS:
            for rep in range(N_REPEATS):
                seed = 80000 + ord(cond.name) * 1000 + int(sf * 1000) + rep
                tasks.append((cond.value, sf, seed))

    logger.info("Launching %d sweep runs (%d conditions x %d fractions x %d repeats)...",
                len(tasks), 4, len(SEED_FRACTIONS), N_REPEATS)
    t0 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            try:
                results.append(f.result())
            except Exception as e:
                logger.error("Failed: %s", e)
            if done % 50 == 0 or done == len(tasks):
                logger.info("  [%d/%d] (%.1fs)", done, len(tasks), time.time() - t0)

    elapsed = time.time() - t0
    logger.info("Sweep done in %.1fs", elapsed)

    # aggregate
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r["condition"]][r["seed_fraction"]].append(r)

    summary = {}
    for cond_name, sf_data in agg.items():
        summary[cond_name] = {}
        for sf, runs in sorted(sf_data.items()):
            coops = [r["final_cooperation"] for r in runs]
            norms = [r["final_norm_adoption"] for r in runs]
            summary[cond_name][str(sf)] = {
                "coop_mean": float(np.mean(coops)),
                "coop_std": float(np.std(coops)),
                "norm_mean": float(np.mean(norms)),
                "norm_std": float(np.std(norms)),
                "n_runs": len(runs),
            }

    out = {
        "meta": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "n_repeats": N_REPEATS,
            "seed_fractions": SEED_FRACTIONS,
        },
        "summary": summary,
        "runs": sorted(
            results,
            key=lambda row: (row["condition"], row["seed_fraction"], row["seed"]),
        ),
        "elapsed": elapsed,
        "n_total": len(results),
    }
    (ABM_DIR / "critical_mass_sweep.json").write_text(json.dumps(out, indent=2))
    logger.info("Saved: %s", ABM_DIR / "critical_mass_sweep.json")

    return summary


def plot_critical_mass_all(summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "dyadic_baseline": "#E24A33",
        "dyadic_reciprocity": "#FFA500",
        "triad_hyperedge": "#348ABD",
        "pentad_hyperedge": "#2ca02c",
    }
    labels = {
        "dyadic_baseline": "Condition A (Dyadic)",
        "dyadic_reciprocity": "Condition B (Reciprocity)",
        "triad_hyperedge": "Condition C (Triad)",
        "pentad_hyperedge": "Condition D (Pentad)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for cond_name in ["dyadic_baseline", "dyadic_reciprocity", "triad_hyperedge", "pentad_hyperedge"]:
        if cond_name not in summary:
            continue
        sf_data = summary[cond_name]
        sfs = sorted([float(k) for k in sf_data.keys()])
        coop_means = [sf_data[str(s)]["coop_mean"] for s in sfs]
        coop_stds = [sf_data[str(s)]["coop_std"] for s in sfs]
        norm_means = [sf_data[str(s)]["norm_mean"] for s in sfs]
        norm_stds = [sf_data[str(s)]["norm_std"] for s in sfs]

        color = colors[cond_name]
        label = labels[cond_name]

        ax1.errorbar(sfs, coop_means, yerr=coop_stds, fmt="o-", color=color,
                     linewidth=2, markersize=6, capsize=4, label=label)
        ax2.errorbar(sfs, norm_means, yerr=norm_stds, fmt="o-", color=color,
                     linewidth=2, markersize=6, capsize=4, label=label)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Initial seed fraction $\\rho_0$", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    ax1.set_ylabel("Final cooperation rate $c_\\infty$", fontsize=12)
    ax1.set_title("(a) Cooperation: Topology Has Modest Effect", fontsize=13)

    ax2.set_ylabel("Final norm adoption $\\rho_\\infty$", fontsize=12)
    ax2.set_title("(b) Norm Adoption: Higher-Order Requires Critical Mass", fontsize=13)

    fig.suptitle("Critical Mass Analysis: Continuous (Dyadic) vs. Discontinuous (Hyperedge) Transitions\n"
                 f"N={N_AGENTS} agents, T={N_ROUNDS} rounds, R={N_REPEATS} repeats per point",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = str(FIG_DIR / "Fig_ABM_critical_mass_all.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", out)

    # also plot a bistability heatmap for Condition C
    _plot_bistability_heatmap(summary)


def _plot_bistability_heatmap(summary: dict):
    """2D heatmap: seed_fraction x condition -> norm adoption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = ["dyadic_baseline", "dyadic_reciprocity", "triad_hyperedge", "pentad_hyperedge"]
    cond_labels = ["A: Dyadic", "B: Reciprocity", "C: Triad", "D: Pentad"]

    all_sfs = sorted(set(float(k) for c in summary.values() for k in c.keys()))
    matrix = np.zeros((len(conds), len(all_sfs)))

    for i, cond in enumerate(conds):
        if cond not in summary:
            continue
        for j, sf in enumerate(all_sfs):
            key = str(sf)
            if key in summary[cond]:
                matrix[i, j] = summary[cond][key]["norm_mean"]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(cond_labels, fontsize=11)
    ax.set_xticks(range(len(all_sfs)))
    ax.set_xticklabels([f"{s:.2f}" for s in all_sfs], rotation=45, fontsize=9)
    ax.set_xlabel("Initial seed fraction $\\rho_0$", fontsize=12)
    ax.set_title("Norm Adoption Phase Diagram: Topology × Seed Fraction", fontsize=13)

    # annotate values
    for i in range(len(conds)):
        for j in range(len(all_sfs)):
            v = matrix[i, j]
            color = "white" if v < 0.4 or v > 0.7 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Final norm adoption $\\rho_\\infty$", shrink=0.9)
    fig.tight_layout()
    out = str(FIG_DIR / "Fig_ABM_bistability_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Critical Mass Sweep ===")
    summary = run_sweep(n_workers=6)

    logger.info("=== Generating Figures ===")
    plot_critical_mass_all(summary)

    logger.info("=== Done ===")
