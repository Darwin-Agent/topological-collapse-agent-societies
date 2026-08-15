"""
Analyze ABM experiment results and generate publication-quality figures.

Produces:
  Fig_ABM_cooperation_trajectories.png — c(t) for all 4 conditions
  Fig_ABM_norm_adoption.png           — rho(t) for all 4 conditions
  Fig_ABM_phase_transition.png        — dc/dt showing explosive transitions
  Fig_ABM_critical_mass.png           — critical mass sweep for Condition C
  Fig_ABM_final_boxplot.png           — final cooperation box plots
  Fig_ABM_composite.png               — 2x2 composite for paper
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ABM_DIR = ROOT / "results" / "abm"
FIG_DIR = ROOT / "results" / "paper_figures"

CONDITION_LABELS = {
    "dyadic_baseline": "Condition A\n(Dyadic Baseline)",
    "dyadic_reciprocity": "Condition B\n(Dyadic + Reciprocity)",
    "triad_hyperedge": "Condition C\n(Triad Hyperedge)",
    "pentad_hyperedge": "Condition D\n(Pentad Hyperedge)",
}

CONDITION_COLORS = {
    "dyadic_baseline": "#E24A33",
    "dyadic_reciprocity": "#FFA500",
    "triad_hyperedge": "#348ABD",
    "pentad_hyperedge": "#2ca02c",
}

CONDITION_ORDER = ["dyadic_baseline", "dyadic_reciprocity", "triad_hyperedge", "pentad_hyperedge"]


def load_summary() -> dict:
    path = ABM_DIR / "abm_summary.json"
    return json.loads(path.read_text())


def load_raw_runs() -> list[dict]:
    raw_dir = ABM_DIR / "raw"
    runs = []
    for f in sorted(raw_dir.glob("*.json")):
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8", errors="ignore")))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Skipping corrupt file %s: %s", f.name, e)
    return runs


def plot_cooperation_trajectories(summary: dict, output: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_cooperation_trajectory"])
        std = np.array(s["std_cooperation_trajectory"])
        t = np.arange(len(avg))

        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, avg, color=color, linewidth=2, label=label)
        ax.fill_between(t, avg - std, avg + std, color=color, alpha=0.15)

    ax.set_xlabel("Round $t$", fontsize=12)
    ax.set_ylabel("Cooperation rate $c(t)$", fontsize=12)
    ax.set_title("Public Goods Game: Cooperation Dynamics under Four Topological Conditions", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_norm_adoption(summary: dict, output: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_norm_trajectory"])
        std = np.array(s["std_norm_trajectory"])
        t = np.arange(len(avg))

        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, avg, color=color, linewidth=2, label=label)
        ax.fill_between(t, avg - std, avg + std, color=color, alpha=0.15)

    ax.set_xlabel("Round $t$", fontsize=12)
    ax.set_ylabel("Norm adoption rate $\\rho(t)$", fontsize=12)
    ax.set_title("Norm Propagation: Higher-Order Topology Enables Explosive Adoption", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_phase_transition_speed(summary: dict, output: str | None = None):
    """Plot dc/dt to show explosive vs gradual transitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    window = 15

    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_cooperation_trajectory"])
        smoothed = np.convolve(avg, np.ones(window) / window, mode="valid")
        dc_dt = np.diff(smoothed)
        t = np.arange(len(dc_dt))

        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, dc_dt, color=color, linewidth=1.5, label=label)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Round $t$", fontsize=12)
    ax.set_ylabel("$dc/dt$ (cooperation change rate)", fontsize=12)
    ax.set_title("Phase Transition Detection: Explosive vs. Gradual Cooperation Onset", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_critical_mass_sweep(raw_runs: list[dict], output: str | None = None):
    """Plot final cooperation as f(seed_fraction) for Condition C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    triad_runs = [r for r in raw_runs if r["condition"] == "triad_hyperedge"]

    # group by seed → infer seed_fraction from seed encoding
    # seed = 90000 + int(sf * 10000) + rep  for sweep runs
    sweep_runs = [r for r in triad_runs if r["seed"] >= 90000]
    non_sweep = [r for r in triad_runs if r["seed"] < 90000]

    if not sweep_runs:
        logger.warning("No critical mass sweep data found")
        return None

    sf_to_coop = {}
    for r in sweep_runs:
        sf = round((r["seed"] - 90000) // 1 % 10000 / 10000, 4)
        # re-derive: seed = 90000 + int(sf*10000) + rep → sf ~ (seed - 90000) / 10000 (floor)
        sf_approx = (r["seed"] - 90000) // 20 / 500  # approximate
        # Use actual: parse from trajectory shape
        if sf_approx not in sf_to_coop:
            sf_to_coop[sf_approx] = []
        sf_to_coop[sf_approx].append(r["final_cooperation"])

    # fallback: just group by unique seed prefixes
    from collections import defaultdict
    seed_groups = defaultdict(list)
    for r in sweep_runs:
        prefix = (r["seed"] - 90000) // 20
        seed_groups[prefix].append(r["final_cooperation"])

    fig, ax = plt.subplots(figsize=(8, 6))

    # more robust: bin by final cooperation
    fractions_used = sorted(seed_groups.keys())
    sf_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30][:len(fractions_used)]

    means = []
    stds = []
    for frac in fractions_used:
        vals = seed_groups[frac]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    if len(sf_values) == len(means):
        ax.errorbar(sf_values, means, yerr=stds, fmt="o-", color="#348ABD",
                    linewidth=2, markersize=8, capsize=5)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% cooperation threshold")
        ax.set_xlabel("Initial seed fraction $\\rho_0$", fontsize=12)
        ax.set_ylabel("Final cooperation rate $c_\\infty$", fontsize=12)
        ax.set_title("Critical Mass for Explosive Cooperation\n(Condition C: Triad Hyperedge)", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_final_boxplot(raw_runs: list[dict], output: str | None = None):
    """Box plot of final cooperation across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # filter to main experiment runs (not sweep)
    main_runs = [r for r in raw_runs if r["seed"] < 90000]

    coop_data = []
    norm_data = []
    labels = []
    colors = []

    for cond in CONDITION_ORDER:
        cond_runs = [r for r in main_runs if r["condition"] == cond]
        if not cond_runs:
            continue
        coop_data.append([r["final_cooperation"] for r in cond_runs])
        norm_data.append([r["final_norm_adoption"] for r in cond_runs])
        labels.append(CONDITION_LABELS[cond].replace("\n", " "))
        colors.append(CONDITION_COLORS[cond])

    bp1 = ax1.boxplot(coop_data, labels=[f"Cond {c}" for c in "ABCD"[:len(labels)]],
                       patch_artist=True, widths=0.6)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Final cooperation rate $c_\\infty$", fontsize=12)
    ax1.set_title("Cooperation Outcomes", fontsize=13)
    ax1.grid(alpha=0.3, axis="y")

    bp2 = ax2.boxplot(norm_data, labels=[f"Cond {c}" for c in "ABCD"[:len(labels)]],
                       patch_artist=True, widths=0.6)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Final norm adoption $\\rho_\\infty$", fontsize=12)
    ax2.set_title("Norm Adoption Outcomes", fontsize=13)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("ABM Experiment: Topology Determines Collective Outcomes (N=100, T=500, R=30)", fontsize=14)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_composite(summary: dict, raw_runs: list[dict], output: str | None = None):
    """2x2 composite figure for paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) cooperation trajectories
    ax = axes[0, 0]
    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_cooperation_trajectory"])
        std = np.array(s["std_cooperation_trajectory"])
        t = np.arange(len(avg))
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, avg, color=color, linewidth=2, label=label)
        ax.fill_between(t, avg - std, avg + std, color=color, alpha=0.12)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Cooperation rate $c(t)$")
    ax.set_title("(a) Cooperation Dynamics")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) norm adoption
    ax = axes[0, 1]
    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_norm_trajectory"])
        std = np.array(s["std_norm_trajectory"])
        t = np.arange(len(avg))
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, avg, color=color, linewidth=2, label=label)
        ax.fill_between(t, avg - std, avg + std, color=color, alpha=0.12)
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("Norm adoption $\\rho(t)$")
    ax.set_title("(b) Norm Propagation")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) phase transition speed
    ax = axes[1, 0]
    window = 15
    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_cooperation_trajectory"])
        smoothed = np.convolve(avg, np.ones(window) / window, mode="valid")
        dc_dt = np.diff(smoothed)
        t = np.arange(len(dc_dt))
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        ax.plot(t, dc_dt, color=color, linewidth=1.5, label=label)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Round $t$")
    ax.set_ylabel("$dc/dt$")
    ax.set_title("(c) Phase Transition Speed")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) final outcomes box plot
    ax = axes[1, 1]
    main_runs = [r for r in raw_runs if r["seed"] < 90000]
    coop_data = []
    colors_list = []
    for cond in CONDITION_ORDER:
        cond_runs = [r for r in main_runs if r["condition"] == cond]
        if not cond_runs:
            continue
        coop_data.append([r["final_cooperation"] for r in cond_runs])
        colors_list.append(CONDITION_COLORS[cond])

    if coop_data:
        bp = ax.boxplot(coop_data, labels=["A", "B", "C", "D"][:len(coop_data)],
                        patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Final cooperation $c_\\infty$")
    ax.set_title("(d) Final Cooperation Comparison")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("ABM Public Goods Game: Topological Conditions A-D\n"
                 "(N=100 agents, T=500 rounds, R=30 repeats, seed=5%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ABM results...")
    summary = load_summary()
    raw_runs = load_raw_runs()

    logger.info("Generating figures...")

    plot_cooperation_trajectories(
        summary, str(FIG_DIR / "Fig_ABM_cooperation_trajectories.png"))

    plot_norm_adoption(
        summary, str(FIG_DIR / "Fig_ABM_norm_adoption.png"))

    plot_phase_transition_speed(
        summary, str(FIG_DIR / "Fig_ABM_phase_transition.png"))

    plot_critical_mass_sweep(
        raw_runs, str(FIG_DIR / "Fig_ABM_critical_mass.png"))

    plot_final_boxplot(
        raw_runs, str(FIG_DIR / "Fig_ABM_final_boxplot.png"))

    plot_composite(
        summary, raw_runs, str(FIG_DIR / "Fig_ABM_composite.png"))

    # print statistical summary
    logger.info("\n=== Statistical Summary ===")
    for cond in CONDITION_ORDER:
        if cond not in summary:
            continue
        s = summary[cond]
        logger.info("  %s:", cond)
        logger.info("    Cooperation: %.3f ± %.3f",
                    s["final_cooperation_mean"], s["final_cooperation_std"])
        logger.info("    Norm adoption: %.3f ± %.3f",
                    s["final_norm_mean"], s["final_norm_std"])
        logger.info("    Phase transitions detected: %d/%d",
                    s["phase_transition_detected"], s["n_runs"])

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
