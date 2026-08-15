"""
Analyze LLM experiment results and compare with ABM + theory.

Produces:
  Fig_LLM_cooperation.png          — c(t) across conditions
  Fig_LLM_contribution_heatmap.png — per-agent contribution over time
  Fig_LLM_vs_ABM.png               — LLM vs ABM trajectory comparison
  Fig_LLM_vs_Theory.png            — LLM data vs theoretical prediction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = ROOT / "results" / "llm_experiment"
ABM_DIR = ROOT / "results" / "abm"
FIG_DIR = ROOT / "results" / "paper_figures"

COLORS = {"A": "#E24A33", "B": "#FFA500", "C": "#348ABD", "D": "#2ca02c"}
LABELS = {
    "A": "Cond A (Dyadic)",
    "B": "Cond B (Reciprocity)",
    "C": "Cond C (Triad)",
    "D": "Cond D (Pentad)",
}

COND_TO_ABM = {
    "A": "dyadic_baseline",
    "B": "dyadic_reciprocity",
    "C": "triad_hyperedge",
    "D": "pentad_hyperedge",
}


def load_llm_summary() -> dict:
    return json.loads((LLM_DIR / "llm_summary.json").read_text())


def load_abm_summary() -> dict:
    path = ABM_DIR / "abm_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def plot_llm_cooperation(summary: dict, output: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in ["A", "B", "C", "D"]:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_cooperation_trajectory"])
        std = np.array(s["std_cooperation_trajectory"])
        t = np.arange(len(avg))

        ax.plot(t, avg, color=COLORS[cond], linewidth=2, label=LABELS[cond])
        ax.fill_between(t, avg - std, avg + std, color=COLORS[cond], alpha=0.15)

    ax.set_xlabel("Round $t$", fontsize=12)
    ax.set_ylabel("Cooperation rate $c(t)$", fontsize=12)
    ax.set_title("LLM Multi-Agent PGG: Cooperation under Topological Constraints", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_llm_vs_abm(llm_summary: dict, abm_summary: dict, output: str | None = None):
    """Compare LLM and ABM cooperation trajectories side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [c for c in ["A", "B", "C", "D"] if c in llm_summary]
    if not conditions:
        logger.warning("No LLM data to compare")
        return None

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        # LLM
        ls = llm_summary[cond]
        llm_avg = np.array(ls["avg_cooperation_trajectory"])
        llm_std = np.array(ls["std_cooperation_trajectory"])
        t_llm = np.arange(len(llm_avg))
        ax.plot(t_llm, llm_avg, color=COLORS[cond], linewidth=2, label="LLM agents")
        ax.fill_between(t_llm, llm_avg - llm_std, llm_avg + llm_std,
                        color=COLORS[cond], alpha=0.12)

        # ABM
        abm_cond = COND_TO_ABM.get(cond, "")
        if abm_cond in abm_summary:
            abm_s = abm_summary[abm_cond]
            abm_avg = np.array(abm_s["avg_cooperation_trajectory"])
            abm_std = np.array(abm_s["std_cooperation_trajectory"])
            # normalize ABM time to match LLM rounds
            t_abm = np.linspace(0, len(llm_avg) - 1, len(abm_avg))
            ax.plot(t_abm, abm_avg, color=COLORS[cond], linewidth=2,
                    linestyle="--", label="ABM agents", alpha=0.7)
            ax.fill_between(t_abm, abm_avg - abm_std, abm_avg + abm_std,
                            color=COLORS[cond], alpha=0.06)

        ax.set_xlabel("Round $t$")
        ax.set_title(f"Condition {cond}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Cooperation rate $c(t)$")
    fig.suptitle("LLM vs ABM: Do Real Language Agents Match Simulation?", fontsize=14)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_contribution_heatmap(output: str | None = None):
    """Per-agent contribution heatmap from raw LLM data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_dir = LLM_DIR / "raw"
    if not raw_dir.exists():
        logger.warning("No raw LLM data")
        return None

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        return None

    # pick one run per condition for visualization
    shown = {}
    for f in files:
        data = json.loads(f.read_text())
        cond = data["condition"]
        if cond not in shown:
            shown[cond] = data

    n_conds = len(shown)
    fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 5))
    if n_conds == 1:
        axes = [axes]

    for ax, (cond, data) in zip(axes, sorted(shown.items())):
        histories = data.get("agent_histories", {})
        if not histories:
            continue

        n_agents = len(histories)
        n_rounds = len(next(iter(histories.values())).get("contributions", []))
        matrix = np.zeros((n_agents, n_rounds))

        for i, (aid, info) in enumerate(sorted(histories.items(), key=lambda x: int(x[0]))):
            contribs = info.get("contributions", [])
            for j, c in enumerate(contribs):
                matrix[i, j] = c

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=20)
        ax.set_xlabel("Round")
        ax.set_ylabel("Agent ID")
        ax.set_title(f"Condition {cond}")

    fig.colorbar(im, ax=axes, label="Contribution", shrink=0.8)
    fig.suptitle("Per-Agent Contribution Patterns", fontsize=14)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def plot_llm_contribution(summary: dict, output: str | None = None):
    """Plot mean contribution trajectory across conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in ["A", "B", "C", "D"]:
        if cond not in summary:
            continue
        s = summary[cond]
        avg = np.array(s["avg_contribution_trajectory"])
        t = np.arange(len(avg))
        ax.plot(t, avg, color=COLORS[cond], linewidth=2, label=LABELS[cond])

    ax.set_xlabel("Round $t$", fontsize=12)
    ax.set_ylabel("Mean contribution", fontsize=12)
    ax.set_title("LLM PGG: Mean Contribution per Round (endowment=20)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", output)
    return fig


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading results...")
    llm_summary = load_llm_summary()
    abm_summary = load_abm_summary()

    logger.info("Generating LLM figures...")
    plot_llm_cooperation(llm_summary, str(FIG_DIR / "Fig_LLM_cooperation.png"))
    plot_llm_contribution(llm_summary, str(FIG_DIR / "Fig_LLM_contribution.png"))
    plot_contribution_heatmap(str(FIG_DIR / "Fig_LLM_contribution_heatmap.png"))

    if abm_summary:
        logger.info("Generating LLM vs ABM comparison...")
        plot_llm_vs_abm(llm_summary, abm_summary,
                        str(FIG_DIR / "Fig_LLM_vs_ABM.png"))

    logger.info("=== Done ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
