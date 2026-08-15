"""
Generate the key paper figure: ABM experimental data vs theoretical model predictions.

This figure validates the LLM-adapted contagion model by overlaying:
  - ABM experimental trajectories (data points with error bars)
  - Fitted mean-field model predictions (smooth curves)
  - Microscopic MC simulation (dashed curves)

Produces:
  Fig_ABM_vs_Theory_trajectories.png — 4-panel trajectory comparison
  Fig_ABM_vs_Theory_critical_mass.png — critical mass theory vs experiment
  Fig_ABM_vs_Theory_composite.png — combined 2x3 for paper
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ABM_DIR = ROOT / "results" / "abm"
FIT_DIR = ROOT / "results" / "model_fitting"
MICRO_DIR = ROOT / "results" / "micro_contagion"
FIG_DIR = ROOT / "results" / "paper_figures"

COLORS = {
    "A": "#E24A33", "B": "#FFA500", "C": "#348ABD", "D": "#2ca02c",
}
COND_MAP = {
    "dyadic_baseline": "A", "dyadic_reciprocity": "B",
    "triad_hyperedge": "C", "pentad_hyperedge": "D",
}
COND_LABELS = {
    "A": "Condition A\n(Dyadic Baseline)",
    "B": "Condition B\n(Dyadic + Reciprocity)",
    "C": "Condition C\n(Triad Hyperedge)",
    "D": "Condition D\n(Pentad Hyperedge)",
}


def load_data():
    abm = json.loads((ABM_DIR / "abm_summary.json").read_text())
    fit = json.loads((FIT_DIR / "fitting_abm.json").read_text())
    micro_path = MICRO_DIR / "micro_results.json"
    micro = json.loads(micro_path.read_text()) if micro_path.exists() else {}
    return abm, fit, micro


def generate_model_trajectory(params: dict, n_steps: int, rho0: float = 0.05) -> np.ndarray:
    """Generate mean-field model trajectory from fitted parameters."""
    from src.models.contagion import LLMContagionModel
    model = LLMContagionModel(
        beta1=params["beta1"],
        beta2=params["beta2"],
        mu=params["mu"],
        lam=params.get("lam", 0),
        C=params.get("C", 1),
    )
    t, rho = model.simulate(T=float(n_steps), rho0=rho0, dt=1.0)
    # resample to match ABM steps
    if len(rho) != n_steps:
        rho = np.interp(np.linspace(0, 1, n_steps), np.linspace(0, 1, len(rho)), rho)
    return rho


def plot_trajectory_comparison(abm: dict, fit: dict, micro: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    conds = ["dyadic_baseline", "dyadic_reciprocity", "triad_hyperedge", "pentad_hyperedge"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes = axes.flatten()

    for ax, cond_name in zip(axes, conds):
        letter = COND_MAP[cond_name]
        color = COLORS[letter]

        # ABM data
        if cond_name in abm:
            abm_avg = np.array(abm[cond_name]["avg_cooperation_trajectory"])
            abm_std = np.array(abm[cond_name]["std_cooperation_trajectory"])
            t_abm = np.arange(len(abm_avg))
            ax.plot(t_abm, abm_avg, color=color, linewidth=2.5, label="ABM experiment",
                    alpha=0.9)
            ax.fill_between(t_abm, abm_avg - abm_std, abm_avg + abm_std,
                            color=color, alpha=0.12)

        # fitted model prediction
        if letter in fit.get("fits", {}):
            params = fit["fits"][letter]
            n_steps = len(abm_avg) if cond_name in abm else 500
            model_rho = generate_model_trajectory(params, n_steps)
            ax.plot(np.arange(n_steps), model_rho, color="black", linewidth=1.5,
                    linestyle="--", label="Mean-field model (fitted)", alpha=0.8)

        # micro MC data (only for rho0=0.05)
        if letter in micro and "0.05" in micro[letter]:
            mc = micro[letter]["0.05"]
            mc_mean = np.array(mc["mean"])
            mc_std = np.array(mc["std"])
            t_mc = np.linspace(0, len(abm_avg) - 1 if cond_name in abm else 499, len(mc_mean))
            ax.plot(t_mc, mc_mean, color="gray", linewidth=1.5,
                    linestyle=":", label="Monte Carlo micro", alpha=0.7)

        ax.set_xlabel("Round $t$")
        ax.set_ylabel("$c(t)$ / $\\rho(t)$" if ax == axes[0] or ax == axes[2] else "")
        ax.set_title(f"{COND_LABELS[letter].replace(chr(10), ' ')}", fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

        # add fitted parameter text
        if letter in fit.get("fits", {}):
            p = fit["fits"][letter]
            txt = f"$\\beta_1$={p['beta1']:.3f}"
            if p['beta2'] > 0:
                txt += f", $\\beta_2$={p['beta2']:.3f}\n$\\lambda$={p.get('lam',0):.2f}, C={p.get('C',1):.1f}"
            txt += f"\nMSE={p['loss']:.2e}"
            ax.text(0.98, 0.15, txt, transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.suptitle("ABM Experiment vs. Theoretical Model: Quantitative Validation\n"
                 "(N=100 agents, T=500 rounds, R=30 repeats, ρ₀=5%)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = str(FIG_DIR / "Fig_ABM_vs_Theory_trajectories.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", out)


def plot_model_prediction_diagram(fit: dict):
    """Show how the model predicts critical mass for each condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.models.contagion import LLMContagionModel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: rho(t) for different rho0 under Condition C parameters
    if "C" in fit.get("fits", {}):
        params = fit["fits"]["C"]
        rho0_values = [0.03, 0.05, 0.08, 0.10, 0.15, 0.25]
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(rho0_values)))

        for rho0, c in zip(rho0_values, cmap):
            model = LLMContagionModel(
                beta1=params["beta1"], beta2=params["beta2"],
                mu=params["mu"], lam=params.get("lam", 0), C=params.get("C", 1))
            t, rho = model.simulate(T=500, rho0=rho0)
            ax1.plot(t, rho, color=c, linewidth=1.5, label=f"$\\rho_0$={rho0:.2f}")

        ax1.set_xlabel("Time $t$", fontsize=12)
        ax1.set_ylabel("Norm adoption $\\rho(t)$", fontsize=12)
        ax1.set_title("(a) Model Prediction: Condition C Dynamics\n"
                       "by Initial Seed Fraction", fontsize=12)
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

    # Panel 2: final rho vs rho0 for each condition's fitted model
    rho0_scan = np.linspace(0.01, 0.50, 50)
    for letter in ["A", "B", "C", "D"]:
        if letter not in fit.get("fits", {}):
            continue
        params = fit["fits"][letter]
        finals = []
        for rho0 in rho0_scan:
            model = LLMContagionModel(
                beta1=params["beta1"], beta2=params["beta2"],
                mu=params["mu"], lam=params.get("lam", 0), C=params.get("C", 1))
            _, rho = model.simulate(T=500, rho0=rho0, dt=1.0)
            finals.append(rho[-1])

        ax2.plot(rho0_scan, finals, color=COLORS[letter], linewidth=2,
                 label=f"Cond {letter}")

    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Initial seed fraction $\\rho_0$", fontsize=12)
    ax2.set_ylabel("Final adoption $\\rho_\\infty$", fontsize=12)
    ax2.set_title("(b) Model-Predicted Phase Diagram\nFitted Parameters from ABM Data", fontsize=12)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("Theoretical Model Predictions Using ABM-Fitted Parameters",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = str(FIG_DIR / "Fig_ABM_vs_Theory_predictions.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", out)


def main():
    logger.info("Loading data...")
    abm, fit, micro = load_data()

    logger.info("Generating trajectory comparison...")
    plot_trajectory_comparison(abm, fit, micro)

    logger.info("Generating model prediction diagram...")
    plot_model_prediction_diagram(fit)

    logger.info("=== Done ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
