"""
Study 3 extension: Find parameter regime exhibiting bistability
and explosive phase transitions.

The key is: β₁ must be low enough that dyadic transmission alone
is subcritical (ρ→0), but β₂ high enough that higher-order effects
push the system past a tipping point.

This maps to the paper's central prediction:
  "Without hyperedge constraints, norms die. With them, they explode."
"""

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.contagion import LLMContagionModel


def scan_bistability(
    beta1_range=(0.01, 0.15),
    beta2_range=(0.5, 5.0),
    mu: float = 0.1,
    lam: float = 2.0,
    C: float = 8.0,
    n_beta1: int = 30,
    n_beta2: int = 30,
):
    """Scan β₁-β₂ space for bistability."""
    beta1s = np.linspace(*beta1_range, n_beta1)
    beta2s = np.linspace(*beta2_range, n_beta2)

    bistable_map = np.zeros((n_beta1, n_beta2))
    n_eq_map = np.zeros((n_beta1, n_beta2))

    for i, b1 in enumerate(beta1s):
        for j, b2 in enumerate(beta2s):
            model = LLMContagionModel(beta1=b1, beta2=b2, mu=mu, lam=lam, C=C)
            eq = model.find_equilibria()
            n_eq_map[i, j] = len(eq)
            bistable_map[i, j] = 1 if model.is_bistable() else 0

    n_bistable = int(bistable_map.sum())
    logger.info("Bistability scan: %d/%d parameter combos are bistable",
                n_bistable, n_beta1 * n_beta2)

    return {
        "beta1s": beta1s,
        "beta2s": beta2s,
        "bistable_map": bistable_map,
        "n_eq_map": n_eq_map,
    }


def find_best_bistable_params(scan_result: dict, mu=0.1, lam=2.0, C=8.0):
    """Find the 'most dramatic' bistable parameter set."""
    beta1s = scan_result["beta1s"]
    beta2s = scan_result["beta2s"]
    bmap = scan_result["bistable_map"]

    best = None
    best_gap = 0

    for i in range(len(beta1s)):
        for j in range(len(beta2s)):
            if bmap[i, j] == 0:
                continue
            model = LLMContagionModel(beta1=beta1s[i], beta2=beta2s[j],
                                       mu=mu, lam=lam, C=C)
            eq = model.find_equilibria()
            if len(eq) >= 3:
                gap = eq[-1] - eq[1]
                if gap > best_gap:
                    best_gap = gap
                    best = (beta1s[i], beta2s[j], eq)

    return best


def demo_explosive_transition(beta1, beta2, mu=0.1, lam=2.0, C=8.0,
                               output_path=None):
    """Generate the key figure showing explosive vs. gradual transition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho0_values = np.linspace(0.01, 0.4, 30)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Dynamics at different ρ₀
    ax = axes[0]
    model_ho = LLMContagionModel(beta1=beta1, beta2=beta2, mu=mu, lam=lam, C=C)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rho0_values)))
    for r0, color in zip(rho0_values, cmap):
        t, rho = model_ho.simulate(T=300, rho0=r0)
        ax.plot(t, rho, color=color, alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Norm adoption $\\rho(t)$")
    ax.set_title(f"LLM + Hyperedge ($\\beta_2={beta2:.2f}$)\nColor = $\\rho_0$")
    ax.set_ylim(-0.02, 1.02)

    # Panel 2: Same but without hyperedge (β₂=0)
    ax = axes[1]
    model_no = LLMContagionModel(beta1=beta1, beta2=0, mu=mu, lam=lam, C=C)
    for r0, color in zip(rho0_values, cmap):
        t, rho = model_no.simulate(T=300, rho0=r0)
        ax.plot(t, rho, color=color, alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Time $t$")
    ax.set_title(f"LLM Dyadic Only ($\\beta_2=0$)\nColor = $\\rho_0$")
    ax.set_ylim(-0.02, 1.02)

    # Panel 3: Bifurcation diagram (final ρ vs ρ₀)
    ax = axes[2]
    rho0_fine = np.linspace(0.001, 0.5, 100)
    final_ho, final_no = [], []
    for r0 in rho0_fine:
        _, rho = model_ho.simulate(T=500, rho0=r0, dt=1.0)
        final_ho.append(rho[-1])
        _, rho = model_no.simulate(T=500, rho0=r0, dt=1.0)
        final_no.append(rho[-1])

    ax.plot(rho0_fine, final_ho, "o-", color="#2ca02c", markersize=3,
            label=f"With hyperedge ($\\beta_2={beta2:.2f}$)")
    ax.plot(rho0_fine, final_no, "s-", color="#E24A33", markersize=3,
            label="Dyadic only ($\\beta_2=0$)")
    ax.set_xlabel("Initial density $\\rho_0$")
    ax.set_ylabel("Final adoption $\\rho_\\infty$")
    ax.set_title("Bistability: Critical Mass Threshold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.02)

    fig.suptitle("Explosive Cooperation Phase Transition", fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved explosive transition figure to %s", output_path)
    return fig


def main():
    outdir = ROOT / "results" / "study3"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Bistability Parameter Search ===")

    # Scan with attention decay
    scan = scan_bistability(
        beta1_range=(0.01, 0.12),
        beta2_range=(0.5, 6.0),
        mu=0.1, lam=2.0, C=8.0,
        n_beta1=40, n_beta2=40,
    )

    best = find_best_bistable_params(scan)
    if best:
        b1, b2, eq = best
        logger.info("Best bistable params: β₁=%.4f, β₂=%.4f, eq=%s",
                    b1, b2, [f"{e:.3f}" for e in eq])
        demo_explosive_transition(b1, b2, output_path=str(outdir / "fig_explosive_transition.png"))
    else:
        logger.info("No bistability found in default range, trying wider...")
        scan2 = scan_bistability(
            beta1_range=(0.001, 0.10),
            beta2_range=(1.0, 10.0),
            mu=0.1, lam=3.0, C=4.0,
            n_beta1=50, n_beta2=50,
        )
        best2 = find_best_bistable_params(scan2, mu=0.1, lam=3.0, C=4.0)
        if best2:
            b1, b2, eq = best2
            logger.info("Found: β₁=%.4f, β₂=%.4f, eq=%s", b1, b2, [f"{e:.3f}" for e in eq])
            demo_explosive_transition(b1, b2, mu=0.1, lam=3.0, C=4.0,
                                       output_path=str(outdir / "fig_explosive_transition.png"))
        else:
            logger.warning("No bistability found. Try adjusting μ or λ/C ratio.")

    # Plot bistability map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(scan["beta2s"], scan["beta1s"], scan["bistable_map"],
                  cmap="RdYlGn", shading="auto")
    ax.set_xlabel("Higher-order rate $\\beta_2$")
    ax.set_ylabel("Dyadic rate $\\beta_1$")
    ax.set_title("Bistability Region (green = bistable)")
    fig.savefig(str(outdir / "fig_bistability_map.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved bistability map")

    logger.info("=== Bistability search complete ===")


if __name__ == "__main__":
    main()
