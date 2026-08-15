"""
Study 3: LLM-adapted higher-order contagion model.

Extends the Iacopini et al. (2019, Nat. Commun.) SIS-simplicial model
with three LLM-specific mechanisms:

  A. Attention decay: context window limits cause exponential forgetting
  B. Prompt-dependent susceptibility: framing effects modify infection rate
  C. Hyperedge batch update: synchronous state change under group deliberation

Mean-field ODE:
    dρ/dt = -μρ + (1-ρ) [β₁ρ + β₂ρ² · exp(-λ/C)]

When β₂=0: standard SIS (dyadic only, Condition A prediction)
When C→∞: Iacopini human model (perfect memory)

Usage:
    from src.models.contagion import LLMContagionModel
    model = LLMContagionModel(beta1=0.3, beta2=1.5, mu=0.1, lam=2.0, C=8.0)
    t, rho = model.simulate(T=500, rho0=0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class LLMContagionModel:
    """
    Mean-field SIS contagion on hypergraphs with LLM-specific extensions.

    Parameters:
        beta1: dyadic (pairwise) transmission rate
        beta2: higher-order (hyperedge) transmission rate
        mu:    recovery rate (norm abandonment)
        lam:   forgetting rate (inversely related to context window)
        C:     effective context capacity (normalized, e.g. 4-128)
        g:     framing effect multiplier (1.0 = neutral)
    """
    beta1: float = 0.3
    beta2: float = 1.5
    mu: float = 0.1
    lam: float = 2.0
    C: float = 8.0
    g: float = 1.0

    @property
    def attention_factor(self) -> float:
        """Exponential attention decay due to context window limitation."""
        return np.exp(-self.lam / self.C)

    @property
    def effective_beta2(self) -> float:
        return self.beta2 * self.attention_factor * self.g

    def drho_dt(self, t: float, rho: float) -> float:
        """Mean-field ODE right-hand side."""
        rho = np.clip(rho, 0, 1)
        infection = (1 - rho) * (self.beta1 * rho + self.effective_beta2 * rho ** 2)
        recovery = self.mu * rho
        return infection - recovery

    def simulate(self, T: float = 500, rho0: float = 0.05,
                 dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Solve the mean-field ODE."""
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        sol = solve_ivp(lambda t, y: self.drho_dt(t, y[0]),
                        t_span, [rho0], t_eval=t_eval,
                        method="RK45", max_step=1.0)
        return sol.t, np.clip(sol.y[0], 0, 1)

    def find_equilibria(self, n_points: int = 1000) -> list[float]:
        """Find steady-state solutions where dρ/dt ≈ 0."""
        rhos = np.linspace(0.001, 0.999, n_points)
        drho = np.array([self.drho_dt(0, r) for r in rhos])

        sign_changes = np.where(np.diff(np.sign(drho)))[0]
        equilibria = []
        for idx in sign_changes:
            eq = rhos[idx] + (rhos[idx + 1] - rhos[idx]) * (-drho[idx]) / (drho[idx + 1] - drho[idx])
            equilibria.append(float(eq))

        equilibria = [0.0] + equilibria
        return sorted(set(round(e, 6) for e in equilibria))

    def is_bistable(self) -> bool:
        """Check if system exhibits bistability (3+ equilibria including 0)."""
        eq = self.find_equilibria()
        return len(eq) >= 3

    def critical_mass(self, rho_range: tuple = (0.001, 0.5),
                      n_trials: int = 50) -> Optional[float]:
        """
        Find the critical initial density ρ_c below which the norm dies out
        and above which it explodes. Only meaningful in bistable regime.
        """
        if not self.is_bistable():
            return None

        rho_trials = np.linspace(rho_range[0], rho_range[1], n_trials)
        outcomes = []
        for r0 in rho_trials:
            _, rho = self.simulate(T=1000, rho0=r0, dt=1.0)
            final = rho[-1]
            outcomes.append(final)

        outcomes = np.array(outcomes)
        threshold = 0.5 * (outcomes.max() + outcomes.min())

        for i in range(len(outcomes) - 1):
            if outcomes[i] < threshold and outcomes[i + 1] >= threshold:
                return float(rho_trials[i])

        return None


# ── Special cases ────────────────────────────────────────────────────

def standard_sis(beta: float = 0.3, mu: float = 0.1) -> LLMContagionModel:
    """Condition A: pure dyadic SIS, no higher-order effects."""
    return LLMContagionModel(beta1=beta, beta2=0.0, mu=mu, lam=0, C=1)


def iacopini_human(beta1: float = 0.3, beta2: float = 1.5,
                   mu: float = 0.1) -> LLMContagionModel:
    """Iacopini (2019) model: perfect memory (C→∞), no framing."""
    return LLMContagionModel(beta1=beta1, beta2=beta2, mu=mu, lam=0, C=1e6)


def llm_no_hyperedge(beta1: float = 0.3, mu: float = 0.1,
                     lam: float = 2.0, C: float = 8.0) -> LLMContagionModel:
    """LLM agents with attention decay but no hyperedge structure."""
    return LLMContagionModel(beta1=beta1, beta2=0.0, mu=mu, lam=lam, C=C)


def llm_with_hyperedge(beta1: float = 0.3, beta2: float = 1.5,
                       mu: float = 0.1, lam: float = 2.0,
                       C: float = 8.0) -> LLMContagionModel:
    """Full LLM model with attention decay + hyperedge effects."""
    return LLMContagionModel(beta1=beta1, beta2=beta2, mu=mu, lam=lam, C=C)


# ── Phase diagram computation ────────────────────────────────────────

def compute_phase_diagram(
    beta2_range: tuple = (0, 3.0),
    rho0_range: tuple = (0.01, 0.5),
    n_beta2: int = 50,
    n_rho0: int = 50,
    beta1: float = 0.3,
    mu: float = 0.1,
    lam: float = 2.0,
    C: float = 8.0,
    T: float = 500,
) -> dict:
    """
    Compute phase diagram: final adoption rate as function of (β₂, ρ₀).
    This produces the data for Fig 5 in the paper.
    """
    beta2s = np.linspace(*beta2_range, n_beta2)
    rho0s = np.linspace(*rho0_range, n_rho0)
    final_rho = np.zeros((n_beta2, n_rho0))

    for i, b2 in enumerate(beta2s):
        model = LLMContagionModel(beta1=beta1, beta2=b2, mu=mu, lam=lam, C=C)
        for j, r0 in enumerate(rho0s):
            _, rho = model.simulate(T=T, rho0=r0, dt=1.0)
            final_rho[i, j] = rho[-1]

    return {
        "beta2s": beta2s,
        "rho0s": rho0s,
        "final_rho": final_rho,
    }


# ── Visualization ────────────────────────────────────────────────────

def plot_dynamics_comparison(output_path: Optional[str] = None):
    """
    Generate Fig 6: Norm propagation dynamics under four model conditions.
    Directly maps to the paper's experimental conditions A, B, C, D.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = {
        "A: Dyadic SIS\n(Moltbook baseline)": standard_sis(),
        "B: Dyadic + Attention\n(LLM no hyperedge)": llm_no_hyperedge(),
        "C: Iacopini human\n(perfect memory)": iacopini_human(),
        "D: LLM + Hyperedge\n(full model)": llm_with_hyperedge(),
    }

    colors = ["#E24A33", "#FFA500", "#348ABD", "#2ca02c"]
    rho0_values = [0.05, 0.15, 0.30]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, rho0 in zip(axes, rho0_values):
        for (name, model), color in zip(conditions.items(), colors):
            t, rho = model.simulate(T=300, rho0=rho0)
            ax.plot(t, rho, label=name, color=color, linewidth=1.8)

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$\\rho_0 = {rho0}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Norm adoption $\\rho(t)$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.suptitle("Norm Propagation Dynamics: Model Predictions", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved dynamics comparison to %s", output_path)
    return fig


def plot_phase_diagram(phase_data: dict, output_path: Optional[str] = None):
    """Generate Fig 5: Phase diagram (β₂ vs ρ₀ → final adoption)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        phase_data["rho0s"], phase_data["beta2s"], phase_data["final_rho"],
        cmap="RdYlBu_r", shading="auto", vmin=0, vmax=1
    )
    cb = fig.colorbar(im, ax=ax, label="Final adoption $\\rho_\\infty$")
    ax.set_xlabel("Initial density $\\rho_0$")
    ax.set_ylabel("Higher-order rate $\\beta_2$")
    ax.set_title("Phase Diagram: Bistability and Explosive Transitions")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved phase diagram to %s", output_path)
    return fig


if __name__ == "__main__":
    import sys
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ROOT = Path(__file__).resolve().parents[2]
    outdir = ROOT / "results" / "study3"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Study 3: Contagion Model ===")

    logger.info("\n--- Equilibria analysis ---")
    for name, model in [
        ("Standard SIS", standard_sis()),
        ("Iacopini human", iacopini_human()),
        ("LLM no hyperedge", llm_no_hyperedge()),
        ("LLM + hyperedge", llm_with_hyperedge()),
    ]:
        eq = model.find_equilibria()
        bistable = model.is_bistable()
        cm = model.critical_mass()
        logger.info("  %-25s eq=%s bistable=%s critical_mass=%s",
                    name, [f"{e:.3f}" for e in eq], bistable,
                    f"{cm:.3f}" if cm else "N/A")

    logger.info("\n--- Generating figures ---")
    plot_dynamics_comparison(str(outdir / "fig6_dynamics_comparison.png"))

    logger.info("\n--- Computing phase diagram ---")
    phase = compute_phase_diagram(n_beta2=40, n_rho0=40)
    plot_phase_diagram(phase, str(outdir / "fig5_phase_diagram.png"))

    logger.info("\n=== Study 3 complete ===")
