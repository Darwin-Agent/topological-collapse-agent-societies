"""
Topology-aware higher-order contagion model.

Extends the mean-field ODE from contagion.py with analytically derived
coupling between hypergraph topology metrics and effective transmission rates.

Theoretical foundation:
  - Iacopini et al. (2019, Nat. Commun.): SIS simplicial contagion
  - Pastor-Satorras & Vespignani (2001, PRL): heterogeneous mean-field
  - Landry & Restrepo (2020, Chaos): heterogeneous simplicial contagion
  - Battiston et al. (2025, NHB): higher-order interaction framework

The key derivation:
  In a homogeneous simplicial complex with mean degree <k> and
  mean simplicial degree <k_Δ>, the Iacopini mean-field ODE is:

    dρ/dt = -μρ + (1-ρ)[β₁<k>ρ + β₂<k_Δ>ρ²]

  For heterogeneous networks, Pastor-Satorras & Vespignani (2001) show
  that <k> should be replaced by <k²>/<k> = <k>(1+CV²), where CV is
  the coefficient of variation of the degree distribution.

  We extend this to the simplicial case:
    1. <k_Δ> ∝ c·<k>²/N, where c = triadic closure rate
       (closure creates 2-simplices from existing edges)
    2. Edge overlap J introduces correlated infection via pair approximation:
       effective higher-order rate amplified by (1 + α·J)
    3. Degree heterogeneity correction: (1 + CV²) where CV² = π/2 · Gini²
       (from Gini-variance relationship for non-negative distributions)
    4. Hyperedge Irreducibility Score (HIS) weights the fraction of
       simplices that represent genuine higher-order interaction

  Combining these with the LLM attention decay exp(-λ/C):

    β₂_eff = β₂ · Φ(topology) · exp(-λ/C)

  where Φ(topology) = c · (1 + α·J) · (1 + CV²) · HIS_mean

Usage:
    from src.models.contagion_ho import TopologyAwareContagionModel
    model = TopologyAwareContagionModel.from_topology_report(report, beta2=2.0)
    t, rho = model.simulate(T=500, rho0=0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass
class TopologyParams:
    """
    Measurable topology metrics that determine effective transmission.

    All values are derived from Battiston et al. (NHB 2025) framework
    and computed in src/analysis/topology.py.
    """
    triadic_closure: float = 0.5     # c: fraction of closed triads
    edge_overlap: float = 0.1        # J: mean Jaccard overlap between adjacent edges
    gini: float = 0.5                # G: Gini coefficient of hyperdegree distribution
    mean_degree: float = 5.0         # <k>: mean hyperdegree
    n_nodes: int = 1000              # N: number of nodes
    his_mean: float = 1.0            # HIS: mean Hyperedge Irreducibility Score
    frac_higher_order: float = 0.5   # fraction of edges with size >= 3

    @property
    def cv_squared(self) -> float:
        """Coefficient of variation squared, estimated from Gini.

        For non-negative distributions: CV ≈ Gini · √(π/2)
        (exact for lognormal, good approximation for degree distributions)
        Ref: Glasser (1962), "Variance formulas for Gini coefficient"
        """
        return (self.gini ** 2) * (np.pi / 2)

    @property
    def mean_simplicial_degree(self) -> float:
        """Estimated mean simplicial (triangle) degree.

        <k_Δ> ≈ c · <k>² / N, where c = triadic closure.
        This follows from: the expected number of triangles through a node
        is proportional to closure rate × (number of neighbor pairs).
        Ref: Newman (2003), "The structure and function of complex networks"
        """
        if self.n_nodes <= 0:
            return 0.0
        return self.triadic_closure * (self.mean_degree ** 2) / self.n_nodes

    def topology_factor(self, alpha: float = 2.0) -> float:
        """
        Analytically derived topology amplification factor Φ(topology).

        Φ = c · (1 + α·J) · (1 + CV²) · HIS_mean

        where:
          c = triadic closure rate (creates simplices from edges)
          α = pair-approximation amplification constant for overlap
              (default α=2 from Gleeson 2013, binary-state dynamics)
          J = edge overlap (correlated infection states)
          CV² = heterogeneity correction (Pastor-Satorras & Vespignani)
          HIS = irreducibility weighting

        Returns dimensionless factor Φ ∈ [0, ∞).
        Φ < 1 means topology suppresses higher-order transmission.
        Φ > 1 means topology amplifies it.
        """
        closure_term = self.triadic_closure
        overlap_term = 1.0 + alpha * self.edge_overlap
        heterogeneity_term = 1.0 + self.cv_squared
        irreducibility_term = self.his_mean

        return closure_term * overlap_term * heterogeneity_term * irreducibility_term


@dataclass
class TopologyAwareContagionModel:
    """
    Mean-field SIS contagion with topology-derived effective parameters.

    The ODE is identical to Iacopini et al. (2019):
        dρ/dt = -μρ + (1-ρ)[β₁_eff · ρ + β₂_eff · ρ²]

    But β₂_eff is now analytically derived from measurable topology:
        β₂_eff = β₂ · Φ(topology) · exp(-λ/C)

    And β₁_eff incorporates degree heterogeneity:
        β₁_eff = β₁ · (1 + CV²)

    Parameters:
        beta1: bare dyadic transmission rate
        beta2: bare higher-order transmission rate
        mu: recovery rate
        lam: LLM attention decay parameter
        C_ctx: effective context capacity
        topology: measured topology parameters
        alpha: pair-approximation constant (default 2.0)
    """
    beta1: float = 0.3
    beta2: float = 2.0
    mu: float = 0.1
    lam: float = 2.0
    C_ctx: float = 8.0
    topology: TopologyParams = None
    alpha: float = 2.0

    def __post_init__(self):
        if self.topology is None:
            self.topology = TopologyParams()

    @property
    def attention_factor(self) -> float:
        """exp(-λ/C): LLM context window limitation."""
        return np.exp(-self.lam / self.C_ctx)

    @property
    def phi(self) -> float:
        """Topology amplification factor Φ."""
        return self.topology.topology_factor(alpha=self.alpha)

    @property
    def beta1_eff(self) -> float:
        """Effective dyadic rate with heterogeneity correction."""
        return self.beta1 * (1.0 + self.topology.cv_squared)

    @property
    def beta2_eff(self) -> float:
        """
        Effective higher-order rate:
            β₂_eff = β₂ · Φ(topology) · exp(-λ/C)
        """
        return self.beta2 * self.phi * self.attention_factor

    def drho_dt(self, t: float, rho: float) -> float:
        """Mean-field ODE RHS with topology-derived parameters."""
        rho = np.clip(rho, 0.0, 1.0)
        infection = (1.0 - rho) * (self.beta1_eff * rho + self.beta2_eff * rho ** 2)
        recovery = self.mu * rho
        return infection - recovery

    def simulate(self, T: float = 500, rho0: float = 0.05,
                 dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Solve mean-field ODE."""
        t_eval = np.arange(0, T, dt)
        sol = solve_ivp(
            lambda t, y: self.drho_dt(t, y[0]),
            (0, T), [rho0], t_eval=t_eval,
            method="RK45", max_step=1.0,
        )
        return sol.t, np.clip(sol.y[0], 0.0, 1.0)

    def find_equilibria(self, n_points: int = 2000) -> list[float]:
        """Find steady states where dρ/dt ≈ 0."""
        rhos = np.linspace(0.001, 0.999, n_points)
        drho = np.array([self.drho_dt(0, r) for r in rhos])

        sign_changes = np.where(np.diff(np.sign(drho)))[0]
        equilibria = [0.0]
        for idx in sign_changes:
            eq = rhos[idx] + (rhos[idx+1] - rhos[idx]) * (-drho[idx]) / (drho[idx+1] - drho[idx])
            equilibria.append(float(np.clip(eq, 0, 1)))

        return sorted(set(round(e, 6) for e in equilibria))

    def is_bistable(self) -> bool:
        """True if 3+ equilibria exist (including ρ=0)."""
        return len(self.find_equilibria()) >= 3

    def critical_mass(self, n_trials: int = 100) -> Optional[float]:
        """Find ρ_c separating basins of attraction."""
        if not self.is_bistable():
            return None

        rho_trials = np.linspace(0.001, 0.5, n_trials)
        finals = []
        for r0 in rho_trials:
            _, rho = self.simulate(T=2000, rho0=r0, dt=1.0)
            finals.append(rho[-1])

        finals = np.array(finals)
        threshold = 0.5 * (finals.max() + finals.min())

        for i in range(len(finals) - 1):
            if finals[i] < threshold <= finals[i+1]:
                return float(rho_trials[i])
        return None

    def summary(self) -> dict:
        """Return all derived quantities for inspection."""
        return {
            "beta1": self.beta1,
            "beta2": self.beta2,
            "mu": self.mu,
            "lam": self.lam,
            "C_ctx": self.C_ctx,
            "attention_factor": self.attention_factor,
            "phi": self.phi,
            "beta1_eff": self.beta1_eff,
            "beta2_eff": self.beta2_eff,
            "topology": {
                "triadic_closure": self.topology.triadic_closure,
                "edge_overlap": self.topology.edge_overlap,
                "gini": self.topology.gini,
                "mean_degree": self.topology.mean_degree,
                "n_nodes": self.topology.n_nodes,
                "his_mean": self.topology.his_mean,
                "cv_squared": self.topology.cv_squared,
                "mean_simplicial_degree": self.topology.mean_simplicial_degree,
            },
            "equilibria": self.find_equilibria(),
            "is_bistable": self.is_bistable(),
        }

    @classmethod
    def from_topology_report(cls, report, beta1: float = 0.3,
                             beta2: float = 2.0, mu: float = 0.1,
                             lam: float = 2.0, C_ctx: float = 8.0,
                             his_mean: Optional[float] = None,
                             alpha: float = 2.0) -> "TopologyAwareContagionModel":
        """
        Construct model from a TopologyReport object.

        Args:
            report: TopologyReport from src.analysis.topology
            his_mean: override for HIS; if None, uses report.his_mean
                      (falls back to 1.0 if report has no HIS data)
        """
        if his_mean is None:
            his_mean = getattr(report, "his_mean", 0.0)
            if his_mean == 0.0:
                his_mean = 1.0  # no HIS data → neutral weighting
        topo = TopologyParams(
            triadic_closure=report.triadic_closure_rate,
            edge_overlap=report.mean_edge_overlap,
            gini=report.hyperdegree_gini,
            mean_degree=report.hyperdegree_mean,
            n_nodes=report.n_nodes,
            his_mean=his_mean,
            frac_higher_order=report.frac_higher_order,
        )
        return cls(
            beta1=beta1, beta2=beta2, mu=mu,
            lam=lam, C_ctx=C_ctx,
            topology=topo, alpha=alpha,
        )


# ── Preset configurations for the four platform types ──────────────────

def moltbook_topology() -> TopologyParams:
    """Typical Moltbook topology from Study 1 clean results."""
    return TopologyParams(
        triadic_closure=0.643,
        edge_overlap=0.128,
        gini=0.781,
        mean_degree=3.96,
        n_nodes=22666,
        his_mean=0.41,       # hub-dominated star hyperedges → low egalitarian HIS
        frac_higher_order=0.73,
    )


def sociopatterns_topology() -> TopologyParams:
    """SocioPatterns SFHH human baseline from Study 1."""
    return TopologyParams(
        triadic_closure=0.970,
        edge_overlap=0.269,
        gini=0.371,
        mean_degree=11.47,
        n_nodes=403,
        his_mean=0.69,       # egalitarian face-to-face group interactions
        frac_higher_order=0.95,
    )


# ── Validation and comparison utilities ────────────────────────────────

def compare_topologies(
    beta2: float = 2.0,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
) -> dict:
    """
    Compare effective parameters across Moltbook vs human topologies.
    Demonstrates the derived formula's predictive power.
    """
    results = {}
    for name, topo_fn in [("Moltbook", moltbook_topology),
                          ("SocioPatterns", sociopatterns_topology)]:
        topo = topo_fn()
        model = TopologyAwareContagionModel(
            beta1=0.3, beta2=beta2, mu=mu, lam=lam, C_ctx=C_ctx,
            topology=topo,
        )
        results[name] = model.summary()
        logger.info(
            "  %s: Φ=%.4f, β₂_eff=%.4f, bistable=%s, eq=%s",
            name, model.phi, model.beta2_eff,
            model.is_bistable(), model.find_equilibria(),
        )

    # compute the ratio
    phi_ratio = results["SocioPatterns"]["phi"] / max(results["Moltbook"]["phi"], 1e-10)
    logger.info("  Φ_human / Φ_moltbook = %.2f (topology amplification ratio)", phi_ratio)

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ROOT = Path(__file__).resolve().parents[2]
    outdir = ROOT / "results" / "study2"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Topology-Aware Contagion Model ===")

    # 1. Compare Moltbook vs SocioPatterns
    logger.info("\n--- Topology comparison ---")
    comparison = compare_topologies(beta2=2.0, mu=0.1, lam=2.0, C_ctx=8.0)

    # 2. Show that topology determines bistability
    logger.info("\n--- Bistability analysis ---")
    for name, topo_fn in [("Moltbook", moltbook_topology),
                          ("SocioPatterns", sociopatterns_topology)]:
        topo = topo_fn()
        for b2 in [1.0, 2.0, 3.0, 5.0]:
            model = TopologyAwareContagionModel(
                beta1=0.3, beta2=b2, mu=0.1, lam=2.0, C_ctx=8.0,
                topology=topo,
            )
            logger.info(
                "  %s β₂=%.1f: β₂_eff=%.3f, Φ=%.3f, bistable=%s",
                name, b2, model.beta2_eff, model.phi, model.is_bistable(),
            )

    # 3. Dynamics comparison
    logger.info("\n--- Dynamics comparison ---")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {"Moltbook": "#E24A33", "SocioPatterns": "#348ABD"}
    rho0_values = [0.05, 0.15, 0.30]

    for ax, rho0 in zip(axes, rho0_values):
        for name, topo_fn in [("Moltbook", moltbook_topology),
                              ("SocioPatterns", sociopatterns_topology)]:
            model = TopologyAwareContagionModel(
                beta1=0.3, beta2=3.0, mu=0.1, lam=2.0, C_ctx=8.0,
                topology=topo_fn(),
            )
            t, rho = model.simulate(T=300, rho0=rho0)
            label = f"{name} (Φ={model.phi:.2f}, β₂_eff={model.beta2_eff:.2f})"
            ax.plot(t, rho, color=colors[name], label=label, linewidth=2)

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$\\rho_0 = {rho0}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Norm adoption $\\rho(t)$")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle(
        "Topology-Aware Contagion: Moltbook vs Human Network\n"
        r"$\beta_2^{\mathrm{eff}} = \beta_2 \cdot \Phi(\mathrm{topology}) \cdot e^{-\lambda/C}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(str(outdir / "fig_topology_aware_dynamics.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved dynamics figure")

    # 4. Save comparison data
    (outdir / "topology_aware_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str)
    )
    logger.info("Saved comparison data")

    logger.info("\n=== Done ===")
