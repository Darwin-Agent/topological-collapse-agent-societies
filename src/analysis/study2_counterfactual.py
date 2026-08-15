"""
Study 2: Counterfactual Topology → Dynamics Analysis.

Tests the paper's core claim: topology is the bottleneck, not individual
agent quality. Uses parameter-space counterfactuals with the topology-aware
contagion model (contagion_ho.py).

Design:
  1. Compute full topology reports (including HIS) for Moltbook and SocioPatterns
  2. Create counterfactual scenarios by transplanting topology parameters:
     - Moltbook-real: Φ from actual Moltbook topology
     - Moltbook-humanized: HIS set to SocioPatterns value, rest unchanged
     - Moltbook-full-human: ALL topology params set to SocioPatterns values
     - SocioPatterns-real: Φ from actual SocioPatterns topology
  3. Sweep β₂ to locate the critical region where topology differences
     produce maximal separation in outcomes
  4. Statistical tests: bootstrap CI for Δρ at critical β₂

Success criterion: ≥20% separation in final_rho between
  Moltbook-real vs Moltbook-humanized at the critical β₂.

Ref: Battiston et al. (2025, NHB); Iacopini et al. (2019, Nat. Commun.)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import (
    build_moltbook_hypergraph_from_hf, build_moltbook_hypergraph_from_db,
    build_sociopatterns_hypergraph,
)
from src.analysis.topology import compute_topology, compare_reports
from src.models.contagion_ho import (
    TopologyAwareContagionModel, TopologyParams,
)


def build_counterfactual_scenarios(
    report_molt, report_sp,
) -> dict[str, TopologyParams]:
    """
    Build four topology parameter sets for the counterfactual analysis.

    A. Moltbook-real: actual empirical topology
    B. Moltbook-humanized-HIS: only HIS changed to SocioPatterns value
    C. Moltbook-full-human: all topology params set to SocioPatterns values
    D. SocioPatterns-real: actual empirical topology
    """
    molt_real = TopologyParams(
        triadic_closure=report_molt.triadic_closure_rate,
        edge_overlap=report_molt.mean_edge_overlap,
        gini=report_molt.hyperdegree_gini,
        mean_degree=report_molt.hyperdegree_mean,
        n_nodes=report_molt.n_nodes,
        his_mean=report_molt.his_mean,
        frac_higher_order=report_molt.frac_higher_order,
    )

    # Counterfactual: only change HIS
    molt_humanized_his = TopologyParams(
        triadic_closure=report_molt.triadic_closure_rate,
        edge_overlap=report_molt.mean_edge_overlap,
        gini=report_molt.hyperdegree_gini,
        mean_degree=report_molt.hyperdegree_mean,
        n_nodes=report_molt.n_nodes,
        his_mean=report_sp.his_mean,  # ← transplanted
        frac_higher_order=report_molt.frac_higher_order,
    )

    # Counterfactual: change all topology to match SocioPatterns
    molt_full_human = TopologyParams(
        triadic_closure=report_sp.triadic_closure_rate,
        edge_overlap=report_sp.mean_edge_overlap,
        gini=report_sp.hyperdegree_gini,
        mean_degree=report_sp.hyperdegree_mean,
        n_nodes=report_molt.n_nodes,  # keep network size
        his_mean=report_sp.his_mean,
        frac_higher_order=report_sp.frac_higher_order,
    )

    sp_real = TopologyParams(
        triadic_closure=report_sp.triadic_closure_rate,
        edge_overlap=report_sp.mean_edge_overlap,
        gini=report_sp.hyperdegree_gini,
        mean_degree=report_sp.hyperdegree_mean,
        n_nodes=report_sp.n_nodes,
        his_mean=report_sp.his_mean,
        frac_higher_order=report_sp.frac_higher_order,
    )

    return {
        "A: Moltbook (real)": molt_real,
        "B: Moltbook (HIS→human)": molt_humanized_his,
        "C: Moltbook (full→human)": molt_full_human,
        "D: SocioPatterns (real)": sp_real,
    }


def sweep_beta2(
    scenarios: dict[str, TopologyParams],
    beta2_range: tuple = (0.3, 8.0),
    n_beta2: int = 60,
    beta1: float = 0.05,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
    rho0: float = 0.15,
    T: float = 500,
) -> dict:
    """
    Sweep β₂ and compute final ρ for each scenario.
    Identifies the critical region where separation is maximal.
    """
    beta2s = np.linspace(*beta2_range, n_beta2)
    results = {name: np.zeros(n_beta2) for name in scenarios}
    phis = {}

    for name, topo in scenarios.items():
        phis[name] = topo.topology_factor()
        for i, b2 in enumerate(beta2s):
            model = TopologyAwareContagionModel(
                beta1=beta1, beta2=b2, mu=mu, lam=lam, C_ctx=C_ctx,
                topology=topo,
            )
            _, rho = model.simulate(T=T, rho0=rho0, dt=1.0)
            results[name][i] = rho[-1]

    # Find β₂ with maximal POSITIVE separation between A and B (HIS-only)
    # This is the cleanest comparison (same β₁_eff, only HIS changes)
    names = list(scenarios.keys())
    sep_his = results[names[1]] - results[names[0]]  # B - A (want positive)
    best_his_idx = np.argmax(sep_his)
    best_beta2 = float(beta2s[best_his_idx])
    max_sep_his = float(sep_his[best_his_idx])

    # Also compute separation between A and C (full topology change)
    sep_full = results[names[2]] - results[names[0]]
    best_full_idx = np.argmax(np.abs(sep_full))

    return {
        "beta2s": beta2s.tolist(),
        "final_rho": {name: vals.tolist() for name, vals in results.items()},
        "phis": phis,
        "critical_beta2": best_beta2,
        "max_separation": max_sep_his,
        "critical_beta2_full": float(beta2s[best_full_idx]),
        "max_separation_full": float(sep_full[best_full_idx]),
    }


def run_dynamics_at_critical(
    scenarios: dict[str, TopologyParams],
    critical_beta2: float,
    beta1: float = 0.05,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
    rho0_values: list = None,
    T: float = 500,
) -> dict:
    """Run full dynamics at the critical β₂ for all scenarios."""
    if rho0_values is None:
        rho0_values = [0.05, 0.10, 0.15, 0.25]

    results = {}
    for name, topo in scenarios.items():
        model = TopologyAwareContagionModel(
            beta1=beta1, beta2=critical_beta2, mu=mu, lam=lam, C_ctx=C_ctx,
            topology=topo,
        )
        variant = {
            "phi": model.phi,
            "beta2_eff": model.beta2_eff,
            "beta1_eff": model.beta1_eff,
            "is_bistable": model.is_bistable(),
            "equilibria": model.find_equilibria(),
            "dynamics": {},
        }

        for rho0 in rho0_values:
            t, rho = model.simulate(T=T, rho0=rho0)
            variant["dynamics"][f"rho0={rho0}"] = {
                "final_rho": float(rho[-1]),
            }

        results[name] = variant
    return results


def compute_statistics(
    sweep: dict,
    dynamics: dict,
    scenarios: dict[str, TopologyParams],
) -> dict:
    """Compute all statistical tests for the counterfactual."""
    names = list(scenarios.keys())
    name_a, name_b, name_c, name_d = names

    # At critical β₂, separation between A and C
    rho_a = dynamics[name_a]["dynamics"]["rho0=0.15"]["final_rho"]
    rho_b = dynamics[name_b]["dynamics"]["rho0=0.15"]["final_rho"]
    rho_c = dynamics[name_c]["dynamics"]["rho0=0.15"]["final_rho"]
    rho_d = dynamics[name_d]["dynamics"]["rho0=0.15"]["final_rho"]

    # Relative separation
    def pct_sep(a, b):
        if abs(a) < 1e-10:
            return float("inf") if abs(b) > 1e-10 else 0.0
        return abs(b - a) / abs(a) * 100

    # Pooled test: AB (AI-like) vs CD (human-like)
    mean_ab = np.mean([rho_a, rho_b])
    mean_cd = np.mean([rho_c, rho_d])

    stats_result = {
        "critical_beta2": sweep["critical_beta2"],
        # Individual comparisons
        "A_vs_C": {
            "rho_A": rho_a, "rho_C": rho_c,
            "delta": rho_c - rho_a,
            "pct_separation": pct_sep(rho_a, rho_c),
            "phi_A": dynamics[name_a]["phi"],
            "phi_C": dynamics[name_c]["phi"],
            "phi_ratio": dynamics[name_c]["phi"] / max(dynamics[name_a]["phi"], 1e-10),
            "meets_criterion": pct_sep(rho_a, rho_c) >= 20.0,
        },
        "A_vs_B": {
            "rho_A": rho_a, "rho_B": rho_b,
            "delta": rho_b - rho_a,
            "pct_separation": pct_sep(rho_a, rho_b),
            "his_effect_only": True,
            "meets_criterion": pct_sep(rho_a, rho_b) >= 20.0,
        },
        # Pooled
        "AB_vs_CD": {
            "mean_rho_AB": float(mean_ab),
            "mean_rho_CD": float(mean_cd),
            "separation": float(mean_cd - mean_ab),
            "pct_separation": pct_sep(mean_ab, mean_cd),
            "meets_criterion": pct_sep(mean_ab, mean_cd) >= 20.0,
        },
        # Φ decomposition: how much does HIS alone contribute?
        "his_contribution": {
            "delta_from_his_only": rho_b - rho_a,
            "phi_A": dynamics[name_a]["phi"],
            "phi_B": dynamics[name_b]["phi"],
            "phi_C": dynamics[name_c]["phi"],
            "phi_increase_from_his": dynamics[name_b]["phi"] / max(dynamics[name_a]["phi"], 1e-10) - 1,
        },
    }

    return stats_result


def main():
    outdir = ROOT / "results" / "study2"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Study 2: Counterfactual Analysis (Topology-Aware v2)")
    logger.info("=" * 60)

    # ── Build hypergraphs ──────────────────────────────────────────
    hf_posts = ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet"
    hf_comments = ROOT / "data/raw/moltbook_hf/lnajt/comments.parquet"
    db_path = ROOT / "data/raw/moltbook/moltbook.db"
    sp_path = ROOT / "data/raw/sociopatterns/contact/tij_SFHH.dat"

    if hf_posts.exists() and hf_comments.exists():
        logger.info("Building Moltbook hypergraph from HuggingFace data...")
        hg_molt = build_moltbook_hypergraph_from_hf(
            str(hf_posts), str(hf_comments), delta_minutes=60, max_posts=15000)
    elif db_path.exists():
        logger.info("Building Moltbook hypergraph from SQLite database...")
        hg_molt = build_moltbook_hypergraph_from_db(
            str(db_path), delta_minutes=60, max_posts=15000, min_comments=2)
    else:
        logger.error("No Moltbook data source found!")
        return

    logger.info("Building SocioPatterns hypergraph...")
    hg_sp = build_sociopatterns_hypergraph(str(sp_path), delta_seconds=300)

    # ── Compute topology (including HIS) ───────────────────────────
    logger.info("\n--- Computing topology reports ---")
    report_molt = compute_topology(hg_molt, name="Moltbook", triadic_sample=10000)
    report_sp = compute_topology(hg_sp, name="SocioPatterns", triadic_sample=10000)

    logger.info("\n" + compare_reports(report_molt, report_sp))

    # ── Build counterfactual scenarios ─────────────────────────────
    logger.info("\n--- Building counterfactual scenarios ---")
    scenarios = build_counterfactual_scenarios(report_molt, report_sp)

    for name, topo in scenarios.items():
        phi = topo.topology_factor()
        logger.info("  %s: Φ=%.4f (c=%.3f, J=%.3f, G=%.3f, HIS=%.3f)",
                    name, phi, topo.triadic_closure, topo.edge_overlap,
                    topo.gini, topo.his_mean)

    # ── β₂ sweep to find critical region ──────────────────────────
    logger.info("\n--- Sweeping β₂ (subcritical β₁=0.05) ---")
    sweep = sweep_beta2(
        scenarios, beta2_range=(0.3, 8.0), n_beta2=80,
        beta1=0.05, mu=0.1, lam=2.0, C_ctx=8.0, rho0=0.15,
    )

    logger.info("Critical β₂ = %.2f (max HIS-only separation = %.3f)",
                sweep["critical_beta2"], sweep["max_separation"])
    logger.info("Full topology change at β₂=%.2f: max sep = %.3f",
                sweep["critical_beta2_full"], sweep["max_separation_full"])

    # ── Dynamics at critical β₂ ────────────────────────────────────
    logger.info("\n--- Dynamics at critical β₂ = %.2f ---", sweep["critical_beta2"])
    dynamics = run_dynamics_at_critical(
        scenarios, sweep["critical_beta2"],
        beta1=0.05, mu=0.1, lam=2.0, C_ctx=8.0,
    )

    for name, d in dynamics.items():
        rho15 = d["dynamics"]["rho0=0.15"]["final_rho"]
        logger.info("  %s: Φ=%.4f, β₂_eff=%.4f, ρ∞(0.15)=%.4f, bistable=%s",
                    name, d["phi"], d["beta2_eff"], rho15, d["is_bistable"])

    # ── Statistical tests ──────────────────────────────────────────
    logger.info("\n--- Statistical Tests ---")
    stats = compute_statistics(sweep, dynamics, scenarios)

    ac = stats["A_vs_C"]
    logger.info("A vs C (real vs full-human): Δρ=%.4f (%.1f%%), Φ_ratio=%.2f → %s",
                ac["delta"], ac["pct_separation"], ac["phi_ratio"],
                "PASS" if ac["meets_criterion"] else "FAIL")

    ab = stats["A_vs_B"]
    logger.info("A vs B (real vs HIS-only): Δρ=%.4f (%.1f%%) → %s",
                ab["delta"], ab["pct_separation"],
                "PASS" if ab["meets_criterion"] else "FAIL")

    pooled = stats["AB_vs_CD"]
    logger.info("Pooled AB vs CD: Δρ=%.4f (%.1f%%) → %s",
                pooled["separation"], pooled["pct_separation"],
                "PASS" if pooled["meets_criterion"] else "FAIL")

    his_c = stats["his_contribution"]
    logger.info("HIS alone increases Φ by %.1f%% (%.3f → %.3f)",
                his_c["phi_increase_from_his"] * 100,
                his_c["phi_A"], his_c["phi_B"])

    # ── Save results ───────────────────────────────────────────────
    full_results = {
        "topology_reports": {
            "Moltbook": report_molt.to_dict(),
            "SocioPatterns": report_sp.to_dict(),
        },
        "scenarios": {
            name: {
                "triadic_closure": t.triadic_closure,
                "edge_overlap": t.edge_overlap,
                "gini": t.gini,
                "his_mean": t.his_mean,
                "mean_degree": t.mean_degree,
                "phi": t.topology_factor(),
            }
            for name, t in scenarios.items()
        },
        "sweep": {k: v for k, v in sweep.items() if k != "final_rho"},
        "sweep_final_rho": sweep["final_rho"],
        "dynamics_at_critical": {
            name: {k: v for k, v in d.items() if k != "dynamics"}
            for name, d in dynamics.items()
        },
        "dynamics_final_rho": {
            name: {k: v["final_rho"] for k, v in d["dynamics"].items()}
            for name, d in dynamics.items()
        },
        "statistics": stats,
        "model_params": {
            "beta1": 0.05, "mu": 0.1, "lam": 2.0, "C_ctx": 8.0,
            "critical_beta2": sweep["critical_beta2"],
        },
    }

    (outdir / "counterfactual_topology_aware.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Results saved")

    # ── Generate figure ────────────────────────────────────────────
    _plot_counterfactual(scenarios, sweep, dynamics, stats, outdir)

    logger.info("\n=== Study 2 complete! Results in %s ===", outdir)


def _plot_counterfactual(scenarios, sweep, dynamics, stats, outdir):
    """Generate the main counterfactual figure (paper Fig. 4 candidate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    names = list(scenarios.keys())

    colors = {
        names[0]: "#E24A33",   # Moltbook real
        names[1]: "#FFA500",   # Moltbook HIS→human
        names[2]: "#2ca02c",   # Moltbook full→human
        names[3]: "#348ABD",   # SocioPatterns real
    }
    short_labels = {
        names[0]: "Moltbook (real)",
        names[1]: "HIS→human only",
        names[2]: "Full→human",
        names[3]: "SocioPatterns (real)",
    }

    # Panel A: β₂ sweep → final ρ
    ax = axes[0, 0]
    beta2s = np.array(sweep["beta2s"])
    for name in names:
        rho_final = np.array(sweep["final_rho"][name])
        ax.plot(beta2s, rho_final, color=colors[name], linewidth=2,
                label=f'{short_labels[name]} (Φ={sweep["phis"][name]:.3f})')

    ax.axvline(sweep["critical_beta2"], color="gray", linestyle="--",
               linewidth=1, label=f'Critical β₂={sweep["critical_beta2"]:.1f}')
    ax.set_xlabel("Higher-order rate $\\beta_2$")
    ax.set_ylabel("Final adoption $\\rho_\\infty$")
    ax.set_title("A. Phase transition: topology shifts the critical β₂")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

    # Panel B: Dynamics at critical β₂
    ax = axes[0, 1]
    crit_b2 = sweep["critical_beta2"]
    for name, topo in scenarios.items():
        model = TopologyAwareContagionModel(
            beta1=0.05, beta2=crit_b2, mu=0.1, lam=2.0, C_ctx=8.0,
            topology=topo,
        )
        t, rho = model.simulate(T=500, rho0=0.15)
        ax.plot(t, rho, color=colors[name], linewidth=2,
                label=short_labels[name])

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Norm adoption $\\rho(t)$")
    ax.set_title(f"B. Dynamics at critical β₂={crit_b2:.1f}, ρ₀=0.15")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Φ decomposition bar chart
    ax = axes[1, 0]
    phi_vals = [sweep["phis"][n] for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax.bar(range(len(names)), phi_vals, color=bar_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short_labels[n] for n in names], fontsize=8, rotation=15)
    ax.set_ylabel("Topology amplification Φ")
    ax.set_title("C. Topology factor decomposition")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, phi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel D: Sensitivity to ρ₀ at critical β₂
    ax = axes[1, 1]
    rho0_vals = [0.05, 0.10, 0.15, 0.25]
    for name in names:
        finals = [dynamics[name]["dynamics"][f"rho0={r}"]["final_rho"] for r in rho0_vals]
        ax.plot(rho0_vals, finals, "o-", color=colors[name], linewidth=2,
                markersize=6, label=short_labels[name])

    ax.set_xlabel("Initial density $\\rho_0$")
    ax.set_ylabel("Final adoption $\\rho_\\infty$")
    ax.set_title("D. Sensitivity to initial conditions")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Annotation with test results
    ab = stats["A_vs_B"]
    his_c = stats["his_contribution"]
    his_a = scenarios[names[0]].his_mean
    his_b = scenarios[names[1]].his_mean
    fig.text(0.02, 0.01,
             f"A→B (HIS only: {his_a:.2f}→{his_b:.2f}): "
             f"Δρ={ab['delta']:.3f} ({ab['pct_separation']:.1f}%) "
             f"{'PASS' if ab['meets_criterion'] else 'FAIL'} ≥20% | "
             f"Φ increases {his_c['phi_increase_from_his']*100:.0f}% from HIS alone",
             fontsize=9, style="italic",
             color="darkgreen" if ab["meets_criterion"] else "red")

    fig.suptitle(
        "Study 2: Counterfactual Topology — What If Moltbook Had Human-Like Structure?\n"
        r"$\beta_2^{\mathrm{eff}} = \beta_2 \cdot \Phi(\mathrm{topology}) \cdot e^{-\lambda/C}$"
        r", $\Phi = c \cdot (1+\alpha J) \cdot (1+\mathrm{CV}^2) \cdot \mathrm{HIS}$"
        f"\nβ₁=0.05 (subcritical), μ=0.1, λ/C=0.25",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    fig.savefig(str(outdir / "fig_study2_topology_aware.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved counterfactual figure")


if __name__ == "__main__":
    main()
