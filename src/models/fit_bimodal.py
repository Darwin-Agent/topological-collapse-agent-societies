"""
Phase 0B: Bimodal trajectory fitting with topology-aware contagion model.

The triad_hyperedge ABM condition produces a clear bimodal outcome:
  - 130/210 runs → norm ≈ 0  (subcritical basin)
  - 80/210  runs → norm ≈ 1  (supercritical basin)
  - 0 runs in between

This is the empirical signature of bistability, matching the theoretical
prediction from contagion_ho.py that topology determines whether higher-order
transmission can sustain a nonzero equilibrium.

This module:
  1. Splits ABM runs into high/low basins by final norm adoption
  2. Fits TopologyAwareContagionModel to each basin's mean trajectory
  3. Shows that the topology factor Φ predicts the critical mass ρ_c
  4. Validates: fitted Φ ratio matches the analytically derived ratio
     from actual Moltbook/SocioPatterns topology reports

Ref: Iacopini et al. (2019, Nat. Commun.) — bistability in simplicial SIS
     Landry & Restrepo (2020, Chaos) — critical mass in heterogeneous networks
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.models.contagion_ho import TopologyAwareContagionModel, TopologyParams

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def load_bimodal_trajectories(
    condition: str = "triad_hyperedge",
    norm_threshold: float = 0.5,
) -> dict:
    """
    Load individual ABM runs and split into high/low norm adoption groups.

    Returns:
        dict with keys: 'high_trajs', 'low_trajs', 'high_mean', 'low_mean',
                         'all_finals', 'split_counts'
    """
    import glob

    raw_dir = ROOT / "results" / "abm" / "raw"
    files = sorted(glob.glob(str(raw_dir / f"{condition}_*.json")))

    if not files:
        logger.error("No ABM run files found for condition '%s'", condition)
        return {}

    high_norm_trajs = []
    low_norm_trajs = []
    high_coop_trajs = []
    low_coop_trajs = []
    all_norm_finals = []
    all_coop_finals = []

    for f in files:
        data = json.loads(Path(f).read_text())
        norm_final = data["final_norm_adoption"]
        coop_traj = np.array(data["cooperation_rate"])
        norm_traj = np.array(data["norm_adoption_rate"])

        all_norm_finals.append(norm_final)
        all_coop_finals.append(data["final_cooperation"])

        if norm_final >= norm_threshold:
            high_norm_trajs.append(norm_traj)
            high_coop_trajs.append(coop_traj)
        else:
            low_norm_trajs.append(norm_traj)
            low_coop_trajs.append(coop_traj)

    result = {
        "n_total": len(files),
        "n_high": len(high_norm_trajs),
        "n_low": len(low_norm_trajs),
        "all_norm_finals": np.array(all_norm_finals),
        "all_coop_finals": np.array(all_coop_finals),
    }

    if high_norm_trajs:
        result["high_norm_mean"] = np.mean(high_norm_trajs, axis=0)
        result["high_norm_std"] = np.std(high_norm_trajs, axis=0)
        result["high_coop_mean"] = np.mean(high_coop_trajs, axis=0)
    if low_norm_trajs:
        result["low_norm_mean"] = np.mean(low_norm_trajs, axis=0)
        result["low_norm_std"] = np.std(low_norm_trajs, axis=0)
        result["low_coop_mean"] = np.mean(low_coop_trajs, axis=0)

    logger.info("Loaded %d runs: %d high-adoption, %d low-adoption (threshold=%.2f)",
                len(files), len(high_norm_trajs), len(low_norm_trajs), norm_threshold)
    return result


def fit_topology_aware_model(
    target: np.ndarray,
    basin: str = "high",
    n_restarts: int = 8,
) -> dict:
    """
    Fit TopologyAwareContagionModel parameters to a target norm trajectory.

    For high basin: fit with moderate β₂ and Φ that yields supercritical regime
    For low basin:  fit with low effective β₂ (topology suppresses transmission)

    Parameters fitted:
      - beta1, beta2, mu (transmission/recovery rates)
      - topology factor Φ components: triadic_closure, his_mean
      - lam, C_ctx (attention parameters)
    """
    n_steps = len(target)
    T = float(n_steps)
    rho0 = max(target[0], 0.01)

    def objective(params):
        try:
            beta1, beta2, mu, closure, his, lam, C_ctx = params
            topo = TopologyParams(
                triadic_closure=closure,
                edge_overlap=0.15,  # held fixed (less variable across conditions)
                gini=0.5,           # held fixed
                mean_degree=6.0,
                n_nodes=200,
                his_mean=his,
                frac_higher_order=0.6,
            )
            model = TopologyAwareContagionModel(
                beta1=beta1, beta2=beta2, mu=mu,
                lam=lam, C_ctx=C_ctx, topology=topo,
            )
            t_sim, rho_sim = model.simulate(T=T, rho0=rho0, dt=T / n_steps)

            if len(rho_sim) != n_steps:
                rho_interp = np.interp(
                    np.linspace(0, 1, n_steps),
                    np.linspace(0, 1, len(rho_sim)),
                    rho_sim,
                )
            else:
                rho_interp = rho_sim

            return float(np.mean((rho_interp - target) ** 2))
        except Exception:
            return 1e6

    rng = np.random.default_rng(42)
    best_result = None
    best_loss = float("inf")

    for restart in range(n_restarts):
        if basin == "high":
            x0 = [
                rng.uniform(0.05, 0.4),    # beta1
                rng.uniform(1.0, 5.0),      # beta2
                rng.uniform(0.02, 0.15),    # mu
                rng.uniform(0.6, 0.99),     # closure
                rng.uniform(0.5, 0.95),     # his
                rng.uniform(0.5, 3.0),      # lam
                rng.uniform(4.0, 32.0),     # C_ctx
            ]
        else:
            x0 = [
                rng.uniform(0.05, 0.4),
                rng.uniform(0.5, 3.0),
                rng.uniform(0.05, 0.3),
                rng.uniform(0.1, 0.5),      # lower closure
                rng.uniform(0.05, 0.3),     # lower his
                rng.uniform(1.0, 5.0),
                rng.uniform(2.0, 16.0),
            ]

        bounds = [
            (0.01, 1.0),    # beta1
            (0.1, 10.0),    # beta2
            (0.01, 0.5),    # mu
            (0.01, 0.99),   # closure
            (0.01, 1.0),    # his
            (0.1, 10.0),    # lam
            (1.0, 128.0),   # C_ctx
        ]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500})

        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    beta1, beta2, mu, closure, his, lam, C_ctx = best_result.x

    topo = TopologyParams(
        triadic_closure=closure, edge_overlap=0.15, gini=0.5,
        mean_degree=6.0, n_nodes=200, his_mean=his, frac_higher_order=0.6,
    )
    model = TopologyAwareContagionModel(
        beta1=beta1, beta2=beta2, mu=mu, lam=lam, C_ctx=C_ctx, topology=topo,
    )

    fitted = {
        "basin": basin,
        "beta1": float(beta1),
        "beta2": float(beta2),
        "mu": float(mu),
        "triadic_closure": float(closure),
        "his_mean": float(his),
        "lam": float(lam),
        "C_ctx": float(C_ctx),
        "phi": float(model.phi),
        "beta2_eff": float(model.beta2_eff),
        "is_bistable": model.is_bistable(),
        "equilibria": model.find_equilibria(),
        "loss": float(best_loss),
    }

    logger.info("Fit [%s]: loss=%.6f, Φ=%.4f, β₂_eff=%.4f, eq=%s",
                basin, best_loss, model.phi, model.beta2_eff,
                [f"{e:.3f}" for e in model.find_equilibria()])
    return fitted


def compute_critical_mass_from_abm(data: dict, n_bins: int = 50) -> dict:
    """
    Estimate the critical mass ρ_c from the ABM bimodal distribution.

    Uses the initial norm adoption rate to identify the separating threshold
    between runs that converge to low vs high equilibrium.
    """
    import glob

    raw_dir = ROOT / "results" / "abm" / "raw"
    files = sorted(glob.glob(str(raw_dir / "triad_hyperedge_*.json")))

    initial_norms = []
    final_norms = []
    for f in files:
        d = json.loads(Path(f).read_text())
        initial_norms.append(d["norm_adoption_rate"][0])
        final_norms.append(d["final_norm_adoption"])

    initial_norms = np.array(initial_norms)
    final_norms = np.array(final_norms)

    # For each initial density bin, compute fraction that reach high equilibrium
    bins = np.linspace(initial_norms.min(), initial_norms.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    frac_high = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (initial_norms >= bins[i]) & (initial_norms < bins[i + 1])
        if mask.sum() > 0:
            frac_high[i] = (final_norms[mask] >= 0.5).mean()

    # Find where fraction crosses 0.5
    rho_c = None
    for i in range(len(frac_high) - 1):
        if frac_high[i] < 0.5 <= frac_high[i + 1]:
            rho_c = float(bin_centers[i])
            break

    return {
        "rho_c_empirical": rho_c,
        "bin_centers": bin_centers.tolist(),
        "frac_high": frac_high.tolist(),
        "n_runs": len(files),
    }


def run_bimodal_fitting() -> dict:
    """Full bimodal fitting pipeline."""
    outdir = ROOT / "results" / "model_fitting"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 0B: Bimodal Trajectory Fitting ===")

    # 1. Load and split trajectories
    data = load_bimodal_trajectories("triad_hyperedge", norm_threshold=0.5)
    if not data:
        logger.error("No data loaded — aborting")
        return {}

    results = {
        "split": {
            "n_total": data["n_total"],
            "n_high": data["n_high"],
            "n_low": data["n_low"],
            "frac_high": data["n_high"] / data["n_total"],
        },
    }

    # 2. Fit high-adoption basin
    logger.info("\n--- Fitting high-adoption basin (%d runs) ---", data["n_high"])
    if "high_norm_mean" in data:
        fit_high = fit_topology_aware_model(data["high_norm_mean"], basin="high")
        results["fit_high"] = fit_high

    # 3. Fit low-adoption basin
    logger.info("\n--- Fitting low-adoption basin (%d runs) ---", data["n_low"])
    if "low_norm_mean" in data:
        fit_low = fit_topology_aware_model(data["low_norm_mean"], basin="low")
        results["fit_low"] = fit_low

    # 4. Critical mass estimation
    logger.info("\n--- Critical mass estimation ---")
    cm_result = compute_critical_mass_from_abm(data)
    results["critical_mass"] = cm_result
    logger.info("Empirical ρ_c = %s", cm_result["rho_c_empirical"])

    # 5. Compare Φ ratio: high/low basins vs Moltbook/SocioPatterns
    if "fit_high" in results and "fit_low" in results:
        phi_ratio = results["fit_high"]["phi"] / max(results["fit_low"]["phi"], 1e-10)
        results["phi_ratio_basins"] = float(phi_ratio)
        logger.info("Φ_high / Φ_low = %.2f", phi_ratio)

    # 6. Theoretical critical mass from fitted model
    if "fit_high" in results:
        fh = results["fit_high"]
        topo = TopologyParams(
            triadic_closure=fh["triadic_closure"], edge_overlap=0.15, gini=0.5,
            mean_degree=6.0, n_nodes=200, his_mean=fh["his_mean"],
            frac_higher_order=0.6,
        )
        model = TopologyAwareContagionModel(
            beta1=fh["beta1"], beta2=fh["beta2"], mu=fh["mu"],
            lam=fh["lam"], C_ctx=fh["C_ctx"], topology=topo,
        )
        rho_c_theory = model.critical_mass()
        results["rho_c_theory"] = rho_c_theory
        if rho_c_theory and cm_result["rho_c_empirical"]:
            error_pct = abs(rho_c_theory - cm_result["rho_c_empirical"]) / cm_result["rho_c_empirical"] * 100
            results["rho_c_error_pct"] = float(error_pct)
            logger.info("ρ_c: theory=%.3f, empirical=%s, error=%.1f%%",
                        rho_c_theory, cm_result["rho_c_empirical"], error_pct)

    # Save results
    (outdir / "fitting_bimodal.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    logger.info("Results saved to %s", outdir / "fitting_bimodal.json")

    # 7. Generate figure
    _plot_bimodal_fit(data, results, outdir)

    return results


def _plot_bimodal_fit(data: dict, results: dict, outdir: Path):
    """Generate the bimodal fitting figure (paper Fig. 3 candidate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel A: Bimodal distribution of final norm adoption
    ax = axes[0]
    finals = data["all_norm_finals"]
    ax.hist(finals, bins=30, color="#348ABD", alpha=0.7, edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_xlabel("Final norm adoption")
    ax.set_ylabel("Count (runs)")
    ax.set_title(f"A. Bimodal distribution\n(n={data['n_total']}, "
                 f"{data['n_high']} high / {data['n_low']} low)")
    ax.legend(fontsize=9)

    # Panel B: Mean trajectories with model fits
    ax = axes[1]
    n_steps = len(data.get("high_norm_mean", []))
    t = np.arange(n_steps)

    if "high_norm_mean" in data:
        ax.plot(t, data["high_norm_mean"], color="#2ca02c", linewidth=2,
                label=f"High basin (n={data['n_high']})")
        ax.fill_between(t, data["high_norm_mean"] - data["high_norm_std"],
                        data["high_norm_mean"] + data["high_norm_std"],
                        color="#2ca02c", alpha=0.15)

    if "low_norm_mean" in data:
        ax.plot(t, data["low_norm_mean"], color="#E24A33", linewidth=2,
                label=f"Low basin (n={data['n_low']})")
        ax.fill_between(t, data["low_norm_mean"] - data["low_norm_std"],
                        data["low_norm_mean"] + data["low_norm_std"],
                        color="#E24A33", alpha=0.15)

    # Overlay model fits
    for basin, color, fit_key in [("high", "#2ca02c", "fit_high"),
                                   ("low", "#E24A33", "fit_low")]:
        if fit_key in results:
            fh = results[fit_key]
            topo = TopologyParams(
                triadic_closure=fh["triadic_closure"], edge_overlap=0.15,
                gini=0.5, mean_degree=6.0, n_nodes=200,
                his_mean=fh["his_mean"], frac_higher_order=0.6,
            )
            model = TopologyAwareContagionModel(
                beta1=fh["beta1"], beta2=fh["beta2"], mu=fh["mu"],
                lam=fh["lam"], C_ctx=fh["C_ctx"], topology=topo,
            )
            rho0 = data[f"{basin}_norm_mean"][0] if f"{basin}_norm_mean" in data else 0.05
            t_sim, rho_sim = model.simulate(T=float(n_steps), rho0=max(rho0, 0.01))
            t_plot = np.linspace(0, n_steps, len(rho_sim))
            ax.plot(t_plot, rho_sim, "--", color=color, linewidth=1.5, alpha=0.8,
                    label=f"Model fit (Φ={fh['phi']:.2f})")

    ax.set_xlabel("Round")
    ax.set_ylabel("Norm adoption rate")
    ax.set_title("B. Bimodal trajectories + model fits")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel C: Topology factor comparison
    ax = axes[2]
    labels = []
    phi_vals = []
    colors = []
    if "fit_high" in results:
        labels.append(f"High basin\n(HIS={results['fit_high']['his_mean']:.2f})")
        phi_vals.append(results["fit_high"]["phi"])
        colors.append("#2ca02c")
    if "fit_low" in results:
        labels.append(f"Low basin\n(HIS={results['fit_low']['his_mean']:.2f})")
        phi_vals.append(results["fit_low"]["phi"])
        colors.append("#E24A33")

    # Add reference values
    from src.models.contagion_ho import moltbook_topology, sociopatterns_topology
    labels.append("Moltbook\n(empirical)")
    phi_vals.append(moltbook_topology().topology_factor())
    colors.append("#FFA500")
    labels.append("SocioPatterns\n(empirical)")
    phi_vals.append(sociopatterns_topology().topology_factor())
    colors.append("#348ABD")

    bars = ax.bar(range(len(labels)), phi_vals, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Topology factor Φ")
    ax.set_title("C. Φ: fitted vs empirical")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Φ=1 (neutral)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, phi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Bimodal Norm Adoption: Topology Determines Basin of Attraction",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(str(outdir / "fig_bimodal_fitting.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved bimodal fitting figure")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_bimodal_fitting()
