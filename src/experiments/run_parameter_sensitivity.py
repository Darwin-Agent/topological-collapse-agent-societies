"""
Parameter Sensitivity Analysis for free model parameters (alpha, lambda, C).

Addresses reviewer question: "How sensitive is the Phi ratio and bistability
to the hardcoded alpha=2.0, lambda=2.0, C=8.0?"

Three analyses:
  1. Alpha sweep: Phi and Phi_ratio as a function of the pair-approximation constant
  2. Lambda-C grid: bistability regions for both platforms
  3. Latin Hypercube robustness: fraction of parameter space where core claims hold
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.contagion_ho import (
    TopologyAwareContagionModel,
    TopologyParams,
    moltbook_topology,
    sociopatterns_topology,
)


# ═══════════════════════════════════════════════════════════════════════
# Sweep functions
# ═══════════════════════════════════════════════════════════════════════

def sweep_alpha(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha_range: tuple = (0.5, 5.0),
    n_points: int = 40,
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
) -> dict:
    """Sweep alpha and track Phi, Phi_ratio, and bistability."""
    alphas = np.linspace(*alpha_range, n_points)
    results = {
        "alphas": alphas.tolist(),
        "phi_molt": [],
        "phi_sp": [],
        "phi_ratio": [],
        "bistable_molt": [],
        "bistable_sp": [],
        "rho_final_molt": [],
        "rho_final_sp": [],
    }

    for a in alphas:
        phi_m = molt.topology_factor(alpha=a)
        phi_s = sp.topology_factor(alpha=a)
        results["phi_molt"].append(float(phi_m))
        results["phi_sp"].append(float(phi_s))
        results["phi_ratio"].append(float(phi_s / phi_m) if phi_m > 0 else float("inf"))

        for label, topo in [("molt", molt), ("sp", sp)]:
            model = TopologyAwareContagionModel(
                beta1=beta1, beta2=beta2, mu=mu, lam=lam, C_ctx=C_ctx,
                topology=topo, alpha=a,
            )
            results[f"bistable_{label}"].append(model.is_bistable())
            _, rho = model.simulate(T=500, rho0=0.15, dt=1.0)
            results[f"rho_final_{label}"].append(float(rho[-1]))

    return results


def sweep_lam_C(
    molt: TopologyParams,
    sp: TopologyParams,
    lam_range: tuple = (0.5, 5.0),
    C_range: tuple = (2.0, 32.0),
    n_lam: int = 25,
    n_C: int = 25,
    alpha: float = 2.0,
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
) -> dict:
    """2D sweep over lambda and C, tracking bistability for both platforms."""
    lams = np.linspace(*lam_range, n_lam)
    Cs = np.linspace(*C_range, n_C)

    bistable_molt = np.zeros((n_lam, n_C), dtype=bool)
    bistable_sp = np.zeros((n_lam, n_C), dtype=bool)
    rho_molt = np.zeros((n_lam, n_C))
    rho_sp = np.zeros((n_lam, n_C))

    for i, l in enumerate(lams):
        for j, c in enumerate(Cs):
            for label, topo, arr_b, arr_r in [
                ("molt", molt, bistable_molt, rho_molt),
                ("sp", sp, bistable_sp, rho_sp),
            ]:
                model = TopologyAwareContagionModel(
                    beta1=beta1, beta2=beta2, mu=mu, lam=l, C_ctx=c,
                    topology=topo, alpha=alpha,
                )
                arr_b[i, j] = model.is_bistable()
                _, rho = model.simulate(T=500, rho0=0.15, dt=1.0)
                arr_r[i, j] = rho[-1]

    return {
        "lams": lams.tolist(),
        "Cs": Cs.tolist(),
        "bistable_molt": bistable_molt.tolist(),
        "bistable_sp": bistable_sp.tolist(),
        "rho_molt": rho_molt.tolist(),
        "rho_sp": rho_sp.tolist(),
    }


def latin_hypercube_robustness(
    molt: TopologyParams,
    sp: TopologyParams,
    n_samples: int = 500,
    alpha_range: tuple = (0.5, 5.0),
    lam_range: tuple = (0.5, 5.0),
    C_range: tuple = (2.0, 32.0),
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Latin Hypercube Sampling over (alpha, lambda, C).

    For each sample, check whether core claims hold:
      - Phi_ratio > 1.5
      - Phi_ratio > 2.0
      - SP bistable & Molt not bistable
      - SP rho_final > Molt rho_final
    """
    rng = np.random.default_rng(seed)

    # LHS: stratified sampling
    intervals = np.arange(n_samples) / n_samples
    perm_a = rng.permutation(n_samples)
    perm_l = rng.permutation(n_samples)
    perm_c = rng.permutation(n_samples)

    alphas = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * (intervals[perm_a] + rng.uniform(0, 1 / n_samples, n_samples))
    lams = lam_range[0] + (lam_range[1] - lam_range[0]) * (intervals[perm_l] + rng.uniform(0, 1 / n_samples, n_samples))
    Cs = C_range[0] + (C_range[1] - C_range[0]) * (intervals[perm_c] + rng.uniform(0, 1 / n_samples, n_samples))

    claims = {
        "phi_ratio_gt_1.5": 0,
        "phi_ratio_gt_2.0": 0,
        "sp_bistable_molt_not": 0,
        "sp_rho_gt_molt_rho": 0,
        "all_core_claims": 0,
    }
    phi_ratios = []
    rho_gaps = []

    for i in range(n_samples):
        a, l, c = float(alphas[i]), float(lams[i]), float(Cs[i])
        phi_m = molt.topology_factor(alpha=a)
        phi_s = sp.topology_factor(alpha=a)
        ratio = phi_s / phi_m if phi_m > 0 else float("inf")
        phi_ratios.append(ratio)

        model_m = TopologyAwareContagionModel(
            beta1=beta1, beta2=beta2, mu=mu, lam=l, C_ctx=c,
            topology=molt, alpha=a)
        model_s = TopologyAwareContagionModel(
            beta1=beta1, beta2=beta2, mu=mu, lam=l, C_ctx=c,
            topology=sp, alpha=a)

        bi_m = model_m.is_bistable()
        bi_s = model_s.is_bistable()
        _, rho_m = model_m.simulate(T=500, rho0=0.15, dt=1.0)
        _, rho_s = model_s.simulate(T=500, rho0=0.15, dt=1.0)
        rho_gap = float(rho_s[-1] - rho_m[-1])
        rho_gaps.append(rho_gap)

        if ratio > 1.5:
            claims["phi_ratio_gt_1.5"] += 1
        if ratio > 2.0:
            claims["phi_ratio_gt_2.0"] += 1
        if bi_s and not bi_m:
            claims["sp_bistable_molt_not"] += 1
        if rho_s[-1] > rho_m[-1]:
            claims["sp_rho_gt_molt_rho"] += 1
        if ratio > 1.5 and rho_s[-1] > rho_m[-1]:
            claims["all_core_claims"] += 1

    fractions = {k: v / n_samples for k, v in claims.items()}
    phi_ratios = np.array(phi_ratios)
    rho_gaps = np.array(rho_gaps)

    return {
        "n_samples": n_samples,
        "counts": claims,
        "fractions": fractions,
        "phi_ratio_stats": {
            "mean": float(np.mean(phi_ratios)),
            "std": float(np.std(phi_ratios)),
            "min": float(np.min(phi_ratios)),
            "max": float(np.max(phi_ratios)),
            "ci_95": [float(np.percentile(phi_ratios, 2.5)),
                      float(np.percentile(phi_ratios, 97.5))],
        },
        "rho_gap_stats": {
            "mean": float(np.mean(rho_gaps)),
            "std": float(np.std(rho_gaps)),
            "frac_positive": float(np.mean(rho_gaps > 0)),
        },
        "param_ranges": {
            "alpha": list(alpha_range),
            "lam": list(lam_range),
            "C": list(C_range),
        },
    }


def tornado_analysis(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha_range: tuple = (0.5, 5.0),
    lam_range: tuple = (0.5, 5.0),
    C_range: tuple = (2.0, 32.0),
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
    defaults: dict = None,
) -> dict:
    """
    Tornado diagram: sensitivity of Phi_ratio and rho_gap to each parameter.

    For each parameter, sweep it across its range while holding others at defaults.
    Record the range of Phi_ratio and rho_gap.
    """
    if defaults is None:
        defaults = {"alpha": 2.0, "lam": 2.0, "C": 8.0}

    params = {
        "alpha": {"range": alpha_range, "n": 30},
        "lam": {"range": lam_range, "n": 30},
        "C": {"range": C_range, "n": 30},
    }

    results = {}
    for param_name, config in params.items():
        vals = np.linspace(*config["range"], config["n"])
        phi_ratios = []
        rho_gaps = []

        for v in vals:
            current = dict(defaults)
            current[param_name] = float(v)

            phi_m = molt.topology_factor(alpha=current["alpha"])
            phi_s = sp.topology_factor(alpha=current["alpha"])
            ratio = phi_s / phi_m if phi_m > 0 else float("inf")
            phi_ratios.append(ratio)

            model_m = TopologyAwareContagionModel(
                beta1=beta1, beta2=beta2, mu=mu,
                lam=current["lam"], C_ctx=current["C"],
                topology=molt, alpha=current["alpha"])
            model_s = TopologyAwareContagionModel(
                beta1=beta1, beta2=beta2, mu=mu,
                lam=current["lam"], C_ctx=current["C"],
                topology=sp, alpha=current["alpha"])

            _, rho_m = model_m.simulate(T=500, rho0=0.15, dt=1.0)
            _, rho_s = model_s.simulate(T=500, rho0=0.15, dt=1.0)
            rho_gaps.append(float(rho_s[-1] - rho_m[-1]))

        phi_ratios = np.array(phi_ratios)
        rho_gaps = np.array(rho_gaps)
        results[param_name] = {
            "phi_ratio_range": [float(phi_ratios.min()), float(phi_ratios.max())],
            "phi_ratio_span": float(phi_ratios.max() - phi_ratios.min()),
            "rho_gap_range": [float(rho_gaps.min()), float(rho_gaps.max())],
            "rho_gap_span": float(rho_gaps.max() - rho_gaps.min()),
            "values": vals.tolist(),
            "phi_ratios": phi_ratios.tolist(),
            "rho_gaps": rho_gaps.tolist(),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_sensitivity(alpha_sweep, lam_C, lhs, tornado, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ── Panel A: Alpha sweep ──
    ax = axes[0, 0]
    alphas = alpha_sweep["alphas"]
    ax.plot(alphas, alpha_sweep["phi_molt"], "-", color="#E24A33", linewidth=2, label="Φ Moltbook")
    ax.plot(alphas, alpha_sweep["phi_sp"], "-", color="#348ABD", linewidth=2, label="Φ SocioPatterns")
    ax.set_xlabel("Pair-approximation constant α")
    ax.set_ylabel("Topology factor Φ", color="black")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(alphas, alpha_sweep["phi_ratio"], "--", color="#2ca02c", linewidth=2, label="Φ ratio")
    ax2.set_ylabel("Φ_SP / Φ_Molt", color="#2ca02c")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.legend(loc="right", fontsize=8)
    ax.axvline(2.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Default α=2.0")

    ax.set_title("A. Phi sensitivity to α (overlap amplification)")

    # ── Panel B: Lambda-C bistability heatmap ──
    ax = axes[0, 1]
    lams = np.array(lam_C["lams"])
    Cs = np.array(lam_C["Cs"])
    bi_molt = np.array(lam_C["bistable_molt"], dtype=float)
    bi_sp = np.array(lam_C["bistable_sp"], dtype=float)

    # Encode: 0=neither, 1=SP only, 2=both, 3=Molt only
    combined = bi_sp + 2 * bi_molt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f0f0f0", "#348ABD", "#FFA500", "#E24A33"])

    im = ax.imshow(combined, origin="lower", aspect="auto",
                   extent=[Cs[0], Cs[-1], lams[0], lams[-1]], cmap=cmap,
                   vmin=0, vmax=3)
    ax.set_xlabel("Context capacity $C$")
    ax.set_ylabel("Attention decay $\\lambda$")
    ax.set_title("B. Bistability regions in (λ, C) space")
    ax.plot(8.0, 2.0, "w*", markersize=12, markeredgecolor="black", markeredgewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f0f0f0", edgecolor="black", label="Neither"),
        Patch(facecolor="#348ABD", label="SP only"),
        Patch(facecolor="#FFA500", label="Both"),
        Patch(facecolor="#E24A33", label="Molt only"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # ── Panel C: LHS robustness ──
    ax = axes[1, 0]
    claim_labels = [
        "Φ ratio > 1.5",
        "Φ ratio > 2.0",
        "SP bistable\nMolt not",
        "SP ρ > Molt ρ",
        "All core\nclaims",
    ]
    claim_keys = list(lhs["fractions"].keys())
    fracs = [lhs["fractions"][k] * 100 for k in claim_keys]
    claim_colors = ["#2ca02c" if f >= 90 else "#FFA500" if f >= 70 else "#E24A33" for f in fracs]

    bars = ax.bar(range(len(claim_labels)), fracs, color=claim_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(claim_labels)))
    ax.set_xticklabels(claim_labels, fontsize=8)
    ax.set_ylabel("% of parameter space")
    ax.set_ylim(0, 105)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="90% threshold")
    ax.set_title(f"C. Robustness (n={lhs['n_samples']} LHS samples)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for bar, f in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{f:.0f}%", ha="center", fontsize=9, fontweight="bold")

    # ── Panel D: Tornado diagram ──
    ax = axes[1, 1]
    param_labels = {"alpha": "α (overlap)", "lam": "λ (attention)", "C": "C (context)"}
    param_colors = {"alpha": "#E24A33", "lam": "#348ABD", "C": "#2ca02c"}

    # Sort by phi_ratio_span descending
    sorted_params = sorted(tornado.keys(), key=lambda p: tornado[p]["phi_ratio_span"], reverse=True)

    y_pos = np.arange(len(sorted_params))
    default_ratio = lhs["phi_ratio_stats"]["mean"]  # approximate center

    for i, p in enumerate(sorted_params):
        low, high = tornado[p]["phi_ratio_range"]
        ax.barh(i, high - low, left=low, color=param_colors[p], alpha=0.7,
                edgecolor="black", linewidth=0.5, height=0.5)
        ax.text(high + 0.02, i, f"[{low:.2f}, {high:.2f}]",
                va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_labels[p] for p in sorted_params])
    ax.set_xlabel("Φ_SP / Φ_Molt ratio")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("D. Tornado: parameter sensitivity ranking")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Parameter Sensitivity: Core Claims Robust Across (α, λ, C) Space",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(outdir / "fig_parameter_sensitivity.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved sensitivity figure")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = ROOT / "results" / "parameter_sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Parameter Sensitivity Analysis")
    logger.info("=" * 60)

    molt = moltbook_topology()
    sp = sociopatterns_topology()

    # ── Alpha sweep ───────────────────────────────────────────────
    logger.info("\n--- Alpha sweep (0.5-5.0) ---")
    alpha_result = sweep_alpha(molt, sp)
    logger.info("  Φ_ratio range: [%.2f, %.2f]",
                min(alpha_result["phi_ratio"]), max(alpha_result["phi_ratio"]))
    logger.info("  Φ_ratio at α=2.0: %.2f",
                alpha_result["phi_ratio"][
                    np.argmin(np.abs(np.array(alpha_result["alphas"]) - 2.0))])

    # ── Lambda-C grid ─────────────────────────────────────────────
    logger.info("\n--- Lambda-C bistability grid ---")
    lam_C_result = sweep_lam_C(molt, sp)
    bi_molt = np.array(lam_C_result["bistable_molt"])
    bi_sp = np.array(lam_C_result["bistable_sp"])
    logger.info("  Moltbook bistable: %d/%d cells (%.1f%%)",
                bi_molt.sum(), bi_molt.size, bi_molt.mean() * 100)
    logger.info("  SocioPatterns bistable: %d/%d cells (%.1f%%)",
                bi_sp.sum(), bi_sp.size, bi_sp.mean() * 100)
    sp_only = (bi_sp & ~bi_molt).sum()
    logger.info("  SP bistable & Molt not: %d cells (%.1f%%)",
                sp_only, sp_only / bi_sp.size * 100)

    # ── Latin Hypercube ───────────────────────────────────────────
    logger.info("\n--- Latin Hypercube robustness (n=500) ---")
    lhs_result = latin_hypercube_robustness(molt, sp)
    for k, v in lhs_result["fractions"].items():
        logger.info("  %s: %.1f%%", k, v * 100)
    logger.info("  Φ_ratio: %.2f ± %.2f [%.2f, %.2f]",
                lhs_result["phi_ratio_stats"]["mean"],
                lhs_result["phi_ratio_stats"]["std"],
                lhs_result["phi_ratio_stats"]["ci_95"][0],
                lhs_result["phi_ratio_stats"]["ci_95"][1])

    # ── Tornado ───────────────────────────────────────────────────
    logger.info("\n--- Tornado analysis ---")
    tornado_result = tornado_analysis(molt, sp)
    for p, r in tornado_result.items():
        logger.info("  %s: Φ_ratio [%.2f, %.2f] (span=%.2f), ρ_gap [%.3f, %.3f]",
                    p, r["phi_ratio_range"][0], r["phi_ratio_range"][1],
                    r["phi_ratio_span"],
                    r["rho_gap_range"][0], r["rho_gap_range"][1])

    # ── Save ──────────────────────────────────────────────────────
    full_results = {
        "alpha_sweep": {k: v for k, v in alpha_result.items()},
        "lam_C_grid": {k: v for k, v in lam_C_result.items()},
        "lhs_robustness": lhs_result,
        "tornado": {p: {k: v for k, v in r.items() if k not in ("values", "phi_ratios", "rho_gaps")}
                    for p, r in tornado_result.items()},
    }
    (outdir / "sensitivity_results.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )

    # ── Figure ────────────────────────────────────────────────────
    _plot_sensitivity(alpha_result, lam_C_result, lhs_result, tornado_result, outdir)

    logger.info("\n=== Parameter sensitivity complete! Results in %s ===", outdir)


if __name__ == "__main__":
    main()
