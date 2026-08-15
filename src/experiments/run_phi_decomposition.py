"""
Phi Component Decomposition: Shapley values, leave-one-out, and log decomposition.

Addresses reviewer question: "Which topological feature primarily drives
the AI-human Phi gap?"

Phi = c · (1 + alpha·J) · (1 + CV²) · HIS has 4 multiplicative factors.
This script decomposes the Phi_SP / Phi_Molt ratio into per-factor contributions
using three complementary methods:
  1. Exact Shapley values (2^4=16 coalitions)
  2. Leave-one-out (single-factor swaps)
  3. Multiplicative log decomposition

Also includes bootstrap CIs and dynamics comparison for LOO scenarios.
"""

import json
import logging
import sys
from itertools import combinations
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
# Core decomposition functions
# ═══════════════════════════════════════════════════════════════════════

FACTOR_NAMES = ["closure", "overlap", "heterogeneity", "HIS"]


def extract_components(params: TopologyParams, alpha: float = 2.0) -> dict:
    """Extract the 4 multiplicative Phi components from TopologyParams."""
    cv2 = (params.gini ** 2) * (np.pi / 2)
    return {
        "closure": params.triadic_closure,
        "overlap": 1.0 + alpha * params.edge_overlap,
        "heterogeneity": 1.0 + cv2,
        "HIS": params.his_mean,
    }


def phi_from_dict(components: dict) -> float:
    """Compute Phi from a components dict."""
    return components["closure"] * components["overlap"] * components["heterogeneity"] * components["HIS"]


def shapley_decomposition(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha: float = 2.0,
) -> dict:
    """
    Exact Shapley values for each factor's contribution to the Phi gap.

    For each factor i, the Shapley value is the weighted average marginal
    contribution across all 2^(n-1) coalitions that include i.

    Coalition rule: factors IN the coalition use SocioPatterns values,
    factors NOT in the coalition use Moltbook values.
    """
    c_molt = extract_components(molt, alpha)
    c_sp = extract_components(sp, alpha)
    n = len(FACTOR_NAMES)

    phi_molt = phi_from_dict(c_molt)
    phi_sp = phi_from_dict(c_sp)
    phi_gap = phi_sp - phi_molt

    def coalition_phi(coalition_set):
        """Phi with coalition factors set to SP values, rest to Molt."""
        components = {}
        for f in FACTOR_NAMES:
            components[f] = c_sp[f] if f in coalition_set else c_molt[f]
        return phi_from_dict(components)

    # Exact Shapley: for each factor i, average marginal contribution
    # across all coalitions S ⊆ N\{i}
    from math import factorial

    shapley = {}
    for i, factor in enumerate(FACTOR_NAMES):
        others = [f for f in FACTOR_NAMES if f != factor]
        sv = 0.0
        for size in range(len(others) + 1):
            for combo in combinations(others, size):
                S = set(combo)
                S_with_i = S | {factor}
                marginal = coalition_phi(S_with_i) - coalition_phi(S)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                sv += weight * marginal
        shapley[factor] = sv

    return {
        "shapley_values": shapley,
        "phi_molt": phi_molt,
        "phi_sp": phi_sp,
        "phi_gap": phi_gap,
        "sum_shapley": sum(shapley.values()),
        "pct_contributions": {f: v / phi_gap * 100 for f, v in shapley.items()},
        "components_molt": c_molt,
        "components_sp": c_sp,
    }


def leave_one_out(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha: float = 2.0,
) -> dict:
    """
    Leave-one-out: swap each factor individually to SP value.

    For each factor, compute Phi with that one factor at SP value
    and all others at Moltbook values.
    """
    c_molt = extract_components(molt, alpha)
    c_sp = extract_components(sp, alpha)
    phi_molt = phi_from_dict(c_molt)
    phi_sp = phi_from_dict(c_sp)

    results = {}
    for factor in FACTOR_NAMES:
        swapped = dict(c_molt)
        swapped[factor] = c_sp[factor]
        phi_swapped = phi_from_dict(swapped)
        delta = phi_swapped - phi_molt
        pct_of_gap = delta / (phi_sp - phi_molt) * 100 if (phi_sp - phi_molt) != 0 else 0
        results[factor] = {
            "phi_swapped": phi_swapped,
            "delta_phi": delta,
            "pct_increase": delta / phi_molt * 100 if phi_molt != 0 else 0,
            "pct_of_total_gap": pct_of_gap,
            "component_molt": c_molt[factor],
            "component_sp": c_sp[factor],
            "component_ratio": c_sp[factor] / c_molt[factor] if c_molt[factor] != 0 else float("inf"),
        }

    return {"phi_molt": phi_molt, "phi_sp": phi_sp, "factors": results}


def log_decomposition(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha: float = 2.0,
) -> dict:
    """
    Multiplicative log decomposition.

    log(Phi_SP / Phi_Molt) = sum_i log(component_i_SP / component_i_Molt)

    This gives an exact additive partition of the log-ratio.
    """
    c_molt = extract_components(molt, alpha)
    c_sp = extract_components(sp, alpha)
    phi_molt = phi_from_dict(c_molt)
    phi_sp = phi_from_dict(c_sp)

    log_total = np.log(phi_sp / phi_molt) if phi_molt > 0 else float("inf")

    log_contributions = {}
    for factor in FACTOR_NAMES:
        if c_molt[factor] > 0:
            log_ratio = np.log(c_sp[factor] / c_molt[factor])
        else:
            log_ratio = float("inf")
        log_contributions[factor] = {
            "log_ratio": float(log_ratio),
            "pct_of_log_total": float(log_ratio / log_total * 100) if log_total != 0 else 0,
        }

    return {
        "log_phi_ratio": float(log_total),
        "phi_ratio": float(phi_sp / phi_molt) if phi_molt > 0 else float("inf"),
        "contributions": log_contributions,
        "sum_log_contributions": float(sum(c["log_ratio"] for c in log_contributions.values())),
    }


def bootstrap_shapley(
    molt: TopologyParams,
    sp: TopologyParams,
    n_bootstrap: int = 5000,
    noise_frac: float = 0.10,
    alpha: float = 2.0,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CIs on Shapley values by perturbing topology parameters.

    Perturbs each metric by ±noise_frac (uniform) independently.
    """
    rng = np.random.default_rng(seed)

    def perturb(params: TopologyParams) -> TopologyParams:
        def p(val):
            return val * (1.0 + rng.uniform(-noise_frac, noise_frac))
        return TopologyParams(
            triadic_closure=np.clip(p(params.triadic_closure), 0, 1),
            edge_overlap=np.clip(p(params.edge_overlap), 0, 1),
            gini=np.clip(p(params.gini), 0, 1),
            mean_degree=max(1.0, p(params.mean_degree)),
            n_nodes=params.n_nodes,
            his_mean=np.clip(p(params.his_mean), 0, 1),
            frac_higher_order=np.clip(p(params.frac_higher_order), 0, 1),
        )

    all_shapley = {f: [] for f in FACTOR_NAMES}
    all_pct = {f: [] for f in FACTOR_NAMES}

    for _ in range(n_bootstrap):
        m_pert = perturb(molt)
        s_pert = perturb(sp)
        result = shapley_decomposition(m_pert, s_pert, alpha)
        for f in FACTOR_NAMES:
            all_shapley[f].append(result["shapley_values"][f])
            all_pct[f].append(result["pct_contributions"][f])

    bootstrap_result = {}
    for f in FACTOR_NAMES:
        vals = np.array(all_shapley[f])
        pcts = np.array(all_pct[f])
        bootstrap_result[f] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "pct_mean": float(np.mean(pcts)),
            "pct_ci_95": [float(np.percentile(pcts, 2.5)), float(np.percentile(pcts, 97.5))],
        }

    return bootstrap_result


def run_loo_dynamics(
    molt: TopologyParams,
    sp: TopologyParams,
    alpha: float = 2.0,
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
    rho0: float = 0.15,
    T: float = 500,
) -> dict:
    """Run ODE dynamics for each LOO scenario at a fixed beta2."""
    c_molt = extract_components(molt, alpha)
    c_sp = extract_components(sp, alpha)

    scenarios = {"Moltbook (original)": dict(c_molt)}
    for factor in FACTOR_NAMES:
        swapped = dict(c_molt)
        swapped[factor] = c_sp[factor]
        scenarios[f"Swap {factor}"] = swapped
    scenarios["SocioPatterns (target)"] = dict(c_sp)

    results = {}
    for name, components in scenarios.items():
        # Reconstruct TopologyParams from components
        # Reverse-engineer raw values from component values
        closure = components["closure"]
        J = (components["overlap"] - 1.0) / alpha
        cv2 = components["heterogeneity"] - 1.0
        gini = np.sqrt(cv2 / (np.pi / 2)) if cv2 > 0 else 0.0
        his = components["HIS"]

        topo = TopologyParams(
            triadic_closure=closure,
            edge_overlap=J,
            gini=gini,
            mean_degree=molt.mean_degree,
            n_nodes=molt.n_nodes,
            his_mean=his,
            frac_higher_order=molt.frac_higher_order,
        )
        model = TopologyAwareContagionModel(
            beta1=beta1, beta2=beta2, mu=mu, lam=lam, C_ctx=C_ctx,
            topology=topo, alpha=alpha,
        )
        t, rho = model.simulate(T=T, rho0=rho0, dt=1.0)
        results[name] = {
            "phi": float(model.phi),
            "beta2_eff": float(model.beta2_eff),
            "final_rho": float(rho[-1]),
            "is_bistable": model.is_bistable(),
            "trajectory": rho.tolist(),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_decomposition(shapley, loo, log_dec, bootstrap, dynamics, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors = {
        "closure": "#E24A33",
        "overlap": "#348ABD",
        "heterogeneity": "#FFA500",
        "HIS": "#2ca02c",
    }

    # ── Panel A: Shapley bars with bootstrap CIs ──
    ax = axes[0, 0]
    x = np.arange(len(FACTOR_NAMES))
    vals = [shapley["shapley_values"][f] for f in FACTOR_NAMES]
    errs = np.array([
        [shapley["shapley_values"][f] - bootstrap[f]["ci_95"][0],
         bootstrap[f]["ci_95"][1] - shapley["shapley_values"][f]]
        for f in FACTOR_NAMES
    ]).T

    bar_colors = [colors[f] for f in FACTOR_NAMES]
    ax.bar(x, vals, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5,
           yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(["Triadic\nclosure", "Edge\noverlap", "Degree\nheterogeneity", "HIS"])
    ax.set_ylabel("Shapley value (Φ units)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"A. Shapley decomposition of Φ gap\n"
                 f"(Φ_Molt={shapley['phi_molt']:.3f} → Φ_SP={shapley['phi_sp']:.3f}, "
                 f"gap={shapley['phi_gap']:.3f})")
    ax.grid(axis="y", alpha=0.3)

    for i, f in enumerate(FACTOR_NAMES):
        pct = shapley["pct_contributions"][f]
        ax.text(i, vals[i] + (0.01 if vals[i] >= 0 else -0.03),
                f"{pct:+.0f}%", ha="center", fontsize=9, fontweight="bold")

    # ── Panel B: LOO waterfall ──
    ax = axes[0, 1]
    phi_molt = loo["phi_molt"]
    phi_sp = loo["phi_sp"]

    # Sort factors by delta_phi descending
    sorted_factors = sorted(FACTOR_NAMES, key=lambda f: loo["factors"][f]["delta_phi"], reverse=True)
    labels = ["Moltbook\n(baseline)"]
    cumulative = [phi_molt]
    bar_vals = [phi_molt]
    bar_bottoms = [0]
    bar_colors_wf = ["#E24A33"]

    running = phi_molt
    for f in sorted_factors:
        delta = loo["factors"][f]["delta_phi"]
        labels.append(f"+ {f}")
        bar_bottoms.append(running)
        bar_vals.append(delta)
        bar_colors_wf.append(colors[f])
        running += delta
        cumulative.append(running)

    labels.append("SP\n(target)")
    bar_bottoms.append(0)
    bar_vals.append(phi_sp)
    bar_colors_wf.append("#348ABD")

    x = np.arange(len(labels))
    bars = ax.bar(x, bar_vals, bottom=bar_bottoms, color=bar_colors_wf, alpha=0.8,
                  edgecolor="black", linewidth=0.5)

    # Connect bars with lines
    for i in range(1, len(cumulative)):
        ax.plot([i - 0.4, i + 0.4], [cumulative[i - 1]] * 2,
                color="gray", linewidth=0.8, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Topology factor Φ")
    ax.set_title("B. Leave-one-out: individual factor impact")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel C: Log decomposition ──
    ax = axes[1, 0]
    log_vals = [log_dec["contributions"][f]["pct_of_log_total"] for f in FACTOR_NAMES]
    labels_log = ["Triadic\nclosure", "Edge\noverlap", "Degree\nheterog.", "HIS"]
    pie_colors = [colors[f] for f in FACTOR_NAMES]

    # Handle negative contributions (heterogeneity works against)
    positive = [max(0, v) for v in log_vals]
    negative = [min(0, v) for v in log_vals]

    if any(v < 0 for v in log_vals):
        # Use horizontal bar chart instead of pie for mixed signs
        ax.barh(range(len(FACTOR_NAMES)), log_vals, color=pie_colors, alpha=0.8,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(FACTOR_NAMES)))
        ax.set_yticklabels(labels_log, fontsize=9)
        ax.set_xlabel("% of log(Φ_SP/Φ_Molt)")
        ax.axvline(0, color="black", linewidth=0.8)
        for i, v in enumerate(log_vals):
            ax.text(v + (2 if v >= 0 else -2), i, f"{v:+.1f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
    else:
        wedges, texts, autotexts = ax.pie(
            log_vals, labels=labels_log, colors=pie_colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75)
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight("bold")

    ax.set_title(f"C. Log decomposition\n"
                 f"Φ_SP/Φ_Molt = {log_dec['phi_ratio']:.2f}×")

    # ── Panel D: LOO dynamics ──
    ax = axes[1, 1]
    dyn_colors = {
        "Moltbook (original)": "#E24A33",
        "SocioPatterns (target)": "#348ABD",
    }
    for f in FACTOR_NAMES:
        dyn_colors[f"Swap {f}"] = colors[f]

    for name, d in dynamics.items():
        traj = d["trajectory"]
        t = np.arange(len(traj))
        style = "-" if "Moltbook" in name or "SocioPatterns" in name else "--"
        lw = 2.5 if "Moltbook" in name or "SocioPatterns" in name else 1.5
        ax.plot(t, traj, style, color=dyn_colors.get(name, "gray"),
                linewidth=lw, label=f'{name} (ρ∞={d["final_rho"]:.3f})')

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Norm adoption $\\rho(t)$")
    ax.set_title("D. Dynamics: which swap moves the needle?")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Φ Component Decomposition: Which Topology Feature Drives the AI–Human Gap?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(outdir / "fig_phi_decomposition.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved decomposition figure")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = ROOT / "results" / "phi_decomposition"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phi Component Decomposition Analysis")
    logger.info("=" * 60)

    molt = moltbook_topology()
    sp = sociopatterns_topology()

    # ── Shapley decomposition ─────────────────────────────────────
    logger.info("\n--- Exact Shapley values (16 coalitions) ---")
    shapley = shapley_decomposition(molt, sp)
    for f in FACTOR_NAMES:
        logger.info("  %s: Shapley=%.4f (%+.1f%%)",
                    f, shapley["shapley_values"][f], shapley["pct_contributions"][f])
    logger.info("  Sum of Shapley values: %.4f (gap: %.4f)",
                shapley["sum_shapley"], shapley["phi_gap"])

    # ── Leave-one-out ─────────────────────────────────────────────
    logger.info("\n--- Leave-one-out analysis ---")
    loo = leave_one_out(molt, sp)
    for f in FACTOR_NAMES:
        d = loo["factors"][f]
        logger.info("  Swap %s: Φ %.3f → %.3f (Δ=%.3f, %.1f%% of gap)",
                    f, loo["phi_molt"], d["phi_swapped"],
                    d["delta_phi"], d["pct_of_total_gap"])

    # ── Log decomposition ─────────────────────────────────────────
    logger.info("\n--- Log decomposition ---")
    log_dec = log_decomposition(molt, sp)
    logger.info("  Φ_SP/Φ_Molt = %.3f (log ratio = %.4f)",
                log_dec["phi_ratio"], log_dec["log_phi_ratio"])
    for f in FACTOR_NAMES:
        c = log_dec["contributions"][f]
        logger.info("  %s: log_ratio=%.4f (%.1f%%)", f, c["log_ratio"], c["pct_of_log_total"])

    # ── Bootstrap CIs ─────────────────────────────────────────────
    logger.info("\n--- Bootstrap Shapley CIs (n=5000) ---")
    bootstrap = bootstrap_shapley(molt, sp, n_bootstrap=5000)
    for f in FACTOR_NAMES:
        b = bootstrap[f]
        logger.info("  %s: %.4f [%.4f, %.4f] (%.1f%% [%.1f, %.1f])",
                    f, b["mean"], b["ci_95"][0], b["ci_95"][1],
                    b["pct_mean"], b["pct_ci_95"][0], b["pct_ci_95"][1])

    # ── LOO dynamics ──────────────────────────────────────────────
    logger.info("\n--- LOO dynamics at β₂=3.5 ---")
    dynamics = run_loo_dynamics(molt, sp, beta2=3.5)
    for name, d in dynamics.items():
        logger.info("  %s: Φ=%.3f, ρ∞=%.3f, bistable=%s",
                    name, d["phi"], d["final_rho"], d["is_bistable"])

    # ── Save results ──────────────────────────────────────────────
    full_results = {
        "shapley": shapley,
        "leave_one_out": loo,
        "log_decomposition": log_dec,
        "bootstrap": bootstrap,
        "dynamics": {name: {k: v for k, v in d.items() if k != "trajectory"}
                     for name, d in dynamics.items()},
    }
    (outdir / "decomposition_results.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )

    # ── Figure ────────────────────────────────────────────────────
    _plot_decomposition(shapley, loo, log_dec, bootstrap, dynamics, outdir)

    logger.info("\n=== Phi decomposition complete! Results in %s ===", outdir)


if __name__ == "__main__":
    main()
