"""
Phase 3A: Mini-Moltbook in-silico validation experiment.

Generates synthetic AI agent networks with controlled topology parameters
matching real Moltbook observations, then runs the topology-aware contagion
model to validate that:

  1. Broadcast condition (star-like topology, low HIS) reproduces
     Moltbook-like dynamics within 20%
  2. Humanized condition (clique-like topology, high HIS) produces
     qualitatively different dynamics matching SocioPatterns predictions
  3. The topology factor Φ accurately predicts which condition produces
     higher norm adoption

Verification criterion: Broadcast condition topology metrics match
real Moltbook within 20% on: triadic closure, Gini, HIS, edge overlap.

Ref: Iacopini et al. (2019); our contagion_ho.py derivation
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

from src.analysis.hypergraph_builder import Hypergraph
from src.analysis.topology import compute_topology
from src.models.contagion_ho import (
    TopologyAwareContagionModel, TopologyParams,
    moltbook_topology, sociopatterns_topology,
)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic network generators
# ═══════════════════════════════════════════════════════════════════════

def generate_star_network(
    n_nodes: int = 2000,
    n_threads: int = 5000,
    hub_frac: float = 0.02,
    max_commenters: int = 12,
    seed: int = 42,
) -> Hypergraph:
    """
    Generate Moltbook-like star topology at realistic scale.

    2000 nodes allows sufficient degree range: hubs at 500-2000 threads,
    pool followers at 30-70, lurkers at 2-10. This produces the needed
    hub/commenter ratio (>20×) for HIS≈0.41.

    Targets: closure≈0.79, Gini≈0.82, HIS≈0.41, overlap≈0.16
    """
    rng = np.random.default_rng(seed)
    n_hubs = max(5, int(n_nodes * hub_frac))

    nodes = {f"n{i}" for i in range(n_nodes)}
    hub_ids = list(range(n_hubs))
    non_hub_ids = list(range(n_hubs, n_nodes))

    # Power law -2.0: top hub ~30%, smooth falloff
    hub_weights = (np.arange(1, n_hubs + 1, dtype=float)) ** (-2.0)
    hub_weights /= hub_weights.sum()

    # Pool size ~80: tight enough for closure, large enough for HIS
    follower_pools = {}
    for h in hub_ids:
        pool_size = rng.integers(65, 95)
        follower_pools[h] = rng.choice(non_hub_ids, size=min(pool_size, len(non_hub_ids)),
                                       replace=False).tolist()

    hyperedges = []
    for _ in range(n_threads):
        hub = rng.choice(hub_ids, p=hub_weights)
        n_comm = rng.integers(2, max_commenters + 1)

        # 85% from follower pool, 15% random
        pool = follower_pools[hub]
        n_from_pool = max(1, int(n_comm * 0.85))
        n_random = n_comm - n_from_pool

        from_pool = rng.choice(pool, size=min(n_from_pool, len(pool)), replace=False)
        from_random = rng.choice(non_hub_ids, size=n_random, replace=False)

        members = set([hub])
        members.update(int(x) for x in from_pool)
        members.update(int(x) for x in from_random)

        edge = frozenset(f"n{m}" for m in members)
        if len(edge) >= 2:
            hyperedges.append(edge)

    return Hypergraph(
        nodes=nodes,
        hyperedges=hyperedges,
        metadata={"source": "mini_moltbook_star", "n_hubs": n_hubs},
    )


def generate_clique_network(
    n_nodes: int = 2000,
    n_groups: int = 12000,
    group_size_range: tuple = (4, 10),
    seed: int = 42,
) -> Hypergraph:
    """
    Generate SocioPatterns-like clique topology at realistic scale.

    25 communities of ~80 nodes each. Many groups (12000) saturate
    within-community projected graph → very high closure.
    40% group memory creates temporal recurring encounters → high overlap.

    Targets: closure≈0.97, Gini≈0.37, HIS≈0.69, overlap≈0.27
    """
    rng = np.random.default_rng(seed)
    nodes = {f"n{i}" for i in range(n_nodes)}

    # 20 communities of ~100 nodes
    n_communities = 20
    community_ids = np.arange(n_nodes) % n_communities
    rng.shuffle(community_ids)
    communities = {}
    for i, c in enumerate(community_ids):
        communities.setdefault(c, []).append(i)

    # Activity variation for moderate Gini (~0.37)
    activity_weights = np.ones(n_nodes)
    n_active = n_nodes // 6
    active = rng.choice(n_nodes, size=n_active, replace=False)
    activity_weights[active] = 4.0

    # Track recent groups per community for "memory" mechanism
    recent_groups: dict[int, list[list[int]]] = {c: [] for c in range(n_communities)}

    hyperedges = []
    for _ in range(n_groups):
        size = rng.integers(group_size_range[0], group_size_range[1] + 1)

        # 96% within-community
        if rng.random() < 0.96:
            comm = rng.integers(0, n_communities)
            pool = communities[comm]
            pool_w = activity_weights[pool]
            pool_w = pool_w / pool_w.sum()

            # 50% chance: modify a recent group (high overlap)
            if recent_groups[comm] and rng.random() < 0.50:
                base = list(recent_groups[comm][rng.integers(len(recent_groups[comm]))])
                n_remove = rng.integers(1, min(3, len(base)))
                n_add = rng.integers(1, 3)
                remove_idx = rng.choice(len(base), size=n_remove, replace=False)
                remaining = [base[i] for i in range(len(base)) if i not in remove_idx]
                new_pool = [p for p in pool if p not in remaining]
                if new_pool:
                    new_w = activity_weights[new_pool]
                    new_w = new_w / new_w.sum()
                    added = rng.choice(new_pool, size=min(n_add, len(new_pool)),
                                       replace=False, p=new_w)
                    members = remaining + list(added)
                else:
                    members = remaining
            else:
                if len(pool) >= size:
                    members = list(rng.choice(pool, size=size, replace=False, p=pool_w))
                else:
                    members = list(pool[:size])

            # Update recent groups (keep last 15 per community)
            if len(members) >= 3:
                recent_groups[comm].append(members)
                if len(recent_groups[comm]) > 15:
                    recent_groups[comm].pop(0)
        else:
            # Cross-community
            comm = rng.integers(0, n_communities)
            pool = communities[comm] + communities[(comm + 1) % n_communities]
            pool_w = activity_weights[pool]
            pool_w = pool_w / pool_w.sum()
            members = list(rng.choice(pool, size=min(size, len(pool)),
                                      replace=False, p=pool_w))

        edge = frozenset(f"n{m}" for m in members)
        if len(edge) >= 2:
            hyperedges.append(edge)

    return Hypergraph(
        nodes=nodes,
        hyperedges=hyperedges,
        metadata={"source": "mini_moltbook_clique", "n_communities": n_communities},
    )


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_topology_match(
    report,
    target: dict,
    tolerance: float = 0.20,
) -> dict:
    """
    Check if synthetic topology matches target within tolerance.

    Returns dict with per-metric pass/fail and overall result.
    """
    metrics = [
        ("triadic_closure_rate", "triadic_closure"),
        ("hyperdegree_gini", "gini"),
        ("his_mean", "his_mean"),
        ("mean_edge_overlap", "edge_overlap"),
    ]

    checks = {}
    all_pass = True
    for report_key, target_key in metrics:
        observed = getattr(report, report_key)
        expected = target[target_key]
        if expected == 0:
            error = abs(observed)
        else:
            error = abs(observed - expected) / abs(expected)

        passed = error <= tolerance
        if not passed:
            all_pass = False

        checks[report_key] = {
            "observed": float(observed),
            "expected": float(expected),
            "error_pct": float(error * 100),
            "pass": passed,
        }

    return {"all_pass": all_pass, "tolerance_pct": tolerance * 100, "checks": checks}


def run_contagion_comparison(
    report_star, report_clique,
    beta2_values: list = None,
) -> dict:
    """
    Run contagion on both topologies across β₂ values.
    Show that topology determines which condition achieves norm adoption.
    """
    if beta2_values is None:
        beta2_values = [0.5, 1.0, 2.0, 3.0, 5.0]

    results = {}
    for name, report in [("Star (AI-like)", report_star),
                          ("Clique (human-like)", report_clique)]:
        results[name] = {}
        for b2 in beta2_values:
            model = TopologyAwareContagionModel.from_topology_report(
                report, beta1=0.05, beta2=b2, mu=0.1, lam=2.0, C_ctx=8.0,
            )
            _, rho = model.simulate(T=500, rho0=0.15, dt=1.0)

            results[name][f"b2={b2}"] = {
                "phi": model.phi,
                "beta2_eff": model.beta2_eff,
                "final_rho": float(rho[-1]),
                "is_bistable": model.is_bistable(),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = ROOT / "results" / "mini_moltbook"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 3A: Mini-Moltbook Validation")
    logger.info("=" * 60)

    # ── Reference targets ──────────────────────────────────────────
    molt_target = {
        "triadic_closure": 0.79,
        "gini": 0.82,
        "his_mean": 0.41,
        "edge_overlap": 0.16,
    }
    sp_target = {
        "triadic_closure": 0.97,
        "gini": 0.37,
        "his_mean": 0.69,
        "edge_overlap": 0.27,
    }

    # ── Generate synthetic networks ────────────────────────────────
    logger.info("\n--- Generating star (Moltbook-like) network ---")
    hg_star = generate_star_network(n_nodes=2000, n_threads=6000)
    logger.info("  %s", hg_star.summary())

    logger.info("\n--- Generating clique (SocioPatterns-like) network ---")
    hg_clique = generate_clique_network(n_nodes=2000, n_groups=15000)
    logger.info("  %s", hg_clique.summary())

    # ── Compute topology ───────────────────────────────────────────
    logger.info("\n--- Computing topology ---")
    report_star = compute_topology(hg_star, name="Star (AI)", triadic_sample=10000)
    report_clique = compute_topology(hg_clique, name="Clique (Human)", triadic_sample=10000)

    logger.info("Star:   closure=%.3f, Gini=%.3f, HIS=%.3f, overlap=%.3f",
                report_star.triadic_closure_rate, report_star.hyperdegree_gini,
                report_star.his_mean, report_star.mean_edge_overlap)
    logger.info("Clique: closure=%.3f, Gini=%.3f, HIS=%.3f, overlap=%.3f",
                report_clique.triadic_closure_rate, report_clique.hyperdegree_gini,
                report_clique.his_mean, report_clique.mean_edge_overlap)

    # ── Validate topology match ────────────────────────────────────
    logger.info("\n--- Validation: Star vs Moltbook target (20% tolerance) ---")
    val_star = validate_topology_match(report_star, molt_target)
    for metric, check in val_star["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        logger.info("  %s: %.3f vs %.3f (%.1f%% error) → %s",
                    metric, check["observed"], check["expected"],
                    check["error_pct"], status)
    logger.info("  Overall: %s", "PASS" if val_star["all_pass"] else "FAIL")

    logger.info("\n--- Validation: Clique vs SocioPatterns target ---")
    val_clique = validate_topology_match(report_clique, sp_target)
    for metric, check in val_clique["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        logger.info("  %s: %.3f vs %.3f (%.1f%% error) → %s",
                    metric, check["observed"], check["expected"],
                    check["error_pct"], status)
    logger.info("  Overall: %s", "PASS" if val_clique["all_pass"] else "FAIL")

    # ── Contagion dynamics comparison ──────────────────────────────
    logger.info("\n--- Contagion comparison ---")
    dynamics = run_contagion_comparison(report_star, report_clique)

    for name, d in dynamics.items():
        logger.info("  %s:", name)
        for bkey, vals in d.items():
            logger.info("    %s: Φ=%.3f, β₂_eff=%.3f, ρ∞=%.3f",
                        bkey, vals["phi"], vals["beta2_eff"], vals["final_rho"])

    # ── Save results ───────────────────────────────────────────────
    full_results = {
        "star_topology": report_star.to_dict(),
        "clique_topology": report_clique.to_dict(),
        "validation_star": val_star,
        "validation_clique": val_clique,
        "dynamics": dynamics,
        "targets": {"moltbook": molt_target, "sociopatterns": sp_target},
    }

    (outdir / "mini_moltbook_results.json").write_text(
        json.dumps(_sanitize(full_results), indent=2, default=str)
    )

    # ── Generate figure ────────────────────────────────────────────
    _plot_mini_moltbook(report_star, report_clique, val_star, val_clique,
                         dynamics, molt_target, sp_target, outdir)

    logger.info("\n=== Phase 3A complete! Results in %s ===", outdir)


def _sanitize(obj):
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _plot_mini_moltbook(report_star, report_clique, val_star, val_clique,
                          dynamics, molt_target, sp_target, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Topology comparison bar chart
    ax = axes[0, 0]
    metrics = ["triadic_closure_rate", "hyperdegree_gini", "his_mean", "mean_edge_overlap"]
    labels = ["Triadic\nclosure", "Degree\nGini", "HIS", "Edge\noverlap"]
    target_keys = ["triadic_closure", "gini", "his_mean", "edge_overlap"]

    x = np.arange(len(metrics))
    w = 0.2
    star_vals = [getattr(report_star, m) for m in metrics]
    clique_vals = [getattr(report_clique, m) for m in metrics]
    molt_vals = [molt_target[k] for k in target_keys]
    sp_vals = [sp_target[k] for k in target_keys]

    ax.bar(x - 1.5*w, star_vals, w, label="Synthetic star", color="#E24A33", alpha=0.8)
    ax.bar(x - 0.5*w, molt_vals, w, label="Moltbook (target)", color="#E24A33",
           alpha=0.3, edgecolor="#E24A33", linewidth=2)
    ax.bar(x + 0.5*w, clique_vals, w, label="Synthetic clique", color="#348ABD", alpha=0.8)
    ax.bar(x + 1.5*w, sp_vals, w, label="SocioPatterns (target)", color="#348ABD",
           alpha=0.3, edgecolor="#348ABD", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("A. Topology validation: synthetic vs real")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Validation error bars
    ax = axes[0, 1]
    checks_star = val_star["checks"]
    checks_clique = val_clique["checks"]
    metric_names = list(checks_star.keys())
    x = np.arange(len(metric_names))

    errors_star = [checks_star[m]["error_pct"] for m in metric_names]
    errors_clique = [checks_clique[m]["error_pct"] for m in metric_names]
    colors_star = ["#2ca02c" if checks_star[m]["pass"] else "#E24A33" for m in metric_names]
    colors_clique = ["#2ca02c" if checks_clique[m]["pass"] else "#E24A33" for m in metric_names]

    ax.bar(x - 0.2, errors_star, 0.35, color=colors_star, alpha=0.7, label="Star vs Moltbook")
    ax.bar(x + 0.2, errors_clique, 0.35, color=colors_clique, alpha=0.4, label="Clique vs SP",
           hatch="//")
    ax.axhline(20, color="red", linestyle="--", linewidth=1.5, label="20% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Error (%)")
    ax.set_title("B. Validation error (target < 20%)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Contagion dynamics across β₂
    ax = axes[1, 0]
    b2_vals = [0.5, 1.0, 2.0, 3.0, 5.0]
    star_rho = [dynamics["Star (AI-like)"][f"b2={b}"]["final_rho"] for b in b2_vals]
    clique_rho = [dynamics["Clique (human-like)"][f"b2={b}"]["final_rho"] for b in b2_vals]

    ax.plot(b2_vals, star_rho, "o-", color="#E24A33", linewidth=2, markersize=6,
            label="Star (AI-like)")
    ax.plot(b2_vals, clique_rho, "s-", color="#348ABD", linewidth=2, markersize=6,
            label="Clique (human-like)")
    ax.set_xlabel("Higher-order rate $\\beta_2$")
    ax.set_ylabel("Final adoption $\\rho_\\infty$")
    ax.set_title("C. Norm adoption: topology determines outcome")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel D: Φ comparison
    ax = axes[1, 1]
    phi_star = dynamics["Star (AI-like)"]["b2=3.0"]["phi"]
    phi_clique = dynamics["Clique (human-like)"]["b2=3.0"]["phi"]
    phi_molt = moltbook_topology().topology_factor()
    phi_sp = sociopatterns_topology().topology_factor()

    labels_phi = ["Synthetic\nStar", "Real\nMoltbook", "Synthetic\nClique", "Real\nSP"]
    vals_phi = [phi_star, phi_molt, phi_clique, phi_sp]
    colors_phi = ["#E24A33", "#E24A33", "#348ABD", "#348ABD"]
    alphas = [0.8, 0.4, 0.8, 0.4]

    bars = ax.bar(range(4), vals_phi, color=[
        (*plt.matplotlib.colors.to_rgb(c), a) for c, a in zip(colors_phi, alphas)
    ], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_phi, fontsize=8)
    ax.set_ylabel("Topology factor Φ")
    ax.set_title("D. Φ: synthetic reproduces empirical ranking")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals_phi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    star_n_pass = sum(1 for c in val_star["checks"].values() if c["pass"])
    clique_n_pass = sum(1 for c in val_clique["checks"].values() if c["pass"])
    phi_correct = phi_clique > phi_star  # clique should have higher Φ
    fig.suptitle(
        f"Mini-Moltbook Validation: Star {star_n_pass}/4, Clique {clique_n_pass}/4 metrics within 20%\n"
        f"Φ direction {'correct' if phi_correct else 'INCORRECT'}: "
        f"Φ_clique={phi_clique:.3f} > Φ_star={phi_star:.3f} "
        f"(real: SP={phi_sp:.3f} > Molt={phi_molt:.3f})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig_mini_moltbook.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved mini-Moltbook figure")


if __name__ == "__main__":
    main()
