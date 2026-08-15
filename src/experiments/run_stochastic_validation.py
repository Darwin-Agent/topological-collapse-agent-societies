"""
Stochastic Validation: ODE mean-field vs Monte Carlo on empirical hypergraphs.

Addresses reviewer question: "Does the mean-field approximation hold
for these specific networks?"

Runs MicroContagionModel on actual SocioPatterns (403 nodes, full) and
subsampled Moltbook (~500 nodes) hypergraphs, comparing trajectories
against the TopologyAwareContagionModel mean-field ODE.
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

from src.analysis.hypergraph_builder import (
    Hypergraph,
    build_moltbook_hypergraph_from_hf,
    build_moltbook_hypergraph_from_db,
    build_sociopatterns_hypergraph,
)
from src.analysis.topology import compute_topology
from src.models.contagion_ho import TopologyAwareContagionModel
from src.models.contagion_micro import MicroContagionModel


# ═══════════════════════════════════════════════════════════════════════
# Adapter functions
# ═══════════════════════════════════════════════════════════════════════

def hypergraph_to_micro_input(hg: Hypergraph) -> tuple[int, list[tuple], list[tuple]]:
    """
    Convert Hypergraph (string nodes, frozenset edges) to micro model format.

    Returns:
        n_nodes: int
        pairwise_edges: list of (int, int) tuples (from clique projection)
        hyperedges: list of int tuples (size >= 3 only)
    """
    node_list = sorted(hg.nodes)
    node_map = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)

    pairwise_set = set()
    hyperedges = []

    for edge in hg.hyperedges:
        members = tuple(sorted(node_map[n] for n in edge))
        if len(members) == 2:
            pairwise_set.add(members)
        elif len(members) >= 3:
            hyperedges.append(members)
            # Also project to pairwise
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairwise_set.add((members[i], members[j]))

    pairwise_edges = list(pairwise_set)
    return n_nodes, pairwise_edges, hyperedges


def subsample_hypergraph(
    hg: Hypergraph,
    n_target: int = 500,
    seed: int = 42,
) -> Hypergraph:
    """
    Subsample to top-degree nodes + their induced hyperedges.

    Uses the highest-degree nodes to preserve hub structure.
    """
    degrees = {}
    for edge in hg.hyperedges:
        for node in edge:
            degrees[node] = degrees.get(node, 0) + 1

    sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)
    selected = set(sorted_nodes[:n_target])

    # Induced hyperedges: keep only edges where ALL members are selected
    induced_edges = []
    for edge in hg.hyperedges:
        if edge.issubset(selected):
            induced_edges.append(edge)

    return Hypergraph(
        nodes=selected,
        hyperedges=induced_edges,
        metadata={"source": "subsampled", "n_original": len(hg.nodes),
                  "n_target": n_target},
    )


# ═══════════════════════════════════════════════════════════════════════
# Core validation
# ═══════════════════════════════════════════════════════════════════════

def run_ode_vs_micro(
    hg: Hypergraph,
    hg_name: str,
    report,
    n_repeats: int = 30,
    rho0_values: list[float] = None,
    beta1: float = 0.05,
    beta2: float = 3.5,
    mu: float = 0.1,
    lam: float = 2.0,
    C_ctx: float = 8.0,
    T: int = 300,
) -> dict:
    """
    Compare ODE mean-field vs stochastic MC on an empirical hypergraph.

    Parameter mapping:
    - ODE: beta2_eff = beta2 * Phi * exp(-lam/C)
    - Micro: effective_beta2 = beta2_micro * exp(-lam/C)
    - To match: beta2_micro = beta2 * Phi
    """
    if rho0_values is None:
        rho0_values = [0.05, 0.15, 0.25]

    # Build ODE model
    ode_model = TopologyAwareContagionModel.from_topology_report(
        report, beta1=beta1, beta2=beta2, mu=mu, lam=lam, C_ctx=C_ctx)

    # Set micro beta2 to match ODE effective rate
    beta2_micro = beta2 * ode_model.phi
    micro_model = MicroContagionModel(
        beta1=beta1, beta2=beta2_micro, mu=mu, lam=lam, C=C_ctx, g=1.0)

    logger.info("  %s: Φ=%.3f, β₂_eff=%.4f, micro_β₂=%.3f",
                hg_name, ode_model.phi, ode_model.beta2_eff, beta2_micro)

    # Convert hypergraph
    n_nodes, pairwise_edges, hyperedges = hypergraph_to_micro_input(hg)
    logger.info("  Topology: %d nodes, %d pairwise, %d hyperedges",
                n_nodes, len(pairwise_edges), len(hyperedges))

    results = {}
    for rho0 in rho0_values:
        logger.info("  ρ₀=%.2f: running ODE + %d MC repeats...", rho0, n_repeats)

        # ODE
        t_ode, rho_ode = ode_model.simulate(T=float(T), rho0=rho0, dt=1.0)

        # MC ensemble
        mc_trajectories = []
        mc_finals = []
        for rep in range(n_repeats):
            r = micro_model.simulate(
                n_nodes=n_nodes,
                pairwise_edges=pairwise_edges,
                hyperedges=hyperedges,
                T=T, rho0=rho0, seed=42 + rep, dt=1.0,
            )
            mc_trajectories.append(r["rho_t"])
            mc_finals.append(r["final_rho"])

        mc_arr = np.array(mc_trajectories)
        mc_mean = mc_arr.mean(axis=0)
        mc_std = mc_arr.std(axis=0)
        mc_finals = np.array(mc_finals)

        # Discrepancy metrics
        # Align lengths (ODE and MC may differ slightly)
        min_len = min(len(rho_ode), len(mc_mean))
        ode_aligned = rho_ode[:min_len]
        mc_mean_aligned = mc_mean[:min_len]
        mc_std_aligned = mc_std[:min_len]

        rmse = float(np.sqrt(np.mean((ode_aligned - mc_mean_aligned) ** 2)))
        mae = float(np.mean(np.abs(ode_aligned - mc_mean_aligned)))

        # Fraction of time ODE within 1-sigma band
        within_band = np.abs(ode_aligned - mc_mean_aligned) <= mc_std_aligned
        frac_within = float(np.mean(within_band))

        # KS test on final rho
        from scipy.stats import kstest, norm
        ode_final = float(rho_ode[-1])
        if mc_finals.std() > 1e-10:
            ks_stat, ks_p = kstest(mc_finals, norm(loc=ode_final, scale=mc_finals.std()).cdf)
        else:
            ks_stat, ks_p = 0.0, 1.0

        results[f"rho0={rho0}"] = {
            "ode_final": ode_final,
            "mc_final_mean": float(mc_finals.mean()),
            "mc_final_std": float(mc_finals.std()),
            "rmse": rmse,
            "mae": mae,
            "frac_within_1sigma": frac_within,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "ode_trajectory": rho_ode.tolist(),
            "mc_mean": mc_mean.tolist(),
            "mc_std": mc_std.tolist(),
        }

        logger.info("    ODE final=%.4f, MC mean=%.4f±%.4f, RMSE=%.4f, "
                    "within_band=%.1f%%",
                    ode_final, mc_finals.mean(), mc_finals.std(),
                    rmse, frac_within * 100)

    return {
        "name": hg_name,
        "n_nodes": n_nodes,
        "n_pairwise": len(pairwise_edges),
        "n_hyperedges": len(hyperedges),
        "phi": float(ode_model.phi),
        "beta2_eff": float(ode_model.beta2_eff),
        "n_repeats": n_repeats,
        "rho0_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_validation(sp_results, molt_results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho0_vals = [0.05, 0.15, 0.25]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, (results, name, color) in enumerate([
        (sp_results, "SocioPatterns", "#348ABD"),
        (molt_results, "Moltbook (subsampled)", "#E24A33"),
    ]):
        for col, rho0 in enumerate(rho0_vals):
            ax = axes[row, col]
            key = f"rho0={rho0}"
            if key not in results["rho0_results"]:
                continue
            d = results["rho0_results"][key]

            ode_traj = np.array(d["ode_trajectory"])
            mc_mean = np.array(d["mc_mean"])
            mc_std = np.array(d["mc_std"])
            t_ode = np.arange(len(ode_traj))
            t_mc = np.arange(len(mc_mean))

            ax.plot(t_ode, ode_traj, "-", color=color, linewidth=2.5,
                    label=f'ODE (final={d["ode_final"]:.3f})')
            ax.plot(t_mc, mc_mean, "--", color="black", linewidth=1.5,
                    label=f'MC mean (final={d["mc_final_mean"]:.3f})')
            ax.fill_between(t_mc, mc_mean - mc_std, mc_mean + mc_std,
                           color="gray", alpha=0.25, label="MC ±1σ")

            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{name}, ρ₀={rho0}\n"
                        f"RMSE={d['rmse']:.4f}, "
                        f"in-band={d['frac_within_1sigma']:.0%}")
            if col == 0:
                ax.set_ylabel("Adoption ρ(t)")
            if row == 1:
                ax.set_xlabel("Time $t$")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Mean-Field ODE vs Stochastic MC on Empirical Hypergraphs\n"
        f"SP: Φ={sp_results['phi']:.3f}, {sp_results['n_nodes']} nodes | "
        f"Molt: Φ={molt_results['phi']:.3f}, {molt_results['n_nodes']} nodes",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig_stochastic_validation.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved validation figure")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = ROOT / "results" / "stochastic_validation"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stochastic Validation: ODE vs MC on Empirical Hypergraphs")
    logger.info("=" * 60)

    # ── Build hypergraphs ─────────────────────────────────────────
    sp_path = ROOT / "data/raw/sociopatterns/contact/tij_SFHH.dat"
    hf_posts = ROOT / "data/raw/moltbook_hf/lnajt/posts.parquet"
    hf_comments = ROOT / "data/raw/moltbook_hf/moltnet/data/v2026-02-28/comments.parquet"
    db_path = ROOT / "data/raw/moltbook/moltbook.db"

    logger.info("\n--- Building SocioPatterns hypergraph ---")
    hg_sp = build_sociopatterns_hypergraph(str(sp_path), delta_seconds=300)
    report_sp = compute_topology(hg_sp, name="SocioPatterns", triadic_sample=10000)
    logger.info("  %s", hg_sp.summary())

    logger.info("\n--- Building Moltbook hypergraph ---")
    if hf_posts.exists() and hf_comments.exists():
        hg_molt = build_moltbook_hypergraph_from_hf(
            str(hf_posts), str(hf_comments), delta_minutes=60, max_posts=15000)
    elif db_path.exists():
        hg_molt = build_moltbook_hypergraph_from_db(
            str(db_path), delta_minutes=60, max_posts=15000, min_comments=2)
    else:
        logger.error("No Moltbook data source found!")
        return

    logger.info("  Full Moltbook: %s", hg_molt.summary())

    # Subsample for MC tractability
    logger.info("\n--- Subsampling Moltbook to ~500 nodes ---")
    hg_molt_sub = subsample_hypergraph(hg_molt, n_target=500)
    logger.info("  Subsampled: %d nodes, %d edges",
                len(hg_molt_sub.nodes), len(hg_molt_sub.hyperedges))

    report_molt_sub = compute_topology(hg_molt_sub, name="Moltbook (sub)", triadic_sample=10000)
    logger.info("  Sub topology: closure=%.3f, Gini=%.3f, HIS=%.3f",
                report_molt_sub.triadic_closure_rate,
                report_molt_sub.hyperdegree_gini,
                report_molt_sub.his_mean)

    # ── SocioPatterns validation ──────────────────────────────────
    logger.info("\n--- SocioPatterns: ODE vs MC (full, %d nodes) ---", len(hg_sp.nodes))
    sp_results = run_ode_vs_micro(hg_sp, "SocioPatterns", report_sp,
                                   n_repeats=20, T=300)

    # ── Moltbook subsampled validation ────────────────────────────
    logger.info("\n--- Moltbook: ODE vs MC (subsampled, %d nodes) ---",
                len(hg_molt_sub.nodes))
    molt_results = run_ode_vs_micro(hg_molt_sub, "Moltbook (sub)", report_molt_sub,
                                     n_repeats=20, T=300)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n--- Summary ---")
    for name, results in [("SocioPatterns", sp_results), ("Moltbook (sub)", molt_results)]:
        rmses = [results["rho0_results"][k]["rmse"]
                 for k in results["rho0_results"]]
        in_bands = [results["rho0_results"][k]["frac_within_1sigma"]
                    for k in results["rho0_results"]]
        logger.info("  %s: mean RMSE=%.4f, mean in-band=%.1f%%",
                    name, np.mean(rmses), np.mean(in_bands) * 100)

    # ── Save ──────────────────────────────────────────────────────
    def strip_trajectories(results):
        """Keep only summary stats for JSON (trajectories too large)."""
        out = {k: v for k, v in results.items() if k != "rho0_results"}
        out["rho0_results"] = {}
        for rho_key, d in results["rho0_results"].items():
            out["rho0_results"][rho_key] = {
                k: v for k, v in d.items()
                if k not in ("ode_trajectory", "mc_mean", "mc_std")
            }
        return out

    full_results = {
        "sociopatterns": strip_trajectories(sp_results),
        "moltbook_sub": strip_trajectories(molt_results),
    }
    (outdir / "validation_results.json").write_text(
        json.dumps(full_results, indent=2, default=str), encoding="utf-8"
    )

    # ── Figure ────────────────────────────────────────────────────
    _plot_validation(sp_results, molt_results, outdir)

    logger.info("\n=== Stochastic validation complete! Results in %s ===", outdir)


if __name__ == "__main__":
    main()
