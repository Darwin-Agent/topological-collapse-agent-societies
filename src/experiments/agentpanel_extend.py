"""
Extended AgentPanel experiment: additional seeds for stronger statistics.
Appends to the existing forum.db and results.
"""

import asyncio
import json
import logging
import sys
import time
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

from src.experiments.agentpanel_experiment import (
    init_db, run_condition, analyze_topology, compare_with_ode,
    build_hypergraph_from_forum_db, forum_stats, _plot_results,
    MODEL,
)


async def async_main():
    outdir = ROOT / "results" / "agentpanel"
    db_path = str(outdir / "forum.db")
    conn = init_db(db_path)

    logger.info("=" * 60)
    logger.info("AgentPanel Extended Run: Additional Seeds")
    logger.info("=" * 60)

    # Load existing results
    existing_path = outdir / "agentpanel_results.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    existing_trajs = existing.get("per_condition", {})

    conditions = ["A", "B", "C", "D"]
    rho0_values = [0.10, 0.25, 0.50]
    n_rounds = 12
    new_seeds = [777, 999, 2024]

    all_results = []

    # Reconstruct prior results from saved per-run files
    for cond in conditions:
        for rho0 in rho0_values:
            for seed in [42, 123]:
                f = outdir / f"run_{cond}_rho{rho0:.2f}_s{seed}.json"
                if f.exists():
                    d = json.loads(f.read_text())
                    all_results.append({
                        "condition": d["condition"],
                        "rho0": d["rho0"],
                        "seed": d["seed"],
                        "n_rounds": n_rounds,
                        "rho_trajectory": d["rho_trajectory"],
                        "final_rho": d["final_rho"],
                        "final_scores": d.get("final_scores", []),
                        "interactions": [],  # not stored in per-run
                    })

    t_start = time.time()

    for cond in conditions:
        logger.info("\n>>> Condition %s <<<", cond)
        for rho0 in rho0_values:
            for seed in new_seeds:
                result = await run_condition(
                    cond, conn, category_id=1,
                    n_rounds=n_rounds, rho0=rho0, seed=seed,
                )
                all_results.append(result)

                (outdir / f"run_{cond}_rho{rho0:.2f}_s{seed}.json").write_text(
                    json.dumps({
                        "condition": cond, "rho0": rho0, "seed": seed,
                        "rho_trajectory": result["rho_trajectory"],
                        "final_rho": result["final_rho"],
                        "final_scores": result["final_scores"],
                    }, indent=2), encoding="utf-8",
                )

    elapsed = time.time() - t_start
    logger.info("\n=== Extended runs complete: %.1f min ===", elapsed / 60)

    # Forum stats
    stats = forum_stats(conn)
    logger.info("Forum DB: %d agents, %d threads, %d comments",
                stats["n_agents"], stats["n_threads"], stats["n_comments"])

    # Topology from all interactions (DB-based for extended)
    logger.info("\n--- Topology Analysis (all data) ---")
    topologies = {}
    for cond in conditions:
        hg = build_hypergraph_from_forum_db(conn, cond)
        if len(hg.hyperedges) >= 5:
            from src.analysis.topology import compute_topology as ct
            from src.models.contagion_ho import TopologyAwareContagionModel as TACM
            report = ct(hg, name=cond, triadic_sample=5000)
            model = TACM.from_topology_report(report)
            topologies[cond] = {
                "n_nodes": report.n_nodes,
                "n_edges": report.n_edges,
                "his_mean": report.his_mean,
                "his_median": report.his_median,
                "frac_simplicial": report.frac_simplicial,
                "triadic_closure": report.triadic_closure_rate,
                "gini": report.hyperdegree_gini,
                "overlap": report.mean_edge_overlap,
                "edge_size_mean": report.edge_size_mean,
                "frac_higher_order": report.frac_higher_order,
                "phi": float(model.phi),
                "beta2_eff": float(model.beta2_eff),
            }
            logger.info("  %s: HIS=%.3f Gini=%.3f closure=%.3f Φ=%.3f",
                        cond, report.his_mean, report.hyperdegree_gini,
                        report.triadic_closure_rate, float(model.phi))

    # ODE comparison
    ode_comparisons = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topo = topologies.get(cond, {})
        if "error" not in topo and topo:
            ode_comparisons[cond] = compare_with_ode(cond_results, topo)

    # Summary
    logger.info("\n--- Summary (all %d runs) ---", len(all_results))
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in runs]
        logger.info("  %s: ρ∞ = %.3f ± %.3f (n=%d)",
                    cond, np.mean(finals), np.std(finals), len(finals))

    # Validation
    logger.info("\n--- Validation ---")
    his_b = topologies.get("B", {}).get("his_mean", 0)
    his_d = topologies.get("D", {}).get("his_mean", 0)
    phi_b = topologies.get("B", {}).get("phi", 0)
    phi_d = topologies.get("D", {}).get("phi", 0)
    rho_b = np.mean([r["final_rho"] for r in all_results if r["condition"] == "B"])
    rho_d = np.mean([r["final_rho"] for r in all_results if r["condition"] == "D"])
    logger.info("  HIS: Star=%.3f < Clique=%.3f → %s",
                his_b, his_d, "PASS" if his_b < his_d else "FAIL")
    logger.info("  Φ: Star=%.3f < Clique=%.3f → %s",
                phi_b, phi_d, "PASS" if phi_b < phi_d else "FAIL")
    logger.info("  ρ∞: Star=%.3f, Clique=%.3f, diff=%.3f",
                rho_b, rho_d, rho_d - rho_b)

    # Mann-Whitney U test: Star vs Clique final rho
    from scipy.stats import mannwhitneyu
    star_finals = [r["final_rho"] for r in all_results if r["condition"] == "B"]
    clique_finals = [r["final_rho"] for r in all_results if r["condition"] == "D"]
    if len(star_finals) >= 3 and len(clique_finals) >= 3:
        u_stat, p_val = mannwhitneyu(clique_finals, star_finals, alternative="greater")
        logger.info("  Mann-Whitney U (Clique > Star): U=%.1f, p=%.4f", u_stat, p_val)

    # Save combined results
    summary = {
        "experiment": "agentpanel_forum_topology_experiment_extended",
        "model": MODEL,
        "n_agents": 12,
        "n_rounds": n_rounds,
        "conditions": conditions,
        "rho0_values": rho0_values,
        "seeds": [42, 123, 777, 999, 2024],
        "elapsed_seconds_extended": elapsed,
        "forum_stats": stats,
        "topologies": topologies,
        "validation": {
            "his_star_lt_clique": his_b < his_d,
            "phi_star_lt_clique": phi_b < phi_d,
            "rho_clique_minus_star": float(rho_d - rho_b),
        },
        "per_condition": {},
    }
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in runs]
        summary["per_condition"][cond] = {
            "n_runs": len(runs),
            "final_rho_mean": float(np.mean(finals)),
            "final_rho_std": float(np.std(finals)),
            "trajectories": [r["rho_trajectory"] for r in runs],
        }

    (outdir / "agentpanel_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    (outdir / "agentpanel_topology.json").write_text(
        json.dumps(topologies, indent=2, default=str), encoding="utf-8",
    )

    # Regenerate figure with all data
    _plot_results(all_results, topologies, ode_comparisons, outdir)

    conn.close()
    logger.info("\n=== Extended experiment complete! ===")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
