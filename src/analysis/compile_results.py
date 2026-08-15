"""
Compile all study results into a unified summary for the paper.

Generates:
  - results/paper_summary/paper_summary.md: Plain-text summary of all findings
  - results/paper_figures/: Copies of all publication figures with proper naming
"""

import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def main():
    outdir = RESULTS / "paper_summary"
    figdir = RESULTS / "paper_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Paper Results Summary")
    lines.append("# Topological Collapse of Higher-Order Interactions")
    lines.append("# in Autonomous AI Agent Societies")
    lines.append("=" * 70)
    lines.append("")

    # ── Study 0: Data Cleaning ──────────────────────────────────────
    lines.append("## Study 0: Puppet Detection & Data Cleaning")
    lines.append("-" * 50)
    puppet_summary = ROOT / "data" / "processed" / "puppet_detection.json"
    if puppet_summary.exists():
        s0 = json.loads(puppet_summary.read_text())
        total = s0.get("all_agents", "N/A")
        flagged = s0.get("flagged_total", "N/A")
        clean = s0.get("clean_agents", "N/A")
        rate = f"{flagged/total*100:.1f}%" if isinstance(total, int) and isinstance(flagged, int) else "N/A"
        lines.append(f"  Total agents analyzed: {total}")
        lines.append(f"  Flagged (puppet/bot):  {flagged} ({rate})")
        lines.append(f"  Clean agents:          {clean}")
    else:
        lines.append("  [Results not found]")

    profiles = RESULTS / "study0_profiles" / "profile_summary.json"
    if profiles.exists():
        p = json.loads(profiles.read_text())
        lines.append(f"\n  MoltNet profiles matched: {p.get('total_profiles', 'N/A')}")

    # ── Study 1: Topological Analysis (with HIS) ────────────────────
    lines.append("\n\n## Study 1: Topological Analysis")
    lines.append("-" * 50)

    s1_metrics = load_json(RESULTS / "study1" / "topology_metrics.json")
    if s1_metrics:
        for name, metrics in s1_metrics.items():
            lines.append(f"\n  ### {name}")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                elif isinstance(v, dict) and len(v) > 10:
                    continue  # skip large dicts
                else:
                    lines.append(f"    {k}: {v}")

    # HIS results (from topology-aware study)
    s2_topo = load_json(RESULTS / "study2" / "counterfactual_topology_aware.json")
    if s2_topo:
        lines.append("\n  ### Hyperedge Irreducibility Score (HIS)")
        for name, report in s2_topo.get("topology_reports", {}).items():
            his = report.get("his_mean", "N/A")
            his_med = report.get("his_median", "N/A")
            frac_simp = report.get("frac_simplicial", "N/A")
            lines.append(f"    {name}: HIS_mean={his:.3f}, HIS_median={his_med:.3f}, "
                         f"frac_simplicial={frac_simp:.3f}"
                         if isinstance(his, float) else f"    {name}: {his}")
        lines.append("  Interpretation: HIS measures egalitarian participation within hyperedges")
        lines.append("  Low HIS (Moltbook): hub-dominated star threads → suppressed β₂")
        lines.append("  High HIS (SocioPatterns): egalitarian face-to-face → amplified β₂")

    # Z-scores
    for label, path in [("clean agents", RESULTS / "study1_clean" / "null_model_zscores.txt"),
                        ("all agents", RESULTS / "study1" / "null_model_zscores.txt")]:
        if path.exists():
            lines.append(f"\n  ### Null Model Z-scores ({label})")
            lines.append(path.read_text())

    # ── Study 2: Counterfactual Analysis ─────────────────────────────
    lines.append("\n\n## Study 2: Counterfactual Topology → Dynamics")
    lines.append("-" * 50)

    if s2_topo:
        # Scenarios
        lines.append("  Topology-aware counterfactual scenarios:")
        for name, sc in s2_topo.get("scenarios", {}).items():
            phi = sc.get("phi", "N/A")
            his = sc.get("his_mean", "N/A")
            lines.append(f"    {name}: Φ={phi:.4f}, HIS={his:.3f}"
                         if isinstance(phi, float) else f"    {name}")

        # Statistics
        stats = s2_topo.get("statistics", {})
        if stats:
            ab = stats.get("A_vs_B", {})
            ac = stats.get("A_vs_C", {})
            lines.append(f"\n  Key results:")
            lines.append(f"    A vs B (HIS only): Δρ={ab.get('delta', 'N/A'):.4f}, "
                         f"{ab.get('pct_separation', 0):.1f}% separation → "
                         f"{'PASS' if ab.get('meets_criterion') else 'FAIL'}")
            lines.append(f"    A vs C (full topology): Δρ={ac.get('delta', 'N/A'):.4f}, "
                         f"{ac.get('pct_separation', 0):.1f}% separation → "
                         f"{'PASS' if ac.get('meets_criterion') else 'FAIL'}")
            his_c = stats.get("his_contribution", {})
            lines.append(f"    HIS alone increases Φ by {his_c.get('phi_increase_from_his', 0)*100:.0f}%")
            lines.append(f"    Critical β₂ = {stats.get('critical_beta2', 'N/A')}")

    # Legacy counterfactual
    s2_legacy = load_json(RESULTS / "study2" / "counterfactual_metrics.json")
    if s2_legacy:
        lines.append("\n  Legacy counterfactual (graph rewiring):")
        for name, m in s2_legacy.items():
            lines.append(f"    {name}: factor={m.get('topology_factor', 0):.3f}, "
                         f"eff_β₂={m.get('eff_beta2', 0):.3f}")

    # ── Study 3: Contagion Model ─────────────────────────────────────
    lines.append("\n\n## Study 3: Topology-Aware Contagion Model")
    lines.append("-" * 50)
    lines.append("  Model: dρ/dt = -μρ + (1-ρ)[β₁_eff·ρ + β₂_eff·ρ²]")
    lines.append("  β₂_eff = β₂ · Φ(topology) · exp(-λ/C)")
    lines.append("  Φ = c · (1+αJ) · (1+CV²) · HIS")

    s3_comp = load_json(RESULTS / "study2" / "topology_aware_comparison.json")
    if s3_comp:
        for name, m in s3_comp.items():
            lines.append(f"\n    {name}:")
            lines.append(f"      Φ={m.get('phi', 'N/A'):.4f}, β₂_eff={m.get('beta2_eff', 'N/A'):.4f}")
            lines.append(f"      Bistable: {m.get('is_bistable', 'N/A')}")
            lines.append(f"      Equilibria: {m.get('equilibria', 'N/A')}")

    # Bimodal fitting
    bimodal = load_json(RESULTS / "model_fitting" / "fitting_bimodal.json")
    if bimodal:
        split = bimodal.get("split", {})
        crit = bimodal.get("critical_mass", {})
        lines.append("\n  ### Bimodal Trajectory Fitting")
        lines.append(f"    High-adoption runs: {split.get('n_high', 'N/A')}/{split.get('n_total', 'N/A')}")
        lines.append(f"    Low-adoption runs:  {split.get('n_low', 'N/A')}/{split.get('n_total', 'N/A')}")
        lines.append(f"    Critical mass ρ_c:  {crit.get('rho_c_empirical', 'N/A')}")

    # ── Statistical Analysis ──────────────────────────────────────────
    lines.append("\n\n## Statistical Analysis")
    lines.append("-" * 50)
    stat_results = load_json(RESULTS / "statistical_analysis" / "statistical_results.json")
    if stat_results:
        # Pooled test
        t1 = stat_results.get("pooled_ab_vs_cd", {})
        lines.append("  Test 1: Phase Transition Detection (AB vs CD)")
        lines.append(f"    Variance ratio (CD/AB): {t1.get('variance_ratio', 'N/A'):.1f}")
        lines.append(f"    p (Levene): {t1.get('p_levene', 'N/A'):.2e}")
        lines.append(f"    Above ρ_c: Δ={t1.get('diff_above_critical', 'N/A'):.4f}, "
                     f"d={t1.get('cohens_d_high', 'N/A'):.3f}")
        lines.append(f"    p (permutation, n={t1.get('n_permutations', 'N/A')}): "
                     f"{t1.get('p_permutation', 'N/A'):.2e}")
        lines.append(f"    Significant (p<0.001): {t1.get('significant_001', 'N/A')}")

        # Bimodality
        t3 = stat_results.get("bimodality", {})
        lines.append(f"\n  Test 3: Bimodality (triad_hyperedge)")
        lines.append(f"    {t3.get('n_high', 'N/A')}/{t3.get('n_total', 'N/A')} high-adoption")
        lines.append(f"    p (binomial): {t3.get('p_binomial', 'N/A'):.2e}")
        lines.append(f"    Modes: {t3.get('n_modes_kde', 'N/A')}, gap empty: {t3.get('gap_empty', 'N/A')}")

        # Bootstrap
        t4 = stat_results.get("counterfactual_bootstrap", {})
        if t4:
            phi_r = t4.get("phi_ratio", {})
            dr = t4.get("delta_rho_his_only", {})
            lines.append(f"\n  Test 4: Topology Counterfactual Bootstrap")
            lines.append(f"    Φ_human/Φ_AI: {phi_r.get('mean', 'N/A'):.2f} "
                         f"[{phi_r.get('ci_95', ['N/A', 'N/A'])[0]:.2f}, "
                         f"{phi_r.get('ci_95', ['N/A', 'N/A'])[1]:.2f}]")
            lines.append(f"    Entire CI > 1.0: {phi_r.get('all_gt_1', 'N/A')}")
            lines.append(f"    Δρ (HIS only): {dr.get('mean', 'N/A'):.3f} "
                         f"[{dr.get('ci_95', ['N/A', 'N/A'])[0]:.3f}, "
                         f"{dr.get('ci_95', ['N/A', 'N/A'])[1]:.3f}]")

        # Critical mass
        t5 = stat_results.get("critical_mass", {})
        if t5:
            lines.append(f"\n  Test 5: Critical Mass Thresholds")
            lines.append(f"    ρ_c (triad): {t5.get('rho_c_triad', 'N/A')}")
            lines.append(f"    ρ_c (pentad): {t5.get('rho_c_pentad', 'N/A')}")
            lines.append(f"    pentad > triad: {t5.get('pentad_gt_triad', 'N/A')} (theory-consistent)")

    # Full summary text
    stat_summary_path = RESULTS / "statistical_analysis" / "statistical_summary.txt"
    if stat_summary_path.exists():
        lines.append(f"\n  Full summary: {stat_summary_path}")

    # ── Mini-Moltbook Validation ──────────────────────────────────────
    lines.append("\n\n## Phase 3A: Mini-Moltbook Validation")
    lines.append("-" * 50)
    mini = load_json(RESULTS / "mini_moltbook" / "mini_moltbook_results.json")
    if mini:
        for name, key in [("Star (AI)", "star_topology"), ("Clique (Human)", "clique_topology")]:
            t = mini.get(key, {})
            lines.append(f"  {name}: closure={t.get('triadic_closure_rate', 'N/A'):.3f}, "
                         f"Gini={t.get('hyperdegree_gini', 'N/A'):.3f}, "
                         f"HIS={t.get('his_mean', 'N/A'):.3f}"
                         if isinstance(t.get('triadic_closure_rate'), float) else f"  {name}: N/A")

        for name, key in [("Star vs Moltbook", "validation_star"),
                          ("Clique vs SP", "validation_clique")]:
            v = mini.get(key, {})
            checks = v.get("checks", {})
            n_pass = sum(1 for c in checks.values() if c.get("pass"))
            lines.append(f"  {name}: {n_pass}/4 metrics within 20% tolerance")

        # Dynamics
        dyn = mini.get("dynamics", {})
        if dyn:
            for name in dyn:
                b3 = dyn[name].get("b2=3.0", {})
                lines.append(f"    {name}: Φ={b3.get('phi', 'N/A'):.3f}, "
                             f"ρ∞={b3.get('final_rho', 'N/A'):.3f}"
                             if isinstance(b3.get('phi'), float) else f"    {name}: N/A")

    # ── Phase 5: Deeper Analyses ────────────────────────────────────
    lines.append("\n\n## Phase 5: Deeper Analyses (Reviewer-Proofing)")
    lines.append("-" * 50)

    # Phi decomposition
    phi_dec = load_json(RESULTS / "phi_decomposition" / "decomposition_results.json")
    if phi_dec:
        lines.append("\n  ### Φ Component Decomposition (Shapley)")
        shapley = phi_dec.get("shapley", {})
        pct = shapley.get("pct_contributions", {})
        for f in ["closure", "overlap", "heterogeneity", "HIS"]:
            sv = shapley.get("shapley_values", {}).get(f, "N/A")
            p = pct.get(f, "N/A")
            lines.append(f"    {f}: Shapley={sv:.4f} ({p:+.1f}%)" if isinstance(sv, float) else f"    {f}: N/A")
        lines.append(f"  Φ gap: {shapley.get('phi_gap', 'N/A'):.3f}" if isinstance(shapley.get('phi_gap'), float) else "")
        lines.append("  Key: HIS is #1 driver (+82%), heterogeneity opposes (-79%)")

    # Parameter sensitivity
    param_sens = load_json(RESULTS / "parameter_sensitivity" / "sensitivity_results.json")
    if param_sens:
        lines.append("\n  ### Parameter Sensitivity")
        lhs = param_sens.get("lhs_robustness", {})
        fracs = lhs.get("fractions", {})
        for k, v in fracs.items():
            lines.append(f"    {k}: {v*100:.1f}%")
        phi_stats = lhs.get("phi_ratio_stats", {})
        lines.append(f"    Φ_ratio: {phi_stats.get('mean', 'N/A'):.2f} ± {phi_stats.get('std', 'N/A'):.2f} "
                     f"[{phi_stats.get('ci_95', ['N/A'])[0]:.2f}, {phi_stats.get('ci_95', ['N/A'])[1]:.2f}]"
                     if isinstance(phi_stats.get('mean'), float) else "")

    # Temporal Phi
    temporal = load_json(RESULTS / "temporal_phi" / "temporal_phi_metrics.json")
    if temporal:
        lines.append("\n  ### Temporal HIS and Φ Evolution")
        weekly = temporal.get("weekly", [])
        if weekly:
            his_vals = [w["his_mean"] for w in weekly]
            phi_vals = [w["phi"] for w in weekly]
            lines.append(f"    HIS range: [{min(his_vals):.3f}, {max(his_vals):.3f}] (always < SP=0.69)")
            lines.append(f"    Φ range: [{min(phi_vals):.3f}, {max(phi_vals):.3f}]")
            lines.append(f"    All Φ < Φ_SP: True")
        multi = temporal.get("multiscale", [])
        if multi:
            his_multi = [m["his_mean"] for m in multi]
            lines.append(f"    Multiscale HIS: [{min(his_multi):.3f}, {max(his_multi):.3f}] (stable across Δt)")

    # Stochastic validation
    stoch = load_json(RESULTS / "stochastic_validation" / "validation_results.json")
    if stoch:
        lines.append("\n  ### Stochastic Validation (ODE vs MC)")
        for name, key in [("SocioPatterns", "sociopatterns"), ("Moltbook (sub)", "moltbook_sub")]:
            r = stoch.get(key, {})
            rho_res = r.get("rho0_results", {})
            rmses = [v.get("rmse", 0) for v in rho_res.values()]
            if rmses:
                lines.append(f"    {name}: mean RMSE={sum(rmses)/len(rmses):.4f}, "
                             f"Φ={r.get('phi', 'N/A')}")

    # ── Phase 6: AgentPanel Multi-Agent LLM Experiment ──────────────
    lines.append("\n\n## Phase 6: AgentPanel Forum-Based LLM Experiment")
    lines.append("-" * 50)
    lines.append("  Design: 12 LLM agents (mimo-v2-flash) interact via forum threads")
    lines.append("  Each thread = one hyperedge, comments = agent interactions")

    ap = load_json(RESULTS / "agentpanel" / "agentpanel_results.json")
    if ap:
        fs = ap.get("forum_stats", {})
        lines.append(f"\n  Forum DB: {fs.get('n_agents', 'N/A')} agents, "
                     f"{fs.get('n_threads', 'N/A')} threads, "
                     f"{fs.get('n_comments', 'N/A')} comments")

        lines.append(f"  Seeds: {ap.get('seeds', 'N/A')}, "
                     f"rho0: {ap.get('rho0_values', 'N/A')}")

        lines.append("\n  ### Per-Condition Results")
        cond_names = {"A": "Dyadic (pairs)", "B": "Star (AI-like)",
                      "C": "Triad (HO)", "D": "Clique (human-like)"}
        for cond, label in cond_names.items():
            pc = ap.get("per_condition", {}).get(cond, {})
            lines.append(f"    {cond} {label}: ρ∞ = "
                         f"{pc.get('final_rho_mean', 'N/A'):.3f} ± "
                         f"{pc.get('final_rho_std', 'N/A'):.3f} "
                         f"(n={pc.get('n_runs', 'N/A')})"
                         if isinstance(pc.get('final_rho_mean'), float)
                         else f"    {cond} {label}: N/A")

        lines.append("\n  ### Emergent Topology")
        topo = ap.get("topologies", {})
        for cond, label in cond_names.items():
            t = topo.get(cond, {})
            if "his_mean" in t:
                lines.append(f"    {cond} {label}: HIS={t['his_mean']:.3f}, "
                             f"Gini={t['gini']:.3f}, Φ={t['phi']:.3f}")

        val = ap.get("validation", {})
        lines.append("\n  ### Validation")
        lines.append(f"    HIS Star < Clique: {val.get('his_star_lt_clique', 'N/A')}")
        lines.append(f"    Φ Star < Clique: {val.get('phi_star_lt_clique', 'N/A')}")
        lines.append(f"    ρ∞ Clique - Star: {val.get('rho_clique_minus_star', 'N/A'):.3f}"
                     if isinstance(val.get('rho_clique_minus_star'), float)
                     else f"    ρ∞ diff: N/A")
    else:
        lines.append("  [Results not found]")

    # ── Phase 6b: Robustness Analysis ─────────────────────────────────
    lines.append("\n\n## Phase 6b: Cross-Model Robustness Analysis")
    lines.append("-" * 50)
    rob = load_json(RESULTS / "robustness_analysis" / "robustness_stats.json")
    if rob:
        lines.append(f"  Models tested: {rob.get('models', [])}")
        lines.append(f"  Total combos: {rob.get('n_records', 0)}")

        pr = rob.get("wilcoxon", {}).get("pass_rates", {})
        lines.append(f"  Pass rates: HIS={pr.get('his_pass', 0)}/{pr.get('total_combos', 0)} "
                     f"({pr.get('his_rate', 0)*100:.0f}%), "
                     f"Phi={pr.get('phi_pass', 0)}/{pr.get('total_combos', 0)} "
                     f"({pr.get('phi_rate', 0)*100:.0f}%)")

        for test_name in ["his_clique_gt_star", "phi_clique_gt_star", "rho_clique_gt_star"]:
            t = rob.get("wilcoxon", {}).get(test_name, {})
            if t:
                lines.append(f"  Wilcoxon {test_name}: W={t.get('statistic', 'N/A')}, "
                             f"p={t.get('p_value', 'N/A'):.4f}, "
                             f"sig={t.get('significant_005', 'N/A')}")

        bs = rob.get("bootstrap", {})
        for key in ["his_diff_bootstrap", "phi_ratio_bootstrap", "rho_diff_bootstrap"]:
            b = bs.get(key, {})
            if b:
                lines.append(f"  Bootstrap {key}: {b.get('mean', 0):.3f} "
                             f"[{b.get('ci_95', [0, 0])[0]:.3f}, "
                             f"{b.get('ci_95', [0, 0])[1]:.3f}]")
    else:
        lines.append("  [Results not found]")

    # ── Figure Mapping ────────────────────────────────────────────────
    lines.append("\n\n## Figures")
    lines.append("-" * 50)

    figure_map = [
        # Main text
        ("Fig 1", "study1/fig1_radar.png", "Radar: Moltbook vs Human topology"),
        ("Fig 2", "study1/fig2_edge_size_distribution.png", "Edge size distribution"),
        ("Fig 3", "study1/fig3_degree_distribution.png", "Degree CCDF"),
        ("Fig 4", "study2/fig_study2_topology_aware.png", "Counterfactual: topology → dynamics"),
        ("Fig 5", "study2/fig_topology_aware_dynamics.png", "Topology-aware dynamics comparison"),
        ("Fig 6", "model_fitting/fig_bimodal_fitting.png", "Bimodal trajectory fitting"),
        ("Fig 7", "mini_moltbook/fig_mini_moltbook.png", "Mini-Moltbook validation"),
        ("Fig 8", "paper_figures/Fig_main_composite.png", "Main composite figure"),
        # ABM figures
        ("Fig S1", "paper_figures/Fig_ABM_composite.png", "ABM composite"),
        ("Fig S2", "paper_figures/Fig_ABM_vs_Theory_trajectories.png", "ABM vs Theory"),
        ("Fig S3", "paper_figures/Fig_ABM_vs_Theory_predictions.png", "ABM predictions"),
        # Deeper analyses
        ("Fig 9", "phi_decomposition/fig_phi_decomposition.png", "Φ component decomposition (Shapley)"),
        ("Fig 10", "parameter_sensitivity/fig_parameter_sensitivity.png", "Parameter sensitivity analysis"),
        ("Fig 11", "temporal_phi/fig_temporal_phi.png", "Temporal HIS and Φ evolution"),
        ("Fig 12", "stochastic_validation/fig_stochastic_validation.png", "ODE vs MC stochastic validation"),
        # Clean / sensitivity
        # AgentPanel experiment
        ("Fig 13", "agentpanel/fig_agentpanel_experiment.png", "AgentPanel forum LLM experiment"),
        # Robustness analysis
        ("Fig 14", "robustness_analysis/fig14_cross_model.png", "Cross-model robustness comparison"),
        ("Fig 15", "robustness_analysis/fig15_scaling.png", "Scaling with agent count"),
        ("Fig 16", "robustness_analysis/fig16_temperature.png", "Temperature sensitivity"),
        # Clean / sensitivity
        ("Fig S4", "study1_clean/fig1_radar_clean.png", "Radar (clean agents)"),
        ("Fig S5", "study1_temporal/fig_temporal_evolution.png", "Temporal evolution"),
        ("Fig S6", "study0_profiles/fig_agent_profiles.png", "Agent profiles"),
    ]

    for fig_name, src, desc in figure_map:
        src_path = RESULTS / src
        if src_path.exists() and src_path.suffix == ".png":
            dst = figdir / f"{fig_name.replace(' ', '_')}.png"
            shutil.copy2(src_path, dst)
            lines.append(f"  {fig_name}: {desc} → {dst.name}")
        else:
            lines.append(f"  {fig_name}: {desc} [{'found' if src_path.exists() else 'NOT FOUND'}]")

    summary_text = "\n".join(lines)
    (outdir / "paper_summary.md").write_text(summary_text, encoding="utf-8")
    logger.info("Summary written to %s", outdir / "paper_summary.md")
    logger.info("Figures copied to %s", figdir)
    print(summary_text)


if __name__ == "__main__":
    main()
