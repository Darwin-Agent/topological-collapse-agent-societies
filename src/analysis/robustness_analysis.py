"""
Phase 6b Robustness Analysis: Cross-model comparison, scaling, temperature sensitivity.

Generates:
  Fig 14: Cross-model comparison (HIS, Phi, rho by model)
  Fig 15: Scaling analysis (HIS, Phi, rho vs n_agents)
  Fig 16: Temperature sensitivity (HIS, Phi, rho vs temperature)
  Statistical tests: Wilcoxon signed-rank, bootstrap CIs
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def load_sweep(sweep_dir: Path) -> dict | None:
    """Load sweep_results.json from a robustness sweep directory."""
    f = sweep_dir / "sweep_results.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


def collect_all_sweeps() -> dict:
    """Collect results from all robustness sweep directories."""
    dirs = {
        "deepseek-v3.1": RESULTS / "agentpanel_gpt4o",
        "Qwen2.5-72B": RESULTS / "agentpanel_claude",
        "mimo-v2-flash": RESULTS / "agentpanel_mimo_extended",
        "gpt-5": RESULTS / "agentpanel_gpt5",
        "gemini-2.5-pro": RESULTS / "agentpanel_gemini",
        "claude-sonnet-4-6": RESULTS / "agentpanel_claude_s46",
    }
    sweeps = {}
    for label, d in dirs.items():
        data = load_sweep(d)
        if data:
            sweeps[label] = data
            logger.info("Loaded %s: %d combos", label, len(data.get("combos", [])))
        else:
            logger.warning("Missing: %s", d)

    # Also load original Phase 6 baseline
    baseline = RESULTS / "agentpanel" / "agentpanel_results.json"
    if baseline.exists():
        sweeps["mimo-v2-flash (baseline)"] = {
            "model": "mimo-v2-flash",
            "combos": [{
                "n_agents": 12, "temperature": 0.8,
                **json.loads(baseline.read_text()),
            }],
        }
        logger.info("Loaded baseline: 1 combo")

    return sweeps


def extract_metrics(sweeps: dict) -> list[dict]:
    """Flatten sweep results into a list of records."""
    records = []
    for label, sweep in sweeps.items():
        model = sweep.get("model", label)
        for combo in sweep.get("combos", []):
            rec = {
                "label": label,
                "model": model,
                "n_agents": combo.get("n_agents", 12),
                "temperature": combo.get("temperature", 0.8),
                "his_star": combo.get("his_star", combo.get("topologies", {}).get("B", {}).get("his_mean", 0)),
                "his_clique": combo.get("his_clique", combo.get("topologies", {}).get("D", {}).get("his_mean", 0)),
                "phi_star": combo.get("phi_star", combo.get("topologies", {}).get("B", {}).get("phi", 0)),
                "phi_clique": combo.get("phi_clique", combo.get("topologies", {}).get("D", {}).get("phi", 0)),
                "rho_star": combo.get("rho_star", combo.get("per_condition", {}).get("B", {}).get("final_rho_mean", 0)),
                "rho_clique": combo.get("rho_clique", combo.get("per_condition", {}).get("D", {}).get("final_rho_mean", 0)),
                "his_pass": combo.get("his_pass", False),
                "phi_pass": combo.get("phi_pass", False),
            }
            rec["rho_diff"] = rec["rho_clique"] - rec["rho_star"]
            rec["his_diff"] = rec["his_clique"] - rec["his_star"]
            rec["phi_ratio"] = rec["phi_clique"] / rec["phi_star"] if rec["phi_star"] > 0 else float("inf")
            records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════

def wilcoxon_tests(records: list[dict]) -> dict:
    """Wilcoxon signed-rank tests: Star vs Clique across all conditions."""
    his_diffs = [r["his_diff"] for r in records if r["his_diff"] != 0]
    phi_diffs = [r["phi_ratio"] - 1 for r in records if r["phi_ratio"] != float("inf")]
    rho_diffs = [r["rho_diff"] for r in records]

    results = {}

    # HIS: Clique > Star
    if len(his_diffs) >= 5:
        stat, p = sp_stats.wilcoxon(his_diffs, alternative="greater")
        results["his_clique_gt_star"] = {
            "statistic": float(stat), "p_value": float(p),
            "n": len(his_diffs),
            "significant_005": p < 0.05,
            "median_diff": float(np.median(his_diffs)),
        }

    # Phi: Clique > Star
    if len(phi_diffs) >= 5:
        stat, p = sp_stats.wilcoxon(phi_diffs, alternative="greater")
        results["phi_clique_gt_star"] = {
            "statistic": float(stat), "p_value": float(p),
            "n": len(phi_diffs),
            "significant_005": p < 0.05,
            "median_ratio": float(np.median([r["phi_ratio"] for r in records if r["phi_ratio"] != float("inf")])),
        }

    # Rho: Clique > Star
    if len(rho_diffs) >= 5:
        stat, p = sp_stats.wilcoxon(rho_diffs, alternative="greater")
        results["rho_clique_gt_star"] = {
            "statistic": float(stat), "p_value": float(p),
            "n": len(rho_diffs),
            "significant_005": p < 0.05,
            "median_diff": float(np.median(rho_diffs)),
        }

    # Pass rate
    n_total = len(records)
    n_his_pass = sum(1 for r in records if r["his_pass"])
    n_phi_pass = sum(1 for r in records if r["phi_pass"])
    n_both_pass = sum(1 for r in records if r["his_pass"] and r["phi_pass"])
    results["pass_rates"] = {
        "total_combos": n_total,
        "his_pass": n_his_pass,
        "phi_pass": n_phi_pass,
        "both_pass": n_both_pass,
        "his_rate": n_his_pass / n_total if n_total else 0,
        "phi_rate": n_phi_pass / n_total if n_total else 0,
        "both_rate": n_both_pass / n_total if n_total else 0,
    }

    return results


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval."""
    arr = np.array(values)
    rng = np.random.default_rng(42)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_95": [float(lo), float(hi)],
        "n": len(arr),
    }


def bootstrap_analysis(records: list[dict]) -> dict:
    """Bootstrap CIs for key metrics across all conditions."""
    results = {}

    his_diffs = [r["his_diff"] for r in records]
    phi_ratios = [r["phi_ratio"] for r in records if r["phi_ratio"] != float("inf")]
    rho_diffs = [r["rho_diff"] for r in records]

    if his_diffs:
        results["his_diff_bootstrap"] = bootstrap_ci(his_diffs)
    if phi_ratios:
        results["phi_ratio_bootstrap"] = bootstrap_ci(phi_ratios)
    if rho_diffs:
        results["rho_diff_bootstrap"] = bootstrap_ci(rho_diffs)

    # Per-model bootstrap
    models = set(r["label"] for r in records)
    results["per_model"] = {}
    for model in sorted(models):
        model_recs = [r for r in records if r["label"] == model]
        if len(model_recs) >= 3:
            results["per_model"][model] = {
                "his_diff": bootstrap_ci([r["his_diff"] for r in model_recs]),
                "phi_ratio": bootstrap_ci([r["phi_ratio"] for r in model_recs
                                           if r["phi_ratio"] != float("inf")]),
                "rho_diff": bootstrap_ci([r["rho_diff"] for r in model_recs]),
            }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════

def plot_fig14_cross_model(records: list[dict], outdir: Path):
    """Fig 14: Cross-model comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = sorted(set(r["label"] for r in records))
    _palette = ["#E24A33", "#348ABD", "#2ca02c", "#FFA500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    colors = {m: _palette[i % len(_palette)] for i, m in enumerate(models)}

    # Panel A: HIS Star vs Clique by model
    ax = axes[0]
    x_pos = np.arange(len(models))
    width = 0.35
    for i, model in enumerate(models):
        recs = [r for r in records if r["label"] == model]
        his_star = [r["his_star"] for r in recs]
        his_clique = [r["his_clique"] for r in recs]
        ax.bar(i - width/2, np.mean(his_star), width, yerr=np.std(his_star),
               color=colors[model], alpha=0.5, edgecolor="black", linewidth=0.5)
        ax.bar(i + width/2, np.mean(his_clique), width, yerr=np.std(his_clique),
               color=colors[model], alpha=0.9, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.split(" ")[0][:12] for m in models], fontsize=8, rotation=15)
    ax.set_ylabel("HIS")
    ax.set_title("A. HIS: Star (light) vs Clique (dark)")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: Phi ratio by model
    ax = axes[1]
    for i, model in enumerate(models):
        recs = [r for r in records if r["label"] == model]
        ratios = [r["phi_ratio"] for r in recs if r["phi_ratio"] != float("inf")]
        if ratios:
            bp = ax.boxplot([ratios], positions=[i], widths=0.5, patch_artist=True)
            bp["boxes"][0].set_facecolor(colors[model])
            bp["boxes"][0].set_alpha(0.7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Ratio = 1")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split(" ")[0][:12] for m in models], fontsize=8, rotation=15)
    ax.set_ylabel("Phi_Clique / Phi_Star")
    ax.set_title("B. Topology amplification ratio")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Final rho difference by model
    ax = axes[2]
    for i, model in enumerate(models):
        recs = [r for r in records if r["label"] == model]
        diffs = [r["rho_diff"] for r in recs]
        bp = ax.boxplot([diffs], positions=[i], widths=0.5, patch_artist=True)
        bp["boxes"][0].set_facecolor(colors[model])
        bp["boxes"][0].set_alpha(0.7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split(" ")[0][:12] for m in models], fontsize=8, rotation=15)
    ax.set_ylabel("rho_Clique - rho_Star")
    ax.set_title("C. Adoption advantage (Clique vs Star)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 14. Cross-Model Robustness: Topology -> Dynamics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig14_cross_model.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig14_cross_model.png")


def plot_fig15_scaling(records: list[dict], outdir: Path):
    """Fig 15: Scaling with n_agents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = sorted(set(r["label"] for r in records))
    _palette = ["#E24A33", "#348ABD", "#2ca02c", "#FFA500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    _markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = {m: _palette[i % len(_palette)] for i, m in enumerate(models)}
    markers = {m: _markers[i % len(_markers)] for i, m in enumerate(models)}

    agent_counts = sorted(set(r["n_agents"] for r in records))

    for model in models:
        recs = [r for r in records if r["label"] == model]
        for ax, metric, ylabel, title_letter in [
            (axes[0], "his_diff", "HIS_Clique - HIS_Star", "A"),
            (axes[1], "phi_ratio", "Phi_Clique / Phi_Star", "B"),
            (axes[2], "rho_diff", "rho_Clique - rho_Star", "C"),
        ]:
            means, stds, xs = [], [], []
            for n in agent_counts:
                vals = [r[metric] for r in recs if r["n_agents"] == n
                        and r[metric] != float("inf")]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    xs.append(n)
            if means:
                ax.errorbar(xs, means, yerr=stds, fmt=f"{markers[model]}-",
                            color=colors[model], label=model.split(" ")[0][:12],
                            linewidth=1.5, markersize=6, capsize=3)

    for ax, ylabel, title_letter, ref in [
        (axes[0], "HIS_Clique - HIS_Star", "A", 0),
        (axes[1], "Phi_Clique / Phi_Star", "B", 1),
        (axes[2], "rho_Clique - rho_Star", "C", 0),
    ]:
        ax.axhline(ref, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_letter}. {ylabel} vs Scale")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Fig 15. Scaling Robustness: Effect of Agent Count",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig15_scaling.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig15_scaling.png")


def plot_fig16_temperature(records: list[dict], outdir: Path):
    """Fig 16: Temperature sensitivity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models = sorted(set(r["label"] for r in records))
    _palette = ["#E24A33", "#348ABD", "#2ca02c", "#FFA500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    _markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = {m: _palette[i % len(_palette)] for i, m in enumerate(models)}
    markers = {m: _markers[i % len(_markers)] for i, m in enumerate(models)}

    temps = sorted(set(r["temperature"] for r in records))

    for model in models:
        recs = [r for r in records if r["label"] == model]
        for ax, metric, ylabel in [
            (axes[0], "his_diff", "HIS_Clique - HIS_Star"),
            (axes[1], "phi_ratio", "Phi_Clique / Phi_Star"),
            (axes[2], "rho_diff", "rho_Clique - rho_Star"),
        ]:
            means, stds, xs = [], [], []
            for t in temps:
                vals = [r[metric] for r in recs if r["temperature"] == t
                        and r[metric] != float("inf")]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    xs.append(t)
            if means:
                ax.errorbar(xs, means, yerr=stds, fmt=f"{markers[model]}-",
                            color=colors[model], label=model.split(" ")[0][:12],
                            linewidth=1.5, markersize=6, capsize=3)

    for ax, ylabel, ref in [
        (axes[0], "HIS_Clique - HIS_Star", 0),
        (axes[1], "Phi_Clique / Phi_Star", 1),
        (axes[2], "rho_Clique - rho_Star", 0),
    ]:
        ax.axhline(ref, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Temperature")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Fig 16. Temperature Sensitivity: LLM Stochasticity Effects",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig16_temperature.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: fig16_temperature.png")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    outdir = RESULTS / "robustness_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 6b Robustness Analysis")
    logger.info("=" * 60)

    # Collect data
    sweeps = collect_all_sweeps()
    if not sweeps:
        logger.error("No sweep data found!")
        sys.exit(1)

    records = extract_metrics(sweeps)
    logger.info("Total records: %d", len(records))

    # Statistical tests
    logger.info("\n--- Wilcoxon Signed-Rank Tests ---")
    wilcoxon = wilcoxon_tests(records)
    for test_name, result in wilcoxon.items():
        if isinstance(result, dict) and "p_value" in result:
            logger.info("  %s: W=%.1f, p=%.4f, sig=%s",
                        test_name, result["statistic"], result["p_value"],
                        result["significant_005"])

    pr = wilcoxon.get("pass_rates", {})
    logger.info("  Pass rates: HIS=%d/%d (%.0f%%), Phi=%d/%d (%.0f%%), Both=%d/%d (%.0f%%)",
                pr.get("his_pass", 0), pr.get("total_combos", 0),
                pr.get("his_rate", 0) * 100,
                pr.get("phi_pass", 0), pr.get("total_combos", 0),
                pr.get("phi_rate", 0) * 100,
                pr.get("both_pass", 0), pr.get("total_combos", 0),
                pr.get("both_rate", 0) * 100)

    logger.info("\n--- Bootstrap Analysis ---")
    bootstrap = bootstrap_analysis(records)
    for key, val in bootstrap.items():
        if key != "per_model" and isinstance(val, dict):
            logger.info("  %s: %.3f [%.3f, %.3f]",
                        key, val["mean"], val["ci_95"][0], val["ci_95"][1])

    # Save results
    all_stats = {
        "n_records": len(records),
        "models": list(set(r["label"] for r in records)),
        "wilcoxon": wilcoxon,
        "bootstrap": bootstrap,
        "records": records,
    }
    (outdir / "robustness_stats.json").write_text(
        json.dumps(all_stats, indent=2, default=str), encoding="utf-8",
    )

    # Figures
    logger.info("\n--- Generating Figures ---")
    plot_fig14_cross_model(records, outdir)
    plot_fig15_scaling(records, outdir)
    plot_fig16_temperature(records, outdir)

    logger.info("\n=== Robustness analysis complete! ===")
    logger.info("Output: %s", outdir)


if __name__ == "__main__":
    main()
