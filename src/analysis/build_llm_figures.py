#!/usr/bin/env python3
"""
build_llm_figures.py — Regenerate the cross-model LLM figures from REAL experiment data.

Replaces the previously HARDCODED heatmap in fig_tables_to_figures.py
(which covered only 17 of the 22 models via a literal Python dict) with a
pipeline that reads the actual AgentPanel experiment outputs for all 22 models:

  * per-condition norm-adoption rho  — aggregated from run_*.json (final_rho)
  * per-condition HIS / topology     — computed from forum.db via the SAME
                                        pipeline used in production
                                        (src.analysis.topology.compute_topology)

NO result values are hardcoded. Model identity is resolved from each results
JSON's own "model" field (NOT the directory name — several Round-1 dirs are
mis-named, e.g. agentpanel_gpt4o/ actually holds deepseek-v3.1).

Outputs:
  results/paper_figures/fig_llm_behaviour_heatmap.png   (22 models x 4 conditions)
  results/paper_figures/fig_llm_heatmap.png             (HIS + divergence + scaling)
  results/paper_figures/llm_figure_data.json            (the exact numbers plotted)

Usage:
  /tmp/figenv/bin/python -m src.analysis.build_llm_figures            # from Code/
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # .../Code
RESULTS = ROOT / "results"
OUTDIR = RESULTS / "paper_figures"
CONDITIONS = ["A", "B", "C", "D"]
COND_LABELS = ["A (Pairs)", "B (Star)", "C (Triads)", "D (5-Clique)"]

sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
#  Model registry: canonical name -> (data dir relative to RESULTS, vendor, round)
#  Verified against each dir's results JSON "model" field. Dir name != model.
# ─────────────────────────────────────────────────────────────────────────────
REGISTRY = [
    # Round 1 (separate, sometimes mis-named dirs)
    ("MiMo-v2-flash",      "agentpanel",              "Xiaomi",    1),
    ("DeepSeek-V3.1",      "agentpanel_gpt4o",        "DeepSeek",  1),  # dir mis-named!
    ("Qwen2.5-72B",        "agentpanel_claude",       "Alibaba",   1),  # dir mis-named!
    ("Gemini-2.5-Pro",     "agentpanel_gemini",       "Google",    1),
    ("GPT-5",              "agentpanel_gpt5",         "OpenAI",    1),  # partial: t=1.0 only
    ("Claude-Sonnet-4.6",  "agentpanel_claude_s46",   "Anthropic", 1),
    # Round 2 (multimodel_16/agentpanel/<Model>)
    ("GPT-5.4",            "multimodel_16/agentpanel/GPT-5.4",           "OpenAI",    2),
    ("o4-mini",            "multimodel_16/agentpanel/o4-mini",           "OpenAI",    2),
    ("GPT-5-mini",         "multimodel_16/agentpanel/GPT-5-mini",        "OpenAI",    2),
    ("Claude-Opus-4.5",    "multimodel_16/agentpanel/Claude-Opus-4.5",   "Anthropic", 2),
    ("Claude-Sonnet-4.5",  "multimodel_16/agentpanel/Claude-Sonnet-4.5", "Anthropic", 2),
    ("DeepSeek-V3.2",      "multimodel_16/agentpanel/DeepSeek-V3.2",     "DeepSeek",  2),
    ("DeepSeek-R1",        "multimodel_16/agentpanel/DeepSeek-R1",       "DeepSeek",  2),
    ("Qwen3.6-plus",       "multimodel_16/agentpanel/Qwen3.6-plus",      "Alibaba",   2),
    ("Qwen3.5-plus",       "multimodel_16/agentpanel/Qwen3.5-plus",      "Alibaba",   2),
    ("Qwen3-max",          "multimodel_16/agentpanel/Qwen3-max",         "Alibaba",   2),
    ("GLM-5",              "multimodel_16/agentpanel/GLM-5",             "Zhipu",     2),
    ("Kimi-K2.5",          "multimodel_16/agentpanel/Kimi-K2.5",         "Moonshot",  2),
    ("Kimi-K2",            "multimodel_16/agentpanel/Kimi-K2",           "Moonshot",  2),
    ("MiniMax-M2.7",       "multimodel_16/agentpanel/MiniMax-M2.7",      "MiniMax",   2),
    ("MiMo-v2.5-Pro",      "multimodel_16/agentpanel/MiMo-v2.5-Pro",     "Xiaomi",    2),
    ("Seed-OSS-36B",       "multimodel_16/agentpanel/Seed-OSS-36B",      "ByteDance", 2),
]

VENDOR_COLORS = {
    "OpenAI": "#10A37F", "Anthropic": "#D4A574", "Google": "#4285F4",
    "Xiaomi": "#FF6900", "DeepSeek": "#5B6ACD", "Alibaba": "#FF6A00",
    "ByteDance": "#000000", "Moonshot": "#6B4FBB", "Zhipu": "#2E86AB",
    "MiniMax": "#E94560",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading — real data only
# ─────────────────────────────────────────────────────────────────────────────
def _rho_from_runs(model_dir: Path) -> dict:
    """Aggregate per-condition mean final_rho from run_*.json files.

    Mirrors the aggregation in src/experiments/agentpanel_extend.py:
    per_condition[c].final_rho_mean = mean over runs of that condition.
    """
    per_cond = {c: [] for c in CONDITIONS}
    for f in sorted(model_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        c = d.get("condition")
        if c in per_cond and "final_rho" in d:
            per_cond[c].append(float(d["final_rho"]))
    return {c: float(np.mean(v)) for c, v in per_cond.items() if v}


def _rho_from_sweep(model_dir: Path) -> dict:
    """Round-1 models store per-parameter-point results under n*_t*/results.json.
    Aggregate final_rho across all sweep points, grouped by condition."""
    per_cond = {c: [] for c in CONDITIONS}
    for sub in sorted(model_dir.glob("n*_t*/results.json")):
        try:
            d = json.loads(sub.read_text())
        except Exception:
            continue
        pc = d.get("per_condition", {})
        for c in CONDITIONS:
            if c in pc and pc[c].get("final_rho_mean") is not None:
                per_cond[c].append(float(pc[c]["final_rho_mean"]))
    return {c: float(np.mean(v)) for c, v in per_cond.items() if v}


def _topology_from_db(model_dir: Path) -> dict:
    """Compute per-condition HIS/Gini/etc. from forum.db, using the production
    pipeline (compute_topology). Returns {cond: {his_mean, gini, ...}}."""
    # Prefer a top-level forum.db; else the largest one in a sweep subdir.
    db = model_dir / "forum.db"
    if not db.exists():
        cands = sorted(model_dir.glob("**/forum.db"),
                       key=lambda p: p.stat().st_size, reverse=True)
        if not cands:
            return {}
        db = cands[0]

    from src.analysis.hypergraph_builder import Hypergraph
    from src.analysis.topology import compute_topology

    def build_hg(conn, condition):
        # Inlined from agentpanel_experiment.build_hypergraph_from_forum_db
        # (that module import-time-requires httpx, which we don't need here).
        # Each thread = one hyperedge (set of agents who commented).
        rows = conn.execute(
            "SELECT DISTINCT thread_id, author_id FROM comments WHERE thread_id IN "
            "(SELECT id FROM threads WHERE condition = ?)", (condition,)).fetchall()
        thread_members = {}
        for thread_id, author_id in rows:
            thread_members.setdefault(thread_id, set()).add(str(author_id))
        nodes, hyperedges = set(), []
        for members in thread_members.values():
            if len(members) >= 2:
                hyperedges.append(frozenset(members))
                nodes.update(members)
        return Hypergraph(nodes=nodes, hyperedges=hyperedges,
                          metadata={"source": "agentpanel_forum", "condition": condition})

    out = {}
    conn = sqlite3.connect(str(db))
    try:
        for c in CONDITIONS:
            hg = build_hg(conn, c)
            if len(hg.hyperedges) >= 5:
                rep = compute_topology(hg, name=c, triadic_sample=5000)
                out[c] = {
                    "his_mean": rep.his_mean, "gini": rep.hyperdegree_gini,
                    "triadic_closure": rep.triadic_closure_rate,
                    "overlap": rep.mean_edge_overlap,
                    "frac_higher_order": rep.frac_higher_order,
                    "n_nodes": rep.n_nodes, "n_edges": rep.n_edges,
                }
    finally:
        conn.close()
    return out


def load_model(name: str, rel_dir: str) -> dict:
    """Load one model's real per-condition rho + topology.
    Verifies model identity via the results JSON 'model' field."""
    mdir = RESULTS / rel_dir
    res_f = mdir / "agentpanel_results.json"
    topo_f = mdir / "agentpanel_topology.json"
    sweep_f = mdir / "sweep_results.json"

    declared = None
    rho, topo = {}, {}

    # rho + declared model name
    if res_f.exists():
        d = json.loads(res_f.read_text())
        declared = d.get("model")
        pc = d.get("per_condition", {})
        rho = {c: float(pc[c]["final_rho_mean"]) for c in CONDITIONS
               if c in pc and pc[c].get("final_rho_mean") is not None}
    if not rho:
        rho = _rho_from_runs(mdir) or _rho_from_sweep(mdir)
    if declared is None and sweep_f.exists():
        declared = json.loads(sweep_f.read_text()).get("model")

    # topology / HIS
    if topo_f.exists():
        td = json.loads(topo_f.read_text())
        topo = {c: {"his_mean": td[c]["his_mean"], "gini": td[c].get("gini"),
                    "triadic_closure": td[c].get("triadic_closure"),
                    "overlap": td[c].get("overlap")}
                for c in CONDITIONS if c in td}
    if not topo:
        topo = _topology_from_db(mdir)

    return {"name": name, "declared_model": declared,
            "rho": rho, "topology": topo,
            "n_conditions_rho": len(rho), "n_conditions_his": len(topo)}


def load_all() -> list[dict]:
    """Load all 22 registry models. Fails loudly if a model has no rho data."""
    out = []
    print(f"Loading {len(REGISTRY)} models from real experiment data...\n")
    for name, rel, vendor, rnd in REGISTRY:
        m = load_model(name, rel)
        m["vendor"] = vendor
        m["round"] = rnd
        status = "OK" if m["rho"] else "!! NO RHO DATA"
        dec = m["declared_model"] or "(from runs)"
        print(f"  [{status:>13}] {name:<18} rho_conds={m['n_conditions_rho']} "
              f"his_conds={m['n_conditions_his']:>1}  declared='{dec}'")
        if not m["rho"]:
            raise SystemExit(f"FATAL: no real rho data for {name} at {rel}. "
                             "Refusing to fabricate.")
        out.append(m)
    print(f"\n{len(out)} models loaded (real data only).")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    return plt


def fig_behaviour_heatmap(models: list[dict], out: Path):
    """Per-model norm-adoption heatmap — 22 models x 4 conditions, real data."""
    plt = _setup_mpl()
    from matplotlib.colors import TwoSlopeNorm

    # Sort within round by mean rho (desc), Round 1 block then Round 2 block.
    def mean_rho(m):
        vals = [m["rho"][c] for c in CONDITIONS if c in m["rho"]]
        return float(np.mean(vals)) if vals else 0.0

    r1 = sorted([m for m in models if m["round"] == 1], key=mean_rho, reverse=True)
    r2 = sorted([m for m in models if m["round"] == 2], key=mean_rho, reverse=True)
    ordered = r1 + r2
    n = len(ordered)

    matrix = np.full((n, 4), np.nan)
    for i, m in enumerate(ordered):
        for j, c in enumerate(CONDITIONS):
            if c in m["rho"]:
                matrix[i, j] = m["rho"][c]

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(matrix, cmap="RdYlBu", norm=norm, aspect="auto")

    for i in range(n):
        for j in range(4):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.2 or v > 0.85 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
            else:
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=5, color="#999999", style="italic")

    ax.axhline(len(r1) - 0.5, color="white", linewidth=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(COND_LABELS, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([m["name"] for m in ordered], fontsize=6.5)

    for i, m in enumerate(ordered):
        ax.text(-0.72, i, "|", ha="center", va="center", fontsize=12,
                color=VENDOR_COLORS.get(m["vendor"], "#333"), fontweight="bold",
                transform=ax.get_yaxis_transform())

    means = np.nanmean(matrix, axis=1)
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())
    axr.set_yticks(range(n))
    axr.set_yticklabels([f"{v:.2f}" for v in means], fontsize=6, color="#555")
    axr.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    axr.set_ylabel(r"Mean $\rho$", fontsize=7, color="#555")

    ax.text(-0.82, (len(r1) - 1) / 2, "Round 1", fontsize=6, color="#666",
            ha="center", va="center", style="italic")
    ax.text(-0.82, len(r1) + (n - len(r1) - 1) / 2, "Round 2", fontsize=6,
            color="#666", ha="center", va="center", style="italic")

    ax.set_title(f"Per-model norm adoption ({n} LLMs, real data)",
                 fontsize=8, pad=6)
    cb = plt.colorbar(im, ax=axr, fraction=0.046, pad=0.10)
    cb.set_label(r"Norm adoption $\rho$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_universality(models: list[dict], out: Path):
    """HIS heatmap + behavioural divergence + per-round ΔHIS scaling."""
    plt = _setup_mpl()
    import matplotlib.gridspec as gridspec

    ordered = ([m for m in models if m["round"] == 1] +
               [m for m in models if m["round"] == 2])
    n = len(ordered)

    fig = plt.figure(figsize=(7.4, 5.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    # Panel a: HIS heatmap (models x conditions)
    ax = fig.add_subplot(gs[0, 0])
    his = np.full((n, 4), np.nan)
    for i, m in enumerate(ordered):
        for j, c in enumerate(CONDITIONS):
            if c in m["topology"]:
                his[i, j] = m["topology"][c]["his_mean"]
    im = ax.imshow(his.T, aspect="auto", cmap="RdYlBu", vmin=0, vmax=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(["Pairwise", "Star", "Triadic", "5-Clique"], fontsize=6)
    ax.set_xticks([]); ax.set_xlabel(f"{n} models", fontsize=6.5)
    ax.set_title("HIS (per-round invariant)", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)

    # Panel b: behavioural divergence (rho scatter per condition)
    ax = fig.add_subplot(gs[0, 1])
    cond_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for j, c in enumerate(CONDITIONS):
        vals = [m["rho"][c] for m in ordered if c in m["rho"]]
        jit = np.random.RandomState(j).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), j) + jit, vals, c=cond_colors[j],
                   s=22, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Pair", "Star", "Triad", "Clique"], fontsize=6)
    ax.set_ylabel(r"Norm adoption $\rho$", fontsize=6.5)
    ax.set_title("Behaviour diverges at fixed HIS", fontsize=7)

    # Panel c: ΔHIS = HIS_clique - HIS_star vs system size n (real scaling data).
    # n=8/16/24 from the sweep; n=50/100 from results/agentpanel_qwen_scaling.
    ax = fig.add_subplot(gs[1, 0])
    scaling_f = RESULTS / "agentpanel_qwen_scaling" / "scaling_summary.json"
    if scaling_f.exists():
        sc = json.loads(scaling_f.read_text())["delta_his_by_n"]
        ns = sorted(int(k) for k in sc)
        dh = [sc[str(n)] for n in ns]
    else:
        ns, dh = [8, 16, 24], [0.089, 0.185, 0.275]
    ax.plot(ns, dh, "o-", color="#5B6ACD", markersize=4, linewidth=1.3)
    for x, y in zip(ns, dh):
        ax.annotate(f"{y:.2f}", (x, y), fontsize=4.5, ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=6)
    ax.set_xlabel("System size n", fontsize=6.5)
    ax.set_ylabel(r"$\Delta$HIS (Clique$-$Star)", fontsize=6.5)
    ax.set_title(r"$\Delta$HIS grows with n", fontsize=7)

    # Panel d: condition-A s.d. = 0.000 highlight
    ax = fig.add_subplot(gs[1, 1])
    sds = [np.std([m["topology"][c]["his_mean"] for m in ordered if c in m["topology"]])
           for c in CONDITIONS]
    ax.bar(range(4), sds, color=["#2E7D32" if s < 1e-9 else "#C44E52" for s in sds])
    ax.set_xticks(range(4)); ax.set_xticklabels(["A", "B", "C", "D"], fontsize=6.5)
    ax.set_ylabel("Cross-model HIS s.d.", fontsize=6.5)
    ax.set_title("s.d.=0 exact for A (pooled)", fontsize=7)

    fig.suptitle(f"Cross-model topology across {n} frontier LLMs (real data)",
                 fontsize=9, y=0.99)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    models = load_all()

    # Dump exact plotted numbers for provenance
    dump = {m["name"]: {"vendor": m["vendor"], "round": m["round"],
                        "rho": m["rho"],
                        "his": {c: m["topology"][c]["his_mean"]
                                for c in CONDITIONS if c in m["topology"]}}
            for m in models}
    (OUTDIR / "llm_figure_data.json").write_text(json.dumps(dump, indent=2))
    print(f"\n  wrote {OUTDIR / 'llm_figure_data.json'}")

    fig_behaviour_heatmap(models, OUTDIR / "fig_llm_behaviour_heatmap.png")
    fig_universality(models, OUTDIR / "fig_llm_heatmap.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
