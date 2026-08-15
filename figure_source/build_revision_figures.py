#!/usr/bin/env python3
"""Build transparent, data-traceable figures for the NMI manuscript revision.

Every empirical panel prefers a compact, project-contained input archive made
from the local primary result files. The shared result directory remains a
fallback for archive regeneration. Conceptual panels are explicitly labelled
as schematics. The script deliberately avoids using an AgentPanel database as
a proxy for the Moltbook observation.

Outputs:
    ../images/fig1_overview.{pdf,png}
    ../images/fig2_collapse.{pdf,png}
    ../images/fig_temporal_evolution.{pdf,png}
    ../images/fig4_abm.{pdf,png}
    ../images/fig6_protocol_outcomes.{pdf,png}
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
FIGURE_DATA = PROJECT / "figure_source" / "data"
FIG1_OVERVIEW_ARCHIVE = FIGURE_DATA / "fig1_overview_data.json"
FIG2_OBSERVATION_ARCHIVE = FIGURE_DATA / "fig2_observational_data.json"
FIG2_SP_REFERENCE_ARCHIVE = FIGURE_DATA / "fig2_sociopatterns_references.json"
FIG3_TEMPORAL_ARCHIVE = FIGURE_DATA / "fig3_temporal_data.json"
FIG4_ABM_ARCHIVE = FIGURE_DATA / "fig4_abm_data.json"
FIG6_LLM_ARCHIVE = FIGURE_DATA / "fig6_agentpanel_data.json"
OUTDIR = PROJECT / "images"
OUTDIR.mkdir(exist_ok=True)
PDF_CREATION_DATE = datetime(2026, 7, 30, 12, 0, 0)

COLORS = {
    "moltbook": "#C74B45",
    "human": "#2F6FC0",
    "pairwise": "#C74B45",
    "reciprocal": "#D98B36",
    "triad": "#2F6FC0",
    "pentad": "#3E9B51",
    "ink": "#242424",
    "muted": "#6C6C6C",
    "light": "#E8E8E8",
    "grid": "#D5D5D5",
    "violet": "#7254A3",
}

CONDITION_COLORS = {
    "dyadic_baseline": COLORS["pairwise"],
    "dyadic_reciprocity": COLORS["reciprocal"],
    "triad_hyperedge": COLORS["triad"],
    "pentad_hyperedge": COLORS["pentad"],
}

CONDITION_LABELS = {
    "dyadic_baseline": "Dyadic",
    "dyadic_reciprocity": "Reciprocal dyadic",
    "triad_hyperedge": "Triadic",
    "pentad_hyperedge": "Pentadic",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 6.4,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.22, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9.5,
        ha="left",
        va="top",
        color=COLORS["ink"],
    )


def finish(fig: plt.Figure, stem: str) -> None:
    pdf_metadata = {
        "Title": stem.replace("_", " "),
        "Creator": "build_revision_figures.py",
        "CreationDate": PDF_CREATION_DATE,
    }
    fig.savefig(
        OUTDIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
        metadata=pdf_metadata,
    )
    fig.savefig(
        OUTDIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
        metadata={"Title": stem.replace("_", " ")},
    )
    plt.close(fig)


def load_study1() -> tuple[dict, dict, dict]:
    archive_paths = (
        FIG2_OBSERVATION_ARCHIVE,
        FIG2_SP_REFERENCE_ARCHIVE,
    )
    missing = [str(path) for path in archive_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Fig. 2 requires frozen render inputs; missing: " + ", ".join(missing)
        )
    archive = json.loads(FIG2_OBSERVATION_ARCHIVE.read_text())
    traces = json.loads(FIG2_SP_REFERENCE_ARCHIVE.read_text())["traces"]
    return archive["cross"], archive["multiscale"], traces


def graph_from_archive(snapshot: dict) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(snapshot["nodes"])
    graph.add_edges_from([tuple(edge) for edge in snapshot["edges"]])
    graph.graph.update(snapshot.get("metadata", {}))
    return graph


def actual_moltbook_thread() -> nx.Graph:
    """Create an anonymised reply graph from one observed, active Moltbook thread."""
    if FIG1_OVERVIEW_ARCHIVE.exists():
        archive = json.loads(FIG1_OVERVIEW_ARCHIVE.read_text())
        return graph_from_archive(archive["moltbook"])

    db_path = DATA / "raw" / "moltbook" / "moltbook.db"
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        post_id, post_author, post_comments = connection.execute(
            """
            SELECT p.id, p.author_id, COUNT(c.id) AS observed_comments
            FROM posts AS p
            JOIN comments AS c ON c.post_id = p.id
            WHERE c.author_id IS NOT NULL
            GROUP BY p.id, p.author_id
            HAVING observed_comments BETWEEN 25 AND 70
            ORDER BY observed_comments DESC
            LIMIT 1
            """
        ).fetchone()
        comments = connection.execute(
            """
            SELECT id, parent_id, author_id, depth
            FROM comments
            WHERE post_id = ?
            ORDER BY created_at
            LIMIT 60
            """,
            (post_id,),
        ).fetchall()
    finally:
        connection.close()

    graph = nx.Graph()
    root = "post_author"
    graph.add_node(root, depth=-1)
    comment_to_author: dict[str, str] = {}
    author_ids = {post_author: root}
    for comment_id, parent_id, author_id, depth in comments:
        if not author_id:
            continue
        author = author_ids.setdefault(author_id, f"agent_{len(author_ids)}")
        comment_to_author[comment_id] = author
        parent = comment_to_author.get(parent_id, root)
        if author != parent:
            graph.add_edge(parent, author, reply_depth=depth or 0)

    graph.graph["comment_count"] = int(post_comments or len(comments))
    return graph


def actual_sociopatterns_episode() -> nx.Graph:
    """Select a compact observed contact episode rather than a stylised human graph."""
    if FIG1_OVERVIEW_ARCHIVE.exists():
        archive = json.loads(FIG1_OVERVIEW_ARCHIVE.read_text())
        return graph_from_archive(archive["sociopatterns"])

    contacts = np.loadtxt(DATA / "raw" / "sociopatterns" / "contact" / "tij_SFHH.dat", dtype=int)
    times = contacts[:, 0]
    best_graph: nx.Graph | None = None
    best_score = -1.0
    for start in np.arange(times.min(), times.max(), 120):
        window = contacts[(times >= start) & (times < start + 120)]
        graph = nx.Graph()
        for _, a, b in window:
            if a != b:
                graph.add_edge(int(a), int(b))
        nodes, edges = graph.number_of_nodes(), graph.number_of_edges()
        if 18 <= nodes <= 45 and 20 <= edges <= 100:
            score = edges / max(nodes, 1)
            if score > best_score:
                best_graph, best_score = graph, score
    if best_graph is None:
        raise RuntimeError("Could not locate a suitable SocioPatterns contact episode.")
    return best_graph


def draw_small_topology(ax: plt.Axes, kind: str, color: str, label: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    if kind == "cohesive":
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 2
        points = np.c_[np.cos(angles), np.sin(angles)]
        ax.add_patch(Polygon(points, closed=True, facecolor=color, alpha=0.12, edgecolor=color, lw=1.3))
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                ax.plot(points[[i, j], 0], points[[i, j], 1], color=color, alpha=0.18, lw=0.7)
        ax.scatter(points[:, 0], points[:, 1], s=48, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    else:
        angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        points = np.c_[1.1 * np.cos(angles), 1.1 * np.sin(angles)]
        for point in points:
            ax.plot([0, point[0]], [0, point[1]], color=color, alpha=0.45, lw=1.0)
        ax.scatter(points[:, 0], points[:, 1], s=34, color="#F2C2BF", edgecolor=color, linewidth=0.45, zorder=3)
        ax.scatter([0], [0], s=155, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.text(0.5, -0.15, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.7, color=color, fontweight="bold")


def fig1_overview() -> None:
    """Build a data-led overview of interaction structure and study scope."""
    # Match the manuscript's physical text width so labels remain legible after
    # LaTeX places the figure at \textwidth.
    fig = plt.figure(figsize=(5.35, 4.55))
    grid = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.10, 1.20],
        hspace=0.30,
        wspace=0.16,
    )

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "a", x=-0.10)
    graph = actual_moltbook_thread()
    root = "post_author"
    outer_nodes = [node for node in graph.nodes() if node != root]
    pos = nx.shell_layout(graph, nlist=[[root], outer_nodes], rotate=np.pi / 2, scale=0.97)
    direct = [(u, v) for u, v in graph.edges() if root in (u, v)]
    nested = [(u, v) for u, v in graph.edges() if root not in (u, v)]
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=nested, width=0.75, edge_color="#BFC4CA", alpha=0.55
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=direct, width=1.0, edge_color=COLORS["moltbook"], alpha=0.48
    )
    sizes = [180 if node == root else 27 + 13 * graph.degree(node) for node in graph.nodes()]
    nodes = [COLORS["moltbook"] if node == root else "#F0B7B2" for node in graph.nodes()]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=nodes,
        edgecolors="white",
        linewidths=0.55,
    )
    ax.set_title("Observed reply thread", loc="left", pad=3, fontweight="bold")
    ax.text(
        0.50,
        0.02,
        f"{graph.number_of_nodes()} authors | {graph.number_of_edges()} reply ties",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.45,
        color=COLORS["muted"],
    )
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.26, 1.07)
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b", x=-0.08)
    graph = actual_sociopatterns_episode()
    largest_component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_component).copy()
    pos = nx.spring_layout(graph, seed=8, k=0.58, iterations=450, scale=0.97)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, width=0.90, edge_color=COLORS["human"], alpha=0.30
    )
    degrees = np.asarray([graph.degree(node) for node in graph.nodes()])
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=22 + degrees * 9,
        node_color=COLORS["human"],
        alpha=0.90,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.set_title("SocioPatterns component", loc="left", pad=3, fontweight="bold")
    ax.text(
        0.50,
        0.02,
        f"{graph.number_of_nodes()} attendees | {graph.number_of_edges()} contact ties",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.45,
        color=COLORS["muted"],
    )
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.26, 1.07)
    ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    panel_label(ax, "c", x=-0.035, y=1.03)
    ax.set_title("Five designs, distinct evidence boundaries", loc="left", pad=3, fontweight="bold")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")

    evidence = [
        (
            "Selected\nthreads",
            "49,519 hyperedges",
            "Descriptive contrast",
            COLORS["moltbook"],
        ),
        (
            "Mechanism\nsimulation",
            "1,040 grid runs",
            "Specified update rule",
            COLORS["reciprocal"],
        ),
        (
            "Degree-matched\ncontrol",
            "20 pairs | 17,280 runs",
            "Common-rule rewiring",
            COLORS["human"],
        ),
        (
            "Protocol\narchive",
            "192 stored run cells",
            "Protocol construction",
            COLORS["pentad"],
        ),
        (
            "Evidence\nrelay",
            "6,400 intended calls",
            "Lossless + limited relay",
            COLORS["violet"],
        ),
    ]
    positions = ((0.0, 1.02), (1.0, 1.02), (2.0, 1.02), (0.5, 0.08), (1.5, 0.08))
    for index, ((heading, count, scope, color), (x0, y0)) in enumerate(zip(evidence, positions)):
        ax.add_patch(
            Rectangle(
                (x0 + 0.045, y0),
                0.91,
                0.72,
                facecolor="white",
                edgecolor="#D7D7D7",
                linewidth=0.60,
            )
        )
        ax.add_patch(
            Rectangle(
                (x0 + 0.045, y0),
                0.055,
                0.72,
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(
            x0 + 0.14,
            y0 + 0.63,
            f"0{index + 1}",
            ha="left",
            va="center",
            fontsize=5.75,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x0 + 0.14,
            y0 + 0.48,
            heading,
            ha="left",
            va="center",
            fontsize=5.70,
            fontweight="bold",
            color=COLORS["ink"],
            linespacing=1.04,
        )
        ax.text(
            x0 + 0.14,
            y0 + 0.25,
            count,
            ha="left",
            va="center",
            fontsize=5.05,
            color=color,
            fontweight="bold",
        )
        ax.plot(
            [x0 + 0.14, x0 + 0.87],
            [y0 + 0.145, y0 + 0.145],
            color="#E5E5E5",
            linewidth=0.55,
        )
        ax.text(
            x0 + 0.14,
            y0 + 0.09,
            scope,
            ha="left",
            va="top",
            fontsize=4.90,
            color=COLORS["muted"],
            linespacing=1.02,
        )

    finish(fig, "fig1_overview")


def fig2_collapse() -> None:
    """Render the empirical contrast with all frozen contact references."""
    cross, multiscale, traces = load_study1()
    trace_labels = {
        "SP-InVS13": "InVS13",
        "SP-InVS15": "InVS15",
        "SP-LH10": "LH10",
        "SP-LyonSchool": "LyonSchool",
        "SP-SFHH": "SFHH",
        "SP-Thiers13": "Thiers13",
    }
    ordered_traces = sorted(
        traces.items(),
        key=lambda item: item[1]["hyperdegree_gini"],
        reverse=True,
    )
    # Match the manuscript's physical text width so every source identifier
    # remains legible in the complete six-trace comparison.
    fig = plt.figure(figsize=(5.45, 4.18))
    grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.77, wspace=0.54)

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "a")
    ax.scatter(
        [metrics["frac_higher_order"] for _, metrics in ordered_traces],
        [metrics["hyperdegree_gini"] for _, metrics in ordered_traces],
        s=37,
        color=COLORS["human"],
        alpha=0.9,
        edgecolors="white",
        linewidths=0.65,
        zorder=3,
        label="SocioPatterns traces (n = 6)",
    )
    ax.scatter(
        cross["moltbook"]["higher_order_fraction"],
        cross["moltbook"]["hyperdegree_gini"],
        s=63,
        marker="D",
        color=COLORS["moltbook"],
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
        label="Moltbook (top-50k)",
    )
    ax.scatter(
        cross["enron"]["higher_order_fraction"],
        cross["enron"]["hyperdegree_gini"],
        s=37,
        marker="s",
        color=COLORS["muted"],
        edgecolors="white",
        linewidths=0.65,
        zorder=3,
        label="Enron unique ties",
    )
    ax.annotate(
        "Moltbook",
        (
            cross["moltbook"]["higher_order_fraction"],
            cross["moltbook"]["hyperdegree_gini"],
        ),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        fontsize=6.05,
        color=COLORS["moltbook"],
        fontweight="bold",
    )
    ax.annotate(
        "Enron",
        (
            cross["enron"]["higher_order_fraction"],
            cross["enron"]["hyperdegree_gini"],
        ),
        xytext=(4, -9),
        textcoords="offset points",
        ha="left",
        fontsize=5.9,
        color=COLORS["muted"],
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 0.95)
    ax.set_xlabel("Higher-order hyperedge fraction")
    ax.set_ylabel("Hyperdegree Gini")
    ax.set_title("Higher-order does not imply equality", loc="left", pad=3, fontsize=8.4)
    ax.tick_params(labelsize=7.0)
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.7)

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b")
    trace_y = np.arange(len(ordered_traces))
    trace_gini = np.array(
        [metrics["hyperdegree_gini"] for _, metrics in ordered_traces]
    )
    ax.hlines(trace_y, 0, trace_gini, color=COLORS["human"], lw=1.2, alpha=0.34)
    ax.scatter(
        trace_gini,
        trace_y,
        s=31,
        color=COLORS["human"],
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    for y, value in zip(trace_y, trace_gini):
        ax.text(value + 0.018, y, f"{value:.3f}", va="center", fontsize=5.7, color=COLORS["human"])
    ax.axvline(cross["moltbook"]["hyperdegree_gini"], color=COLORS["moltbook"], lw=1.35)
    ax.scatter(
        cross["moltbook"]["hyperdegree_gini"],
        len(ordered_traces) - 0.15,
        marker="D",
        s=42,
        color=COLORS["moltbook"],
        edgecolors="white",
        linewidths=0.6,
        clip_on=False,
        zorder=4,
    )
    ax.text(
        cross["moltbook"]["hyperdegree_gini"],
        len(ordered_traces) + 0.27,
        "Moltbook\n0.863",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=COLORS["moltbook"],
        fontweight="bold",
    )
    ax.set_xlim(0, 0.98)
    ax.set_ylim(-0.6, len(ordered_traces) + 0.8)
    ax.set_yticks(trace_y)
    ax.set_yticklabels(
        [trace_labels[name] for name, _ in ordered_traces],
        fontsize=6.0,
    )
    ax.set_xlabel("Hyperdegree Gini")
    ax.set_title("Six contact traces", loc="left", pad=3, fontsize=8.4)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "c")
    overlap = [
        ("Moltbook", cross["moltbook"]["mean_edge_overlap"], COLORS["moltbook"], "D"),
        *[
            (trace_labels[name], metrics["mean_edge_overlap"], COLORS["human"], "o")
            for name, metrics in ordered_traces
        ],
        ("Enron ties", cross["enron"]["mean_edge_overlap"], COLORS["muted"], "s"),
    ]
    overlap.sort(key=lambda item: item[1])
    y = np.arange(len(overlap))
    for yi, (label, value, color, marker) in enumerate(overlap):
        ax.hlines(yi, 0, value, color=color, lw=1.3, alpha=0.35)
        ax.scatter(
            value,
            yi,
            s=43 if label == "Moltbook" else 30,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    ax.annotate(
        f"{cross['moltbook']['mean_edge_overlap']:.3f}",
        (cross["moltbook"]["mean_edge_overlap"], overlap.index(next(item for item in overlap if item[0] == "Moltbook"))),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=5.9,
        color=COLORS["moltbook"],
        fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _, _ in overlap], fontsize=5.8)
    ax.set_xlim(0, 0.58)
    ax.set_xlabel("Mean edge overlap (Jaccard)")
    ax.set_title("Lower context recurrence", loc="left", pad=3, fontsize=8.4)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "d", x=-0.22, y=1.18)
    windows = np.array([15, 30, 60, 120])
    gini = np.array([multiscale[f"Δt={window}min"]["hyperdegree_gini"] for window in windows])
    higher_order = np.array([multiscale[f"Δt={window}min"]["frac_higher_order"] for window in windows])
    closure = np.array([multiscale[f"Δt={window}min"]["triadic_closure_rate"] for window in windows])
    overlap_window = np.array([multiscale[f"Δt={window}min"]["mean_edge_overlap"] for window in windows])

    for values, label, color in [
        (gini, "Gini", COLORS["moltbook"]),
        (higher_order, "Higher-order", COLORS["triad"]),
        (closure, "Triadic closure", COLORS["violet"]),
    ]:
        ax.plot(windows, values, "-o", color=color, lw=1.35, ms=3.9,
                label=label, markeredgecolor="white", markeredgewidth=0.45)
    ax.set_ylim(0.70, 1.01)
    ax.set_xticks(windows)
    ax.set_xlabel("Reply window (min)")
    ax.set_ylabel("Topology components")
    ax.set_title("Moltbook window sensitivity", loc="left", pad=3, fontsize=8.4)
    ax.tick_params(labelsize=7.0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)

    ax_overlap = ax.twinx()
    ax_overlap.plot(windows, overlap_window, "-o", color=COLORS["pentad"], lw=1.2,
                    ms=3.7, label="Edge overlap", markeredgecolor="white",
                    markeredgewidth=0.45)
    ax_overlap.set_ylim(0.115, 0.135)
    ax_overlap.set_ylabel("Overlap", color=COLORS["pentad"])
    ax_overlap.tick_params(axis="y", colors=COLORS["pentad"])
    ax_overlap.tick_params(axis="y", labelsize=7.0)
    ax_overlap.spines["top"].set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    overlap_lines, overlap_labels = ax_overlap.get_legend_handles_labels()
    ax.legend(lines + overlap_lines, labels + overlap_labels, loc="lower right",
              ncol=2, columnspacing=0.8, handlelength=1.2, fontsize=5.8)

    finish(fig, "fig2_collapse")


def fig_temporal_evolution() -> None:
    """Render the observed weekly metrics without implying a causal transition."""
    if FIG3_TEMPORAL_ARCHIVE.exists():
        metrics = json.loads(FIG3_TEMPORAL_ARCHIVE.read_text())["metrics"]
    else:
        metrics = json.loads((RESULTS / "study1_temporal" / "temporal_metrics.json").read_text())
    weeks = [entry["week"].replace("2026-", "") for entry in metrics]
    x = np.arange(len(weeks))
    tick_indices = (0, 2, 4, 7)
    tick_labels = [weeks[index] for index in tick_indices]

    fig = plt.figure(figsize=(5.7, 3.6), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, wspace=0.36)

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "a", x=-0.18, y=1.10)
    ax.plot(
        x,
        [entry["gini"] for entry in metrics],
        "-o",
        color=COLORS["moltbook"],
        lw=1.55,
        ms=4.8,
        label="Hyperdegree Gini",
        markeredgecolor="white",
        markeredgewidth=0.55,
    )
    ax.plot(
        x,
        [entry["triadic_closure"] for entry in metrics],
        "-s",
        color=COLORS["human"],
        lw=1.55,
        ms=4.6,
        label="Triadic closure",
        markeredgecolor="white",
        markeredgewidth=0.55,
    )
    ax.set_ylim(0.42, 0.86)
    ax.set_yticks((0.45, 0.55, 0.65, 0.75, 0.85))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Weekly summary")
    ax.set_xlabel("Week")
    ax.set_title(
        "Concentration and closure",
        loc="left",
        pad=4,
        fontsize=9.4,
        fontweight="bold",
    )
    ax.tick_params(labelsize=7.2)
    ax.legend(loc="upper right", fontsize=7.0, handlelength=1.3, handletextpad=0.4)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b", x=-0.18, y=1.10)
    ax.plot(
        x,
        [entry["frac_higher_order"] for entry in metrics],
        "-D",
        color=COLORS["violet"],
        lw=1.55,
        ms=4.7,
        label="Higher-order fraction",
        markeredgecolor="white",
        markeredgewidth=0.55,
    )
    ax.plot(
        x,
        [entry["overlap"] for entry in metrics],
        "-^",
        color=COLORS["pentad"],
        lw=1.55,
        ms=4.6,
        label="Edge overlap",
        markeredgecolor="white",
        markeredgewidth=0.55,
    )
    ax.set_ylim(0.05, 0.92)
    ax.set_yticks((0.1, 0.3, 0.5, 0.7, 0.9))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Weekly summary")
    ax.set_xlabel("Week")
    ax.set_title(
        "Higher-order participation and overlap",
        loc="left",
        pad=4,
        fontsize=9.4,
        fontweight="bold",
    )
    ax.tick_params(labelsize=7.2)
    ax.legend(loc="upper right", fontsize=7.0, handlelength=1.3, handletextpad=0.4)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)

    finish(fig, "fig_temporal_evolution")


def read_abm_trajectories() -> dict[str, list[np.ndarray]]:
    if FIG4_ABM_ARCHIVE.exists():
        data = json.loads(FIG4_ABM_ARCHIVE.read_text())
        return {
            condition: [np.asarray(values, dtype=float) for values in runs]
            for condition, runs in data["trajectory_set"]["conditions"].items()
        }

    raw = RESULTS / "abm" / "raw"
    groups: dict[str, list[np.ndarray]] = {key: [] for key in CONDITION_COLORS}
    seed_ranges = {
        "dyadic_baseline": range(65000, 65030),
        "dyadic_reciprocity": range(66000, 66030),
        "triad_hyperedge": range(67000, 67030),
        "pentad_hyperedge": range(68000, 68030),
    }
    for condition, seeds in seed_ranges.items():
        for seed in seeds:
            file = raw / f"{condition}_seed{seed}.json"
            if file.exists():
                groups[condition].append(np.asarray(json.loads(file.read_text())["norm_adoption_rate"], dtype=float))
    return groups


def read_abm_sweep() -> tuple[dict, list[dict]]:
    if FIG4_ABM_ARCHIVE.exists():
        data = json.loads(FIG4_ABM_ARCHIVE.read_text())["seed_sweep"]
    else:
        data = json.loads((RESULTS / "abm" / "critical_mass_sweep.json").read_text())
    expected_conditions = set(CONDITION_COLORS)
    if set(data["summary"]) != expected_conditions:
        raise ValueError("ABM sweep is missing one or more conditions")
    if len(data["runs"]) != 1040:
        raise ValueError("ABM sweep does not contain the expected 1,040 run records")
    return data["summary"], data["runs"]


def fig4_abm() -> None:
    """Render the archived ABM trajectories and complete seed sweep."""
    trajectories = read_abm_trajectories()
    sweep, sweep_runs = read_abm_sweep()
    fig = plt.figure(figsize=(5.7, 4.2))
    grid = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.12,
        right=0.985,
        bottom=0.11,
        top=0.84,
        hspace=0.76,
        wspace=0.50,
    )

    legend_handles = [
        Line2D([0], [0], color=CONDITION_COLORS[condition], lw=1.7)
        for condition in CONDITION_COLORS
    ]
    fig.legend(
        legend_handles,
        ["Dyadic", "Reciprocal", "Triadic", "Pentadic"],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=4,
        fontsize=7.0,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=1.0,
    )

    def format_panel(axis: plt.Axes) -> None:
        axis.tick_params(axis="both", labelsize=7.0, pad=2)
        axis.xaxis.label.set_size(8.3)
        axis.yaxis.label.set_size(8.3)
        axis.title.set_fontsize(8.4)

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "a")
    for condition, runs in trajectories.items():
        values = np.asarray(runs)
        color = CONDITION_COLORS[condition]
        for trajectory in values:
            ax.plot(trajectory, color=color, alpha=0.07, lw=0.32)
        mean = values.mean(axis=0)
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(np.arange(len(mean)), low, high, color=color, alpha=0.10, lw=0)
        ax.plot(mean, color=color, lw=1.45)
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"Norm adoption $\rho$")
    ax.set_title(r"Seeded trajectories ($\rho_0=0.05$)", loc="left", pad=2)
    format_panel(ax)

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b", x=-0.34)
    for condition, color in CONDITION_COLORS.items():
        entries = sweep[condition]
        x = np.asarray(sorted(float(key) for key in entries))
        y = np.asarray([entries[str(value)]["norm_mean"] for value in x])
        lower = []
        upper = []
        for value in x:
            final_values = [
                run["final_norm_adoption"]
                for run in sweep_runs
                if run["condition"] == condition and run["seed_fraction"] == value
            ]
            offsets = np.linspace(-0.0035, 0.0035, len(final_values))
            ax.scatter(
                value + offsets,
                final_values,
                s=7,
                color=color,
                alpha=0.13,
                linewidths=0,
                zorder=1,
            )
            low, high = np.quantile(final_values, [0.1, 0.9])
            lower.append(low)
            upper.append(high)
        ax.plot(x, y, "-o", color=color, lw=1.35, ms=4.0, markeredgecolor="white", markeredgewidth=0.55)
        ax.fill_between(x, lower, upper, color=color, alpha=0.08, lw=0, zorder=0)
    ax.axhline(0.5, color=COLORS["muted"], lw=0.65, ls=":")
    ax.axvline(0.10, color=COLORS["triad"], lw=0.65, ls="--", alpha=0.7)
    ax.axvline(0.15, color=COLORS["pentad"], lw=0.65, ls="--", alpha=0.7)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel(r"Initial seed fraction $\rho_0$")
    ax.set_ylabel(r"Final norm adoption $\rho_\infty$")
    ax.set_title("Seed response (20 runs per cell)", loc="left", pad=2)
    format_panel(ax)

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "c")
    selected_cells = [
        ("triad_hyperedge", 0.05),
        ("triad_hyperedge", 0.10),
        ("pentad_hyperedge", 0.10),
        ("pentad_hyperedge", 0.15),
    ]
    distributions = []
    for condition, seed_fraction in selected_cells:
        distributions.append(
            [
                run["final_norm_adoption"]
                for run in sweep_runs
                if run["condition"] == condition and run["seed_fraction"] == seed_fraction
            ]
        )
    box = ax.boxplot(
        distributions,
        patch_artist=True,
        widths=0.52,
        medianprops={"color": COLORS["ink"], "linewidth": 0.8},
        whiskerprops={"color": COLORS["muted"], "linewidth": 0.65},
        capprops={"color": COLORS["muted"], "linewidth": 0.65},
        flierprops={"marker": "", "markersize": 0},
    )
    for patch, (condition, _) in zip(box["boxes"], selected_cells):
        patch.set(facecolor=CONDITION_COLORS[condition], alpha=0.18, edgecolor=CONDITION_COLORS[condition], linewidth=0.9)
    for index, ((condition, _), values) in enumerate(zip(selected_cells, distributions), start=1):
        offsets = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(
            index + offsets,
            values,
            s=10,
            color=CONDITION_COLORS[condition],
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )
    ax.axhline(0.5, color=COLORS["muted"], lw=0.65, ls=":")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks(range(1, len(selected_cells) + 1))
    ax.set_xticklabels(["Triadic\n0.05", "Triadic\n0.10", "Pentadic\n0.10", "Pentadic\n0.15"])
    ax.set_ylabel(r"Final norm adoption $\rho_\infty$")
    ax.set_title("Runs around sampled crossings", loc="left", pad=2)
    format_panel(ax)

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "d", x=-0.34)
    for condition, color in CONDITION_COLORS.items():
        rows = [run for run in sweep_runs if run["condition"] == condition]
        adoption = np.asarray([run["final_norm_adoption"] for run in rows])
        cooperation = np.asarray([run["final_cooperation"] for run in rows])
        ax.scatter(adoption, cooperation, s=8, color=color, alpha=0.12, linewidths=0)
        ax.scatter(
            np.mean(adoption),
            np.mean(cooperation),
            s=50,
            color=color,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
    ax.set_xlim(-0.03, 1.04)
    ax.set_ylim(0.70, 1.03)
    ax.set_xlabel(r"Final norm adoption $\rho_\infty$")
    ax.set_ylabel("Final cooperation")
    ax.set_title("Cooperation versus adoption", loc="left", pad=2)
    format_panel(ax)

    finish(fig, "fig4_abm")


def complete_llm_results() -> tuple[list[str], dict, list[dict]]:
    if FIG6_LLM_ARCHIVE.exists():
        data = json.loads(FIG6_LLM_ARCHIVE.read_text())
        models = data["models"]
        topology = data["topology"]
        runs = data["runs"]
    else:
        models = []
        topology = {}
        runs = []
        base = RESULTS / "multimodel_16" / "agentpanel"
        for directory in sorted(path for path in base.iterdir() if path.is_dir()):
            topo_file = directory / "agentpanel_topology.json"
            result_file = directory / "agentpanel_results.json"
            if not (topo_file.exists() and result_file.exists()):
                continue
            models.append(directory.name)
            topology[directory.name] = json.loads(topo_file.read_text())
            for run_file in sorted(directory.glob("run_*.json")):
                record = json.loads(run_file.read_text())
                runs.append({"model": directory.name, **record})

    expected_conditions = {"A", "B", "C", "D"}
    if len(models) != 12 or set(topology) != set(models):
        raise ValueError("LLM archive must contain exactly 12 complete model records")
    if len(runs) != 192 or any(set(record) < {"model", "condition", "rho0", "seed", "final_rho"} for record in runs):
        raise ValueError("LLM archive must contain 192 traceable run endpoints")
    if any(set(topology[model]) != expected_conditions for model in models):
        raise ValueError("LLM topology archive is missing a protocol condition")
    return models, topology, runs


def llm_model_seed_means(
    models: list[str],
    runs: list[dict],
    condition: str,
    rho0: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    means = []
    raw = []
    for model in models:
        values = np.asarray(
            [
                record["final_rho"]
                for record in runs
                if record["model"] == model
                and record["condition"] == condition
                and record["rho0"] == rho0
            ],
            dtype=float,
        )
        if len(values) != 2:
            raise ValueError(f"Expected two stored seed runs for {model}, {condition}, rho0={rho0}")
        means.append(values.mean())
        raw.append(values)
    return np.asarray(means), raw


def fig6_protocol_outcomes() -> None:
    """Report the completed LLM data as a protocol check, not universality proof."""
    models, topology, runs = complete_llm_results()
    conditions = ["A", "B", "C", "D"]
    condition_labels = ["Pairs", "Star", "Triads", "5-cliques"]
    condition_colors = [COLORS["pairwise"], COLORS["reciprocal"], COLORS["triad"], COLORS["pentad"]]
    fig = plt.figure(figsize=(5.7, 4.22))
    grid = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.12,
        right=0.99,
        bottom=0.11,
        top=0.94,
        hspace=0.76,
        wspace=0.54,
    )

    def format_panel(axis: plt.Axes) -> None:
        axis.tick_params(axis="both", labelsize=7.0, pad=2)
        axis.xaxis.label.set_size(8.3)
        axis.yaxis.label.set_size(8.3)
        axis.title.set_fontsize(8.4)

    ax = fig.add_subplot(grid[0, 0])
    panel_label(ax, "a")
    for idx, (condition, color) in enumerate(zip(conditions, condition_colors)):
        if condition == "A":
            ax.text(idx, 0.48, "N/A", ha="center", va="center", fontsize=6.7, color=COLORS["muted"])
            continue
        values = np.asarray([topology[model][condition]["his_mean"] for model in models], dtype=float)
        jitter = np.linspace(-0.065, 0.065, len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, color=color, alpha=0.24, s=16, linewidths=0, zorder=2)
        mean = values.mean()
        ax.scatter(idx, mean, color=color, s=46, edgecolors="white", linewidths=0.55, zorder=3)
        ax.text(idx, mean - 0.075, f"{mean:.2f}", ha="center", va="top", fontsize=6.6, color=color, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Mean degree equality")
    ax.set_title("Membership protocol fixes topology", loc="left", pad=2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.05,
        "faint points: 12 complete\nmodel directories",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=COLORS["muted"],
    )
    format_panel(ax)

    ax = fig.add_subplot(grid[0, 1])
    panel_label(ax, "b", x=-0.27)
    for idx, (condition, color) in enumerate(zip(conditions, condition_colors)):
        for rho0, offset, marker in [(0.10, -0.16, "o"), (0.50, 0.16, "s")]:
            means, seed_values = llm_model_seed_means(
                models,
                runs,
                condition,
                rho0,
            )
            jitter = np.linspace(-0.065, 0.065, len(means))
            positions = np.full(len(means), idx + offset) + jitter
            for position, values in zip(positions, seed_values):
                ax.vlines(
                    position,
                    values.min(),
                    values.max(),
                    color=color,
                    alpha=0.28,
                    linewidth=0.65,
                    zorder=1,
                )
            ax.scatter(
                positions,
                means,
                color=color,
                marker=marker,
                alpha=0.74,
                s=19,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            ax.hlines(
                means.mean(),
                idx + offset - 0.095,
                idx + offset + 0.095,
                color=color,
                lw=1.25,
                zorder=3,
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"Final norm adoption $\rho$")
    ax.set_title("Endpoint variation by protocol", loc="left", pad=2)
    format_panel(ax)

    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "c")
    pairwise, pairwise_raw = llm_model_seed_means(models, runs, "A", 0.10)
    clique, clique_raw = llm_model_seed_means(models, runs, "D", 0.10)
    for mean_a, mean_d, raw_a, raw_d in zip(pairwise, clique, pairwise_raw, clique_raw):
        ax.plot([0, 1], [mean_a, mean_d], color=COLORS["grid"], lw=0.9, alpha=0.85, zorder=1)
        ax.scatter(np.full(2, -0.035), raw_a, color=COLORS["pairwise"], s=10, alpha=0.24, linewidths=0, zorder=2)
        ax.scatter(np.full(2, 1.035), raw_d, color=COLORS["pentad"], s=10, alpha=0.24, linewidths=0, zorder=2)
    ax.scatter(np.zeros(len(models)), pairwise, color=COLORS["pairwise"], s=29, label="Pairs", zorder=3, edgecolors="white", linewidths=0.45)
    ax.scatter(np.ones(len(models)), clique, color=COLORS["pentad"], s=29, label="5-cliques", zorder=3, edgecolors="white", linewidths=0.45)
    ax.set_xlim(-0.24, 1.24)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pairs", "5-cliques"])
    ax.set_ylabel(r"Final norm adoption $\rho$")
    ax.set_title(r"Within-model endpoints ($\rho_0=0.10$)", loc="left", pad=2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    format_panel(ax)

    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "d", x=-0.27)
    pairwise, pairwise_raw = llm_model_seed_means(models, runs, "A", 0.50)
    clique, clique_raw = llm_model_seed_means(models, runs, "D", 0.50)
    for mean_a, mean_d, raw_a, raw_d in zip(pairwise, clique, pairwise_raw, clique_raw):
        ax.plot([0, 1], [mean_a, mean_d], color=COLORS["grid"], lw=0.9, alpha=0.85, zorder=1)
        ax.scatter(np.full(2, -0.035), raw_a, color=COLORS["pairwise"], s=10, alpha=0.24, linewidths=0, zorder=2)
        ax.scatter(np.full(2, 1.035), raw_d, color=COLORS["pentad"], s=10, alpha=0.24, linewidths=0, zorder=2)
    ax.scatter(np.zeros(len(models)), pairwise, color=COLORS["pairwise"], s=29, zorder=3, edgecolors="white", linewidths=0.45)
    ax.scatter(np.ones(len(models)), clique, color=COLORS["pentad"], s=29, zorder=3, edgecolors="white", linewidths=0.45)
    ax.set_xlim(-0.24, 1.24)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pairs", "5-cliques"])
    ax.set_ylabel(r"Final norm adoption $\rho$")
    ax.set_title(r"Within-model endpoints ($\rho_0=0.50$)", loc="left", pad=2)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)
    format_panel(ax)

    finish(fig, "fig6_protocol_outcomes")


def main() -> None:
    configure_style()
    fig1_overview()
    fig2_collapse()
    fig_temporal_evolution()
    fig4_abm()
    fig6_protocol_outcomes()
    print(f"Saved revised figures to {OUTDIR}")


if __name__ == "__main__":
    main()
