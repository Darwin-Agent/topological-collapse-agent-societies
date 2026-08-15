"""
Higher-order topological metrics for hypergraphs.

Implements the analysis framework from Battiston et al. (NHB 2025) Box 1,
quantifying the structural differences between AI agent and human
interaction networks.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import stats

from .hypergraph_builder import Hypergraph

logger = logging.getLogger(__name__)


@dataclass
class TopologyReport:
    """Container for all computed topological metrics."""
    name: str
    n_nodes: int
    n_edges: int

    # Edge size distribution
    edge_size_mean: float = 0.0
    edge_size_median: float = 0.0
    edge_size_std: float = 0.0
    edge_size_max: int = 0
    edge_size_distribution: dict[int, int] = field(default_factory=dict)

    # Hyperdegree distribution
    hyperdegree_mean: float = 0.0
    hyperdegree_median: float = 0.0
    hyperdegree_max: int = 0
    hyperdegree_gini: float = 0.0

    # Higher-order structure
    frac_dyadic: float = 0.0         # fraction of edges with |e|=2
    frac_triadic: float = 0.0        # fraction with |e|=3
    frac_higher_order: float = 0.0   # fraction with |e|>=3

    # Triadic closure
    triadic_closure_rate: float = 0.0
    n_open_triads: int = 0
    n_closed_triads: int = 0

    # Edge overlap
    mean_edge_overlap: float = 0.0

    # Hyperedge Irreducibility Score (HIS)
    his_mean: float = 0.0            # mean HIS across all higher-order edges
    his_median: float = 0.0
    his_std: float = 0.0
    his_by_size: dict[int, float] = field(default_factory=dict)  # mean HIS per edge size
    n_simplicial: int = 0            # edges with HIS=1 (fully irreducible)
    frac_simplicial: float = 0.0     # fraction of higher-order edges that are simplicial

    # Reciprocity (for directed projections)
    reciprocity: float = 0.0

    # Power-law fit for edge size
    edge_size_alpha: float = 0.0
    edge_size_ks_pvalue: float = 0.0

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, dict) and len(v) > 50:
                continue
            d[k] = v
        return d


def compute_topology(hg: Hypergraph, name: str = "unnamed",
                     triadic_sample: int = 50000) -> TopologyReport:
    """Compute full topology report for a hypergraph."""
    logger.info("Computing topology for '%s' (%d nodes, %d edges)",
                name, hg.n_nodes, hg.n_edges)

    report = TopologyReport(name=name, n_nodes=hg.n_nodes, n_edges=hg.n_edges)

    if hg.n_edges == 0:
        return report

    # ── Edge size distribution ──────────────────────────────────────
    sizes = hg.edge_sizes()
    report.edge_size_mean = float(sizes.mean())
    report.edge_size_median = float(np.median(sizes))
    report.edge_size_std = float(sizes.std())
    report.edge_size_max = int(sizes.max())

    size_counts = Counter(sizes)
    report.edge_size_distribution = dict(sorted(size_counts.items()))

    report.frac_dyadic = float((sizes == 2).mean())
    report.frac_triadic = float((sizes == 3).mean())
    report.frac_higher_order = float((sizes >= 3).mean())

    # Power-law fit (for sizes >= 2)
    if len(sizes[sizes >= 2]) > 10:
        try:
            s_fit = sizes[sizes >= 2].astype(float)
            report.edge_size_alpha = 1 + len(s_fit) / np.sum(np.log(s_fit / 1.5))
        except Exception:
            pass

    # ── Hyperdegree distribution ────────────────────────────────────
    degrees = hg.node_degrees()
    deg_vals = np.array(list(degrees.values()))
    report.hyperdegree_mean = float(deg_vals.mean())
    report.hyperdegree_median = float(np.median(deg_vals))
    report.hyperdegree_max = int(deg_vals.max())
    report.hyperdegree_gini = _gini(deg_vals)

    # ── Triadic closure ─────────────────────────────────────────────
    logger.info("  Computing triadic closure (sample=%d)...", triadic_sample)
    tc_open, tc_closed = _triadic_closure(hg, sample_size=triadic_sample)
    report.n_open_triads = tc_open
    report.n_closed_triads = tc_closed
    report.triadic_closure_rate = (
        tc_closed / (tc_open + tc_closed) if (tc_open + tc_closed) > 0 else 0.0
    )

    # ── Edge overlap ────────────────────────────────────────────────
    logger.info("  Computing edge overlap...")
    report.mean_edge_overlap = _mean_edge_overlap(hg)

    # ── Hyperedge Irreducibility Score (HIS) ──────────────────────
    logger.info("  Computing HIS (Hyperedge Irreducibility Score)...")
    his_result = _compute_his(hg)
    report.his_mean = his_result["mean"]
    report.his_median = his_result["median"]
    report.his_std = his_result["std"]
    report.his_by_size = his_result["by_size"]
    report.n_simplicial = his_result["n_simplicial"]
    report.frac_simplicial = his_result["frac_simplicial"]

    logger.info("  Topology computation complete for '%s'", name)
    return report


def _gini(values: np.ndarray) -> float:
    """Gini coefficient (0=perfect equality, 1=max inequality)."""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


def _triadic_closure(hg: Hypergraph, sample_size: int = 50000) -> tuple[int, int]:
    """
    Estimate triadic closure rate on the 2-section (clique projection).

    For each hyperedge of size >= 3, project to pairwise links.
    Then sample node triads and check if they form triangles.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in hg.hyperedges:
        if len(edge) < 2:
            continue
        # Hyperedges are frozensets, so sort membership before constructing
        # the sampled projection to make fixed-seed metrics process-stable.
        members = sorted(edge)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                adjacency[members[i]].add(members[j])
                adjacency[members[j]].add(members[i])

    nodes_with_neighbors = sorted(n for n, nb in adjacency.items() if len(nb) >= 2)
    if len(nodes_with_neighbors) < 3:
        return 0, 0

    rng = np.random.default_rng(42)
    open_triads = 0
    closed_triads = 0

    sample_nodes = rng.choice(nodes_with_neighbors,
                              size=min(sample_size, len(nodes_with_neighbors)),
                              replace=False)

    for node in sample_nodes:
        neighbors = sorted(adjacency[node])
        if len(neighbors) < 2:
            continue
        n_pairs = min(10, len(neighbors) * (len(neighbors) - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng.choice(len(neighbors), size=2, replace=False)
            ni, nj = neighbors[i], neighbors[j]
            if nj in adjacency[ni]:
                closed_triads += 1
            else:
                open_triads += 1

    return open_triads, closed_triads


def _mean_edge_overlap(hg: Hypergraph, sample_size: int = 10000) -> float:
    """
    Mean Jaccard overlap between hyperedges sharing at least one node.
    Sampled for efficiency.
    """
    if hg.n_edges < 2:
        return 0.0

    node_to_edges: dict[str, list[int]] = defaultdict(list)
    for idx, edge in enumerate(hg.hyperedges):
        for n in edge:
            node_to_edges[n].append(idx)

    rng = np.random.default_rng(42)
    overlaps = []
    attempts = 0

    while len(overlaps) < sample_size and attempts < sample_size * 5:
        attempts += 1
        node = rng.choice(sorted(node_to_edges))
        edge_ids = node_to_edges[node]
        if len(edge_ids) < 2:
            continue
        i, j = rng.choice(edge_ids, size=2, replace=False)
        ei, ej = hg.hyperedges[i], hg.hyperedges[j]
        intersection = len(ei & ej)
        union = len(ei | ej)
        if union > 0:
            overlaps.append(intersection / union)

    return float(np.mean(overlaps)) if overlaps else 0.0


def _compute_his(hg: Hypergraph) -> dict:
    """
    Compute the Hyperedge Irreducibility Score (HIS) for all higher-order edges.

    Measures egalitarian participation within each hyperedge using
    the internal Gini coefficient of node hyperdegrees.

    For a hyperedge e = {v1, ..., vk}, k >= 3:

        HIS(e) = 1 - Gini({deg(v) : v ∈ e})

    where deg(v) is the global hyperdegree (# hyperedges containing v).

    Interpretation:
      - HIS ≈ 0: one hub node dominates (star pattern, typical of
                  AI agent comment threads where one author + many
                  independent commenters)
      - HIS ≈ 1: all members participate equally (clique-like,
                  typical of genuine face-to-face group interactions)

    Dynamical relevance:
      Hub-dominated hyperedges create single points of failure for
      higher-order contagion (β₂ρ² term requires multiple infected
      co-members). Egalitarian hyperedges provide redundant infection
      paths, amplifying the effective β₂.

    Ref: Aksoy et al. (2020), "Hypernetwork science via high-order
         tensor eigenvalue problems" — hub dominance concept.
         Battiston et al. (2025, NHB) — higher-order interaction framework.
    """
    # Precompute global hyperdegrees
    degrees = hg.node_degrees()

    his_values = []
    his_by_size: dict[int, list[float]] = defaultdict(list)
    n_egalitarian = 0  # edges with HIS >= 0.8

    for edge in hg.hyperedges:
        k = len(edge)
        if k < 3:
            continue

        # Get hyperdegrees of all members
        member_degrees = np.array([degrees.get(n, 1) for n in edge], dtype=float)

        # HIS = 1 - Gini(degrees within this edge)
        edge_gini = _gini(member_degrees)
        his = 1.0 - edge_gini

        his_values.append(his)
        his_by_size[k].append(his)

        if his >= 0.8:
            n_egalitarian += 1

    if not his_values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "by_size": {}, "n_simplicial": 0, "frac_simplicial": 0.0,
        }

    arr = np.array(his_values)
    n_ho = len(his_values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "by_size": {k: float(np.mean(v)) for k, v in sorted(his_by_size.items())},
        "n_simplicial": n_egalitarian,
        "frac_simplicial": n_egalitarian / n_ho,
    }


# ─── Comparison utilities ───────────────────────────────────────────────

def compare_reports(*reports: TopologyReport) -> str:
    """Pretty-print comparison table of multiple topology reports."""
    if not reports:
        return ""

    keys = [
        ("n_nodes", "Nodes"),
        ("n_edges", "Hyperedges"),
        ("edge_size_mean", "Edge size (mean)"),
        ("edge_size_median", "Edge size (median)"),
        ("edge_size_max", "Edge size (max)"),
        ("frac_dyadic", "% dyadic (s=2)"),
        ("frac_triadic", "% triadic (s=3)"),
        ("frac_higher_order", "% higher-order (s≥3)"),
        ("hyperdegree_mean", "Hyperdegree (mean)"),
        ("hyperdegree_gini", "Hyperdegree Gini"),
        ("triadic_closure_rate", "Triadic closure"),
        ("mean_edge_overlap", "Edge overlap (Jaccard)"),
        ("his_mean", "HIS (mean)"),
        ("his_median", "HIS (median)"),
        ("frac_simplicial", "% simplicial (HIS=1)"),
    ]

    col_width = max(len(r.name) for r in reports) + 2
    header = f"{'Metric':<28}" + "".join(f"{r.name:>{col_width}}" for r in reports)
    lines = ["=" * len(header), header, "-" * len(header)]

    for attr, label in keys:
        row = f"{label:<28}"
        for r in reports:
            val = getattr(r, attr)
            if isinstance(val, float):
                if "frac" in attr or attr in ("triadic_closure_rate", "mean_edge_overlap",
                                               "hyperdegree_gini", "reciprocity"):
                    row += f"{val:>{col_width}.4f}"
                else:
                    row += f"{val:>{col_width}.2f}"
            else:
                row += f"{val:>{col_width},}"
        lines.append(row)

    lines.append("=" * len(header))
    return "\n".join(lines)
