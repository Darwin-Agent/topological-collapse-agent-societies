"""
Null models for hypergraph statistical significance testing.

Implements the configuration model (Chodrow 2020, J. Complex Networks):
generates random hypergraphs preserving the degree sequence and edge
size sequence of the original, but randomizing connectivity.

Used to compute z-scores for all topological metrics.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .hypergraph_builder import Hypergraph
from .topology import TopologyReport, compute_topology

logger = logging.getLogger(__name__)


def configuration_model_sample(hg: Hypergraph, seed: int = 0) -> Hypergraph:
    """
    Generate one random hypergraph from the configuration model.

    Preserves:
      - Number of hyperedges
      - Edge size sequence (each edge keeps its original size)
      - Node degree sequence (each node appears in the same number of edges)

    Method: stub-matching (Chodrow 2020).
    """
    rng = np.random.default_rng(seed)

    sizes = [len(e) for e in hg.hyperedges]

    stubs = []
    for node, deg in hg.node_degrees().items():
        stubs.extend([node] * deg)

    rng.shuffle(stubs)

    new_edges = []
    idx = 0
    for s in sizes:
        if idx + s > len(stubs):
            break
        edge_nodes = frozenset(stubs[idx:idx + s])
        new_edges.append(edge_nodes)
        idx += s

    nodes = set()
    for e in new_edges:
        nodes.update(e)

    return Hypergraph(
        nodes=nodes,
        hyperedges=new_edges,
        timestamps=[None] * len(new_edges),
        metadata={"source": "configuration_model", "seed": seed},
    )


def null_model_ensemble(
    hg: Hypergraph,
    n_samples: int = 100,
    name_prefix: str = "null",
    triadic_sample: int = 20000,
) -> list[TopologyReport]:
    """
    Generate an ensemble of null-model hypergraphs and compute topology.
    """
    logger.info("Generating %d null model samples...", n_samples)
    reports = []
    for i in range(n_samples):
        hg_null = configuration_model_sample(hg, seed=i)
        report = compute_topology(hg_null, name=f"{name_prefix}_{i}",
                                  triadic_sample=triadic_sample)
        reports.append(report)
        if (i + 1) % 10 == 0:
            logger.info("  Completed %d/%d null samples", i + 1, n_samples)
    return reports


def compute_zscores(
    observed: TopologyReport,
    null_reports: list[TopologyReport],
) -> dict[str, tuple[float, float, float]]:
    """
    Compute z-scores for observed metrics against null model ensemble.

    Returns dict of metric_name -> (observed_value, z_score, p_value).
    """
    metrics = [
        "edge_size_mean", "edge_size_median", "frac_dyadic", "frac_higher_order",
        "hyperdegree_mean", "hyperdegree_gini",
        "triadic_closure_rate", "mean_edge_overlap",
    ]

    results = {}
    for metric in metrics:
        obs_val = getattr(observed, metric)
        null_vals = np.array([getattr(r, metric) for r in null_reports])
        null_mean = null_vals.mean()
        null_std = null_vals.std()

        if null_std > 1e-10:
            z = (obs_val - null_mean) / null_std
            p = 2 * (1 - _standard_normal_cdf(abs(z)))
        else:
            z = 0.0
            p = 1.0

        results[metric] = (obs_val, z, p)

    return results


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def format_zscores(
    zscores: dict[str, tuple[float, float, float]],
    name: str = "",
) -> str:
    """Pretty-print z-score table."""
    lines = [f"Z-scores for {name}" if name else "Z-scores", "-" * 65]
    lines.append(f"{'Metric':<28} {'Observed':>10} {'z-score':>10} {'p-value':>10}")
    lines.append("-" * 65)
    for metric, (obs, z, p) in zscores.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"{metric:<28} {obs:>10.4f} {z:>10.2f} {p:>10.4f} {sig}")
    lines.append("-" * 65)
    return "\n".join(lines)
