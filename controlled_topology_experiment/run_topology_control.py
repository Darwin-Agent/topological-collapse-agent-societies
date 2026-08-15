#!/usr/bin/env python3
"""Run a topology-only hypergraph diffusion control experiment.

The two constructions have exactly the same node set, hyperdegree sequence,
number of hyperedges and hyperedge size (all groups have four members). They
differ only in degree mixing inside groups:

* high-equality: predominantly degree-homogeneous groups, with sparse
  degree-preserving cross-class swaps to keep the construction connected;
* low-equality: degree-heterogeneous groups assembled from the same incidence
  budgets.

Both conditions use the same synchronous, monotonic reinforcement update.
Results are deterministic under the declared master seed and are written next
to this script. No external data or LLM calls are used.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "controlled_topology_experiment" / "results"
IMAGES = ROOT / "images"
OUTDIR.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260730
N_ROUNDS = 60
N_GRAPH_PAIRS = 20
N_DYNAMICS_REPLICATES = 24
SEED_FRACTIONS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)
MAJORITY_PROBABILITY = 0.10
MAJORITY_SENSITIVITY = (0.10, 0.20, 0.30)

COLORS = {
    "high_equality": "#2F6FC0",
    "low_equality": "#C74B45",
    "ink": "#242424",
    "grid": "#D5D5D5",
    "muted": "#6C6C6C",
}


@dataclass(frozen=True)
class Hypergraph:
    """A compact immutable representation for the controlled simulations."""

    name: str
    degrees: dict[int, int]
    edges: tuple[tuple[int, ...], ...]


def _class_nodes() -> dict[str, list[int]]:
    """Create a heterogeneous, fixed hyperdegree sequence.

    Incidence totals are H=384, M=512 and L=320, summing to 1,216 slots.
    With four members per hyperedge this gives exactly 304 hyperedges.
    """

    high = list(range(0, 48))  # 48 x 8 = 384 slots
    medium = list(range(48, 176))  # 128 x 4 = 512 slots
    low = list(range(176, 336))  # 160 x 2 = 320 slots
    return {"H": high, "M": medium, "L": low}


def _degrees(classes: dict[str, list[int]]) -> dict[int, int]:
    result = {}
    for label, degree in (("H", 8), ("M", 4), ("L", 2)):
        result.update({node: degree for node in classes[label]})
    return result


def _assign_label_slots(
    edges: list[list[int]],
    edge_specs: list[tuple[str, ...]],
    label: str,
    nodes: list[int],
    degree: int,
    rng: np.random.Generator,
) -> None:
    """Assign one label's exact degree budget without rejection sampling.

    Each permutation layer uses every node exactly once. Slots belonging to
    repeated copies of the same class in one hyperedge are kept in the same
    layer, which makes duplicate members impossible by construction.
    """

    singleton_edges = []
    paired_edges = []
    for edge_id, spec in enumerate(edge_specs):
        count = spec.count(label)
        if count == 1:
            singleton_edges.append(edge_id)
        elif count == 2:
            paired_edges.append(edge_id)
        elif count > 2:
            raise RuntimeError("This controlled construction supports at most two class copies per edge.")

    if len(paired_edges) % degree or len(singleton_edges) % degree:
        raise RuntimeError(f"{label} slots cannot be distributed over degree layers.")
    pairs_per_layer = len(paired_edges) // degree
    singletons_per_layer = len(singleton_edges) // degree
    if 2 * pairs_per_layer + singletons_per_layer != len(nodes):
        raise RuntimeError(f"{label} layer does not consume each node exactly once.")

    rng.shuffle(paired_edges)
    rng.shuffle(singleton_edges)
    for layer in range(degree):
        layer_slots = []
        pair_start = layer * pairs_per_layer
        for edge_id in paired_edges[pair_start : pair_start + pairs_per_layer]:
            layer_slots.extend((edge_id, edge_id))
        single_start = layer * singletons_per_layer
        layer_slots.extend(
            singleton_edges[single_start : single_start + singletons_per_layer]
        )
        if len(layer_slots) != len(nodes):
            raise RuntimeError(f"{label} layer has the wrong number of slots.")
        for edge_id, node in zip(layer_slots, rng.permutation(nodes)):
            edges[edge_id].append(int(node))


def _build_low_equality(
    classes: dict[str, list[int]], rng: np.random.Generator
) -> tuple[tuple[int, ...], ...]:
    """Build degree-heterogeneous groups from exact incidence budgets.

    The edge-type counts solve the fixed incidence constraints exactly:
    208 x HMML, 80 x HHML and 16 x HMLL. Layered permutations consume every
    node's incidence budget exactly and avoid duplicates without retries.
    """

    edge_specs = [("H", "M", "M", "L")] * 208
    edge_specs += [("H", "H", "M", "L")] * 80
    edge_specs += [("H", "M", "L", "L")] * 16
    rng.shuffle(edge_specs)
    edges: list[list[int]] = [[] for _ in edge_specs]
    for label, degree in (("H", 8), ("M", 4), ("L", 2)):
        _assign_label_slots(edges, edge_specs, label, classes[label], degree, rng)

    if any(len(edge) != 4 or len(set(edge)) != 4 for edge in edges):
        raise RuntimeError("Low-equality construction produced an invalid hyperedge.")
    return tuple(tuple(sorted(edge)) for edge in edges)


def _homogeneous_groups(
    nodes: list[int], degree: int, rng: np.random.Generator
) -> list[list[int]]:
    """Create exact-degree four-node groups using independent permutations."""

    groups = []
    if len(nodes) % 4:
        raise RuntimeError("Class size must be divisible by the group size.")
    for _ in range(degree):
        permutation = rng.permutation(nodes)
        groups.extend(
            permutation[start : start + 4].astype(int).tolist()
            for start in range(0, len(nodes), 4)
        )
    return groups


def _build_high_equality(
    classes: dict[str, list[int]], rng: np.random.Generator
) -> tuple[tuple[int, ...], ...]:
    """Build mostly degree-homogeneous groups with sparse cross-class swaps."""

    edges = (
        _homogeneous_groups(classes["H"], 8, rng)
        + _homogeneous_groups(classes["M"], 4, rng)
        + _homogeneous_groups(classes["L"], 2, rng)
    )

    # Preserve every node degree and every edge size while joining degree classes.
    edges = [list(edge) for edge in edges]
    class_of = {
        **{node: "H" for node in classes["H"]},
        **{node: "M" for node in classes["M"]},
        **{node: "L" for node in classes["L"]},
    }
    edge_classes = [class_of[edge[0]] for edge in edges]
    swaps = (("H", "M"), ("M", "L"), ("H", "L"))
    for left_class, right_class in swaps:
        left_ids = [i for i, label in enumerate(edge_classes) if label == left_class]
        right_ids = [i for i, label in enumerate(edge_classes) if label == right_class]
        for _ in range(12):
            for _attempt in range(1_000):
                left_idx = int(rng.choice(left_ids))
                right_idx = int(rng.choice(right_ids))
                left_pos = int(rng.integers(0, 4))
                right_pos = int(rng.integers(0, 4))
                left_node = edges[left_idx][left_pos]
                right_node = edges[right_idx][right_pos]
                if (
                    right_node not in edges[left_idx]
                    and left_node not in edges[right_idx]
                ):
                    edges[left_idx][left_pos] = right_node
                    edges[right_idx][right_pos] = left_node
                    break
            else:
                raise RuntimeError("Could not find a valid degree-preserving swap.")

    if any(len(edge) != 4 or len(set(edge)) != 4 for edge in edges):
        raise RuntimeError("High-equality construction produced an invalid hyperedge.")
    return tuple(tuple(sorted(edge)) for edge in edges)


def _node_degrees(edges: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    degrees: dict[int, int] = defaultdict(int)
    for edge in edges:
        for node in edge:
            degrees[node] += 1
    return dict(degrees)


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, len(values) + 1)
    return float((2 * np.sum(index * values) / (len(values) * values.sum())) - (len(values) + 1) / len(values))


def _degree_equality_values(hg: Hypergraph) -> np.ndarray:
    """Return one minus the standard Gini coefficient within each hyperedge."""
    values = []
    for edge in hg.edges:
        values.append(1.0 - _gini(np.array([hg.degrees[node] for node in edge])))
    return np.asarray(values)


def _edge_overlap(hg: Hypergraph) -> float:
    node_to_edges: dict[int, list[int]] = defaultdict(list)
    for edge_id, edge in enumerate(hg.edges):
        for node in edge:
            node_to_edges[node].append(edge_id)

    pairs: set[tuple[int, int]] = set()
    for edge_ids in node_to_edges.values():
        pairs.update(combinations(sorted(edge_ids), 2))

    if not pairs:
        return 0.0
    overlaps = []
    for left, right in pairs:
        a, b = set(hg.edges[left]), set(hg.edges[right])
        overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps))


def _components(hg: Hypergraph) -> int:
    adjacency: dict[int, set[int]] = {node: set() for node in hg.degrees}
    for edge in hg.edges:
        for node in edge:
            adjacency[node].update(member for member in edge if member != node)

    unseen = set(adjacency)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbours = adjacency[node] & unseen
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
    return count


def topology_summary(hg: Hypergraph) -> dict[str, float | int]:
    degree_equality = _degree_equality_values(hg)
    return {
        "nodes": len(hg.degrees),
        "hyperedges": len(hg.edges),
        "hyperedge_size": len(hg.edges[0]),
        "incidences": sum(len(edge) for edge in hg.edges),
        "hyperdegree_gini": _gini(np.array(list(hg.degrees.values()))),
        "mean_degree_equality": float(degree_equality.mean()),
        "median_degree_equality": float(np.median(degree_equality)),
        "mean_overlap": _edge_overlap(hg),
        "components": _components(hg),
    }


def simulate(
    hg: Hypergraph,
    seed_fraction: float,
    majority_probability: float,
    initial_seed: int,
    update_seed: int,
) -> np.ndarray:
    """Run a common synchronous, monotonic group-reinforcement process."""

    # Node identifiers are deliberately contiguous in this synthetic design.
    n_nodes = len(hg.degrees)
    indexed_edges = np.asarray(hg.edges, dtype=int)

    initial_rng = np.random.default_rng(initial_seed)
    update_rng = np.random.default_rng(update_seed)
    states = np.zeros(n_nodes, dtype=bool)
    n_initial = max(1, int(round(n_nodes * seed_fraction)))
    states[initial_rng.choice(n_nodes, size=n_initial, replace=False)] = True
    trajectory = np.empty(N_ROUNDS + 1, dtype=float)
    trajectory[0] = states.mean()

    for round_index in range(N_ROUNDS):
        adopted_per_edge = states[indexed_edges].sum(axis=1)
        exposures = np.zeros(n_nodes, dtype=float)

        # With groups of four, exactly three adopters supplies certain
        # reinforcement to the one remaining susceptible member.
        all_reinforced = indexed_edges[adopted_per_edge == 3]
        if len(all_reinforced):
            exposures[all_reinforced.ravel()] = 1.0

        # Exactly two adopters satisfy strict-majority reinforcement for a
        # susceptible group member. Exposures from distinct groups combine.
        majority_reinforced = indexed_edges[adopted_per_edge == 2]
        if len(majority_reinforced):
            n_majority_exposures = np.bincount(
                majority_reinforced.ravel(), minlength=n_nodes
            )
            p_majority = 1.0 - (1.0 - majority_probability) ** n_majority_exposures
            exposures = np.maximum(exposures, p_majority)

        states |= update_rng.random(n_nodes) < exposures
        trajectory[round_index + 1] = states.mean()

    return trajectory


def simulate_batch(
    hg: Hypergraph,
    seed_fraction: float,
    majority_probability: float,
    initial_uniforms: np.ndarray,
    update_uniforms: np.ndarray,
) -> np.ndarray:
    """Run independent paired trajectories together using supplied uniforms."""

    n_replicates, n_nodes = initial_uniforms.shape
    if n_nodes != len(hg.degrees):
        raise ValueError("Initial uniforms do not match the graph size.")
    if update_uniforms.shape != (N_ROUNDS, n_replicates, n_nodes):
        raise ValueError("Update uniforms have an unexpected shape.")

    indexed_edges = np.asarray(hg.edges, dtype=int)
    n_initial = max(1, int(round(n_nodes * seed_fraction)))
    states = np.zeros((n_replicates, n_nodes), dtype=bool)
    initial_indices = np.argpartition(initial_uniforms, n_initial - 1, axis=1)[
        :, :n_initial
    ]
    states[np.arange(n_replicates)[:, None], initial_indices] = True
    trajectories = np.empty((n_replicates, N_ROUNDS + 1), dtype=float)
    trajectories[:, 0] = states.mean(axis=1)

    for round_index in range(N_ROUNDS):
        adopted_per_edge = states[:, indexed_edges].sum(axis=2)
        exposures = np.zeros((n_replicates, n_nodes), dtype=float)

        fully_reinforced = np.nonzero(adopted_per_edge == 3)
        if fully_reinforced[0].size:
            exposures[
                fully_reinforced[0][:, None], indexed_edges[fully_reinforced[1]]
            ] = 1.0

        majority_reinforced = np.nonzero(adopted_per_edge == 2)
        if majority_reinforced[0].size:
            exposure_counts = np.zeros((n_replicates, n_nodes), dtype=np.int16)
            np.add.at(
                exposure_counts,
                (
                    majority_reinforced[0][:, None],
                    indexed_edges[majority_reinforced[1]],
                ),
                1,
            )
            p_majority = 1.0 - (1.0 - majority_probability) ** exposure_counts
            exposures = np.maximum(exposures, p_majority)

        states |= update_uniforms[round_index] < exposures
        trajectories[:, round_index + 1] = states.mean(axis=1)

    return trajectories


def _bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI across graph pairs."""

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(5_000, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, (0.025, 0.975)))


def _make_graph_pair(pair_index: int) -> tuple[Hypergraph, Hypergraph]:
    """Construct one independently randomised, degree-matched topology pair."""

    classes = _class_nodes()
    expected_degrees = _degrees(classes)
    high_equality = Hypergraph(
        "high_equality",
        expected_degrees,
        _build_high_equality(
            classes, np.random.default_rng(MASTER_SEED + 10_000 * pair_index + 1)
        ),
    )
    low_equality = Hypergraph(
        "low_equality",
        expected_degrees,
        _build_low_equality(
            classes, np.random.default_rng(MASTER_SEED + 10_000 * pair_index + 2)
        ),
    )
    for hg in (high_equality, low_equality):
        if _node_degrees(hg.edges) != expected_degrees:
            raise RuntimeError(f"{hg.name} does not preserve the fixed degree sequence.")
        if any(len(edge) != 4 or len(set(edge)) != 4 for edge in hg.edges):
            raise RuntimeError(f"{hg.name} has an invalid hyperedge.")
        if _components(hg) != 1:
            raise RuntimeError(f"{hg.name} is not connected.")
    return high_equality, low_equality


def run_experiment() -> tuple[dict, dict[str, list[np.ndarray]]]:
    """Run common-random-number simulations across independent graph pairs."""

    collected: dict[str, dict[str, dict[str, list]]] = {
        f"{probability:.2f}": {
            f"{seed_fraction:.2f}": {
                "high_final": [],
                "low_final": [],
                "high_trajectory": [],
                "low_trajectory": [],
                "graph_pair_differences": [],
                "graph_pair_high_means": [],
                "graph_pair_low_means": [],
            }
            for seed_fraction in SEED_FRACTIONS
        }
        for probability in MAJORITY_SENSITIVITY
    }
    topology_samples = {"high_equality": [], "low_equality": []}

    for pair_index in range(N_GRAPH_PAIRS):
        high_equality, low_equality = _make_graph_pair(pair_index)
        topology_samples["high_equality"].append(_degree_equality_values(high_equality))
        topology_samples["low_equality"].append(_degree_equality_values(low_equality))

        for probability in MAJORITY_SENSITIVITY:
            probability_label = f"{probability:.2f}"
            for seed_fraction in SEED_FRACTIONS:
                seed_label = f"{seed_fraction:.2f}"
                seed_base = int(
                    MASTER_SEED
                    + pair_index * 1_000_000
                    + round(probability * 10_000) * 10_000
                    + round(seed_fraction * 10_000) * 100
                )
                paired_rng = np.random.default_rng(seed_base)
                initial_uniforms = paired_rng.random(
                    (N_DYNAMICS_REPLICATES, len(high_equality.degrees))
                )
                update_uniforms = paired_rng.random(
                    (N_ROUNDS, N_DYNAMICS_REPLICATES, len(high_equality.degrees))
                )
                # Both conditions receive identical initial adopters and
                # node-level random uniforms at every synchronous update.
                high_trajectories = simulate_batch(
                    high_equality,
                    seed_fraction=seed_fraction,
                    majority_probability=probability,
                    initial_uniforms=initial_uniforms,
                    update_uniforms=update_uniforms,
                )
                low_trajectories = simulate_batch(
                    low_equality,
                    seed_fraction=seed_fraction,
                    majority_probability=probability,
                    initial_uniforms=initial_uniforms,
                    update_uniforms=update_uniforms,
                )
                high_finals = high_trajectories[:, -1]
                low_finals = low_trajectories[:, -1]

                entry = collected[probability_label][seed_label]
                entry["high_final"].extend(high_finals.tolist())
                entry["low_final"].extend(low_finals.tolist())
                entry["high_trajectory"].extend(high_trajectories)
                entry["low_trajectory"].extend(low_trajectories)
                entry["graph_pair_high_means"].append(float(high_finals.mean()))
                entry["graph_pair_low_means"].append(float(low_finals.mean()))
                entry["graph_pair_differences"].append(
                    float((high_finals - low_finals).mean())
                )

    summaries: dict[str, dict] = {}
    for probability in MAJORITY_SENSITIVITY:
        probability_label = f"{probability:.2f}"
        summaries[probability_label] = {}
        for seed_fraction in SEED_FRACTIONS:
            seed_label = f"{seed_fraction:.2f}"
            entry = collected[probability_label][seed_label]
            high_final = np.asarray(entry["high_final"])
            low_final = np.asarray(entry["low_final"])
            graph_high = np.asarray(entry["graph_pair_high_means"])
            graph_low = np.asarray(entry["graph_pair_low_means"])
            graph_differences = np.asarray(entry["graph_pair_differences"])
            summaries[probability_label][seed_label] = {
                "high_equality_final_mean": float(high_final.mean()),
                "high_equality_final_sd": float(high_final.std(ddof=1)),
                "high_equality_final_graph_pair_ci95": _bootstrap_ci(
                    graph_high, MASTER_SEED + 17
                ),
                "low_equality_final_mean": float(low_final.mean()),
                "low_equality_final_sd": float(low_final.std(ddof=1)),
                "low_equality_final_graph_pair_ci95": _bootstrap_ci(
                    graph_low, MASTER_SEED + 23
                ),
                "mean_difference_high_equality_minus_low_equality": float(
                    graph_differences.mean()
                ),
                "difference_graph_pair_ci95": _bootstrap_ci(
                    graph_differences, MASTER_SEED + 29
                ),
                "fraction_graph_pairs_high_gt_low": float(
                    np.mean(graph_differences > 0)
                ),
                "high_equality_trajectory_mean": np.mean(
                    entry["high_trajectory"], axis=0
                ).tolist(),
                "low_equality_trajectory_mean": np.mean(
                    entry["low_trajectory"], axis=0
                ).tolist(),
                "n_graph_pairs": N_GRAPH_PAIRS,
                "n_dynamics_replicates_per_graph_pair": N_DYNAMICS_REPLICATES,
                "n_runs_per_condition": len(high_final),
            }
    return summaries, topology_samples


def _plot(results: dict, topology_samples: dict[str, list[np.ndarray]]) -> None:
    """Render a compact publication figure with an explicit control audit."""

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.7, 4.25), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    low_pair_means = np.asarray(
        [values.mean() for values in topology_samples["low_equality"]]
    )
    high_pair_means = np.asarray(
        [values.mean() for values in topology_samples["high_equality"]]
    )
    pair_jitter = np.linspace(-0.060, 0.060, len(low_pair_means))
    for low_value, high_value, jitter in zip(
        low_pair_means, high_pair_means, pair_jitter
    ):
        ax_a.plot(
            [jitter, 1 + jitter],
            [low_value, high_value],
            color="#C9C9C9",
            linewidth=0.65,
            alpha=0.75,
            zorder=1,
        )
    ax_a.scatter(
        pair_jitter,
        low_pair_means,
        color=COLORS["low_equality"],
        edgecolors="white",
        linewidths=0.35,
        s=18,
        alpha=0.82,
        zorder=2,
    )
    ax_a.scatter(
        1 + pair_jitter,
        high_pair_means,
        color=COLORS["high_equality"],
        edgecolors="white",
        linewidths=0.35,
        s=18,
        alpha=0.82,
        zorder=2,
    )
    for position, values, color in (
        (0, low_pair_means, COLORS["low_equality"]),
        (1, high_pair_means, COLORS["high_equality"]),
    ):
        mean = values.mean()
        interval = np.quantile(values, (0.025, 0.975))
        ax_a.errorbar(
            position,
            mean,
            yerr=np.array([[mean - interval[0]], [interval[1] - mean]]),
            color=color,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.25,
            markersize=6.1,
            capsize=2.6,
            linewidth=1.15,
            zorder=3,
        )
        ax_a.text(
            position,
            interval[1] + 0.010,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
            fontweight="bold",
        )
    ax_a.set_xlim(-0.30, 1.30)
    ax_a.set_xticks([0, 1], ["Low equality", "High equality"])
    ax_a.set_ylabel("Within-group degree equality")
    ax_a.set_ylim(0.63, 1.03)
    ax_a.set_title("Matched group equality", loc="left")

    ax_b.axis("off")
    ax_b.set_title("Exact design controls", loc="left")
    rows = [
        ("Nodes", "336"),
        ("Hyperedges", "304"),
        ("Hyperedge size", "4"),
        ("Incidences", "1,216"),
        ("Node hyperdegree", "identical"),
        ("Connected components", "1"),
    ]
    ax_b.text(
        0.04,
        0.93,
        "Low equality",
        color=COLORS["low_equality"],
        ha="left",
        va="center",
        fontweight="bold",
        fontsize=7.2,
        transform=ax_b.transAxes,
    )
    ax_b.text(
        0.50,
        0.93,
        "=",
        color=COLORS["muted"],
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10.0,
        transform=ax_b.transAxes,
    )
    ax_b.text(
        0.96,
        0.93,
        "High equality",
        color=COLORS["high_equality"],
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=7.2,
        transform=ax_b.transAxes,
    )
    ax_b.text(
        0.04,
        0.81,
        "Matched feature",
        color=COLORS["muted"],
        ha="left",
        va="center",
        fontweight="bold",
        fontsize=7.0,
        transform=ax_b.transAxes,
    )
    ax_b.text(
        0.96,
        0.81,
        "Fixed value",
        color=COLORS["muted"],
        ha="right",
        va="center",
        fontweight="bold",
        fontsize=7.0,
        transform=ax_b.transAxes,
    )
    for index, (label, value) in enumerate(rows):
        y = 0.70 - index * 0.115
        ax_b.plot(
            (0.03, 0.97),
            (y - 0.056, y - 0.056),
            color="#E4E4E4",
            linewidth=0.55,
            transform=ax_b.transAxes,
            clip_on=False,
        )
        ax_b.text(0.04, y, label, ha="left", va="center", transform=ax_b.transAxes)
        ax_b.text(0.96, y, value, ha="right", va="center", transform=ax_b.transAxes)

    focal = results[f"{MAJORITY_PROBABILITY:.2f}"]
    for topology, color, label in (
        ("low_equality", COLORS["low_equality"], "Low equality"),
        ("high_equality", COLORS["high_equality"], "High equality"),
    ):
        means = []
        low_ci = []
        high_ci = []
        for seed_fraction in SEED_FRACTIONS:
            entry = focal[f"{seed_fraction:.2f}"]
            means.append(entry[f"{topology}_final_mean"])
            ci = entry[f"{topology}_final_graph_pair_ci95"]
            low_ci.append(ci[0])
            high_ci.append(ci[1])
        means_array = np.asarray(means)
        ax_c.plot(
            SEED_FRACTIONS,
            means_array,
            "o-",
            color=color,
            label=label,
            linewidth=1.6,
            markersize=3.8,
        )
        ax_c.fill_between(
            SEED_FRACTIONS,
            low_ci,
            high_ci,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax_c.axhline(0.5, linestyle=":", color=COLORS["muted"], linewidth=0.8)
    ax_c.set(
        xlabel="Initial adopter fraction",
        ylabel="Final adoption",
        ylim=(-0.03, 1.03),
    )
    ax_c.set_xlim(0.015, 0.255)
    ax_c.set_xticks((0.02, 0.05, 0.10, 0.20, 0.25))
    ax_c.set_xticklabels(("0.02", "0.05", "0.10", "0.20", "0.25"), fontsize=6.9)
    ax_c.get_xticklabels()[0].set_ha("right")
    ax_c.get_xticklabels()[1].set_ha("left")
    ax_c.set_title(r"Common rule at $p_\mathrm{majority}=0.10$", loc="left")
    ax_c.legend(loc="lower right", frameon=False, handlelength=1.4)

    difference_matrix = np.array(
        [
            [
                results[f"{probability:.2f}"][f"{seed_fraction:.2f}"][
                    "mean_difference_high_equality_minus_low_equality"
                ]
                for seed_fraction in SEED_FRACTIONS
            ]
            for probability in MAJORITY_SENSITIVITY
        ]
    )
    limit = max(0.10, float(np.max(np.abs(difference_matrix))))
    image = ax_d.imshow(
        difference_matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row in range(difference_matrix.shape[0]):
        for column in range(difference_matrix.shape[1]):
            value = difference_matrix[row, column]
            text_color = "white" if abs(value) > limit * 0.52 else COLORS["ink"]
            ax_d.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.3,
            )
    ax_d.set_xticks(
        range(len(SEED_FRACTIONS)), [f"{x:.02f}" for x in SEED_FRACTIONS]
    )
    ax_d.set_yticks(
        range(len(MAJORITY_SENSITIVITY)),
        [f"{x:.02f}" for x in MAJORITY_SENSITIVITY],
    )
    ax_d.set_xticks(np.arange(-0.5, len(SEED_FRACTIONS), 1), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(MAJORITY_SENSITIVITY), 1), minor=True)
    ax_d.grid(which="minor", color="white", linewidth=0.8)
    ax_d.tick_params(which="minor", bottom=False, left=False)
    ax_d.set_xlabel("Initial adopter fraction")
    ax_d.set_ylabel(r"$p_\mathrm{majority}$")
    ax_d.set_title("High-minus-low adoption contrast", loc="left")
    colourbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.04)
    colourbar.ax.tick_params(labelsize=7.0)
    colourbar.set_label(r"$\Delta$ final adoption", fontsize=7.5)

    label_positions = ((-0.15, 1.06), (-0.08, 1.06), (-0.15, 1.06), (-0.31, 1.06))
    for index, (axis, (x, y)) in enumerate(zip((ax_a, ax_b, ax_c, ax_d), label_positions)):
        axis.text(
            x,
            y,
            "abcd"[index],
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10.0,
        )
    for axis in (ax_a, ax_c):
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)

    fig.savefig(
        IMAGES / "fig5_degree_matched_topology_control.pdf",
        facecolor="white",
        pad_inches=0.025,
        metadata={
            "Title": "Degree-matched synthetic topology control",
            "Creator": "run_topology_control.py",
            "CreationDate": datetime(2026, 7, 30, 12, 0, 0),
        },
    )
    fig.savefig(
        IMAGES / "fig5_degree_matched_topology_control.png",
        dpi=300,
        facecolor="white",
        pad_inches=0.025,
        metadata={
            "Title": "Degree-matched synthetic topology control",
            "Software": "run_topology_control.py",
        },
    )
    plt.close(fig)


def main() -> None:
    results, topology_samples = run_experiment()
    high_equality, low_equality = _make_graph_pair(0)
    output = {
        "design": {
            "master_seed": MASTER_SEED,
            "n_rounds": N_ROUNDS,
            "n_graph_pairs": N_GRAPH_PAIRS,
            "n_dynamics_replicates_per_graph_pair": N_DYNAMICS_REPLICATES,
            "n_runs_per_condition": N_GRAPH_PAIRS * N_DYNAMICS_REPLICATES,
            "seed_fractions": list(SEED_FRACTIONS),
            "majority_probabilities": list(MAJORITY_SENSITIVITY),
            "update_rule": (
                "Synchronous monotonic adoption. A susceptible agent adopts with "
                "probability 1 when every other member of any four-agent group has "
                "adopted, and with p_majority when a strict majority of the other "
                "members has adopted. Multiple group exposures combine independently."
            ),
        },
        "topology": {
            "example_pair": {
                "high_equality": topology_summary(high_equality),
                "low_equality": topology_summary(low_equality),
            },
            "degree_sequence_exactly_matched": True,
            "hyperedge_size_exactly_matched": True,
        },
        "results": results,
    }
    (OUTDIR / "topology_control_summary.json").write_text(json.dumps(output, indent=2))
    _plot(results, topology_samples)
    print(json.dumps(output["topology"], indent=2))
    print(f"Saved {OUTDIR / 'topology_control_summary.json'}")
    print(f"Saved {IMAGES / 'fig5_degree_matched_topology_control.pdf'}")


if __name__ == "__main__":
    main()
