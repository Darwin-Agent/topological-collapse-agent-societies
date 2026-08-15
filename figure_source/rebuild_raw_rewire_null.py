#!/usr/bin/env python3
"""Build a degree- and size-preserving rewiring null on the raw-file check.

The test uses the same alternative Moltbook raw-file pairing as Extended Data
Fig. 5. It is deliberately separate from the archived primary observation:
the two public source files do not share a frozen upstream revision.

Each accepted bipartite incidence swap preserves every node's hyperdegree and
every hyperedge's member count while rejecting duplicate node--hyperedge
memberships. The output is a frozen Monte Carlo summary for descriptive use.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


# Restart before importing the raw-data helper, whose construction uses sets.
if os.environ.get("PYTHONHASHSEED") != "0":
    deterministic_environment = os.environ.copy()
    deterministic_environment["PYTHONHASHSEED"] = "0"
    os.execvpe(
        sys.executable,
        [sys.executable, *sys.argv],
        deterministic_environment,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSTS = (
    ROOT.parent
    / "Code"
    / "data"
    / "raw"
    / "moltbook_hf"
    / "lnajt"
    / "posts.parquet"
)
DEFAULT_COMMENTS = (
    ROOT.parent
    / "Code"
    / "data"
    / "raw"
    / "moltbook_hf"
    / "moltnet"
    / "data"
    / "v2026-02-28"
    / "comments.parquet"
)

from rebuild_raw_snapshot_check import build_hyperedges, sha256, stable_mean_edge_overlap


def relative_project_data_path(path: Path) -> str:
    """Record source locations relative to the shared MLM project root."""
    try:
        return str(path.resolve().relative_to(ROOT.parent))
    except ValueError as error:
        raise ValueError(
            "Raw inputs must be located under the shared project root so the "
            "frozen provenance record remains portable."
        ) from error


def node_degrees(hyperedges: list[set[str]]) -> Counter[str]:
    return Counter(member for edge in hyperedges for member in edge)


def rewire_incidence_matrix(
    observed: list[frozenset[str]],
    seed: int,
    accepted_swaps_per_incidence: int,
) -> tuple[list[set[str]], dict[str, int | float]]:
    """Randomize simple hypergraph incidences with degree/size-preserving swaps."""
    if accepted_swaps_per_incidence <= 0:
        raise ValueError("accepted_swaps_per_incidence must be positive.")

    member_lists = [sorted(edge) for edge in observed]
    member_sets = [set(edge) for edge in member_lists]
    expected_sizes = tuple(len(edge) for edge in member_sets)
    expected_degrees = node_degrees(member_sets)
    total_incidences = sum(expected_sizes)
    target_swaps = total_incidences * accepted_swaps_per_incidence
    maximum_attempts = target_swaps * 25
    generator = random.Random(seed)
    randrange = generator.randrange
    edge_sets = member_sets
    edge_members = member_lists
    accepted = 0
    attempts = 0
    n_edges = len(member_sets)

    while accepted < target_swaps and attempts < maximum_attempts:
        attempts += 1
        left_index = randrange(n_edges)
        right_index = randrange(n_edges - 1)
        if right_index >= left_index:
            right_index += 1

        left = edge_sets[left_index]
        right = edge_sets[right_index]
        left_position = randrange(len(left))
        right_position = randrange(len(right))
        left_member = edge_members[left_index][left_position]
        right_member = edge_members[right_index][right_position]
        if (
            left_member == right_member
            or right_member in left
            or left_member in right
        ):
            continue

        left.remove(left_member)
        left.add(right_member)
        right.remove(right_member)
        right.add(left_member)
        edge_members[left_index][left_position] = right_member
        edge_members[right_index][right_position] = left_member
        accepted += 1

    if accepted != target_swaps:
        raise RuntimeError(
            "Could not complete the declared swap chain: "
            f"{accepted}/{target_swaps} accepted after {attempts} attempts."
        )
    if tuple(len(edge) for edge in member_sets) != expected_sizes:
        raise AssertionError("A swap changed the hyperedge-size sequence.")
    if node_degrees(member_sets) != expected_degrees:
        raise AssertionError("A swap changed the node-hyperdegree sequence.")

    return member_sets, {
        "seed": seed,
        "accepted_swaps": accepted,
        "attempts": attempts,
        "acceptance_rate": accepted / attempts,
    }


def empirical_tail_probability(values: np.ndarray, observed: float) -> dict[str, float]:
    """Return finite-ensemble one-sided and two-sided Monte Carlo tail values."""
    n_values = len(values)
    lower = (1 + int(np.count_nonzero(values <= observed))) / (n_values + 1)
    upper = (1 + int(np.count_nonzero(values >= observed))) / (n_values + 1)
    return {
        "lower_tail": lower,
        "upper_tail": upper,
        "two_sided": min(1.0, 2 * min(lower, upper)),
    }


def run_rewire_ensemble(
    observed: list[frozenset[str]],
    observed_overlap: float,
    null_samples: int,
    swaps_per_incidence: int,
) -> dict:
    """Run independently seeded fixed-length chains and freeze their summary."""
    sample_rows = []
    null_overlaps = []
    for seed in range(null_samples):
        rewired, diagnostics = rewire_incidence_matrix(
            observed,
            seed=seed,
            accepted_swaps_per_incidence=swaps_per_incidence,
        )
        overlap = stable_mean_edge_overlap(rewired)
        null_overlaps.append(overlap)
        sample_rows.append({**diagnostics, "mean_edge_overlap": overlap})

    values = np.asarray(null_overlaps, dtype=float)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "central_95_percent_interval": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "all_samples_below_observed": bool(np.all(values < observed_overlap)),
        "tail_probability": empirical_tail_probability(values, observed_overlap),
        "samples": sample_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", type=Path, default=DEFAULT_POSTS)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figure_source" / "data" / "raw_rewire_null.json",
    )
    parser.add_argument("--null-samples", type=int, default=80)
    parser.add_argument("--swaps-per-incidence", type=int, default=3)
    parser.add_argument(
        "--diagnostic-swaps-per-incidence",
        type=int,
        nargs="+",
        default=(1, 5),
        help="Independent chain lengths used only as descriptive diagnostics.",
    )
    parser.add_argument("--diagnostic-null-samples", type=int, default=20)
    parser.add_argument("--max-posts", type=int, default=50000)
    parser.add_argument("--reply-window-minutes", type=int, default=60)
    arguments = parser.parse_args()

    if arguments.null_samples < 20 or arguments.diagnostic_null_samples < 20:
        raise SystemExit("Each declared ensemble must contain at least 20 chains.")
    if any(length <= 0 for length in arguments.diagnostic_swaps_per_incidence):
        raise SystemExit("--diagnostic-swaps-per-incidence values must be positive.")
    diagnostic_lengths = tuple(
        sorted(
            set(arguments.diagnostic_swaps_per_incidence)
            - {arguments.swaps_per_incidence}
        )
    )

    observed, record_counts = build_hyperedges(
        arguments.posts,
        arguments.comments,
        arguments.max_posts,
        arguments.reply_window_minutes,
    )
    observed_overlap = stable_mean_edge_overlap(observed)
    primary_ensemble = run_rewire_ensemble(
        observed,
        observed_overlap,
        arguments.null_samples,
        arguments.swaps_per_incidence,
    )
    chain_length_diagnostics = [
        {
            "accepted_swaps_per_incidence": length,
            "null_samples": arguments.diagnostic_null_samples,
            **run_rewire_ensemble(
                observed,
                observed_overlap,
                arguments.diagnostic_null_samples,
                length,
            ),
        }
        for length in diagnostic_lengths
    ]
    result = {
        "schema_version": 1,
        "description": (
            "Degree- and hyperedge-size-preserving bipartite-incidence rewiring "
            "ensemble on the alternative local Moltbook raw-file pairing. This is "
            "a descriptive null-model check, not a re-estimation of the archived "
            "primary cross-platform analysis."
        ),
        "source_files": [
            {
                "dataset": "lnajt/moltbook",
                "relative_path": relative_project_data_path(arguments.posts),
                "n_records": record_counts["n_post_records"],
                "sha256": sha256(arguments.posts),
            },
            {
                "dataset": "iNLP-Lab/Moltbook-MoltNet",
                "relative_path": relative_project_data_path(arguments.comments),
                "n_records": record_counts["n_comment_records"],
                "sha256": sha256(arguments.comments),
            },
        ],
        "selection": {
            "max_posts_by_recorded_comment_count": arguments.max_posts,
            "reply_window_minutes": arguments.reply_window_minutes,
            "non_singleton_hyperedges_only": True,
        },
        "null_model": {
            "representation": "Simple bipartite node-hyperedge incidence matrix",
            "move": (
                "Swap two node incidences from distinct hyperedges only when both "
                "new memberships are absent."
            ),
            "preserved_exactly": [
                "Node hyperdegree sequence",
                "Hyperedge-size sequence",
                "Total incidence count",
            ],
            "null_samples": arguments.null_samples,
            "accepted_swaps_per_incidence": arguments.swaps_per_incidence,
            "total_incidences": sum(len(edge) for edge in observed),
            "overlap_sampling_seed": 42,
        },
        "chain_length_diagnostic": {
            "description": (
                "Independent shorter and longer fixed-swap chains from the "
                "observed incidence matrix. These summaries inspect sensitivity "
                "to the declared chain length only; they do not establish mixing "
                "or uniform configuration-model sampling."
            ),
            "results": chain_length_diagnostics,
        },
        "observed": {
            "n_nodes": len(node_degrees([set(edge) for edge in observed])),
            "n_hyperedges": len(observed),
            "mean_edge_overlap": observed_overlap,
        },
        "null_overlap": primary_ensemble,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {arguments.output}")
    print(
        f"Observed overlap={observed_overlap:.6f}; "
        f"null mean={primary_ensemble['mean']:.6f}; "
        f"null range=[{primary_ensemble['minimum']:.6f}, "
        f"{primary_ensemble['maximum']:.6f}]"
    )


if __name__ == "__main__":
    main()
