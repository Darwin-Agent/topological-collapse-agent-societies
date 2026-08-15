#!/usr/bin/env python3
"""Rebuild the six-trace SocioPatterns reference summary used by Fig. 2.

Each temporal contact file is aggregated with the same 300-s non-overlapping
window construction used for the original SFHH reference. This helper writes a
compact, frozen JSON summary and does not issue any network or API request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


# The imported topology helper internally stores node memberships in sets.
# Restart before importing it so raw-data reconstruction is stable even when
# launched from an interpreter with randomized hash ordering.
if os.environ.get("PYTHONHASHSEED") != "0":
    deterministic_environment = os.environ.copy()
    deterministic_environment["PYTHONHASHSEED"] = "0"
    os.execvpe(
        sys.executable,
        [sys.executable, *sys.argv],
        deterministic_environment,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT.parent / "Code" / "data" / "raw" / "sociopatterns" / "contact"
CODE_SRC = ROOT.parent / "Code" / "src"

if str(CODE_SRC) not in sys.path:
    sys.path.insert(0, str(CODE_SRC))

from analysis.hypergraph_builder import build_sociopatterns_hypergraph
from analysis.topology import compute_topology


TRACE_SPECS = (
    ("SP-InVS13", "tij_InVS13.dat"),
    ("SP-InVS15", "tij_InVS15.dat"),
    ("SP-LH10", "tij_LH10.dat"),
    ("SP-LyonSchool", "tij_LyonSchool.dat"),
    ("SP-SFHH", "tij_SFHH.dat"),
    ("SP-Thiers13", "tij_Thiers13.dat"),
)

METRICS = (
    "n_nodes",
    "n_edges",
    "edge_size_mean",
    "edge_size_median",
    "edge_size_max",
    "frac_higher_order",
    "hyperdegree_gini",
    "hyperdegree_mean",
    "hyperdegree_median",
    "hyperdegree_max",
    "mean_edge_overlap",
    "triadic_closure_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retained_metrics(report: object) -> dict[str, int | float]:
    report_dict = report.to_dict()
    return {key: report_dict[key] for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figure_source" / "data" / "fig2_sociopatterns_references.json",
    )
    parser.add_argument("--window-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.window_seconds <= 0:
        raise SystemExit("--window-seconds must be positive.")

    traces: dict[str, dict[str, object]] = {}
    for trace_name, filename in TRACE_SPECS:
        raw_path = args.raw_dir / filename
        if not raw_path.is_file():
            raise SystemExit(f"Missing SocioPatterns source file: {raw_path}")

        hypergraph = build_sociopatterns_hypergraph(
            str(raw_path),
            delta_seconds=args.window_seconds,
        )
        report = compute_topology(hypergraph, name=trace_name)
        traces[trace_name] = {
            "raw_filename": filename,
            "sha256": sha256(raw_path),
            "n_contact_events": hypergraph.metadata["n_contacts"],
            **retained_metrics(report),
        }

    output = {
        "schema_version": 1,
        "description": (
            "Frozen six-trace SocioPatterns contact-reference summary for Fig. 2. "
            "All traces use 300-s non-overlapping temporal aggregation from the "
            "first event timestamp."
        ),
        "construction": {
            "source_programme": "SocioPatterns temporal-contact datasets",
            "window_seconds": args.window_seconds,
            "window_alignment": "Non-overlapping windows start at each trace's first timestamp.",
            "hyperedge_rule": (
                "All participants in at least one recorded contact during a non-empty "
                "window form one hyperedge."
            ),
            "topology_random_seed": 42,
        },
        "traces": traces,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    for trace_name, metrics in traces.items():
        print(
            f"{trace_name}: n={metrics['n_nodes']}, m={metrics['n_edges']}, "
            f"HO={metrics['frac_higher_order']:.3f}, "
            f"Gini={metrics['hyperdegree_gini']:.3f}, "
            f"overlap={metrics['mean_edge_overlap']:.3f}"
        )


if __name__ == "__main__":
    main()
