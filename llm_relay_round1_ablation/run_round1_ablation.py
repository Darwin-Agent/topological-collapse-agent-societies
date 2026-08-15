#!/usr/bin/env python3
"""Run a matched, one-round capacity ablation for the relay benchmark.

This is a post hoc mechanism ablation, not an independent held-out replication:
it reuses the frozen task manifest and model labels from llm_relay_benchmark,
while changing only the relay budget from two synchronous rounds to one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_BENCHMARK_DIR = ROOT.parent / "llm_relay_benchmark"
sys.path.insert(0, str(ROOT.parent))

from llm_relay_benchmark import run_limited_relay_benchmark as relay


RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROTOCOL_PATH = RESULTS_DIR / "protocol_round1.json"
DESIGN_PATH = RESULTS_DIR / "design_diagnostics.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
FREEZE_TIMESTAMP_UTC = "2026-07-30T12:00:00+00:00"


def configure_relay_module() -> None:
    relay.N_ROUNDS = 1
    relay.RESULTS_DIR = RESULTS_DIR
    relay.RAW_DIR = RAW_DIR
    relay.DESIGN_PATH = DESIGN_PATH
    relay.SUMMARY_PATH = SUMMARY_PATH
    relay.utc_now = lambda: FREEZE_TIMESTAMP_UTC


def protocol_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "matched_one_round_relay_ablation",
        "status": "post_hoc_mechanism_ablation",
        "created_at_utc": FREEZE_TIMESTAMP_UTC,
        "source_benchmark": "llm_relay_benchmark",
        "source_manifest_sha256": relay.sha256_file(relay.MANIFEST_PATH),
        "source_two_round_protocol_sha256": relay.sha256_file(
            SOURCE_BENCHMARK_DIR / "results" / "protocol_full.json"
        ),
        "task_ids": manifest["task_blocks"]["heldout_confirmatory"],
        "models": [model.__dict__ for model in relay.DEFAULT_MODELS],
        "conditions": list(relay.CONDITIONS),
        "n_agents": relay.N_AGENTS,
        "n_rounds": 1,
        "relay_rule": (
            "At the single synchronous round, every agent broadcasts the "
            "lexicographically first card it has not previously broadcast. "
            "Payloads are selected before delivery to every co-member."
        ),
        "primary_endpoint": "Final-round run-level plurality accuracy; ties are incorrect.",
        "secondary_endpoints": [
            "individual accuracy",
            "sufficient-evidence agent fraction",
            "accuracy conditional on sufficient evidence",
            "Brier score",
            "API and parse failure rates",
        ],
        "temperature": 0,
        "max_tokens": relay.MAX_TOKENS,
        "interpretation_boundary": (
            "This matched ablation reuses the previously analysed task manifest. "
            "It is not an independent held-out replication and does not isolate "
            "topology from group size or centralisation."
        ),
    }


def freeze_protocol(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = protocol_payload(manifest)
    if PROTOCOL_PATH.exists():
        existing = relay.read_json(PROTOCOL_PATH)
        for key in (
            "source_manifest_sha256",
            "source_two_round_protocol_sha256",
            "task_ids",
            "models",
            "conditions",
            "n_rounds",
            "relay_rule",
            "primary_endpoint",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"Frozen protocol differs at {key}.")
    else:
        relay.write_json(PROTOCOL_PATH, payload)
    return payload


def all_results(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        relay.read_json(relay.run_path(model, task["task_id"], condition))
        for model in relay.DEFAULT_MODELS
        for task in relay.selected_tasks(manifest, "full")
        for condition in relay.CONDITIONS
    ]


def write_summary(manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    results = all_results(manifest)
    records = [relay.extract_metrics(result) for result in results]
    summary = {
        "schema_version": 1,
        "benchmark": "matched_one_round_relay_ablation",
        "status": "post_hoc_mechanism_ablation",
        "generated_at_utc": FREEZE_TIMESTAMP_UTC,
        "source_manifest_sha256": relay.sha256_file(relay.MANIFEST_PATH),
        "protocol_sha256": relay.sha256_file(PROTOCOL_PATH),
        "source_two_round_summary": "../../llm_relay_benchmark/results/summary.json",
        "n_expected_cells": len(relay.DEFAULT_MODELS)
        * len(relay.selected_tasks(manifest, "full"))
        * len(relay.CONDITIONS),
        "n_completed_cells": len(records),
        "records": records,
        "by_model_condition": relay.grouped_summary(records),
        "paired_group_accuracy_contrasts": relay.paired_contrasts(records),
        "deterministic_access": relay.deterministic_access_diagnostics(manifest),
        "failure_counts": {
            "api_error_calls": int(
                sum(
                    result["execution"]["n_api_calls_expected"]
                    - result["execution"]["n_api_calls_completed"]
                    for result in results
                )
            ),
            "expected_calls": int(
                sum(result["execution"]["n_api_calls_expected"] for result in results)
            ),
            "parsed_final_decisions": int(
                sum(
                    1
                    for result in results
                    for agent in result["rounds"][-1]["agents"]
                    if agent["parsed"]["parse_status"] == "ok"
                )
            ),
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    relay.write_json(SUMMARY_PATH, summary)
    return summary


async def execute(manifest: dict[str, Any], force: bool, concurrency: int) -> None:
    if not os.environ.get("LLM_GATEWAY_API_KEY"):
        raise RuntimeError("LLM_GATEWAY_API_KEY is required and must be set in the environment.")
    if not relay.API_URL:
        raise RuntimeError(
            "LLM_GATEWAY_API_URL is required and must point to a chat-completions endpoint."
        )

    semaphore = asyncio.Semaphore(concurrency)
    jobs = [
        relay.run_one(task, model, condition, semaphore, force)
        for model in relay.DEFAULT_MODELS
        for task in relay.selected_tasks(manifest, "full")
        for condition in relay.CONDITIONS
    ]
    await asyncio.gather(*jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("design", "full"),
        required=True,
        help="Freeze design artifacts only, or execute the matched one-round cells.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing raw cells.")
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    configure_relay_module()
    manifest = relay.load_or_create_manifest()
    protocol = freeze_protocol(manifest)

    if args.mode == "design":
        diagnostics = {
            "schema_version": 1,
            "benchmark": "matched_one_round_relay_ablation",
            "status": "post_hoc_mechanism_ablation",
            "generated_at_utc": FREEZE_TIMESTAMP_UTC,
            "source_manifest_sha256": relay.sha256_file(relay.MANIFEST_PATH),
            "protocol_sha256": relay.sha256_file(PROTOCOL_PATH),
            "deterministic_access": relay.deterministic_access_diagnostics(manifest),
            "interpretation_boundary": protocol["interpretation_boundary"],
        }
        relay.write_json(DESIGN_PATH, diagnostics)
        print("Frozen one-round ablation protocol and diagnostics; no API calls made.")
        return

    asyncio.run(execute(manifest, args.force, args.concurrency))
    summary = write_summary(manifest, protocol)
    print(
        "Completed "
        f"{summary['n_completed_cells']}/{summary['n_expected_cells']} one-round cells."
    )


if __name__ == "__main__":
    main()
