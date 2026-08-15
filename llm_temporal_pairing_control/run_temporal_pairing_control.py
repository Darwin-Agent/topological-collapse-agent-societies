#!/usr/bin/env python3
"""Run a prospective temporal-pairing control for the evidence relay task.

The control holds the agent count, group size, number of groups per round,
per-agent broadcasts, total pair slots and four-round relay budget fixed. It
changes only whether each task-specific perfect matching repeats or rotates
through distinct partners across rounds.

Run --mode design before any model requests. Pilot tasks are reserved for
model and logging checks; the 16 held-out tasks are used only by --mode full.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from llm_relay_benchmark import run_limited_relay_benchmark as relay


RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "task_manifest.json"
DESIGN_PATH = RESULTS_DIR / "design_diagnostics.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
FREEZE_TIMESTAMP_UTC = "2026-07-31T12:00:00+00:00"
MANIFEST_SEED = 20260731
CONDITIONS = ("repeated_pairs", "rotating_pairs")


def matchings_for_task(task_id: str) -> list[list[list[int]]]:
    """Return a task-specific one-factorization of the eight participants."""
    generator = relay.np.random.default_rng(
        relay.stable_int("temporal_pairing_factorization", task_id)
    )
    circle = generator.permutation(relay.N_AGENTS).tolist()
    schedule: list[list[list[int]]] = []
    for _ in range(relay.N_AGENTS - 1):
        schedule.append(
            [
                sorted((circle[index], circle[-1 - index]))
                for index in range(relay.N_AGENTS // 2)
            ]
        )
        circle = [circle[0], circle[-1], *circle[1:-1]]
    return schedule


def groups_for_round(condition: str, task_id: str, round_index: int) -> list[list[int]]:
    """Repeat one matching or rotate through a common one-factorization."""
    schedule = matchings_for_task(task_id)
    if condition == "repeated_pairs":
        return schedule[0]
    if condition == "rotating_pairs":
        if round_index >= len(schedule):
            raise ValueError("The requested round exceeds the available one-factorization.")
        return schedule[round_index]
    raise ValueError(f"Unknown temporal-pairing condition: {condition}")


def condition_label(_: str) -> str:
    """Avoid exposing a semantic condition label to the responding model."""
    return "controlled pair protocol"


def schedule_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the shared per-round capacity and intended recurrence contrast."""
    heldout_ids = manifest["task_blocks"]["heldout_confirmatory"]
    all_agents = list(range(relay.N_AGENTS))
    for task_id in heldout_ids:
        schedule = matchings_for_task(task_id)
        if len(schedule) != relay.N_AGENTS - 1:
            raise AssertionError("The round-robin factorization has the wrong number of matchings.")
        for matching in schedule:
            members = [agent for pair in matching for agent in pair]
            if (
                len(matching) != relay.N_AGENTS // 2
                or any(len(pair) != 2 for pair in matching)
                or sorted(members) != all_agents
            ):
                raise AssertionError("A pairing schedule violates the per-round controls.")
        if len({tuple(pair) for matching in schedule for pair in matching}) != 28:
            raise AssertionError("The factorization does not enumerate every pair exactly once.")

        repeated = [
            groups_for_round("repeated_pairs", task_id, round_index)
            for round_index in range(relay.N_ROUNDS)
        ]
        rotating = [
            groups_for_round("rotating_pairs", task_id, round_index)
            for round_index in range(relay.N_ROUNDS)
        ]
        if any(matching != repeated[0] for matching in repeated[1:]):
            raise AssertionError("The repeated-pairs condition unexpectedly changes partners.")
        if len({tuple(pair) for matching in rotating for pair in matching}) != (
            relay.N_ROUNDS * relay.N_AGENTS // 2
        ):
            raise AssertionError("The rotating-pairs condition repeats a partner within four rounds.")

    return {
        "n_heldout_tasks": len(heldout_ids),
        "n_agents": relay.N_AGENTS,
        "n_rounds": relay.N_ROUNDS,
        "groups_per_round": relay.N_AGENTS // 2,
        "group_size": 2,
        "agent_group_memberships_per_round": 1,
        "broadcasts_per_agent_per_round": 1,
        "pair_slots_per_round": relay.N_AGENTS // 2,
        "matched_controls": [
            "Agent count",
            "Four dyadic groups per round",
            "Dyad size",
            "Per-agent group membership per round",
            "One card broadcast per agent per round",
            "Four-round relay budget",
        ],
        "condition_difference": (
            "Repeated pairs reuse the task-specific first matching each round; "
            "rotating pairs use four distinct matchings from the same "
            "task-specific one-factorization."
        ),
        "model_prompt_protocol_label": "controlled pair protocol",
    }


def configure_relay_module() -> None:
    """Reuse audited relay mechanics with this control's independent paths."""
    relay.BENCHMARK_NAME = "temporal_pairing_relay_control"
    relay.MANIFEST_SEED = MANIFEST_SEED
    relay.MANIFEST_PATH = MANIFEST_PATH
    relay.RESULTS_DIR = RESULTS_DIR
    relay.RAW_DIR = RAW_DIR
    relay.N_ROUNDS = 4
    relay.CONDITIONS = CONDITIONS
    relay.groups_for_round = groups_for_round
    relay.condition_label = condition_label
    relay.utc_now = lambda: FREEZE_TIMESTAMP_UTC


def protocol_path(mode: str) -> Path:
    return RESULTS_DIR / f"protocol_{mode}.json"


def protocol_payload(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    task_block = "engineering_pilot" if mode == "pilot" else "heldout_confirmatory"
    return {
        "schema_version": 1,
        "benchmark": "temporal_pairing_relay_control",
        "status": "engineering_pilot" if mode == "pilot" else "prospective_heldout_control",
        "created_at_utc": FREEZE_TIMESTAMP_UTC,
        "manifest_sha256": relay.sha256_file(MANIFEST_PATH),
        "mode": mode,
        "task_ids": manifest["task_blocks"][task_block],
        "models": [model.__dict__ for model in relay.DEFAULT_MODELS],
        "conditions": list(CONDITIONS),
        "n_agents": relay.N_AGENTS,
        "n_rounds": relay.N_ROUNDS,
        "relay_rule": (
            "At each round, every agent broadcasts exactly one card: the "
            "lexicographically first card it has not previously broadcast, or "
            "its first card if all known cards were already broadcast. Payloads "
            "are selected before synchronous delivery to the paired co-member."
        ),
        "schedule_rule": (
            "Each round contains four disjoint dyads. Repeated pairs reuse one "
            "task-specific matching; rotating pairs use a new matching from the "
            "same task-specific one-factorization."
        ),
        "prompt_protocol_label": "controlled pair protocol",
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
            "This benchmark tests a four-round temporal-pairing schedule in a "
            "bounded deterministic-card task. It isolates repeated versus "
            "rotating dyadic contacts under the declared relay rule, not a "
            "general effect of communication topology, model scaling or "
            "open-ended collaboration."
        ),
    }


def freeze_protocol(manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    payload = protocol_payload(manifest, mode)
    path = protocol_path(mode)
    if path.exists():
        existing = relay.read_json(path)
        for key in (
            "manifest_sha256",
            "task_ids",
            "models",
            "conditions",
            "n_rounds",
            "relay_rule",
            "schedule_rule",
            "prompt_protocol_label",
            "primary_endpoint",
        ):
            if existing.get(key) != payload.get(key):
                raise ValueError(f"Frozen protocol differs at {key}.")
    else:
        relay.write_json(path, payload)
    return payload


def selected_tasks(manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    block = "engineering_pilot" if mode == "pilot" else "heldout_confirmatory"
    wanted = set(manifest["task_blocks"][block])
    return [task for task in manifest["tasks"] if task["task_id"] in wanted]


def verify_pilot_complete(manifest: dict[str, Any]) -> None:
    missing = [
        f"{model.slug}/{task['task_id']}/{condition}"
        for model in relay.DEFAULT_MODELS
        for task in selected_tasks(manifest, "pilot")
        for condition in CONDITIONS
        if not relay.run_path(model, task["task_id"], condition).exists()
    ]
    if missing:
        raise RuntimeError(
            "The held-out run requires the engineering pilot to complete first. "
            f"Missing {len(missing)} pilot cells."
        )


def all_results(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        relay.read_json(relay.run_path(model, task["task_id"], condition))
        for model in relay.DEFAULT_MODELS
        for task in selected_tasks(manifest, "full")
        for condition in CONDITIONS
    ]


def paired_rotation_contrasts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the predeclared rotating-minus-repeated task-level contrast."""
    index = {
        (record["model"], record["task_id"], record["condition"]): record
        for record in records
    }
    results: dict[str, Any] = {}
    for model in sorted({record["model"] for record in records}):
        differences = [
            float(index[(model, task_id, "rotating_pairs")]["plurality_correct"])
            - float(index[(model, task_id, "repeated_pairs")]["plurality_correct"])
            for task_id in sorted(
                {
                    record["task_id"]
                    for record in records
                    if record["model"] == model
                    and record["condition"] == "rotating_pairs"
                }
            )
            if (model, task_id, "repeated_pairs") in index
        ]
        results[f"{model}::rotating_pairs_minus_repeated_pairs"] = {
            "n_matched_task_cells": len(differences),
            "mean_group_accuracy_difference": float(relay.np.mean(differences)),
            "ci95": relay.bootstrap_ci(
                differences,
                "temporal_pairing",
                model,
                "rotating_pairs_minus_repeated_pairs",
            ),
        }
    return results


def write_summary(manifest: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    results = all_results(manifest)
    records = [relay.extract_metrics(result) for result in results]
    summary = {
        "schema_version": 1,
        "benchmark": "temporal_pairing_relay_control",
        "status": "prospective_heldout_control",
        "generated_at_utc": FREEZE_TIMESTAMP_UTC,
        "manifest_sha256": relay.sha256_file(MANIFEST_PATH),
        "protocol_sha256": relay.sha256_file(protocol_path("full")),
        "n_expected_cells": (
            len(relay.DEFAULT_MODELS) * len(selected_tasks(manifest, "full")) * len(CONDITIONS)
        ),
        "n_completed_cells": len(records),
        "records": records,
        "by_model_condition": relay.grouped_summary(records),
        "paired_group_accuracy_contrasts": paired_rotation_contrasts(records),
        "deterministic_access": relay.deterministic_access_diagnostics(manifest),
        "schedule_diagnostics": schedule_diagnostics(manifest),
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


async def execute(
    manifest: dict[str, Any],
    mode: str,
    force: bool,
    concurrency: int,
    task_ids: list[str] | None,
) -> int:
    if not os.environ.get("LLM_GATEWAY_API_KEY"):
        raise RuntimeError("LLM_GATEWAY_API_KEY is required and must be set in the environment.")
    if not relay.API_URL:
        raise RuntimeError(
            "LLM_GATEWAY_API_URL is required and must point to a chat-completions endpoint."
        )
    if mode == "full":
        verify_pilot_complete(manifest)

    tasks = selected_tasks(manifest, mode)
    if task_ids is not None:
        if mode != "full":
            raise ValueError("--task-ids can only select from the held-out full protocol.")
        requested = list(dict.fromkeys(task_ids))
        available = {task["task_id"] for task in tasks}
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(
                "Requested task IDs fall outside the frozen held-out protocol: "
                + ", ".join(unknown)
            )
        requested_set = set(requested)
        tasks = [task for task in tasks if task["task_id"] in requested_set]

    semaphore = asyncio.Semaphore(concurrency)
    jobs = [
        relay.run_one(task, model, condition, semaphore, force)
        for model in relay.DEFAULT_MODELS
        for task in tasks
        for condition in CONDITIONS
    ]
    await asyncio.gather(*jobs)
    return len(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("design", "pilot", "full"),
        required=True,
        help="Freeze the design, run pilot cells, or run held-out cells.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing raw cells.")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument(
        "--task-ids",
        nargs="+",
        help="Run a resumable subset of the frozen held-out tasks; no summary is written.",
    )
    args = parser.parse_args()

    configure_relay_module()
    manifest = relay.load_or_create_manifest()
    freeze_protocol(manifest, "pilot")
    full_protocol = freeze_protocol(manifest, "full")
    diagnostics = {
        "schema_version": 1,
        "benchmark": "temporal_pairing_relay_control",
        "generated_at_utc": FREEZE_TIMESTAMP_UTC,
        "manifest_sha256": relay.sha256_file(MANIFEST_PATH),
        "protocol_sha256": {
            "pilot": relay.sha256_file(protocol_path("pilot")),
            "full": relay.sha256_file(protocol_path("full")),
        },
        "schedule_diagnostics": schedule_diagnostics(manifest),
        "deterministic_access": relay.deterministic_access_diagnostics(manifest),
        "interpretation_boundary": full_protocol["interpretation_boundary"],
    }
    relay.write_json(DESIGN_PATH, diagnostics)

    if args.mode == "design":
        print("Frozen temporal-pairing manifest and diagnostics; no API calls made.")
        return

    completed_cells = asyncio.run(
        execute(manifest, args.mode, args.force, args.concurrency, args.task_ids)
    )
    if args.mode == "full":
        if args.task_ids is not None:
            print(f"Completed {completed_cells} held-out temporal-pairing batch cells.")
            return
        summary = write_summary(manifest, full_protocol)
        print(
            "Completed "
            f"{summary['n_completed_cells']}/{summary['n_expected_cells']} "
            "held-out temporal-pairing cells."
        )
    else:
        print(f"Completed {completed_cells} engineering-pilot temporal-pairing cells.")


if __name__ == "__main__":
    main()
