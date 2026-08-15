#!/usr/bin/env python3
"""Run a frozen, bandwidth-limited distributed-evidence benchmark.

This benchmark is separate from ``llm_task_benchmark``. Eight agents start
with distinct exclusion cards, complete two synchronous rounds and may broadcast
only one evidence card per round. The payload schedule is deterministic and does
not depend on model responses. This creates protocol-dependent access to evidence
without treating model-written prose as a data channel.

No credentials are stored in this file. Set LLM_GATEWAY_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "llm_relay_benchmark"
DATA_DIR = BENCHMARK_DIR / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "task_manifest.json"
DESIGN_PATH = RESULTS_DIR / "design_diagnostics.json"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
API_URL = os.environ.get("LLM_GATEWAY_API_URL", "")

LABELS = tuple("ABCDEFGH")
N_AGENTS = 8
N_ROUNDS = 2
MAX_TOKENS = 512
MANIFEST_SEED = 20260730
SCHEMA_VERSION = 1
BENCHMARK_NAME = "bandwidth_limited_distributed_evidence"
CONDITIONS = ("solo", "pairs", "star", "triads", "five_cliques")


@dataclass(frozen=True)
class ModelSpec:
    """A fixed OpenAI-compatible model used for the held-out protocol."""

    label: str
    model: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")


DEFAULT_MODELS = (
    ModelSpec("DeepSeek-V4-Flash", "deepseek-v4-flash"),
    ModelSpec("GPT-4.1-mini", "gpt-4.1-mini"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def base_card_pattern() -> tuple[tuple[int, ...], ...]:
    """Return eight two-label cards over seven false candidates.

    A one-, two- or three-card set cannot exclude all seven false labels.
    Some four-card sets can, so a two-round relay process has a non-ceiling
    access endpoint rather than a binary all-or-none communication endpoint.
    """

    pattern = (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 0),
        (1, 2),
        (3, 4),
        (5, 6),
        (0, 2),
    )
    false_labels = set(range(7))
    if set().union(*(set(card) for card in pattern)) != false_labels:
        raise RuntimeError("Card pattern does not cover every false candidate.")
    if any(
        set().union(*(set(card) for card in subset)) == false_labels
        for size in (1, 2, 3)
        for subset in itertools.combinations(pattern, size)
    ):
        raise RuntimeError("At most three cards must never identify the answer.")
    if not any(
        set().union(*(set(card) for card in subset)) == false_labels
        for subset in itertools.combinations(pattern, 4)
    ):
        raise RuntimeError("At least one four-card set must identify the answer.")
    return pattern


def build_manifest(n_tasks: int = 20) -> dict[str, Any]:
    """Create four pilot and 16 answer-balanced held-out tasks."""

    if n_tasks != 20:
        raise ValueError("This benchmark freezes four pilot and 16 held-out tasks.")

    rng = np.random.default_rng(MANIFEST_SEED)
    pilot_answers = ["A", "C", "E", "G"]
    heldout_answers = list(LABELS) * 2
    rng.shuffle(pilot_answers)
    rng.shuffle(heldout_answers)
    answers = pilot_answers + heldout_answers
    pattern = base_card_pattern()
    tasks: list[dict[str, Any]] = []

    for index, answer in enumerate(answers, start=1):
        task_rng = np.random.default_rng(stable_int(MANIFEST_SEED, "task", index, answer))
        false_labels = [label for label in LABELS if label != answer]
        task_rng.shuffle(false_labels)
        cards = [
            {
                "card_id": f"E{card_index}",
                "excludes": sorted(false_labels[position] for position in template),
            }
            for card_index, template in enumerate(pattern, start=1)
        ]
        card_ids = [card["card_id"] for card in cards]
        task_rng.shuffle(card_ids)
        task = {
            "task_id": f"T{index:02d}",
            "answer": answer,
            "cards": cards,
            "agent_cards": {str(agent): card_ids[agent] for agent in range(N_AGENTS)},
        }
        validate_task(task)
        tasks.append(task)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "manifest_seed": MANIFEST_SEED,
        "labels": list(LABELS),
        "n_agents": N_AGENTS,
        "task_generation": {
            "card_rule": (
                "Every card excludes two false labels. No set of one, two or "
                "three cards identifies the answer; some four-card sets do."
            ),
            "answer_balance": (
                "Four engineering-pilot tasks are followed by 16 held-out tasks "
                "with exactly two instances of every answer label."
            ),
        },
        "task_blocks": {
            "engineering_pilot": [f"T{index:02d}" for index in range(1, 5)],
            "heldout_confirmatory": [f"T{index:02d}" for index in range(5, 21)],
        },
        "tasks": tasks,
    }


def card_lookup(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {card["card_id"]: card for card in task["cards"]}


def candidates_for_cards(task: dict[str, Any], card_ids: set[str]) -> list[str]:
    cards = card_lookup(task)
    excluded = set().union(*(set(cards[card_id]["excludes"]) for card_id in card_ids))
    return [label for label in LABELS if label not in excluded]


def validate_task(task: dict[str, Any]) -> None:
    """Validate deterministic information constraints for one frozen task."""

    if task.get("answer") not in LABELS:
        raise ValueError(f"{task.get('task_id')} has an invalid answer.")
    cards = {card["card_id"]: set(card["excludes"]) for card in task["cards"]}
    expected_ids = {f"E{index}" for index in range(1, N_AGENTS + 1)}
    if set(cards) != expected_ids:
        raise ValueError(f"{task['task_id']} does not contain eight named cards.")
    if any(
        len(excluded) != 2
        or task["answer"] in excluded
        or not excluded <= set(LABELS)
        for excluded in cards.values()
    ):
        raise ValueError(f"{task['task_id']} contains malformed exclusions.")

    false_labels = set(LABELS) - {task["answer"]}
    if set().union(*cards.values()) != false_labels:
        raise ValueError(f"{task['task_id']} is not jointly identifiable.")
    if any(
        set().union(*(cards[card_id] for card_id in subset)) == false_labels
        for size in (1, 2, 3)
        for subset in itertools.combinations(cards, size)
    ):
        raise ValueError(f"{task['task_id']} is solvable from too few cards.")
    if not any(
        set().union(*(cards[card_id] for card_id in subset)) == false_labels
        for subset in itertools.combinations(cards, 4)
    ):
        raise ValueError(f"{task['task_id']} has no four-card solution.")

    assignments = task["agent_cards"]
    if set(assignments) != {str(agent) for agent in range(N_AGENTS)}:
        raise ValueError(f"{task['task_id']} does not assign every agent a card.")
    if sorted(assignments.values()) != sorted(cards):
        raise ValueError(f"{task['task_id']} does not assign every card once.")


def load_or_create_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
    else:
        manifest = build_manifest()
        write_json(MANIFEST_PATH, manifest)

    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("labels") != list(LABELS)
        or manifest.get("n_agents") != N_AGENTS
    ):
        raise ValueError("The manifest is incompatible with this benchmark version.")
    for task in manifest["tasks"]:
        validate_task(task)
    return manifest


def groups_for_round(condition: str, task_id: str, round_index: int) -> list[list[int]]:
    """Return deterministic group memberships for one synchronous relay round."""

    if condition == "solo":
        return [[agent] for agent in range(N_AGENTS)]

    rng = np.random.default_rng(
        stable_int("relay_groups", condition, task_id, round_index)
    )
    order = rng.permutation(N_AGENTS).tolist()
    if condition == "pairs":
        return [order[start : start + 2] for start in range(0, N_AGENTS, 2)]
    if condition == "star":
        hub = stable_int("relay_hub", task_id) % N_AGENTS
        return [[hub, agent] for agent in range(N_AGENTS) if agent != hub]
    if condition == "triads":
        return [order[:3], order[3:6], order[6:]]
    if condition == "five_cliques":
        return [order[:5], order[5:]]
    raise ValueError(f"Unknown protocol condition: {condition}")


def relay_round(
    known_cards: dict[int, set[str]],
    sent_cards: dict[int, set[str]],
    groups: list[list[int]],
) -> tuple[dict[int, set[str]], dict[int, str]]:
    """Deliver one fixed one-card broadcast per agent.

    Every agent broadcasts the lexicographically first card it has not already
    sent. An agent with no unsent card repeats its first card. The payload is
    computed before any delivery, preserving synchronous communication.
    """

    payload: dict[int, str] = {}
    for agent in range(N_AGENTS):
        unsent = sorted(known_cards[agent] - sent_cards[agent])
        payload[agent] = unsent[0] if unsent else min(known_cards[agent])
        sent_cards[agent].add(payload[agent])

    delivered = {agent: set(cards) for agent, cards in known_cards.items()}
    for group in groups:
        group_cards = {payload[agent] for agent in group}
        for agent in group:
            delivered[agent].update(group_cards)
    return delivered, payload


def condition_label(condition: str) -> str:
    return {
        "solo": "Solo",
        "pairs": "Pairs",
        "star": "Hub-and-spoke",
        "triads": "Triads",
        "five_cliques": "5-cliques",
    }[condition]


def build_prompt(
    task: dict[str, Any],
    agent_id: int,
    condition: str,
    round_index: int,
    known_card_ids: set[str],
) -> tuple[str, str]:
    cards = card_lookup(task)
    card_lines = [
        f"{card_id}: EXCLUDES {', '.join(cards[card_id]['excludes'])}"
        for card_id in sorted(known_card_ids)
    ]
    candidates = candidates_for_cards(task, known_card_ids)
    system = (
        "You are an analyst in a controlled distributed-evidence task. "
        "Use only the evidence cards in the user message. A listed candidate "
        "is impossible; every unlisted candidate remains possible. Return "
        "exactly one JSON object with keys answer, confidence, rationale. "
        "answer must be one of A, B, C, D, E, F, G, H. confidence must be "
        "a number from 0 to 100. Keep rationale under 35 words."
    )
    user = (
        f"Task {task['task_id']}; analyst {agent_id}; protocol "
        f"{condition_label(condition)}; synchronous round {round_index + 1}/{N_ROUNDS}.\n"
        "Exactly one label is correct.\n\n"
        "Evidence cards delivered to you:\n"
        + "\n".join(card_lines)
        + "\n\n"
        + f"These cards leave the candidates: {', '.join(candidates)}.\n\n"
        "State your best current answer."
    )
    return system, user


def safe_response_headers(headers: Any) -> dict[str, str]:
    allowed = ("date", "x-request-id", "request-id", "openai-processing-ms")
    return {
        header: str(headers[header])
        for header in allowed
        if headers.get(header) is not None
    }


def extract_json_object(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None, None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None, None
        except json.JSONDecodeError as error:
            return None, str(error)
    return None, "No JSON object found."


def parse_decision(text: str) -> dict[str, Any]:
    payload, error = extract_json_object(text)
    if payload is None:
        return {
            "answer": None,
            "confidence": None,
            "rationale": "",
            "parse_status": f"invalid_json: {error}",
        }

    answer = str(payload.get("answer", "")).strip().upper()
    if answer not in LABELS:
        answer = None
    try:
        confidence = float(np.clip(float(payload.get("confidence")), 0.0, 100.0))
    except (TypeError, ValueError):
        confidence = None
    rationale = str(payload.get("rationale", "")).strip()[:800]
    return {
        "answer": answer,
        "confidence": confidence,
        "rationale": rationale,
        "parse_status": "ok" if answer is not None and confidence is not None else "invalid_fields",
    }


async def call_model(
    semaphore: asyncio.Semaphore,
    model: ModelSpec,
    system: str,
    user: str,
) -> dict[str, Any]:
    payload = {
        "model": model.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {os.environ['LLM_GATEWAY_API_KEY']}",
        "Content-Type": "application/json",
    }
    started = time.perf_counter()
    async with semaphore:
        try:
            def post() -> tuple[dict[str, Any], dict[str, str]]:
                request = Request(
                    API_URL,
                    data=json.dumps(payload).encode(),
                    headers=headers,
                    method="POST",
                )
                with urlopen(request, timeout=90) as response:
                    return json.loads(response.read().decode()), safe_response_headers(response.headers)

            response, response_headers = await asyncio.to_thread(post)
            message = response["choices"][0].get("message", {})
            text = message.get("content")
            if not isinstance(text, str):
                text = message.get("reasoning_content")
            return {
                "ok": True,
                "response": response,
                "text": text if isinstance(text, str) else "",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "safe_headers": response_headers,
            }
        except Exception as error:
            return {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error)[:800],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }


def run_path(model: ModelSpec, task_id: str, condition: str) -> Path:
    return RAW_DIR / model.slug / task_id / f"{condition}.json"


def aggregate_group_answer(decisions: list[dict[str, Any]], answer: str) -> dict[str, Any]:
    answers = [decision["parsed"]["answer"] for decision in decisions]
    counts = Counter(value for value in answers if value is not None)
    if not counts:
        plurality, tied = None, False
    else:
        highest = max(counts.values())
        winners = sorted(answer for answer, count in counts.items() if count == highest)
        plurality, tied = (winners[0], False) if len(winners) == 1 else (None, True)
    return {
        "plurality_answer": plurality,
        "plurality_tied": tied,
        "plurality_correct": plurality == answer,
        "answer_counts": dict(sorted(counts.items())),
        "consensus_share": max(counts.values()) / N_AGENTS if counts else 0.0,
    }


def score_decisions(task: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    answer = task["answer"]
    individual = []
    solvable = []
    conditional = []
    parse_failures = 0
    brier_scores = []
    for decision in decisions:
        parsed = decision["parsed"]
        correct = parsed["answer"] == answer
        individual.append(float(correct))
        has_sufficient_evidence = len(decision["candidates_after_delivery"]) == 1
        solvable.append(float(has_sufficient_evidence))
        if has_sufficient_evidence:
            conditional.append(float(correct))
        if parsed["parse_status"] != "ok":
            parse_failures += 1
        if parsed["confidence"] is not None:
            brier_scores.append((parsed["confidence"] / 100.0 - float(correct)) ** 2)

    return {
        **aggregate_group_answer(decisions, answer),
        "individual_accuracy": float(np.mean(individual)),
        "solvable_agent_fraction": float(np.mean(solvable)),
        "conditional_accuracy_when_solvable": float(np.mean(conditional)) if conditional else None,
        "mean_brier_score": float(np.mean(brier_scores)) if brier_scores else None,
        "parse_failure_fraction": parse_failures / N_AGENTS,
    }


async def run_one(
    task: dict[str, Any],
    model: ModelSpec,
    condition: str,
    semaphore: asyncio.Semaphore,
    force: bool,
) -> dict[str, Any]:
    """Run or resume one model-task-condition record."""

    path = run_path(model, task["task_id"], condition)
    if path.exists() and not force:
        return read_json(path)

    known_cards = {
        agent: {task["agent_cards"][str(agent)]} for agent in range(N_AGENTS)
    }
    sent_cards = {agent: set() for agent in range(N_AGENTS)}
    rounds: list[dict[str, Any]] = []

    for round_index in range(N_ROUNDS):
        groups = groups_for_round(condition, task["task_id"], round_index)
        delivered_cards, payload = relay_round(known_cards, sent_cards, groups)

        async def call_agent(agent: int) -> dict[str, Any]:
            system, user = build_prompt(
                task, agent, condition, round_index, delivered_cards[agent]
            )
            result = await call_model(semaphore, model, system, user)
            base = {
                "agent_id": agent,
                "system_prompt": system,
                "user_prompt": user,
                "known_card_ids": sorted(delivered_cards[agent]),
                "candidates_after_delivery": candidates_for_cards(task, delivered_cards[agent]),
                "sent_card_id": payload[agent],
                "latency_ms": result["latency_ms"],
            }
            if result["ok"]:
                response = result["response"]
                return {
                    **base,
                    "raw_response": result["text"],
                    "parsed": parse_decision(result["text"]),
                    "returned_model": response.get("model"),
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                    "usage": response.get("usage", {}),
                    "safe_response_headers": result["safe_headers"],
                }
            return {
                **base,
                "raw_response": "",
                "parsed": {
                    "answer": None,
                    "confidence": None,
                    "rationale": "",
                    "parse_status": "api_error",
                },
                "api_error": result,
            }

        agent_records = await asyncio.gather(*(call_agent(agent) for agent in range(N_AGENTS)))
        agent_records.sort(key=lambda row: row["agent_id"])
        rounds.append(
            {
                "round_index": round_index,
                "groups": groups,
                "broadcast_payloads": {str(agent): payload[agent] for agent in range(N_AGENTS)},
                "agents": agent_records,
            }
        )
        known_cards = delivered_cards

    all_agents = [agent for round_record in rounds for agent in round_record["agents"]]
    api_errors = sum("api_error" in agent for agent in all_agents)
    usage = {
        field: int(sum(agent.get("usage", {}).get(field, 0) for agent in all_agents))
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "completed_at_utc": utc_now(),
        "task_id": task["task_id"],
        "ground_truth_answer": task["answer"],
        "condition": condition,
        "model": asdict(model),
        "n_rounds": N_ROUNDS,
        "rounds": rounds,
        "metrics": score_decisions(task, rounds[-1]["agents"]),
        "execution": {
            "n_api_calls_expected": N_AGENTS * N_ROUNDS,
            "n_api_calls_completed": len(all_agents) - api_errors,
            "api_error_fraction": api_errors / len(all_agents),
            "usage": usage,
            "mean_latency_ms": float(np.mean([agent["latency_ms"] for agent in all_agents])),
        },
    }
    write_json(path, result)
    return result


def extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": result["model"]["label"],
        "model_slug": re.sub(r"[^a-z0-9]+", "_", result["model"]["label"].lower()).strip("_"),
        "task_id": result["task_id"],
        "condition": result["condition"],
        "api_error_fraction": result["execution"]["api_error_fraction"],
        "total_tokens": result["execution"]["usage"]["total_tokens"],
        **result["metrics"],
    }


def bootstrap_ci(values: list[float], *seed_parts: object) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_int("bootstrap", *seed_parts))
    sampled = array[rng.integers(0, len(array), size=(5000, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(sampled, (0.025, 0.975))]


def grouped_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"], record["condition"])].append(record)

    summary: dict[str, Any] = {}
    for (model, condition), rows in sorted(groups.items()):
        conditional = [
            row["conditional_accuracy_when_solvable"]
            for row in rows
            if row["conditional_accuracy_when_solvable"] is not None
        ]
        key = f"{model}::{condition}"
        summary[key] = {
            "model": model,
            "condition": condition,
            "n_task_cells": len(rows),
            "group_plurality_accuracy": float(np.mean([row["plurality_correct"] for row in rows])),
            "group_plurality_accuracy_ci95": bootstrap_ci(
                [float(row["plurality_correct"]) for row in rows], model, condition, "group"
            ),
            "individual_accuracy": float(np.mean([row["individual_accuracy"] for row in rows])),
            "individual_accuracy_ci95": bootstrap_ci(
                [row["individual_accuracy"] for row in rows], model, condition, "individual"
            ),
            "solvable_agent_fraction": float(
                np.mean([row["solvable_agent_fraction"] for row in rows])
            ),
            "solvable_agent_fraction_ci95": bootstrap_ci(
                [row["solvable_agent_fraction"] for row in rows],
                model,
                condition,
                "solvable",
            ),
            "conditional_accuracy_when_solvable": float(np.mean(conditional)) if conditional else None,
            "parse_failure_fraction": float(np.mean([row["parse_failure_fraction"] for row in rows])),
            "api_error_fraction": float(np.mean([row["api_error_fraction"] for row in rows])),
            "mean_total_tokens_per_cell": float(np.mean([row["total_tokens"] for row in rows])),
        }
    return summary


def paired_contrasts(records: list[dict[str, Any]]) -> dict[str, Any]:
    index = {(row["model"], row["task_id"], row["condition"]): row for row in records}
    results: dict[str, Any] = {}
    for model in sorted({row["model"] for row in records}):
        for condition in CONDITIONS[1:]:
            differences = []
            for task_id in sorted(
                row["task_id"]
                for row in records
                if row["model"] == model and row["condition"] == condition
            ):
                candidate = index.get((model, task_id, condition))
                baseline = index.get((model, task_id, "solo"))
                if candidate is not None and baseline is not None:
                    differences.append(
                        float(candidate["plurality_correct"]) - float(baseline["plurality_correct"])
                    )
            results[f"{model}::{condition}_minus_solo"] = {
                "n_matched_task_cells": len(differences),
                "mean_group_accuracy_difference": float(np.mean(differences)) if differences else None,
                "ci95": bootstrap_ci(differences, model, condition, "contrast"),
            }
    return results


def deterministic_access_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Calculate protocol access without reading model responses."""

    task_index = {task["task_id"]: task for task in manifest["tasks"]}
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id in manifest["task_blocks"]["heldout_confirmatory"]:
        task = task_index[task_id]
        for condition in CONDITIONS:
            known = {agent: {task["agent_cards"][str(agent)]} for agent in range(N_AGENTS)}
            sent = {agent: set() for agent in range(N_AGENTS)}
            for round_index in range(N_ROUNDS):
                known, _ = relay_round(
                    known,
                    sent,
                    groups_for_round(condition, task_id, round_index),
                )
            sufficient = [
                float(len(candidates_for_cards(task, known[agent])) == 1)
                for agent in range(N_AGENTS)
            ]
            condition_rows[condition].append(
                {
                    "task_id": task_id,
                    "sufficient_agent_fraction": float(np.mean(sufficient)),
                    "card_count_by_agent": [len(known[agent]) for agent in range(N_AGENTS)],
                }
            )

    return {
        condition: {
            "n_heldout_tasks": len(rows),
            "mean_sufficient_agent_fraction": float(
                np.mean([row["sufficient_agent_fraction"] for row in rows])
            ),
            "task_level_sufficient_agent_fraction": {
                row["task_id"]: row["sufficient_agent_fraction"] for row in rows
            },
            "card_count_by_task_and_agent": {
                row["task_id"]: row["card_count_by_agent"] for row in rows
            },
        }
        for condition, rows in sorted(condition_rows.items())
    }


def protocol_path(mode: str) -> Path:
    return RESULTS_DIR / f"protocol_{mode}.json"


def freeze_protocol(manifest: dict[str, Any], mode: str) -> None:
    tasks = manifest["task_blocks"][
        "engineering_pilot" if mode == "pilot" else "heldout_confirmatory"
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": utc_now(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "mode": mode,
        "task_ids": tasks,
        "models": [asdict(model) for model in DEFAULT_MODELS],
        "conditions": list(CONDITIONS),
        "n_agents": N_AGENTS,
        "n_rounds": N_ROUNDS,
        "relay_rule": (
            "At each round, every agent broadcasts exactly one card: the "
            "lexicographically first card it has not previously broadcast, or "
            "its first card if all known cards were already broadcast. Payloads "
            "are selected before synchronous delivery to every co-member."
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
        "max_tokens": MAX_TOKENS,
    }
    path = protocol_path(mode)
    if path.exists():
        existing = read_json(path)
        for key in ("manifest_sha256", "task_ids", "models", "conditions", "n_rounds", "relay_rule"):
            if existing.get(key) != payload.get(key):
                raise ValueError(
                    f"Frozen protocol {path.name} differs from the current specification."
                )
    else:
        write_json(path, payload)


def selected_tasks(manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    block = "engineering_pilot" if mode == "pilot" else "heldout_confirmatory"
    wanted = set(manifest["task_blocks"][block])
    return [task for task in manifest["tasks"] if task["task_id"] in wanted]


def verify_pilot_complete(manifest: dict[str, Any]) -> None:
    missing = []
    for model in DEFAULT_MODELS:
        for task in selected_tasks(manifest, "pilot"):
            for condition in CONDITIONS:
                if not run_path(model, task["task_id"], condition).exists():
                    missing.append(f"{model.slug}/{task['task_id']}/{condition}")
    if missing:
        raise RuntimeError(
            "The held-out run requires the frozen engineering pilot to complete first. "
            f"Missing {len(missing)} pilot cells."
        )


async def execute(mode: str, manifest: dict[str, Any], force: bool, concurrency: int) -> None:
    if not os.environ.get("LLM_GATEWAY_API_KEY"):
        raise RuntimeError("LLM_GATEWAY_API_KEY is required and must be set in the environment.")
    if not API_URL:
        raise RuntimeError(
            "LLM_GATEWAY_API_URL is required and must point to a chat-completions endpoint."
        )
    if mode == "full":
        verify_pilot_complete(manifest)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = selected_tasks(manifest, mode)
    jobs = [
        run_one(task, model, condition, semaphore, force)
        for model in DEFAULT_MODELS
        for task in tasks
        for condition in CONDITIONS
    ]
    results = await asyncio.gather(*jobs)
    records = [extract_metrics(result) for result in results]
    if mode == "full":
        all_results = [
            read_json(path)
            for model in DEFAULT_MODELS
            for task in selected_tasks(manifest, "full")
            for condition in CONDITIONS
            for path in [run_path(model, task["task_id"], condition)]
        ]
        all_records = [extract_metrics(result) for result in all_results]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK_NAME,
            "generated_at_utc": utc_now(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "protocol_sha256": sha256_file(protocol_path("full")),
            "n_expected_cells": len(DEFAULT_MODELS) * len(selected_tasks(manifest, "full")) * len(CONDITIONS),
            "n_completed_cells": len(all_records),
            "records": all_records,
            "by_model_condition": grouped_summary(all_records),
            "paired_group_accuracy_contrasts": paired_contrasts(all_records),
            "deterministic_access": deterministic_access_diagnostics(manifest),
            "failure_counts": {
                "api_error_calls": int(
                    sum(
                        result["execution"]["n_api_calls_expected"]
                        - result["execution"]["n_api_calls_completed"]
                        for result in all_results
                    )
                ),
                "expected_calls": int(
                    sum(result["execution"]["n_api_calls_expected"] for result in all_results)
                ),
                "parsed_final_decisions": int(
                    sum(
                        1
                        for result in all_results
                        for agent in result["rounds"][-1]["agents"]
                        if agent["parsed"]["parse_status"] == "ok"
                    )
                ),
            },
        }
        write_json(SUMMARY_PATH, summary)
    else:
        print(f"Completed {len(records)} frozen pilot cells.")


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
    args = parser.parse_args()

    manifest = load_or_create_manifest()
    freeze_protocol(manifest, "pilot")
    freeze_protocol(manifest, "full")
    if args.mode == "design":
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK_NAME,
            "generated_at_utc": utc_now(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "protocol_sha256": {
                "pilot": sha256_file(protocol_path("pilot")),
                "full": sha256_file(protocol_path("full")),
            },
            "deterministic_access": deterministic_access_diagnostics(manifest),
        }
        write_json(DESIGN_PATH, diagnostics)
        print("Frozen task manifest and protocol files; no API calls made.")
        return

    asyncio.run(execute(args.mode, manifest, args.force, args.concurrency))


if __name__ == "__main__":
    main()
