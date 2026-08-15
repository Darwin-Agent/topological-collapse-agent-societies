"""
LLM Multi-Agent Public Goods Game Arena.

Orchestrates real LLM agents (via API) playing PGG under different
topological conditions (A-D). Each agent is an independent LLM session
with a persistent system prompt defining its role and the game rules.

Supports direct providers and configurable OpenAI-compatible endpoints.

Cost-optimized design:
  - Minimal prompt length (~500 tokens input per turn)
  - Structured JSON output to reduce output tokens
  - Async batch processing to maximize throughput
  - Configurable model selection per cost tier
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── System Prompts ───────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are participating in a public goods game as Agent {agent_id}.

RULES:
- Each round, you are endowed with {endowment} tokens.
- You decide how many of these {endowment} tokens (0 to {endowment}) to contribute to a shared project.
- The remaining tokens are kept in your private account.
- Your payoff each round: ({endowment} - your contribution) + (sum of all group members' contributions) * {multiplier} / group size.
- You are randomly assigned to a new group each round.

RESPOND WITH ONLY a JSON object: {{"contribution": <integer 0-{endowment}>, "reasoning": "<brief>"}}"""

NORM_SEED_ADDITION = """

OBSERVATION from past rounds: Groups where members contributed 15 or more tokens each achieved higher average payoffs per person than groups with low contributions."""

GROUP_DELIBERATION_PROMPT = """
IMPORTANT: This round, your group of {group_size} members uses a GROUP NORM RULE.
Each member first states their intended contribution. Then:
- If the MAJORITY (more than half) contribute 50% or more of their endowment, ALL members contribute their full endowment.
- Otherwise, ALL members contribute 0.

Group members' declared intentions: {intentions}

Given this rule, state your intended contribution. Your choice affects the majority outcome."""


# ── LLM Client Abstraction ──────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: str = "deepseek"  # "deepseek", "openai", "anthropic", "gateway", "gptplus5"
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 100
    temperature: float = 0.7

    def __post_init__(self):
        if not self.api_key:
            key_env = {
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "gateway": "LLM_GATEWAY_API_KEY",
                "gptplus5": "GPTPLUS5_API_KEY",
            }
            self.api_key = os.environ.get(key_env.get(self.provider, ""), "")

        if not self.base_url:
            urls = {
                "deepseek": "https://api.deepseek.com",
                "openai": "https://api.openai.com/v1",
                "gateway": os.environ.get("LLM_GATEWAY_API_BASE_URL", ""),
                "gptplus5": os.environ.get("GPTPLUS5_API_BASE", ""),
            }
            self.base_url = urls.get(self.provider, "")


# ── Global token usage tracker ──────────────────────────────────────
_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

def get_token_usage() -> dict:
    return dict(_token_usage)

def reset_token_usage():
    _token_usage.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0})


async def call_llm(config: LLMConfig, system: str, user: str, sem: asyncio.Semaphore) -> str:
    """Call LLM API. Supports OpenAI-compatible endpoints (DeepSeek, OpenAI, etc.)."""
    async with sem:
        if config.provider == "anthropic":
            return await _call_anthropic(config, system, user)
        else:
            return await _call_openai_compat(config, system, user)


async def _call_openai_compat(config: LLMConfig, system: str, user: str) -> str:
    """Call an OpenAI-compatible API."""
    import httpx

    if not config.api_key:
        raise RuntimeError(f"No API key configured for provider {config.provider!r}.")
    if not config.base_url:
        raise RuntimeError(f"No API base URL configured for provider {config.provider!r}.")

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    url = f"{config.base_url}/chat/completions"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                if usage:
                    _token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    _token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    _token_usage["total_tokens"] += usage.get("total_tokens", 0)
                _token_usage["api_calls"] += 1
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt == max_retries - 1:
                logger.error("LLM call failed after %d attempts: %s", max_retries, e)
                return '{"contribution": 0, "reasoning": "API error"}'
            logger.warning("LLM call attempt %d failed (%s), retrying in %ds...", attempt + 1, type(e).__name__, wait)
            await asyncio.sleep(wait)

    return '{"contribution": 0, "reasoning": "API error"}'


async def _call_anthropic(config: LLMConfig, system: str, user: str) -> str:
    """Call Anthropic Claude API."""
    import httpx

    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]
        except Exception as e:
            if attempt == 2:
                logger.error("Anthropic call failed: %s", e)
                return '{"contribution": 0, "reasoning": "API error"}'
            await asyncio.sleep(2 ** attempt)

    return '{"contribution": 0, "reasoning": "API error"}'


# ── Agent & Game State ───────────────────────────────────────────────

@dataclass
class LLMAgent:
    agent_id: int
    is_norm_seed: bool = False
    cumulative_payoff: float = 0.0
    history: list[dict] = field(default_factory=list)

    def get_system_prompt(self, endowment: int = 10, multiplier: int = 3) -> str:
        prompt = AGENT_SYSTEM_PROMPT.format(
            agent_id=self.agent_id,
            endowment=endowment,
            multiplier=multiplier,
        )
        if self.is_norm_seed:
            prompt += NORM_SEED_ADDITION
        return prompt


def parse_contribution(response: str, endowment: int = 10) -> int:
    """Extract contribution from LLM response, with fallback parsing."""
    try:
        # try JSON parse
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        data = json.loads(clean)
        c = int(data.get("contribution", 0))
        return max(0, min(endowment, c))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # fallback: find first number
    import re
    numbers = re.findall(r'\d+', response)
    if numbers:
        c = int(numbers[0])
        return max(0, min(endowment, c))

    return 0


# ── Conditions ───────────────────────────────────────────────────────

def make_dyadic_pairs(n_agents: int, rng: np.random.Generator) -> list[list[int]]:
    """Condition A: random pairwise grouping each round."""
    order = rng.permutation(n_agents).tolist()
    pairs = []
    for i in range(0, n_agents - 1, 2):
        pairs.append([order[i], order[i + 1]])
    return pairs


def make_reciprocal_pairs(n_agents: int, agents: list[LLMAgent],
                          rng: np.random.Generator) -> list[list[int]]:
    """Condition B: pairs only form if both had non-zero contribution last round."""
    order = rng.permutation(n_agents).tolist()
    pairs = []
    unpaired = []
    for i in range(0, n_agents - 1, 2):
        a, b = order[i], order[i + 1]
        a_willing = (not agents[a].history) or agents[a].history[-1].get("contribution", 0) > 0
        b_willing = (not agents[b].history) or agents[b].history[-1].get("contribution", 0) > 0
        if a_willing and b_willing:
            pairs.append([a, b])
        else:
            unpaired.extend([a, b])

    # pair up remaining
    rng.shuffle(unpaired)
    for i in range(0, len(unpaired) - 1, 2):
        pairs.append([unpaired[i], unpaired[i + 1]])

    return pairs


def make_triad_groups(n_agents: int, rng: np.random.Generator) -> list[list[int]]:
    """Condition C: random 3-agent groups."""
    order = rng.permutation(n_agents).tolist()
    groups = []
    for i in range(0, n_agents - 2, 3):
        groups.append(order[i:i + 3])
    # handle remainder
    if len(order) % 3 == 1 and groups:
        groups[-1].append(order[-1])
    elif len(order) % 3 == 2 and groups:
        groups[-1].extend(order[-2:])
    return groups


def make_pentad_groups(n_agents: int, rng: np.random.Generator) -> list[list[int]]:
    """Condition D: random 5-agent groups."""
    order = rng.permutation(n_agents).tolist()
    groups = []
    for i in range(0, n_agents - 4, 5):
        groups.append(order[i:i + 5])
    # handle remainder
    remainder = order[len(groups) * 5:]
    if remainder and groups:
        groups[-1].extend(remainder)
    elif remainder:
        groups.append(remainder)
    return groups


# ── Main Game Loop ───────────────────────────────────────────────────

@dataclass
class GameConfig:
    condition: str = "A"  # A, B, C, D
    n_agents: int = 20
    n_rounds: int = 200
    endowment: int = 20
    multiplier: float = 1.6
    seed_fraction: float = 0.05
    seed: int = 42
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    concurrency: int = 10


@dataclass
class GameResult:
    condition: str
    n_agents: int
    n_rounds: int
    seed: int
    round_data: list[dict]
    cooperation_rate: list[float]
    mean_contribution: list[float]
    mean_payoff: list[float]
    total_llm_calls: int
    elapsed_seconds: float
    agent_histories: dict


async def run_game(config: GameConfig) -> GameResult:
    """Run one complete PGG game."""
    rng = np.random.default_rng(config.seed)
    sem = asyncio.Semaphore(config.concurrency)

    agents = [LLMAgent(agent_id=i) for i in range(config.n_agents)]

    # seed norm agents
    n_seeds = max(1, int(config.n_agents * config.seed_fraction))
    seed_ids = rng.choice(config.n_agents, size=n_seeds, replace=False)
    for idx in seed_ids:
        agents[idx].is_norm_seed = True

    round_data = []
    cooperation_rate = []
    mean_contribution = []
    mean_payoff = []
    total_calls = 0
    t0 = time.time()

    for round_num in range(config.n_rounds):
        # form groups based on condition
        if config.condition == "A":
            groups = make_dyadic_pairs(config.n_agents, rng)
        elif config.condition == "B":
            groups = make_reciprocal_pairs(config.n_agents, agents, rng)
        elif config.condition == "C":
            groups = make_triad_groups(config.n_agents, rng)
        elif config.condition == "D":
            groups = make_pentad_groups(config.n_agents, rng)
        else:
            raise ValueError(f"Unknown condition: {config.condition}")

        round_contributions = {}
        round_payoffs = {}

        # Phase 1: all agents decide concurrently across all groups
        all_tasks = []
        for group in groups:
            for agent_id in group:
                agent = agents[agent_id]
                system = agent.get_system_prompt(config.endowment, config.multiplier)
                if agent.history:
                    last = agent.history[-1]
                    prev_info = (
                        f"Round {round_num + 1}/{config.n_rounds}. "
                        f"Your cumulative payoff so far: {agent.cumulative_payoff:.1f}. "
                        f"Last round you contributed {last.get('contribution', '?')}, "
                        f"others in your group contributed {last.get('group_contributions', '?')}, "
                        f"your payoff was {last.get('payoff', '?'):.1f}."
                    )
                else:
                    prev_info = (
                        f"Round 1/{config.n_rounds}. "
                        f"This is the first round. You have no information about other players."
                    )
                user_msg = f"{prev_info}\n\nDecide: how many tokens (0-{config.endowment}) do you put into the pool?"
                all_tasks.append((agent_id, system, user_msg))

        raw_responses = await asyncio.gather(*[
            call_llm(config.llm_config, sys, usr, sem)
            for _, sys, usr in all_tasks
        ])
        total_calls += len(all_tasks)

        responses = {}
        for (agent_id, _, _), raw in zip(all_tasks, raw_responses):
            responses[agent_id] = parse_contribution(raw, config.endowment)

        # Phase 2: for Condition C/D, group deliberation + majority rule enforcement
        if config.condition in ("C", "D"):
            delib_tasks = []
            agent_to_group = {}
            for group in groups:
                if len(group) < 3:
                    continue
                intentions = {aid: responses[aid] for aid in group}
                intention_str = ", ".join(
                    f"Agent {aid}: {c}" for aid, c in intentions.items()
                )
                for agent_id in group:
                    agent = agents[agent_id]
                    system = agent.get_system_prompt(config.endowment, config.multiplier)
                    delib_msg = GROUP_DELIBERATION_PROMPT.format(
                        group_size=len(group),
                        intentions=intention_str,
                    )
                    delib_tasks.append((agent_id, system, delib_msg))
                    agent_to_group[agent_id] = tuple(group)

            if delib_tasks:
                delib_responses = await asyncio.gather(*[
                    call_llm(config.llm_config, sys, usr, sem)
                    for _, sys, usr in delib_tasks
                ])
                total_calls += len(delib_tasks)
                for (agent_id, _, _), raw in zip(delib_tasks, delib_responses):
                    responses[agent_id] = parse_contribution(raw, config.endowment)

            # Majority rule enforcement (matching ABM's higher-order contagion):
            # In Iacopini's framework, hyperedge interaction amplifies the
            # dominant signal: if majority cooperate, ALL cooperate; otherwise
            # ALL defect. This is the mechanism that creates discontinuous
            # phase transitions absent in pairwise interactions.
            coop_threshold = config.endowment * 0.5
            for group in groups:
                if len(group) < 3:
                    continue
                group_contribs = [responses[aid] for aid in group]
                n_cooperators = sum(1 for c in group_contribs if c >= coop_threshold)
                majority_cooperates = n_cooperators > len(group) / 2
                if majority_cooperates:
                    for aid in group:
                        responses[aid] = config.endowment
                else:
                    for aid in group:
                        responses[aid] = 0

        # Compute payoffs per group
        for group in groups:
            group_contributions = [responses[aid] for aid in group]
            pool = sum(group_contributions) * config.multiplier
            share = pool / len(group)

            for aid in group:
                payoff = (config.endowment - responses[aid]) + share
                round_contributions[aid] = responses[aid]
                round_payoffs[aid] = payoff
                agents[aid].cumulative_payoff += payoff
                agents[aid].history.append({
                    "round": round_num,
                    "contribution": responses[aid],
                    "payoff": payoff,
                    "group_contributions": group_contributions,
                    "group_ids": group,
                })

        # record round metrics
        contribs = list(round_contributions.values())
        payoffs = list(round_payoffs.values())
        coop = sum(1 for c in contribs if c >= config.endowment * 0.75) / len(contribs)

        cooperation_rate.append(coop)
        mean_contribution.append(float(np.mean(contribs)))
        mean_payoff.append(float(np.mean(payoffs)))

        round_data.append({
            "round": round_num,
            "contributions": round_contributions,
            "payoffs": round_payoffs,
            "cooperation_rate": coop,
            "mean_contribution": float(np.mean(contribs)),
        })

        if (round_num + 1) % 5 == 0 or round_num == 0:
            logger.info("  [%s] Round %d/%d: coop=%.2f, mean_contrib=%.1f, calls=%d",
                        config.condition, round_num + 1, config.n_rounds, coop,
                        np.mean(contribs), total_calls)

    elapsed = time.time() - t0

    agent_histories = {}
    for a in agents:
        agent_histories[a.agent_id] = {
            "is_seed": a.is_norm_seed,
            "final_payoff": a.cumulative_payoff,
            "history": a.history,
        }

    return GameResult(
        condition=config.condition,
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
        seed=config.seed,
        round_data=round_data,
        cooperation_rate=cooperation_rate,
        mean_contribution=mean_contribution,
        mean_payoff=mean_payoff,
        total_llm_calls=total_calls,
        elapsed_seconds=elapsed,
        agent_histories=agent_histories,
    )


def save_result(result: GameResult, output_dir: Path) -> Path:
    """Save game result to JSON with full token usage for archiving."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"llm_game_{result.condition}_seed{result.seed}.json"
    path = output_dir / fname

    data = {
        "condition": result.condition,
        "n_agents": result.n_agents,
        "n_rounds": result.n_rounds,
        "seed": result.seed,
        "cooperation_rate": result.cooperation_rate,
        "mean_contribution": result.mean_contribution,
        "mean_payoff": result.mean_payoff,
        "total_llm_calls": result.total_llm_calls,
        "elapsed_seconds": result.elapsed_seconds,
        "token_usage": get_token_usage(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "round_data": result.round_data,
        "agent_histories": {
            str(k): {
                "is_seed": v["is_seed"],
                "final_payoff": v["final_payoff"],
                "contributions": [h["contribution"] for h in v["history"]],
            }
            for k, v in result.agent_histories.items()
        },
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved: %s", path)
    return path
