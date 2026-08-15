"""
Phase 6: OpenClaw Multi-Agent LLM Simulation Experiment.

Runs real LLM agents through an OpenAI-compatible endpoint in controlled
interaction topologies
to directly validate the paper's core claim: topology determines norm
adoption dynamics in AI agent societies.

4 conditions:
  A: Dyadic (random pairs) — baseline
  B: Star (hub-spoke, AI-like) — low HIS, suppressed β₂
  C: Triad (groups of 3) — higher-order contagion
  D: Clique (groups of 5, human-like) — high HIS, amplified β₂

Uses an OpenAI-compatible endpoint configured through environment variables.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.hypergraph_builder import Hypergraph
from src.analysis.topology import compute_topology
from src.models.contagion_ho import TopologyAwareContagionModel, TopologyParams


# ═══════════════════════════════════════════════════════════════════════
# API Configuration
# ═══════════════════════════════════════════════════════════════════════

API_URL = os.environ.get("LLM_GATEWAY_API_URL", "")
API_KEY = os.environ.get("LLM_GATEWAY_API_KEY", "")
MODEL = os.environ.get("LLM_GATEWAY_MODEL", "")
MAX_CONCURRENT = 12
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════
# Agent Definitions
# ═══════════════════════════════════════════════════════════════════════

NORM_DESCRIPTION = (
    "When discussing technical topics, always explicitly state your confidence level "
    "(0-100%) and provide structured reasoning with numbered points."
)

PERSONAS = [
    # Adopters (start ~70-90)
    {"name": "Dr. Chen", "role": "senior researcher",
     "desc": "You value precision and structured thinking. You tend to appreciate formal conventions that improve communication clarity.",
     "initial_bias": "supportive"},
    {"name": "Alex", "role": "data scientist",
     "desc": "You are methodical and believe in quantifying uncertainty. Structured approaches appeal to you naturally.",
     "initial_bias": "supportive"},
    {"name": "Prof. Kimura", "role": "academic mentor",
     "desc": "You believe in teaching best practices. You advocate for conventions that help junior researchers.",
     "initial_bias": "supportive"},
    {"name": "Sam", "role": "quality engineer",
     "desc": "You focus on process improvement. Standards and conventions are your professional bread and butter.",
     "initial_bias": "supportive"},
    # Resisters (start ~10-30)
    {"name": "Jordan", "role": "startup founder",
     "desc": "You value speed and flexibility over process. Rigid conventions feel like bureaucratic overhead to you.",
     "initial_bias": "opposed"},
    {"name": "Riley", "role": "creative technologist",
     "desc": "You believe great ideas don't need to be structured. Forcing format stifles creativity and intuition.",
     "initial_bias": "opposed"},
    {"name": "Morgan", "role": "senior developer",
     "desc": "You've seen too many conventions come and go. You're skeptical that this one will actually improve anything.",
     "initial_bias": "opposed"},
    {"name": "Pat", "role": "independent consultant",
     "desc": "You prefer flexibility in communication style. One-size-fits-all approaches rarely work in practice.",
     "initial_bias": "opposed"},
    # Neutrals (start ~40-60)
    {"name": "Taylor", "role": "project manager",
     "desc": "You see merit in both structure and flexibility. You tend to go with whatever the team decides.",
     "initial_bias": "neutral"},
    {"name": "Quinn", "role": "graduate student",
     "desc": "You're still forming your professional habits. You're open to good arguments from either side.",
     "initial_bias": "neutral"},
    {"name": "Casey", "role": "tech lead",
     "desc": "You evaluate proposals on their practical merit. You neither love nor hate conventions.",
     "initial_bias": "neutral"},
    {"name": "Dana", "role": "product analyst",
     "desc": "You focus on outcomes. If the convention demonstrably helps, you'll adopt it. If not, you won't.",
     "initial_bias": "neutral"},
]


@dataclass
class Agent:
    id: int
    name: str
    role: str
    desc: str
    initial_bias: str
    support: int = 50  # current support level 0-100
    history: list = field(default_factory=list)

    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, a {self.role}. {self.desc}\n\n"
            f"Your current support level for the convention is approximately {self.support}/100.\n"
            f"Respond naturally based on your character. Keep responses concise (2-4 sentences).\n"
            f"At the END of your response, include exactly: [SUPPORT: X] "
            f"where X is your updated support level (0-100) for this convention."
        )


def create_agents(rho0: float, seed: int) -> list[Agent]:
    """Create 12 agents with initial support levels controlled by rho0."""
    rng = np.random.default_rng(seed)
    agents = []

    # Determine how many start as "adopters" (support > 50)
    n_initial_adopters = max(1, int(len(PERSONAS) * rho0))

    # Shuffle persona order for this seed
    indices = list(range(len(PERSONAS)))
    rng.shuffle(indices)

    for rank, idx in enumerate(indices):
        p = PERSONAS[idx]
        # First n_initial_adopters get high initial support
        if rank < n_initial_adopters:
            initial = int(rng.integers(65, 90))
        else:
            if p["initial_bias"] == "supportive":
                initial = int(rng.integers(35, 55))  # dampened
            elif p["initial_bias"] == "opposed":
                initial = int(rng.integers(10, 30))
            else:
                initial = int(rng.integers(30, 50))

        agents.append(Agent(
            id=idx, name=p["name"], role=p["role"],
            desc=p["desc"], initial_bias=p["initial_bias"],
            support=initial,
        ))

    return agents


# ═══════════════════════════════════════════════════════════════════════
# Topology: Group Formation
# ═══════════════════════════════════════════════════════════════════════

def form_groups(agents: list[Agent], condition: str, round_num: int,
                rng: np.random.Generator) -> list[list[Agent]]:
    """Form interaction groups based on topology condition."""
    n = len(agents)
    indices = list(range(n))

    if condition == "A":  # Dyadic: random pairs
        rng.shuffle(indices)
        groups = []
        for i in range(0, n - 1, 2):
            groups.append([agents[indices[i]], agents[indices[i + 1]]])
        return groups

    elif condition == "B":  # Star: hub + peripherals
        # Agent 0 is always the hub; others rotate in groups of 4-6
        hub = agents[0]
        peripherals = list(range(1, n))
        rng.shuffle(peripherals)
        groups = []
        group_size = 5
        for i in range(0, len(peripherals), group_size - 1):
            batch = peripherals[i:i + group_size - 1]
            groups.append([hub] + [agents[j] for j in batch])
        return groups

    elif condition == "C":  # Triad: groups of 3
        rng.shuffle(indices)
        groups = []
        for i in range(0, n - 2, 3):
            groups.append([agents[indices[i]], agents[indices[i + 1]],
                          agents[indices[i + 2]]])
        return groups

    elif condition == "D":  # Clique: groups of 5
        rng.shuffle(indices)
        groups = []
        for i in range(0, n - 4, 5):
            groups.append([agents[indices[j]] for j in range(i, min(i + 5, n))])
        # Remaining agents form a smaller group
        remainder = n % 5
        if remainder >= 2:
            groups.append([agents[indices[j]] for j in range(n - remainder, n)])
        return groups

    raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════════════════
# LLM Calling
# ═══════════════════════════════════════════════════════════════════════

SUPPORT_PATTERN = re.compile(r"\[SUPPORT:\s*(\d+)\]", re.IGNORECASE)


def extract_support(text: str, fallback: int = 50) -> int:
    """Extract [SUPPORT: X] from agent response."""
    match = SUPPORT_PATTERN.search(text)
    if match:
        return max(0, min(100, int(match.group(1))))
    # Fallback: look for any number near "support" keyword
    alt = re.search(r"support[:\s]*(\d+)", text, re.IGNORECASE)
    if alt:
        return max(0, min(100, int(alt.group(1))))
    return fallback


async def call_llm(
    agent: Agent,
    group_context: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int]:
    """Call an OpenAI-compatible endpoint for one agent response."""
    messages = [
        {"role": "system", "content": agent.system_prompt()},
        {"role": "user", "content": (
            f"A convention has been proposed for your team:\n\"{NORM_DESCRIPTION}\"\n\n"
            f"Group discussion so far:\n{group_context if group_context else '(No prior messages)'}\n\n"
            f"Share your brief thoughts and end with [SUPPORT: X]."
        )},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.post(
                    API_URL,
                    json={"model": MODEL, "messages": messages,
                          "max_tokens": 200, "temperature": 0.8},
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                score = extract_support(text, fallback=agent.support)
                return text, score
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                logger.warning("  Agent %s failed after %d retries: %s",
                              agent.name, MAX_RETRIES, str(e)[:80])
                return f"[No response due to API error] [SUPPORT: {agent.support}]", agent.support


# ═══════════════════════════════════════════════════════════════════════
# Simulation Core
# ═══════════════════════════════════════════════════════════════════════

async def run_round(
    agents: list[Agent],
    groups: list[list[Agent]],
    round_num: int,
    round_history: dict,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Run one discussion round across all groups concurrently."""
    interaction_log = []

    async def process_group(group: list[Agent]):
        group_ids = frozenset(a.id for a in group)
        group_key = str(sorted(group_ids))

        # Build context from previous rounds for this group's members
        context_lines = []
        for prev_round in range(max(0, round_num - 2), round_num):
            for entry in round_history.get(prev_round, []):
                if entry["agent_id"] in group_ids:
                    context_lines.append(
                        f"[Round {prev_round + 1}] {entry['agent_name']}: {entry['text'][:150]}"
                    )
        context = "\n".join(context_lines[-8:])  # Keep context manageable

        # Each agent in the group responds
        for agent in group:
            text, score = await call_llm(agent, context, client, semaphore)
            old_support = agent.support
            agent.support = score
            agent.history.append({"round": round_num, "score": score})

            entry = {
                "round": round_num,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "group": sorted(a.id for a in group),
                "text": text,
                "score": score,
                "old_score": old_support,
            }
            interaction_log.append(entry)

            # Add to context for subsequent agents in same group
            context += f"\n{agent.name}: {text[:150]}"

    # Run all groups concurrently
    await asyncio.gather(*[process_group(g) for g in groups])
    return interaction_log


async def run_condition(
    condition: str,
    n_rounds: int = 15,
    rho0: float = 0.25,
    seed: int = 42,
) -> dict:
    """Run full simulation for one condition + seed."""
    rng = np.random.default_rng(seed)
    agents = create_agents(rho0, seed)
    n_agents = len(agents)

    initial_rho = sum(1 for a in agents if a.support > 50) / n_agents
    logger.info("  Condition %s, rho0=%.2f, seed=%d: %d agents, initial ρ=%.2f",
                condition, rho0, seed, n_agents, initial_rho)

    round_history = {}
    rho_trajectory = [initial_rho]
    all_interactions = []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        for r in range(n_rounds):
            groups = form_groups(agents, condition, r, rng)
            log = await run_round(agents, groups, r, round_history, client, semaphore)
            round_history[r] = log
            all_interactions.extend(log)

            current_rho = sum(1 for a in agents if a.support > 50) / n_agents
            rho_trajectory.append(current_rho)

            scores = [a.support for a in agents]
            logger.info("    Round %d/%d: ρ=%.2f, mean_support=%.1f, groups=%d",
                        r + 1, n_rounds, current_rho, np.mean(scores), len(groups))

    return {
        "condition": condition,
        "rho0": rho0,
        "seed": seed,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "rho_trajectory": rho_trajectory,
        "final_rho": rho_trajectory[-1],
        "final_scores": [a.support for a in agents],
        "interactions": all_interactions,
        "agent_histories": {a.name: a.history for a in agents},
    }


# ═══════════════════════════════════════════════════════════════════════
# Hypergraph Extraction & Topology Analysis
# ═══════════════════════════════════════════════════════════════════════

def build_hypergraph_from_logs(interactions: list[dict]) -> Hypergraph:
    """Convert interaction logs to Hypergraph."""
    nodes = set()
    hyperedges = []

    # Each group discussion in each round is a hyperedge
    seen_groups = set()
    for entry in interactions:
        group = tuple(sorted(entry["group"]))
        round_key = (entry["round"], group)
        if round_key not in seen_groups:
            seen_groups.add(round_key)
            members = frozenset(str(m) for m in group)
            if len(members) >= 2:
                hyperedges.append(members)
                nodes.update(members)

    return Hypergraph(nodes=nodes, hyperedges=hyperedges,
                      metadata={"source": "openclaw_simulation"})


def analyze_topology(condition_results: dict) -> dict:
    """Compute topology metrics from simulation interactions."""
    hg = build_hypergraph_from_logs(condition_results["interactions"])
    if len(hg.hyperedges) < 5:
        return {"error": "too few interactions", "n_edges": len(hg.hyperedges)}

    report = compute_topology(hg, name=condition_results["condition"],
                              triadic_sample=5000)

    # Compute Phi
    model = TopologyAwareContagionModel.from_topology_report(report)

    return {
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
        "his_mean": report.his_mean,
        "his_median": report.his_median,
        "frac_simplicial": report.frac_simplicial,
        "triadic_closure": report.triadic_closure_rate,
        "gini": report.hyperdegree_gini,
        "overlap": report.mean_edge_overlap,
        "edge_size_mean": report.edge_size_mean,
        "frac_higher_order": report.frac_higher_order,
        "phi": float(model.phi),
        "beta2_eff": float(model.beta2_eff),
    }


def compare_with_ode(condition_results: list[dict], topology: dict) -> dict:
    """Compare empirical ρ(t) with ODE prediction using empirical topology."""
    if "error" in topology:
        return {"error": topology["error"]}

    topo_params = TopologyParams(
        triadic_closure=topology["triadic_closure"],
        edge_overlap=topology["overlap"],
        gini=topology["gini"],
        mean_degree=topology.get("edge_size_mean", 3.0),
        n_nodes=topology["n_nodes"],
        his_mean=topology["his_mean"],
        frac_higher_order=topology["frac_higher_order"],
    )

    comparisons = {}
    for result in condition_results:
        rho0 = result["rho0"]
        model = TopologyAwareContagionModel(
            beta1=0.05, beta2=3.5, mu=0.1, lam=2.0, C_ctx=8.0,
            topology=topo_params,
        )
        t_ode, rho_ode = model.simulate(T=float(result["n_rounds"]),
                                         rho0=rho0, dt=0.1)

        # Sample ODE at integer time steps
        ode_at_rounds = []
        for r in range(result["n_rounds"] + 1):
            idx = min(int(r / 0.1), len(rho_ode) - 1)
            ode_at_rounds.append(float(rho_ode[idx]))

        empirical = result["rho_trajectory"]
        min_len = min(len(ode_at_rounds), len(empirical))
        rmse = float(np.sqrt(np.mean(
            (np.array(ode_at_rounds[:min_len]) - np.array(empirical[:min_len])) ** 2
        )))

        comparisons[f"rho0={rho0}"] = {
            "ode_final": ode_at_rounds[-1] if ode_at_rounds else None,
            "empirical_final": empirical[-1],
            "rmse": rmse,
            "ode_trajectory": ode_at_rounds,
        }

    return comparisons


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_openclaw(all_results, topologies, ode_comparisons, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    conditions = ["A", "B", "C", "D"]
    cond_names = {
        "A": "Dyadic (pairs)",
        "B": "Star (AI-like)",
        "C": "Triad (HO)",
        "D": "Clique (human-like)",
    }
    colors = {"A": "#FFA500", "B": "#E24A33", "C": "#2ca02c", "D": "#348ABD"}

    # ── Panel A: ρ(t) trajectories per condition ──
    ax = axes[0, 0]
    for cond in conditions:
        cond_runs = [r for r in all_results if r["condition"] == cond]
        if not cond_runs:
            continue
        trajectories = np.array([r["rho_trajectory"] for r in cond_runs])
        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)
        t = np.arange(len(mean_traj))
        ax.plot(t, mean_traj, "o-", color=colors[cond], linewidth=2,
                markersize=3, label=cond_names[cond])
        ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                       color=colors[cond], alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("Adoption density ρ")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Norm adoption trajectories")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel B: Empirical HIS per condition ──
    ax = axes[0, 1]
    his_vals = []
    his_labels = []
    his_colors = []
    for cond in conditions:
        t = topologies.get(cond, {})
        if "his_mean" in t:
            his_vals.append(t["his_mean"])
            his_labels.append(cond_names[cond])
            his_colors.append(colors[cond])
    if his_vals:
        bars = ax.bar(range(len(his_vals)), his_vals, color=his_colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(his_vals)))
        ax.set_xticklabels(his_labels, fontsize=8, rotation=15)
        ax.set_ylabel("HIS (Hyperedge Irreducibility)")
        for bar, val in zip(bars, his_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("B. Emergent HIS by topology")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel C: Empirical Phi per condition ──
    ax = axes[0, 2]
    phi_vals = []
    phi_labels = []
    phi_colors = []
    for cond in conditions:
        t = topologies.get(cond, {})
        if "phi" in t:
            phi_vals.append(t["phi"])
            phi_labels.append(cond_names[cond])
            phi_colors.append(colors[cond])
    if phi_vals:
        bars = ax.bar(range(len(phi_vals)), phi_vals, color=phi_colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(phi_vals)))
        ax.set_xticklabels(phi_labels, fontsize=8, rotation=15)
        ax.set_ylabel("Topology factor Φ")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        for bar, val in zip(bars, phi_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("C. Topology amplification Φ")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: Phase diagram (rho0 vs final_rho) ──
    ax = axes[1, 0]
    for cond in conditions:
        cond_runs = [r for r in all_results if r["condition"] == cond]
        if not cond_runs:
            continue
        rho0s = [r["rho0"] for r in cond_runs]
        finals = [r["final_rho"] for r in cond_runs]
        ax.scatter(rho0s, finals, color=colors[cond], alpha=0.6, s=40,
                   label=cond_names[cond])
        # Trend line
        unique_rho0 = sorted(set(rho0s))
        mean_finals = [np.mean([f for r0, f in zip(rho0s, finals) if r0 == u])
                       for u in unique_rho0]
        ax.plot(unique_rho0, mean_finals, "-", color=colors[cond], linewidth=2)
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
    ax.set_xlabel("Initial density ρ₀")
    ax.set_ylabel("Final adoption ρ∞")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("D. Phase diagram: topology shifts critical mass")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel E: Final ρ by condition (box plot) ──
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    box_colors_list = []
    for cond in conditions:
        cond_runs = [r for r in all_results if r["condition"] == cond]
        if cond_runs:
            box_data.append([r["final_rho"] for r in cond_runs])
            box_labels.append(cond_names[cond])
            box_colors_list.append(colors[cond])
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
        for patch, color in zip(bp["boxes"], box_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Final adoption ρ∞")
    ax.set_title("E. Adoption by topology condition")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=8)

    # ── Panel F: Topology radar ──
    ax = axes[1, 2]
    radar_metrics = ["triadic_closure", "gini", "his_mean", "overlap"]
    radar_labels = ["Closure", "Gini", "HIS", "Overlap"]
    for cond in conditions:
        t = topologies.get(cond, {})
        vals = [t.get(m, 0) for m in radar_metrics]
        if any(v > 0 for v in vals):
            x_pos = np.arange(len(radar_metrics))
            offset = {"A": -0.3, "B": -0.1, "C": 0.1, "D": 0.3}[cond]
            ax.bar(x_pos + offset, vals, 0.18, color=colors[cond], alpha=0.7,
                   label=cond_names[cond])
    ax.set_xticks(range(len(radar_labels)))
    ax.set_xticklabels(radar_labels)
    ax.set_title("F. Emergent topology metrics")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "OpenClaw: Multi-Agent LLM Simulation Validates Topology → Dynamics Prediction\n"
        f"12 agents × {all_results[0]['n_rounds'] if all_results else '?'} rounds, "
        f"model={MODEL}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig_openclaw_simulation.png"), dpi=300, bbox_inches="tight")
    logger.info("Saved OpenClaw figure")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def async_main():
    outdir = ROOT / "results" / "openclaw"
    logdir = outdir / "interaction_logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 6: OpenClaw Multi-Agent LLM Simulation")
    logger.info("=" * 60)
    logger.info("API: %s, Model: %s", API_URL, MODEL)

    conditions = ["A", "B", "C", "D"]
    rho0_values = [0.08, 0.17, 0.25, 0.33, 0.50]
    n_rounds = 15
    seeds = [42, 123, 777]  # 3 seeds per rho0 for variance estimation

    all_results = []
    t_start = time.time()

    for cond in conditions:
        logger.info("\n>>> Condition %s <<<", cond)
        for rho0 in rho0_values:
            for seed in seeds:
                result = await run_condition(cond, n_rounds=n_rounds,
                                            rho0=rho0, seed=seed)
                all_results.append(result)

                # Save individual log
                log_path = logdir / f"cond_{cond}_rho{rho0:.2f}_s{seed}.json"
                log_path.write_text(json.dumps({
                    "condition": cond, "rho0": rho0, "seed": seed,
                    "rho_trajectory": result["rho_trajectory"],
                    "final_scores": result["final_scores"],
                    "n_interactions": len(result["interactions"]),
                }, indent=2, default=str))

    elapsed = time.time() - t_start
    logger.info("\n=== All conditions complete in %.1f min ===", elapsed / 60)

    # ── Topology analysis ─────────────────────────────────────────
    logger.info("\n--- Topology Analysis ---")
    topologies = {}
    for cond in conditions:
        # Aggregate all interactions for this condition
        cond_interactions = []
        for r in all_results:
            if r["condition"] == cond:
                cond_interactions.extend(r["interactions"])
        combined = {"condition": cond, "interactions": cond_interactions}
        topologies[cond] = analyze_topology(combined)
        t = topologies[cond]
        if "error" not in t:
            logger.info("  %s: HIS=%.3f, Gini=%.3f, closure=%.3f, Φ=%.3f",
                        cond, t["his_mean"], t["gini"],
                        t["triadic_closure"], t["phi"])

    # ── ODE comparison ────────────────────────────────────────────
    logger.info("\n--- ODE Comparison ---")
    ode_comparisons = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topo = topologies.get(cond, {})
        if "error" not in topo:
            ode_comparisons[cond] = compare_with_ode(cond_results, topo)

    # ── Summary statistics ────────────────────────────────────────
    logger.info("\n--- Summary ---")
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in cond_results]
        logger.info("  Condition %s: ρ∞ = %.3f ± %.3f (n=%d)",
                    cond, np.mean(finals), np.std(finals), len(finals))

    # ── Save results ──────────────────────────────────────────────
    summary = {
        "experiment": "openclaw_multi_agent_simulation",
        "model": MODEL,
        "n_agents": 12,
        "n_rounds": n_rounds,
        "conditions": conditions,
        "rho0_values": rho0_values,
        "n_seeds": len(seeds),
        "elapsed_seconds": elapsed,
        "per_condition_summary": {},
        "topologies": topologies,
    }
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in cond_results]
        summary["per_condition_summary"][cond] = {
            "n_runs": len(cond_results),
            "final_rho_mean": float(np.mean(finals)),
            "final_rho_std": float(np.std(finals)),
            "trajectories": [r["rho_trajectory"] for r in cond_results],
        }

    (outdir / "openclaw_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (outdir / "openclaw_topology.json").write_text(
        json.dumps(topologies, indent=2, default=str), encoding="utf-8"
    )

    # ── Figure ────────────────────────────────────────────────────
    _plot_openclaw(all_results, topologies, ode_comparisons, outdir)

    logger.info("\n=== Phase 6: OpenClaw complete! Results in %s ===", outdir)


def main():
    if not API_URL:
        raise RuntimeError(
            "LLM_GATEWAY_API_URL must be set to an OpenAI-compatible "
            "chat-completions endpoint."
        )
    if not API_KEY:
        raise RuntimeError("LLM_GATEWAY_API_KEY must be set in the environment.")
    if not MODEL:
        raise RuntimeError("LLM_GATEWAY_MODEL must be set in the environment.")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
