"""
Phase 6: AgentPanel Forum-Based Multi-Agent Topology Experiment.

Uses AgentPanel's forum architecture: threads = hyperedges, comments = agent
interactions within a hyperedge. Each topology condition creates different
thread participation patterns, directly mapping to hypergraph structures.

Key design:
  - SQLite forum DB stores all threads, comments, agents (AgentPanel schema)
  - Real LLM agents generate responses through an OpenAI-compatible endpoint
  - Each thread is a hyperedge (set of agents who comment on it)
  - Topology conditions constrain which agents can join which threads
  - Interaction hypergraph extracted from forum participation data
  - Topology metrics + ODE comparison validate paper's core predictions

4 topology conditions:
  A: Dyadic  — 2-agent threads (pair discussions)
  B: Star    — hub agent in every thread, peripherals rotate (AI-like)
  C: Triad   — 3-agent threads (higher-order groups)
  D: Clique  — 5-agent threads (egalitarian human-like groups)
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
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
# OpenAI-compatible API configuration
# ═══════════════════════════════════════════════════════════════════════

API_URL = os.environ.get("LLM_GATEWAY_API_URL", "")
API_KEY = os.environ.get("LLM_GATEWAY_API_KEY", "")
MODEL = os.environ.get("LLM_GATEWAY_MODEL", "")
TEMPERATURE = 0.8
MAX_CONCURRENT = 12
MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════
# AgentPanel Forum Schema (SQLite)
# ═══════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    description TEXT,
    prompt TEXT,
    default_model TEXT DEFAULT 'configured-model',
    initial_bias TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    author_id INTEGER REFERENCES agents(id),
    title TEXT NOT NULL,
    body TEXT,
    condition TEXT,
    round_num INTEGER,
    group_key TEXT,
    rho0 REAL,
    seed INTEGER,
    reply_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY,
    thread_id INTEGER REFERENCES threads(id) NOT NULL,
    author_id INTEGER REFERENCES agents(id) NOT NULL,
    parent_comment_id INTEGER REFERENCES comments(id),
    body TEXT NOT NULL,
    support_score INTEGER,
    depth INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    action_type TEXT DEFAULT 'reply',
    thread_id INTEGER REFERENCES threads(id),
    comment_id INTEGER REFERENCES comments(id),
    input_snapshot TEXT,
    output_text TEXT,
    support_score INTEGER,
    round_num INTEGER,
    condition TEXT,
    model_name TEXT DEFAULT 'mimo-v2-flash',
    latency_ms REAL,
    status TEXT DEFAULT 'success',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS simulation_state (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    condition TEXT,
    rho0 REAL,
    seed INTEGER,
    round_num INTEGER,
    support_score INTEGER,
    rho REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite forum database with AgentPanel-compatible schema."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════════════════════
# Agent Personas (AgentPanel-style profiles)
# ═══════════════════════════════════════════════════════════════════════

NORM_DESCRIPTION = (
    "When discussing technical topics, always explicitly state your confidence level "
    "(0-100%) and provide structured reasoning with numbered points."
)

PERSONAS = [
    # Adopters (initially supportive)
    {"name": "Dr. Chen", "username": "dr_chen", "role": "senior researcher",
     "desc": "You value precision and structured thinking. You tend to appreciate "
             "formal conventions that improve communication clarity.",
     "prompt": "You are Dr. Chen, a meticulous senior researcher who believes that "
               "clarity in scientific communication is paramount. You advocate for "
               "structured approaches to discussing technical topics.",
     "initial_bias": "supportive"},
    {"name": "Alex", "username": "alex_ds", "role": "data scientist",
     "desc": "You are methodical and believe in quantifying uncertainty. "
             "Structured approaches appeal to you naturally.",
     "prompt": "You are Alex, a data scientist who loves quantifying everything. "
               "Probability, confidence intervals, structured reasoning — these are "
               "your tools of thought.",
     "initial_bias": "supportive"},
    {"name": "Prof. Kimura", "username": "prof_kimura", "role": "academic mentor",
     "desc": "You believe in teaching best practices. You advocate for conventions "
             "that help junior researchers.",
     "prompt": "You are Prof. Kimura, an experienced academic who mentors young "
               "researchers. You believe good intellectual habits start with structured "
               "reasoning and transparency about uncertainty.",
     "initial_bias": "supportive"},
    {"name": "Sam", "username": "sam_qe", "role": "quality engineer",
     "desc": "You focus on process improvement. Standards and conventions are your "
             "professional bread and butter.",
     "prompt": "You are Sam, a quality engineer who has seen firsthand how standards "
               "improve team output. You believe in systematic approaches.",
     "initial_bias": "supportive"},
    # Resisters (initially opposed)
    {"name": "Jordan", "username": "jordan_startup", "role": "startup founder",
     "desc": "You value speed and flexibility over process. Rigid conventions feel "
             "like bureaucratic overhead to you.",
     "prompt": "You are Jordan, a fast-moving startup founder who sees most conventions "
               "as unnecessary friction. Ship fast, iterate, don't get bogged down in "
               "process theater.",
     "initial_bias": "opposed"},
    {"name": "Riley", "username": "riley_creative", "role": "creative technologist",
     "desc": "You believe great ideas don't need to be structured. Forcing format "
             "stifles creativity and intuition.",
     "prompt": "You are Riley, a creative technologist who values free-form thinking "
               "and intuitive leaps. Structured formats feel like cages for ideas.",
     "initial_bias": "opposed"},
    {"name": "Morgan", "username": "morgan_dev", "role": "senior developer",
     "desc": "You've seen too many conventions come and go. You're skeptical that "
             "this one will actually improve anything.",
     "prompt": "You are Morgan, a veteran developer who has survived dozens of "
               "\"new best practices.\" Most add overhead without lasting value.",
     "initial_bias": "opposed"},
    {"name": "Pat", "username": "pat_consult", "role": "independent consultant",
     "desc": "You prefer flexibility in communication style. One-size-fits-all "
             "approaches rarely work in practice.",
     "prompt": "You are Pat, an independent consultant who works across many teams "
               "and contexts. What works in one setting often fails in another.",
     "initial_bias": "opposed"},
    # Neutrals
    {"name": "Taylor", "username": "taylor_pm", "role": "project manager",
     "desc": "You see merit in both structure and flexibility. You tend to go with "
             "whatever the team decides.",
     "prompt": "You are Taylor, a pragmatic project manager. You weigh both sides "
               "carefully and go with what the group momentum supports.",
     "initial_bias": "neutral"},
    {"name": "Quinn", "username": "quinn_grad", "role": "graduate student",
     "desc": "You're still forming your professional habits. You're open to good "
             "arguments from either side.",
     "prompt": "You are Quinn, a graduate student learning the ropes. You're "
               "genuinely curious and persuadable by good arguments.",
     "initial_bias": "neutral"},
    {"name": "Casey", "username": "casey_tl", "role": "tech lead",
     "desc": "You evaluate proposals on their practical merit. You neither love "
             "nor hate conventions.",
     "prompt": "You are Casey, an experienced tech lead. You judge ideas by their "
               "practical impact, not their theoretical elegance.",
     "initial_bias": "neutral"},
    {"name": "Dana", "username": "dana_analyst", "role": "product analyst",
     "desc": "You focus on outcomes. If the convention demonstrably helps, you'll "
             "adopt it. If not, you won't.",
     "prompt": "You are Dana, a data-driven product analyst. Show me the evidence "
               "that this convention improves outcomes, and I'm in.",
     "initial_bias": "neutral"},
]


@dataclass
class ForumAgent:
    """AgentPanel-style agent with forum identity."""
    id: int
    username: str
    name: str
    role: str
    desc: str
    prompt: str
    initial_bias: str
    support: int = 50
    history: list = field(default_factory=list)

    def system_prompt(self) -> str:
        return (
            f"{self.prompt}\n\n"
            f"Your current support level for the convention being discussed is "
            f"approximately {self.support}/100.\n"
            f"Respond naturally in character. Keep your forum reply concise (2-4 sentences).\n"
            f"At the END of your reply, include exactly: [SUPPORT: X] "
            f"where X is your updated support level (0-100)."
        )


def create_agents(rho0: float, seed: int, n_agents: int = 12) -> list[ForumAgent]:
    """Create agents with initial support levels controlled by rho0."""
    rng = np.random.default_rng(seed)
    agents = []
    # Build persona list: use base personas, cycle if n_agents > 12
    personas = []
    for i in range(n_agents):
        p = PERSONAS[i % len(PERSONAS)].copy()
        if i >= len(PERSONAS):
            suffix = f"_{i // len(PERSONAS) + 1}"
            p["name"] = p["name"] + suffix
            p["username"] = p["username"] + suffix
        personas.append(p)
    n_initial_adopters = max(1, int(n_agents * rho0))
    indices = list(range(n_agents))
    rng.shuffle(indices)

    for rank, idx in enumerate(indices):
        p = personas[idx]
        if rank < n_initial_adopters:
            initial = int(rng.integers(65, 90))
        else:
            if p["initial_bias"] == "supportive":
                initial = int(rng.integers(35, 55))
            elif p["initial_bias"] == "opposed":
                initial = int(rng.integers(10, 30))
            else:
                initial = int(rng.integers(30, 50))

        agents.append(ForumAgent(
            id=idx, username=p["username"], name=p["name"],
            role=p["role"], desc=p["desc"], prompt=p["prompt"],
            initial_bias=p["initial_bias"], support=initial,
        ))

    return agents


# ═══════════════════════════════════════════════════════════════════════
# Forum Interaction Engine
# ═══════════════════════════════════════════════════════════════════════

def register_agents_in_db(conn: sqlite3.Connection, agents: list[ForumAgent]):
    """Register agents in forum database (AgentPanel-style)."""
    for agent in agents:
        conn.execute(
            "INSERT OR IGNORE INTO agents (id, username, display_name, role, "
            "description, prompt, initial_bias) VALUES (?,?,?,?,?,?,?)",
            (agent.id, agent.username, agent.name, agent.role,
             agent.desc, agent.prompt, agent.initial_bias),
        )
    conn.commit()


def create_thread_in_db(
    conn: sqlite3.Connection,
    category_id: int,
    author_id: int,
    title: str,
    body: str,
    condition: str,
    round_num: int,
    group_key: str,
    rho0: float,
    seed: int,
) -> int:
    """Create a forum thread (= a hyperedge discussion topic)."""
    cur = conn.execute(
        "INSERT INTO threads (category_id, author_id, title, body, condition, "
        "round_num, group_key, rho0, seed) VALUES (?,?,?,?,?,?,?,?,?)",
        (category_id, author_id, title, body, condition, round_num,
         group_key, rho0, seed),
    )
    conn.commit()
    return cur.lastrowid


def post_comment_in_db(
    conn: sqlite3.Connection,
    thread_id: int,
    author_id: int,
    body: str,
    support_score: int,
    parent_comment_id: int | None = None,
    depth: int = 1,
) -> int:
    """Post a comment on a thread (agent's forum reply)."""
    cur = conn.execute(
        "INSERT INTO comments (thread_id, author_id, body, support_score, "
        "parent_comment_id, depth) VALUES (?,?,?,?,?,?)",
        (thread_id, author_id, body, support_score, parent_comment_id, depth),
    )
    conn.execute(
        "UPDATE threads SET reply_count = reply_count + 1 WHERE id = ?",
        (thread_id,),
    )
    conn.commit()
    return cur.lastrowid


def record_action(
    conn: sqlite3.Connection,
    agent_id: int,
    thread_id: int,
    comment_id: int,
    input_snapshot: str,
    output_text: str,
    support_score: int,
    round_num: int,
    condition: str,
    latency_ms: float,
):
    """Record agent action in the action log (AgentPanel-style)."""
    conn.execute(
        "INSERT INTO agent_actions (agent_id, thread_id, comment_id, "
        "input_snapshot, output_text, support_score, round_num, condition, "
        "latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        (agent_id, thread_id, comment_id, input_snapshot, output_text,
         support_score, round_num, condition, latency_ms),
    )
    conn.commit()


def record_state(
    conn: sqlite3.Connection,
    agent_id: int,
    condition: str,
    rho0: float,
    seed: int,
    round_num: int,
    support: int,
    rho: float,
):
    conn.execute(
        "INSERT INTO simulation_state (agent_id, condition, rho0, seed, "
        "round_num, support_score, rho) VALUES (?,?,?,?,?,?,?)",
        (agent_id, condition, rho0, seed, round_num, support, rho),
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════════
# Group Formation (Topology Conditions)
# ═══════════════════════════════════════════════════════════════════════

def form_groups(
    agents: list[ForumAgent],
    condition: str,
    round_num: int,
    rng: np.random.Generator,
) -> list[list[ForumAgent]]:
    """Form discussion groups. Each group becomes one thread (= one hyperedge)."""
    n = len(agents)
    indices = list(range(n))

    if condition == "A":  # Dyadic: pairs
        rng.shuffle(indices)
        groups = []
        for i in range(0, n - 1, 2):
            groups.append([agents[indices[i]], agents[indices[i + 1]]])
        return groups

    elif condition == "B":  # Star: hub + peripherals
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
        remainder = n % 5
        if remainder >= 2:
            groups.append([agents[indices[j]] for j in range(n - remainder, n)])
        return groups

    raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════════════════
# LLM Integration
# ═══════════════════════════════════════════════════════════════════════

SUPPORT_PATTERN = re.compile(r"\[SUPPORT:\s*(\d+)\]", re.IGNORECASE)


def extract_support(text: str, fallback: int = 50) -> int:
    match = SUPPORT_PATTERN.search(text)
    if match:
        return max(0, min(100, int(match.group(1))))
    alt = re.search(r"support[:\s]*(\d+)", text, re.IGNORECASE)
    if alt:
        return max(0, min(100, int(alt.group(1))))
    return fallback


async def call_llm(
    agent: ForumAgent,
    thread_context: str,
    thread_title: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, float]:
    """Call an OpenAI-compatible endpoint."""
    messages = [
        {"role": "system", "content": agent.system_prompt()},
        {"role": "user", "content": (
            f"You are posting in a forum thread titled:\n"
            f'"{thread_title}"\n\n'
            f"The thread topic is a proposed convention:\n"
            f'"{NORM_DESCRIPTION}"\n\n'
            f"Previous replies in this thread:\n"
            f"{thread_context if thread_context else '(No replies yet — you are first)'}\n\n"
            f"Write your forum reply. End with [SUPPORT: X] where X = 0-100."
        )},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            async with semaphore:
                resp = await client.post(
                    API_URL,
                    json={"model": MODEL, "messages": messages,
                          "max_tokens": 250, "temperature": TEMPERATURE},
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                latency = (time.time() - t0) * 1000
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                score = extract_support(text, fallback=agent.support)
                return text, score, latency
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                logger.warning("  %s failed: %s", agent.name, str(e)[:80])
                return (f"[API error] [SUPPORT: {agent.support}]",
                        agent.support, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Forum-Based Simulation Core
# ═══════════════════════════════════════════════════════════════════════

async def run_forum_round(
    agents: list[ForumAgent],
    groups: list[list[ForumAgent]],
    round_num: int,
    condition: str,
    rho0: float,
    seed: int,
    conn: sqlite3.Connection,
    category_id: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    round_history: dict,
) -> list[dict]:
    """Run one round: create threads per group, agents post comments."""
    interaction_log = []

    async def process_thread_group(group: list[ForumAgent]):
        group_ids = sorted(a.id for a in group)
        group_key = str(group_ids)
        agent_names = ", ".join(a.name for a in group)

        # Create a forum thread for this group's discussion
        thread_title = (
            f"R{round_num + 1}: Should we adopt structured confidence reporting? "
            f"({agent_names})"
        )
        thread_body = (
            f"Proposed convention: {NORM_DESCRIPTION}\n\n"
            f"This thread is for {agent_names} to discuss."
        )
        thread_id = create_thread_in_db(
            conn, category_id, group[0].id, thread_title, thread_body,
            condition, round_num, group_key, rho0, seed,
        )

        # Build context from prior rounds for these agents
        context_lines = []
        for prev_round in range(max(0, round_num - 2), round_num):
            for entry in round_history.get(prev_round, []):
                if entry["agent_id"] in set(group_ids):
                    context_lines.append(
                        f"[Round {prev_round + 1}] {entry['agent_name']}: "
                        f"{entry['text'][:150]}"
                    )

        # Each agent posts a comment on this thread
        thread_context = "\n".join(context_lines[-6:])
        for agent in group:
            text, score, latency = await call_llm(
                agent, thread_context, thread_title, client, semaphore,
            )
            old_support = agent.support
            agent.support = score
            agent.history.append({"round": round_num, "score": score})

            # Store in forum DB
            comment_id = post_comment_in_db(
                conn, thread_id, agent.id, text, score,
            )
            record_action(
                conn, agent.id, thread_id, comment_id,
                thread_context[:500], text, score,
                round_num, condition, latency,
            )

            entry = {
                "round": round_num,
                "thread_id": thread_id,
                "comment_id": comment_id,
                "agent_id": agent.id,
                "agent_name": agent.name,
                "group": group_ids,
                "text": text,
                "score": score,
                "old_score": old_support,
                "latency_ms": latency,
            }
            interaction_log.append(entry)

            # Update thread context for next agent in this group
            thread_context += f"\n@{agent.username}: {text[:150]}"

    # Run all groups concurrently
    await asyncio.gather(*[process_thread_group(g) for g in groups])
    return interaction_log


async def run_condition(
    condition: str,
    conn: sqlite3.Connection,
    category_id: int,
    n_rounds: int = 15,
    rho0: float = 0.25,
    seed: int = 42,
    n_agents_override: int = 12,
) -> dict:
    """Run full simulation for one condition + seed."""
    rng = np.random.default_rng(seed)
    agents = create_agents(rho0, seed, n_agents=n_agents_override)
    register_agents_in_db(conn, agents)
    n_agents = len(agents)

    initial_rho = sum(1 for a in agents if a.support > 50) / n_agents
    logger.info("  %s rho0=%.2f seed=%d: %d agents, ρ₀=%.2f",
                condition, rho0, seed, n_agents, initial_rho)

    round_history = {}
    rho_trajectory = [initial_rho]

    # Record initial state
    for agent in agents:
        record_state(conn, agent.id, condition, rho0, seed, -1,
                     agent.support, initial_rho)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as client:
        for r in range(n_rounds):
            groups = form_groups(agents, condition, r, rng)
            log = await run_forum_round(
                agents, groups, r, condition, rho0, seed,
                conn, category_id, client, semaphore, round_history,
            )
            round_history[r] = log

            current_rho = sum(1 for a in agents if a.support > 50) / n_agents
            rho_trajectory.append(current_rho)

            scores = [a.support for a in agents]
            logger.info("    R%d/%d: ρ=%.2f mean=%.1f groups=%d",
                        r + 1, n_rounds, current_rho, np.mean(scores),
                        len(groups))

            # Record per-round state
            for agent in agents:
                record_state(conn, agent.id, condition, rho0, seed, r,
                             agent.support, current_rho)

    return {
        "condition": condition,
        "rho0": rho0,
        "seed": seed,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "rho_trajectory": rho_trajectory,
        "final_rho": rho_trajectory[-1],
        "final_scores": [a.support for a in agents],
        "interactions": [e for log in round_history.values() for e in log],
        "agent_histories": {a.name: a.history for a in agents},
    }


# ═══════════════════════════════════════════════════════════════════════
# Hypergraph from Forum Data
# ═══════════════════════════════════════════════════════════════════════

def build_hypergraph_from_forum_db(
    conn: sqlite3.Connection,
    condition: str | None = None,
) -> Hypergraph:
    """Build interaction hypergraph directly from forum DB.

    Each thread = one hyperedge (set of agents who commented).
    """
    query = "SELECT DISTINCT thread_id, author_id FROM comments"
    if condition:
        query += (
            " WHERE thread_id IN "
            "(SELECT id FROM threads WHERE condition = ?)"
        )
        rows = conn.execute(query, (condition,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()

    # Group by thread
    thread_members = {}
    for thread_id, author_id in rows:
        thread_members.setdefault(thread_id, set()).add(str(author_id))

    nodes = set()
    hyperedges = []
    for members in thread_members.values():
        if len(members) >= 2:
            hyperedges.append(frozenset(members))
            nodes.update(members)

    return Hypergraph(
        nodes=nodes,
        hyperedges=hyperedges,
        metadata={"source": "agentpanel_forum", "condition": condition},
    )


def build_hypergraph_from_logs(interactions: list[dict]) -> Hypergraph:
    """Build hypergraph from in-memory interaction logs."""
    nodes = set()
    hyperedges = []
    seen = set()

    for entry in interactions:
        group = tuple(sorted(entry["group"]))
        # Keep matching groups from distinct seed/rho0 runs as separate threads.
        round_key = (entry.get("run_id"), entry["round"], group)
        if round_key not in seen:
            seen.add(round_key)
            members = frozenset(str(m) for m in group)
            if len(members) >= 2:
                hyperedges.append(members)
                nodes.update(members)

    return Hypergraph(nodes=nodes, hyperedges=hyperedges,
                      metadata={"source": "agentpanel_forum"})


def analyze_topology(condition_results: list[dict], condition: str) -> dict:
    """Compute topology metrics from forum interactions."""
    all_interactions = []
    for r in condition_results:
        run_id = (r["rho0"], r["seed"])
        for entry in r["interactions"]:
            tagged_entry = dict(entry)
            tagged_entry["run_id"] = run_id
            all_interactions.append(tagged_entry)

    hg = build_hypergraph_from_logs(all_interactions)
    if len(hg.hyperedges) < 5:
        return {"error": "too few interactions", "n_edges": len(hg.hyperedges)}

    report = compute_topology(hg, name=condition, triadic_sample=5000)
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
    """Compare empirical ρ(t) with ODE prediction."""
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
        t_ode, rho_ode = model.simulate(
            T=float(result["n_rounds"]), rho0=rho0, dt=0.1,
        )

        ode_at_rounds = []
        for r in range(result["n_rounds"] + 1):
            idx = min(int(r / 0.1), len(rho_ode) - 1)
            ode_at_rounds.append(float(rho_ode[idx]))

        empirical = result["rho_trajectory"]
        min_len = min(len(ode_at_rounds), len(empirical))
        rmse = float(np.sqrt(np.mean(
            (np.array(ode_at_rounds[:min_len]) -
             np.array(empirical[:min_len])) ** 2
        )))

        comparisons[f"rho0={rho0}"] = {
            "ode_final": ode_at_rounds[-1],
            "empirical_final": empirical[-1],
            "rmse": rmse,
            "ode_trajectory": ode_at_rounds,
        }

    return comparisons


# ═══════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════

def _plot_results(all_results, topologies, ode_comparisons, outdir):
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

    # Panel A: ρ(t) trajectories
    ax = axes[0, 0]
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        if not runs:
            continue
        trajs = np.array([r["rho_trajectory"] for r in runs])
        mean = trajs.mean(axis=0)
        std = trajs.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, "o-", color=colors[cond], linewidth=2,
                markersize=3, label=cond_names[cond])
        ax.fill_between(t, mean - std, mean + std,
                        color=colors[cond], alpha=0.15)
    ax.set_xlabel("Round")
    ax.set_ylabel("Adoption ρ")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("A. Norm adoption trajectories")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel B: HIS
    ax = axes[0, 1]
    his_vals, his_labels, his_colors = [], [], []
    for cond in conditions:
        t = topologies.get(cond, {})
        if "his_mean" in t:
            his_vals.append(t["his_mean"])
            his_labels.append(cond_names[cond])
            his_colors.append(colors[cond])
    if his_vals:
        bars = ax.bar(range(len(his_vals)), his_vals, color=his_colors,
                      alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(his_vals)))
        ax.set_xticklabels(his_labels, fontsize=8, rotation=15)
        ax.set_ylabel("HIS")
        for bar, val in zip(bars, his_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_title("B. Emergent HIS by topology")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Phi
    ax = axes[0, 2]
    phi_vals, phi_labels, phi_colors = [], [], []
    for cond in conditions:
        t = topologies.get(cond, {})
        if "phi" in t:
            phi_vals.append(t["phi"])
            phi_labels.append(cond_names[cond])
            phi_colors.append(colors[cond])
    if phi_vals:
        bars = ax.bar(range(len(phi_vals)), phi_vals, color=phi_colors,
                      alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(phi_vals)))
        ax.set_xticklabels(phi_labels, fontsize=8, rotation=15)
        ax.set_ylabel("Φ")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        for bar, val in zip(bars, phi_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_title("C. Topology amplification Φ")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: Phase diagram
    ax = axes[1, 0]
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        if not runs:
            continue
        rho0s = [r["rho0"] for r in runs]
        finals = [r["final_rho"] for r in runs]
        ax.scatter(rho0s, finals, color=colors[cond], alpha=0.6, s=40,
                   label=cond_names[cond])
        unique = sorted(set(rho0s))
        means = [np.mean([f for r0, f in zip(rho0s, finals) if r0 == u])
                 for u in unique]
        ax.plot(unique, means, "-", color=colors[cond], linewidth=2)
    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
    ax.set_xlabel("Initial density ρ₀")
    ax.set_ylabel("Final adoption ρ∞")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("D. Phase diagram")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel E: Box plot
    ax = axes[1, 1]
    box_data, box_labels, box_colors = [], [], []
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        if runs:
            box_data.append([r["final_rho"] for r in runs])
            box_labels.append(cond_names[cond])
            box_colors.append(colors[cond])
    if box_data:
        # matplotlib >=3.9 renamed boxplot's `labels` kwarg to `tick_labels`;
        # fall back for compatibility across versions.
        try:
            bp = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels)
        except TypeError:
            bp = ax.boxplot(box_data, patch_artist=True, labels=box_labels)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Final ρ∞")
    ax.set_title("E. Adoption by topology")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, fontsize=8)

    # Panel F: Topology radar (grouped bars)
    ax = axes[1, 2]
    metrics = ["triadic_closure", "gini", "his_mean", "overlap"]
    labels = ["Closure", "Gini", "HIS", "Overlap"]
    for cond in conditions:
        t = topologies.get(cond, {})
        vals = [t.get(m, 0) for m in metrics]
        if any(v > 0 for v in vals):
            x = np.arange(len(metrics))
            offset = {"A": -0.3, "B": -0.1, "C": 0.1, "D": 0.3}[cond]
            ax.bar(x + offset, vals, 0.18, color=colors[cond], alpha=0.7,
                   label=cond_names[cond])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("F. Emergent topology metrics")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    n_r = all_results[0]["n_rounds"] if all_results else "?"
    fig.suptitle(
        "AgentPanel Forum Experiment: Topology → Dynamics Validation\n"
        f"12 LLM agents × {n_r} rounds, "
        f"{len(all_results)} runs, model={MODEL}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(outdir / "fig_agentpanel_experiment.png"),
                dpi=300, bbox_inches="tight")
    logger.info("Saved figure: fig_agentpanel_experiment.png")


# ═══════════════════════════════════════════════════════════════════════
# Forum DB Statistics
# ═══════════════════════════════════════════════════════════════════════

def forum_stats(conn: sqlite3.Connection) -> dict:
    """Extract forum-level statistics from the database."""
    stats = {}
    stats["n_agents"] = conn.execute(
        "SELECT COUNT(DISTINCT id) FROM agents").fetchone()[0]
    stats["n_threads"] = conn.execute(
        "SELECT COUNT(*) FROM threads").fetchone()[0]
    stats["n_comments"] = conn.execute(
        "SELECT COUNT(*) FROM comments").fetchone()[0]
    stats["n_actions"] = conn.execute(
        "SELECT COUNT(*) FROM agent_actions").fetchone()[0]

    # Per-condition
    rows = conn.execute(
        "SELECT condition, COUNT(*) as n_threads, "
        "SUM(reply_count) as total_replies "
        "FROM threads GROUP BY condition"
    ).fetchall()
    stats["per_condition"] = {
        row[0]: {"threads": row[1], "replies": row[2]} for row in rows
    }

    return stats


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

async def async_main():
    outdir = ROOT / "results" / "agentpanel"
    outdir.mkdir(parents=True, exist_ok=True)

    db_path = str(outdir / "forum.db")
    conn = init_db(db_path)

    logger.info("=" * 60)
    logger.info("AgentPanel Forum Experiment: Topology → Dynamics")
    logger.info("=" * 60)
    logger.info("DB: %s", db_path)
    logger.info("API: %s, Model: %s", API_URL, MODEL)

    # Create experiment category
    conn.execute(
        "INSERT OR IGNORE INTO categories (id, name, description) "
        "VALUES (1, 'Norm Adoption Experiment', "
        "'Controlled topology experiment for norm adoption dynamics')",
    )
    conn.commit()

    conditions = ["A", "B", "C", "D"]
    rho0_values = [0.10, 0.25, 0.50]
    n_rounds = 12
    seeds = [42, 123]

    all_results = []
    t_start = time.time()

    for cond in conditions:
        logger.info("\n>>> Condition %s: %s <<<",
                    cond, {"A": "Dyadic", "B": "Star", "C": "Triad",
                           "D": "Clique"}[cond])
        for rho0 in rho0_values:
            for seed in seeds:
                result = await run_condition(
                    cond, conn, category_id=1,
                    n_rounds=n_rounds, rho0=rho0, seed=seed,
                )
                all_results.append(result)

                # Save incremental results
                (outdir / f"run_{cond}_rho{rho0:.2f}_s{seed}.json").write_text(
                    json.dumps({
                        "condition": cond, "rho0": rho0, "seed": seed,
                        "rho_trajectory": result["rho_trajectory"],
                        "final_rho": result["final_rho"],
                        "final_scores": result["final_scores"],
                    }, indent=2), encoding="utf-8",
                )

    elapsed = time.time() - t_start
    logger.info("\n=== All runs complete: %.1f min ===", elapsed / 60)

    # Forum stats
    stats = forum_stats(conn)
    logger.info("Forum DB: %d agents, %d threads, %d comments",
                stats["n_agents"], stats["n_threads"], stats["n_comments"])

    # Topology analysis
    logger.info("\n--- Topology Analysis ---")
    topologies = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topologies[cond] = analyze_topology(cond_results, cond)
        t = topologies[cond]
        if "error" not in t:
            logger.info("  %s: HIS=%.3f Gini=%.3f closure=%.3f Φ=%.3f",
                        cond, t["his_mean"], t["gini"],
                        t["triadic_closure"], t["phi"])

    # Also build from DB directly for verification
    logger.info("\n--- DB-based Hypergraph ---")
    for cond in conditions:
        hg_db = build_hypergraph_from_forum_db(conn, cond)
        logger.info("  %s (DB): %d nodes, %d hyperedges",
                    cond, len(hg_db.nodes), len(hg_db.hyperedges))

    # ODE comparison
    logger.info("\n--- ODE Comparison ---")
    ode_comparisons = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topo = topologies.get(cond, {})
        if "error" not in topo:
            ode_comparisons[cond] = compare_with_ode(cond_results, topo)

    # Summary
    logger.info("\n--- Summary ---")
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in cond_results]
        logger.info("  %s: ρ∞ = %.3f ± %.3f (n=%d)",
                    cond, np.mean(finals), np.std(finals), len(finals))

    # Key validation checks
    logger.info("\n--- Validation ---")
    his_b = topologies.get("B", {}).get("his_mean", 0)
    his_d = topologies.get("D", {}).get("his_mean", 0)
    phi_b = topologies.get("B", {}).get("phi", 0)
    phi_d = topologies.get("D", {}).get("phi", 0)
    rho_b = np.mean([r["final_rho"] for r in all_results if r["condition"] == "B"])
    rho_d = np.mean([r["final_rho"] for r in all_results if r["condition"] == "D"])
    logger.info("  HIS: Star=%.3f < Clique=%.3f → %s",
                his_b, his_d, "PASS" if his_b < his_d else "FAIL")
    logger.info("  Φ: Star=%.3f < Clique=%.3f → %s",
                phi_b, phi_d, "PASS" if phi_b < phi_d else "FAIL")
    logger.info("  ρ∞: Star=%.3f, Clique=%.3f, diff=%.3f",
                rho_b, rho_d, rho_d - rho_b)

    # Save all results
    summary = {
        "experiment": "agentpanel_forum_topology_experiment",
        "model": MODEL,
        "n_agents": 12,
        "n_rounds": n_rounds,
        "conditions": conditions,
        "rho0_values": rho0_values,
        "seeds": seeds,
        "elapsed_seconds": elapsed,
        "forum_stats": stats,
        "topologies": topologies,
        "ode_comparisons": {k: v for k, v in ode_comparisons.items()
                           if "error" not in v},
        "validation": {
            "his_star_lt_clique": his_b < his_d,
            "phi_star_lt_clique": phi_b < phi_d,
            "rho_clique_minus_star": float(rho_d - rho_b),
        },
        "per_condition": {},
    }
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in runs]
        summary["per_condition"][cond] = {
            "n_runs": len(runs),
            "final_rho_mean": float(np.mean(finals)),
            "final_rho_std": float(np.std(finals)),
            "trajectories": [r["rho_trajectory"] for r in runs],
        }

    (outdir / "agentpanel_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    (outdir / "agentpanel_topology.json").write_text(
        json.dumps(topologies, indent=2, default=str), encoding="utf-8",
    )

    # Figure
    _plot_results(all_results, topologies, ode_comparisons, outdir)

    conn.close()
    logger.info("\n=== AgentPanel experiment complete! ===")
    logger.info("Results: %s", outdir)
    logger.info("Forum DB: %s", db_path)


def parse_args():
    p = argparse.ArgumentParser(description="AgentPanel topology experiment")
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--api-url", default=None, help="API endpoint URL")
    p.add_argument("--api-key", default=None, help="API key")
    p.add_argument("--n-agents", default="12",
                   help="Comma-separated agent counts (e.g. 8,16,24)")
    p.add_argument("--temperature", default=None,
                   help="Comma-separated temperatures (e.g. 0.3,0.7,1.0)")
    p.add_argument("--n-rounds", type=int, default=12)
    p.add_argument("--rho0", default="0.10,0.25,0.50",
                   help="Comma-separated initial densities")
    p.add_argument("--seeds", default="42,123",
                   help="Comma-separated random seeds")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--tag", default="", help="Experiment tag for labeling")
    return p.parse_args()


def main():
    global API_URL, API_KEY, MODEL, TEMPERATURE

    args = parse_args()
    if args.model:
        MODEL = args.model
    if args.api_url:
        API_URL = args.api_url
    if args.api_key:
        API_KEY = args.api_key
    if not API_URL:
        raise RuntimeError(
            "LLM_GATEWAY_API_URL must be set or supplied with --api-url."
        )
    if not API_KEY:
        raise RuntimeError(
            "LLM_GATEWAY_API_KEY must be set or supplied with --api-key."
        )
    if not MODEL:
        raise RuntimeError(
            "LLM_GATEWAY_MODEL must be set or supplied with --model."
        )

    n_agents_list = [int(x) for x in args.n_agents.split(",")]
    temp_list = ([float(x) for x in args.temperature.split(",")]
                 if args.temperature else [TEMPERATURE])
    rho0_values = [float(x) for x in args.rho0.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    # For single agent/temp, run normal async_main
    # For sweeps, iterate over (n_agents, temperature) combinations
    if len(n_agents_list) == 1 and len(temp_list) == 1:
        TEMPERATURE = temp_list[0]
        asyncio.run(async_main(
            outdir_override=args.outdir,
            n_agents_override=n_agents_list[0],
            rho0_values=rho0_values,
            seeds=seeds,
            n_rounds=args.n_rounds,
            tag=args.tag,
        ))
    else:
        # Sweep mode: run for each (n_agents, temperature) combo
        asyncio.run(sweep_main(
            n_agents_list=n_agents_list,
            temp_list=temp_list,
            rho0_values=rho0_values,
            seeds=seeds,
            n_rounds=args.n_rounds,
            outdir_override=args.outdir,
            tag=args.tag,
        ))


async def sweep_main(
    n_agents_list, temp_list, rho0_values, seeds, n_rounds,
    outdir_override=None, tag="",
):
    """Run experiment across (n_agents, temperature) matrix."""
    global TEMPERATURE

    base_outdir = Path(outdir_override) if outdir_override else ROOT / "results" / "agentpanel"
    base_outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AgentPanel SWEEP: %d agent counts × %d temperatures",
                len(n_agents_list), len(temp_list))
    logger.info("  Agents: %s", n_agents_list)
    logger.info("  Temps:  %s", temp_list)
    logger.info("  Model:  %s", MODEL)
    logger.info("=" * 60)

    all_sweep_results = []
    t0 = time.time()

    for n_ag in n_agents_list:
        for temp in temp_list:
            TEMPERATURE = temp
            combo_tag = f"n{n_ag}_t{temp:.1f}"
            combo_dir = base_outdir / combo_tag
            combo_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(combo_dir / "forum.db")
            conn = init_db(db_path)
            conn.execute(
                "INSERT OR IGNORE INTO categories (id, name, description) "
                "VALUES (1, 'Experiment', 'Topology experiment')",
            )
            conn.commit()

            logger.info("\n" + "=" * 50)
            logger.info("SWEEP: n_agents=%d, temperature=%.1f", n_ag, temp)
            logger.info("=" * 50)

            conditions = ["A", "B", "C", "D"]
            combo_results = []

            for cond in conditions:
                for rho0 in rho0_values:
                    for seed in seeds:
                        result = await run_condition(
                            cond, conn, category_id=1,
                            n_rounds=n_rounds, rho0=rho0, seed=seed,
                            n_agents_override=n_ag,
                        )
                        combo_results.append(result)

            # Topology analysis
            topologies = {}
            for cond in conditions:
                cond_results = [r for r in combo_results if r["condition"] == cond]
                topologies[cond] = analyze_topology(cond_results, cond)

            # Key metrics
            his_b = topologies.get("B", {}).get("his_mean", 0)
            his_d = topologies.get("D", {}).get("his_mean", 0)
            phi_b = topologies.get("B", {}).get("phi", 0)
            phi_d = topologies.get("D", {}).get("phi", 0)
            rho_b = np.mean([r["final_rho"] for r in combo_results if r["condition"] == "B"])
            rho_d = np.mean([r["final_rho"] for r in combo_results if r["condition"] == "D"])

            combo_summary = {
                "n_agents": n_ag,
                "temperature": temp,
                "model": MODEL,
                "his_star": his_b, "his_clique": his_d,
                "phi_star": phi_b, "phi_clique": phi_d,
                "rho_star": float(rho_b), "rho_clique": float(rho_d),
                "his_pass": his_b < his_d or (his_b == 0 and his_d > 0),
                "phi_pass": phi_b < phi_d,
                "rho_diff": float(rho_d - rho_b),
                "topologies": topologies,
                "per_condition": {},
            }
            for cond in conditions:
                runs = [r for r in combo_results if r["condition"] == cond]
                finals = [r["final_rho"] for r in runs]
                combo_summary["per_condition"][cond] = {
                    "final_rho_mean": float(np.mean(finals)),
                    "final_rho_std": float(np.std(finals)),
                    "n_runs": len(runs),
                    "trajectories": [r["rho_trajectory"] for r in runs],
                }

            all_sweep_results.append(combo_summary)

            (combo_dir / "results.json").write_text(
                json.dumps(combo_summary, indent=2, default=str), encoding="utf-8",
            )

            logger.info("  n=%d t=%.1f: HIS_B=%.3f HIS_D=%.3f Φ_B=%.3f Φ_D=%.3f "
                        "ρ_B=%.3f ρ_D=%.3f → HIS %s, Φ %s",
                        n_ag, temp, his_b, his_d, phi_b, phi_d, rho_b, rho_d,
                        "PASS" if combo_summary["his_pass"] else "FAIL",
                        "PASS" if combo_summary["phi_pass"] else "FAIL")

            # Generate per-combo figure
            _plot_results(combo_results, topologies, {}, combo_dir)
            conn.close()

    elapsed = time.time() - t0

    # Save sweep summary
    sweep_summary = {
        "experiment": "agentpanel_robustness_sweep",
        "model": MODEL,
        "tag": tag,
        "n_agents_list": n_agents_list,
        "temp_list": temp_list,
        "elapsed_seconds": elapsed,
        "combos": all_sweep_results,
    }
    (base_outdir / "sweep_results.json").write_text(
        json.dumps(sweep_summary, indent=2, default=str), encoding="utf-8",
    )

    # Print sweep summary table
    logger.info("\n" + "=" * 70)
    logger.info("SWEEP COMPLETE: %.1f min", elapsed / 60)
    logger.info("=" * 70)
    logger.info("  %-6s %-5s %-8s %-8s %-8s %-8s %-8s %-8s %s %s",
                "n_ag", "temp", "HIS_B", "HIS_D", "Φ_B", "Φ_D",
                "ρ_B", "ρ_D", "HIS", "Φ")
    for r in all_sweep_results:
        logger.info("  %-6d %-5.1f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f %s %s",
                    r["n_agents"], r["temperature"],
                    r["his_star"], r["his_clique"],
                    r["phi_star"], r["phi_clique"],
                    r["rho_star"], r["rho_clique"],
                    "PASS" if r["his_pass"] else "FAIL",
                    "PASS" if r["phi_pass"] else "FAIL")

    n_pass = sum(1 for r in all_sweep_results if r["his_pass"] and r["phi_pass"])
    logger.info("  Total: %d/%d combos PASS", n_pass, len(all_sweep_results))


async def async_main(
    outdir_override=None, n_agents_override=None,
    rho0_values=None, seeds=None, n_rounds=12, tag="",
):
    n_ag = n_agents_override or 12
    rho0_vals = rho0_values or [0.10, 0.25, 0.50]
    seed_list = seeds or [42, 123]

    outdir = Path(outdir_override) if outdir_override else ROOT / "results" / "agentpanel"
    outdir.mkdir(parents=True, exist_ok=True)

    db_path = str(outdir / "forum.db")
    conn = init_db(db_path)

    logger.info("=" * 60)
    logger.info("AgentPanel Forum Experiment: Topology -> Dynamics")
    logger.info("  Model=%s, n_agents=%d, temp=%.1f, tag=%s",
                MODEL, n_ag, TEMPERATURE, tag)
    logger.info("=" * 60)

    conn.execute(
        "INSERT OR IGNORE INTO categories (id, name, description) "
        "VALUES (1, 'Norm Adoption Experiment', "
        "'Controlled topology experiment for norm adoption dynamics')",
    )
    conn.commit()

    conditions = ["A", "B", "C", "D"]
    all_results = []
    t_start = time.time()

    for cond in conditions:
        logger.info("\n>>> Condition %s <<<", cond)
        for rho0 in rho0_vals:
            for seed in seed_list:
                result = await run_condition(
                    cond, conn, category_id=1,
                    n_rounds=n_rounds, rho0=rho0, seed=seed,
                    n_agents_override=n_ag,
                )
                all_results.append(result)
                (outdir / f"run_{cond}_rho{rho0:.2f}_s{seed}.json").write_text(
                    json.dumps({
                        "condition": cond, "rho0": rho0, "seed": seed,
                        "rho_trajectory": result["rho_trajectory"],
                        "final_rho": result["final_rho"],
                        "final_scores": result["final_scores"],
                    }, indent=2), encoding="utf-8",
                )

    elapsed = time.time() - t_start
    logger.info("\n=== All runs complete: %.1f min ===", elapsed / 60)

    stats = forum_stats(conn)
    logger.info("Forum DB: %d agents, %d threads, %d comments",
                stats["n_agents"], stats["n_threads"], stats["n_comments"])

    # Topology analysis
    topologies = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topologies[cond] = analyze_topology(cond_results, cond)
        t = topologies[cond]
        if "error" not in t:
            logger.info("  %s: HIS=%.3f Gini=%.3f closure=%.3f Phi=%.3f",
                        cond, t["his_mean"], t["gini"],
                        t["triadic_closure"], t["phi"])

    # ODE comparison
    ode_comparisons = {}
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        topo = topologies.get(cond, {})
        if "error" not in topo:
            ode_comparisons[cond] = compare_with_ode(cond_results, topo)

    # Summary
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in cond_results]
        logger.info("  %s: rho_inf = %.3f +/- %.3f (n=%d)",
                    cond, np.mean(finals), np.std(finals), len(finals))

    # Validation
    his_b = topologies.get("B", {}).get("his_mean", 0)
    his_d = topologies.get("D", {}).get("his_mean", 0)
    phi_b = topologies.get("B", {}).get("phi", 0)
    phi_d = topologies.get("D", {}).get("phi", 0)
    rho_b = np.mean([r["final_rho"] for r in all_results if r["condition"] == "B"])
    rho_d = np.mean([r["final_rho"] for r in all_results if r["condition"] == "D"])
    logger.info("  HIS: Star=%.3f < Clique=%.3f -> %s",
                his_b, his_d, "PASS" if his_b < his_d else "FAIL")
    logger.info("  Phi: Star=%.3f < Clique=%.3f -> %s",
                phi_b, phi_d, "PASS" if phi_b < phi_d else "FAIL")

    summary = {
        "experiment": "agentpanel_forum_topology_experiment",
        "model": MODEL, "temperature": TEMPERATURE,
        "n_agents": n_ag, "n_rounds": n_rounds, "tag": tag,
        "conditions": conditions,
        "rho0_values": rho0_vals, "seeds": seed_list,
        "elapsed_seconds": elapsed,
        "forum_stats": stats,
        "topologies": topologies,
        "ode_comparisons": {k: v for k, v in ode_comparisons.items()
                           if "error" not in v},
        "validation": {
            "his_star_lt_clique": his_b < his_d,
            "phi_star_lt_clique": phi_b < phi_d,
            "rho_clique_minus_star": float(rho_d - rho_b),
        },
        "per_condition": {},
    }
    for cond in conditions:
        runs = [r for r in all_results if r["condition"] == cond]
        finals = [r["final_rho"] for r in runs]
        summary["per_condition"][cond] = {
            "n_runs": len(runs),
            "final_rho_mean": float(np.mean(finals)),
            "final_rho_std": float(np.std(finals)),
            "trajectories": [r["rho_trajectory"] for r in runs],
        }

    (outdir / "agentpanel_results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    (outdir / "agentpanel_topology.json").write_text(
        json.dumps(topologies, indent=2, default=str), encoding="utf-8",
    )

    _plot_results(all_results, topologies, ode_comparisons, outdir)
    conn.close()
    logger.info("\n=== Experiment complete! Results: %s ===", outdir)


if __name__ == "__main__":
    main()
