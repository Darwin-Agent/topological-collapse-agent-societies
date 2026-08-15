"""
ABM Public Goods Game on configurable hypergraph topologies.

Implements Condition A-D from the paper:
  A: Dyadic baseline — random pairwise interactions
  B: Dyadic + reciprocity — pairwise with mutual confirmation
  C: Triad hyperedge — synchronous 3-agent group deliberation
  D: Pentad hyperedge — synchronous 5-agent group deliberation

Uses a minimal agent framework (no Mesa dependency) for maximum control
over the hyperedge synchronous deliberation mechanism.

Key mechanisms:
  - Fermi strategy update with social learning
  - Norm seeding (fraction of initial cooperators)
  - Higher-order group deliberation with majority rule
  - Endowment e=10, multiplier r=3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

ENDOWMENT = 10.0
MULTIPLIER = 3.0
FERMI_BETA = 2.0  # moderate selection intensity for visible condition differences


class Condition(Enum):
    A = "dyadic_baseline"
    B = "dyadic_reciprocity"
    C = "triad_hyperedge"
    D = "pentad_hyperedge"


@dataclass
class PGGAgent:
    uid: int
    contribution: float = 0.0
    payoff: float = 0.0
    is_cooperator: bool = False
    norm_adopted: bool = False
    cooperation_history: list = field(default_factory=list)

    def decide_contribution(self) -> float:
        if self.is_cooperator:
            return ENDOWMENT
        return 0.0

    def record(self):
        self.cooperation_history.append(self.is_cooperator)


# ── Topology generators ─────────────────────────────────────────────

def build_dyadic_graph(n: int, avg_degree: int = 6, rng: np.random.Generator = None) -> list[tuple[int, int]]:
    """Erdos-Renyi random graph as edge list."""
    if rng is None:
        rng = np.random.default_rng()
    p = avg_degree / (n - 1)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return edges


def build_hyperedges(n: int, order: int, avg_membership: int = 4, rng: np.random.Generator = None) -> list[tuple[int, ...]]:
    """
    Build random hyperedges of given order (group size).
    Each agent participates in ~avg_membership hyperedges on average.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_hyperedges = max(1, (n * avg_membership) // order)
    hyperedges = []
    for _ in range(n_hyperedges):
        members = tuple(sorted(rng.choice(n, size=order, replace=False)))
        hyperedges.append(members)
    return hyperedges


def build_adjacency(n: int, edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ── Game mechanics ───────────────────────────────────────────────────

def play_dyadic_pgg(agents: list[PGGAgent], group_indices: list[int]) -> None:
    """Standard 2-player PGG round for a pair."""
    group = [agents[i] for i in group_indices]
    for a in group:
        a.contribution = a.decide_contribution()

    pool = sum(a.contribution for a in group)
    shared = (pool * MULTIPLIER) / len(group)

    for a in group:
        a.payoff += shared - a.contribution


def play_hyperedge_pgg_with_deliberation(agents: list[PGGAgent], group_indices: list[int]) -> None:
    """
    Higher-order PGG with synchronous group deliberation.

    Mechanism (models Iacopini higher-order contagion):
    1. Each agent proposes its individual contribution
    2. Group observes all proposals (complete information)
    3. If majority cooperates, ALL members contribute (group norm enforcement)
    4. Otherwise, all members defect
    """
    group = [agents[i] for i in group_indices]
    proposals = [a.is_cooperator for a in group]
    cooperator_count = sum(proposals)
    majority_cooperates = cooperator_count > len(group) / 2

    if majority_cooperates:
        for a in group:
            a.contribution = ENDOWMENT
    else:
        for a in group:
            a.contribution = 0.0

    pool = sum(a.contribution for a in group)
    shared = (pool * MULTIPLIER) / len(group)

    for a in group:
        a.payoff += shared - a.contribution


def play_reciprocal_pgg(
    agents: list[PGGAgent],
    i: int,
    j: int,
    rng: np.random.Generator,
) -> bool:
    """
    Dyadic PGG with reciprocity requirement.
    Both must agree to interact; if either refuses, no interaction occurs.
    Returns True if interaction happened.
    """
    a, b = agents[i], agents[j]
    a_willing = a.is_cooperator or rng.random() < 0.3
    b_willing = b.is_cooperator or rng.random() < 0.3

    if a_willing and b_willing:
        play_dyadic_pgg(agents, [i, j])
        return True
    return False


# ── Strategy update ──────────────────────────────────────────────────

def fermi_update(agents: list[PGGAgent], adjacency: dict[int, set[int]],
                 rng: np.random.Generator) -> None:
    """
    Fermi pairwise comparison: agent i copies strategy of random neighbor j
    with probability 1 / (1 + exp(-beta * (payoff_j - payoff_i))).
    """
    n = len(agents)
    order = rng.permutation(n)
    for i in order:
        neighbors = list(adjacency.get(i, set()))
        if not neighbors:
            continue
        j = rng.choice(neighbors)
        delta = agents[j].payoff - agents[i].payoff
        prob = 1.0 / (1.0 + np.exp(-FERMI_BETA * delta / ENDOWMENT))
        if rng.random() < prob:
            agents[i].is_cooperator = agents[j].is_cooperator

    # small mutation rate to prevent absorbing states
    for i in range(n):
        if rng.random() < 0.01:
            agents[i].is_cooperator = not agents[i].is_cooperator


def norm_update_from_hyperedge(agents: list[PGGAgent], group_indices: list[int],
                               rng: np.random.Generator = None) -> None:
    """
    After a hyperedge interaction, higher-order contagion applies:
    - If ALL other members are adopters -> guaranteed adoption (strong reinforcement)
    - If majority are adopters -> probabilistic adoption (p=0.5 per non-adopter)
    - Otherwise -> no change
    This implements the Iacopini-style "all-or-nothing" higher-order mechanism.
    """
    group = [agents[i] for i in group_indices]
    n_adopted = sum(1 for a in group if a.norm_adopted)
    n_total = len(group)

    for a in group:
        if a.norm_adopted:
            continue
        others_adopted = n_adopted  # among the whole group (a is not adopted)
        others_total = n_total - 1
        if others_adopted == others_total:
            # all others adopted -> strong higher-order pressure
            a.norm_adopted = True
        elif others_adopted > others_total / 2:
            # majority adopted -> probabilistic
            if rng is not None and rng.random() < 0.3:
                a.norm_adopted = True


# ── Main simulation ──────────────────────────────────────────────────

@dataclass
class SimulationResult:
    condition: str
    n_agents: int
    n_rounds: int
    seed: int
    cooperation_rate: np.ndarray      # c(t) per round
    norm_adoption_rate: np.ndarray    # rho(t) per round
    mean_payoff: np.ndarray           # per round
    final_cooperation: float
    final_norm_adoption: float
    phase_transition_round: Optional[int] = None
    critical_mass: Optional[float] = None


def run_simulation(
    condition: Condition,
    n_agents: int = 100,
    n_rounds: int = 500,
    seed_fraction: float = 0.05,
    seed: int = 42,
    avg_degree: int = 6,
    avg_membership: int = 4,
) -> SimulationResult:
    """
    Run one PGG simulation under specified topological condition.
    """
    rng = np.random.default_rng(seed)

    agents = [PGGAgent(uid=i) for i in range(n_agents)]

    # seed initial cooperators / norm adopters
    n_seeds = max(1, int(n_agents * seed_fraction))
    seed_ids = rng.choice(n_agents, size=n_seeds, replace=False)
    for idx in seed_ids:
        agents[idx].is_cooperator = True
        agents[idx].norm_adopted = True

    # build topology
    if condition in (Condition.A, Condition.B):
        edges = build_dyadic_graph(n_agents, avg_degree, rng)
        adjacency = build_adjacency(n_agents, edges)
        hyperedges = []
    elif condition == Condition.C:
        hyperedges = build_hyperedges(n_agents, order=3, avg_membership=avg_membership, rng=rng)
        edges = []
        for he in hyperedges:
            for k in range(len(he)):
                for l in range(k + 1, len(he)):
                    edges.append((he[k], he[l]))
        adjacency = build_adjacency(n_agents, edges)
    elif condition == Condition.D:
        hyperedges = build_hyperedges(n_agents, order=5, avg_membership=avg_membership, rng=rng)
        edges = []
        for he in hyperedges:
            for k in range(len(he)):
                for l in range(k + 1, len(he)):
                    edges.append((he[k], he[l]))
        adjacency = build_adjacency(n_agents, edges)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    cooperation_rate = np.zeros(n_rounds)
    norm_adoption_rate = np.zeros(n_rounds)
    mean_payoff = np.zeros(n_rounds)

    for t in range(n_rounds):
        # reset round payoffs
        for a in agents:
            a.payoff = 0.0

        if condition == Condition.A:
            # random pairwise interactions
            pairs = list(adjacency.keys())
            rng.shuffle(pairs)
            visited = set()
            for i in pairs:
                if i in visited:
                    continue
                neighbors = list(adjacency[i] - visited)
                if not neighbors:
                    continue
                j = rng.choice(neighbors)
                play_dyadic_pgg(agents, [i, j])
                visited.add(i)
                visited.add(j)

        elif condition == Condition.B:
            pairs = list(adjacency.keys())
            rng.shuffle(pairs)
            visited = set()
            for i in pairs:
                if i in visited:
                    continue
                neighbors = list(adjacency[i] - visited)
                if not neighbors:
                    continue
                j = rng.choice(neighbors)
                play_reciprocal_pgg(agents, i, j, rng)
                visited.add(i)
                visited.add(j)

        elif condition in (Condition.C, Condition.D):
            # each hyperedge plays the group PGG with deliberation
            he_order = list(range(len(hyperedges)))
            rng.shuffle(he_order)
            for he_idx in he_order:
                he = hyperedges[he_idx]
                play_hyperedge_pgg_with_deliberation(agents, list(he))
                norm_update_from_hyperedge(agents, list(he), rng=rng)

        # strategy update via social learning
        fermi_update(agents, adjacency, rng)

        # norm propagation for dyadic conditions (simple contagion, slower)
        if condition in (Condition.A, Condition.B):
            for i in range(n_agents):
                if not agents[i].norm_adopted:
                    for nb in adjacency[i]:
                        if agents[nb].norm_adopted and rng.random() < 0.05:
                            agents[i].norm_adopted = True
                            break

        # norm decay (SIS recovery) — agents can lose the norm
        for i in range(n_agents):
            if agents[i].norm_adopted and rng.random() < 0.02:
                agents[i].norm_adopted = False

        # record
        for a in agents:
            a.record()

        cooperation_rate[t] = sum(a.is_cooperator for a in agents) / n_agents
        norm_adoption_rate[t] = sum(a.norm_adopted for a in agents) / n_agents
        mean_payoff[t] = np.mean([a.payoff for a in agents])

    # detect phase transition: steepest increase in cooperation rate
    if n_rounds > 10:
        smoothed = np.convolve(cooperation_rate, np.ones(10) / 10, mode="valid")
        dc = np.diff(smoothed)
        pt_round = int(np.argmax(dc) + 5) if dc.max() > 0.02 else None
    else:
        pt_round = None

    return SimulationResult(
        condition=condition.value,
        n_agents=n_agents,
        n_rounds=n_rounds,
        seed=seed,
        cooperation_rate=cooperation_rate,
        norm_adoption_rate=norm_adoption_rate,
        mean_payoff=mean_payoff,
        final_cooperation=float(cooperation_rate[-1]),
        final_norm_adoption=float(norm_adoption_rate[-1]),
        phase_transition_round=pt_round,
        critical_mass=seed_fraction,
    )
