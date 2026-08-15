"""
Microscopic Monte Carlo SIS contagion on explicit hypergraph structures.

Unlike the mean-field ODE in contagion.py, this simulates individual
agent states on the actual hypergraph topology, capturing:
  - Local heterogeneity (hubs vs periphery)
  - Stochastic fluctuations
  - Actual higher-order group interactions (not approximated)

Implements both standard SIS and Iacopini-style simplicial contagion
with LLM-specific extensions (attention decay, framing effects).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MicroContagionModel:
    """
    Monte Carlo SIS contagion on an explicit hypergraph.

    Parameters:
        beta1: per-contact pairwise infection rate
        beta2: per-group higher-order infection rate (order >= 3)
        mu: recovery rate per timestep
        lam: attention decay parameter (LLM-specific)
        C: effective context capacity
        g: framing effect multiplier
    """
    beta1: float = 0.3
    beta2: float = 1.5
    mu: float = 0.1
    lam: float = 2.0
    C: float = 8.0
    g: float = 1.0

    @property
    def attention_factor(self) -> float:
        return np.exp(-self.lam / self.C)

    @property
    def effective_beta2(self) -> float:
        return self.beta2 * self.attention_factor * self.g

    def simulate(
        self,
        n_nodes: int,
        pairwise_edges: list[tuple[int, int]],
        hyperedges: list[tuple[int, ...]],
        T: int = 500,
        rho0: float = 0.05,
        seed: int = 42,
        dt: float = 1.0,
    ) -> dict:
        """
        Run Monte Carlo simulation on explicit topology.

        Returns dict with time series of rho(t), individual states, etc.
        """
        rng = np.random.default_rng(seed)

        # initialize states: 0 = susceptible, 1 = infected
        states = np.zeros(n_nodes, dtype=int)
        n_initial = max(1, int(n_nodes * rho0))
        initial_infected = rng.choice(n_nodes, size=n_initial, replace=False)
        states[initial_infected] = 1

        # precompute adjacency
        adj = {i: set() for i in range(n_nodes)}
        for u, v in pairwise_edges:
            adj[u].add(v)
            adj[v].add(u)

        # node to hyperedges mapping
        node_to_he = {i: [] for i in range(n_nodes)}
        for idx, he in enumerate(hyperedges):
            for node in he:
                if node < n_nodes:
                    node_to_he[node].append(idx)

        n_steps = int(T / dt)
        rho_t = np.zeros(n_steps)
        states_history = []

        for step in range(n_steps):
            new_states = states.copy()

            for i in range(n_nodes):
                if states[i] == 1:
                    # recovery
                    if rng.random() < self.mu * dt:
                        new_states[i] = 0
                else:
                    # pairwise infection
                    infected_neighbors = sum(1 for nb in adj[i] if states[nb] == 1)
                    p_pairwise = 1.0 - (1.0 - self.beta1 * dt) ** infected_neighbors

                    # higher-order infection
                    p_higher = 0.0
                    for he_idx in node_to_he[i]:
                        he = hyperedges[he_idx]
                        other_members = [m for m in he if m != i and m < n_nodes]
                        if not other_members:
                            continue
                        all_infected = all(states[m] == 1 for m in other_members)
                        if all_infected:
                            p_higher = 1.0 - (1.0 - p_higher) * (1.0 - self.effective_beta2 * dt)

                    # combined infection probability
                    p_total = 1.0 - (1.0 - p_pairwise) * (1.0 - p_higher)
                    if rng.random() < p_total:
                        new_states[i] = 1

            states = new_states
            rho_t[step] = states.sum() / n_nodes

            if step % 50 == 0:
                states_history.append(states.copy())

        return {
            "rho_t": rho_t,
            "times": np.arange(n_steps) * dt,
            "final_rho": float(rho_t[-1]),
            "states_history": states_history,
            "n_nodes": n_nodes,
            "n_pairwise": len(pairwise_edges),
            "n_hyperedges": len(hyperedges),
        }


def simulate_on_condition(
    condition: str,
    n_nodes: int = 200,
    avg_degree: int = 6,
    model_params: Optional[dict] = None,
    T: int = 500,
    rho0: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Convenience function: build topology for a given condition and simulate.

    Conditions:
        'A': pure pairwise, no hyperedges (standard SIS)
        'B': pairwise with reduced beta1 (reciprocity filter)
        'C': pairwise + order-3 hyperedges
        'D': pairwise + order-5 hyperedges
    """
    rng = np.random.default_rng(seed)

    # build pairwise graph
    p = avg_degree / (n_nodes - 1)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                edges.append((i, j))

    # build hyperedges based on condition
    hyperedges = []
    if condition in ("C", "D"):
        order = 3 if condition == "C" else 5
        n_he = (n_nodes * 4) // order
        for _ in range(n_he):
            members = tuple(sorted(rng.choice(n_nodes, size=order, replace=False)))
            hyperedges.append(members)
        # add projected pairwise edges from hyperedges
        for he in hyperedges:
            for k in range(len(he)):
                for l in range(k + 1, len(he)):
                    edges.append((he[k], he[l]))

    params = model_params or {}
    if condition == "A":
        params.setdefault("beta2", 0.0)
    elif condition == "B":
        params.setdefault("beta1", 0.15)  # reciprocity halves effective rate
        params.setdefault("beta2", 0.0)

    model = MicroContagionModel(**params)

    result = model.simulate(
        n_nodes=n_nodes,
        pairwise_edges=edges,
        hyperedges=hyperedges,
        T=T,
        rho0=rho0,
        seed=seed,
    )
    result["condition"] = condition
    return result


def run_micro_study(
    n_nodes: int = 200,
    T: int = 500,
    n_repeats: int = 20,
    rho0_values: list[float] | None = None,
) -> dict:
    """
    Full micro-simulation study across conditions and initial densities.
    """
    if rho0_values is None:
        rho0_values = [0.03, 0.05, 0.10, 0.15, 0.25]

    results = {}
    for cond in ["A", "B", "C", "D"]:
        results[cond] = {}
        for rho0 in rho0_values:
            trajectories = []
            for rep in range(n_repeats):
                r = simulate_on_condition(
                    condition=cond,
                    n_nodes=n_nodes,
                    T=T,
                    rho0=rho0,
                    seed=1000 * ord(cond) + int(rho0 * 1000) + rep,
                )
                trajectories.append(r["rho_t"])

            arr = np.array(trajectories)
            results[cond][rho0] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "final_mean": float(arr[:, -1].mean()),
                "final_std": float(arr[:, -1].std()),
            }
            logger.info("  Condition %s, rho0=%.2f: final=%.3f±%.3f",
                        cond, rho0, arr[:, -1].mean(), arr[:, -1].std())

    return results


if __name__ == "__main__":
    import json
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ROOT = Path(__file__).resolve().parents[2]
    outdir = ROOT / "results" / "micro_contagion"
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Microscopic Contagion Study ===")

    results = run_micro_study(n_nodes=200, T=500, n_repeats=20)

    # save results (without numpy arrays)
    (outdir / "micro_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    logger.info("Results saved to %s", outdir / "micro_results.json")

    # generate comparison figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len([0.05, 0.15]), figsize=(14, 6), sharey=True)
    colors = {"A": "#E24A33", "B": "#FFA500", "C": "#348ABD", "D": "#2ca02c"}
    labels = {"A": "Cond A (Dyadic)", "B": "Cond B (Reciprocity)",
              "C": "Cond C (Triad)", "D": "Cond D (Pentad)"}

    for ax, rho0 in zip(axes, [0.05, 0.15]):
        for cond in ["A", "B", "C", "D"]:
            if rho0 in results[cond]:
                mean = np.array(results[cond][rho0]["mean"])
                std = np.array(results[cond][rho0]["std"])
                t = np.arange(len(mean))
                ax.plot(t, mean, color=colors[cond], label=labels[cond], linewidth=2)
                ax.fill_between(t, mean - std, mean + std, color=colors[cond], alpha=0.15)

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$\\rho_0 = {rho0}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Infection density $\\rho(t)$")
    axes[0].legend(fontsize=9)
    fig.suptitle("Microscopic Monte Carlo Contagion on Hypergraphs", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(outdir / "micro_contagion_comparison.png"), dpi=300, bbox_inches="tight")
    logger.info("Figure saved.")

    logger.info("=== Done ===")
