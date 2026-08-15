"""
Fit theoretical contagion model parameters from ABM / LLM experiment data.

Strategy:
  1. Load experimental cooperation trajectories c(t) from ABM/LLM results
  2. Map cooperation rate to norm adoption: rho(t) ~ c(t) (monotonic transform)
  3. Fit LLMContagionModel parameters (beta1, beta2, mu, lam, C) via least-squares
  4. Validate: fit on Condition A data, predict Condition C/D transitions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.models.contagion import LLMContagionModel

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def load_abm_trajectories() -> dict:
    """Load average cooperation trajectories from ABM results."""
    path = ROOT / "results" / "abm" / "abm_summary.json"
    if not path.exists():
        logger.error("ABM summary not found")
        return {}

    data = json.loads(path.read_text())
    result = {}
    cond_map = {
        "dyadic_baseline": "A",
        "dyadic_reciprocity": "B",
        "triad_hyperedge": "C",
        "pentad_hyperedge": "D",
    }
    for abm_name, short in cond_map.items():
        if abm_name in data:
            result[short] = np.array(data[abm_name]["avg_cooperation_trajectory"])
    return result


def load_llm_trajectories() -> dict:
    """Load average cooperation trajectories from LLM results."""
    path = ROOT / "results" / "llm_experiment" / "llm_summary.json"
    if not path.exists():
        logger.warning("LLM summary not found")
        return {}

    data = json.loads(path.read_text())
    result = {}
    for cond in ["A", "B", "C", "D"]:
        if cond in data:
            result[cond] = np.array(data[cond]["avg_cooperation_trajectory"])
    return result


def fit_to_trajectory(
    target: np.ndarray,
    condition: str = "A",
    T: Optional[float] = None,
    n_restarts: int = 5,
) -> dict:
    """
    Fit LLMContagionModel to a target rho(t) trajectory.

    For Condition A: beta2=0 (dyadic only), fit beta1, mu
    For Condition B: beta2=0, reduced beta1, fit beta1, mu
    For Condition C/D: fit beta1, beta2, mu, lam, C
    """
    n_steps = len(target)
    if T is None:
        T = float(n_steps)
    rho0 = target[0] if target[0] > 0.001 else 0.05

    target_interp = target.copy()

    def objective(params):
        try:
            if condition in ("A", "B"):
                beta1, mu = params
                model = LLMContagionModel(beta1=beta1, beta2=0.0, mu=mu, lam=0, C=1)
            else:
                beta1, beta2, mu, lam, C = params
                model = LLMContagionModel(beta1=beta1, beta2=beta2, mu=mu, lam=lam, C=C)

            t_sim, rho_sim = model.simulate(T=T, rho0=rho0, dt=T / n_steps)

            # interpolate to match target length
            if len(rho_sim) != n_steps:
                rho_interp = np.interp(
                    np.linspace(0, 1, n_steps),
                    np.linspace(0, 1, len(rho_sim)),
                    rho_sim,
                )
            else:
                rho_interp = rho_sim

            return float(np.mean((rho_interp - target_interp) ** 2))
        except Exception:
            return 1e6

    best_result = None
    best_loss = float("inf")
    rng = np.random.default_rng(42)

    for restart in range(n_restarts):
        if condition in ("A", "B"):
            x0 = [rng.uniform(0.05, 0.5), rng.uniform(0.05, 0.3)]
            bounds = [(0.01, 1.0), (0.01, 0.5)]
        else:
            x0 = [
                rng.uniform(0.05, 0.5),
                rng.uniform(0.5, 3.0),
                rng.uniform(0.05, 0.3),
                rng.uniform(0.5, 4.0),
                rng.uniform(2.0, 32.0),
            ]
            bounds = [
                (0.01, 1.0),   # beta1
                (0.1, 10.0),   # beta2
                (0.01, 0.5),   # mu
                (0.1, 10.0),   # lam
                (1.0, 128.0),  # C
            ]

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500})

        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    if condition in ("A", "B"):
        beta1, mu = best_result.x
        fitted = {"beta1": beta1, "beta2": 0.0, "mu": mu, "lam": 0.0, "C": 1.0}
    else:
        beta1, beta2, mu, lam, C = best_result.x
        fitted = {"beta1": beta1, "beta2": beta2, "mu": mu, "lam": lam, "C": C}

    fitted["loss"] = float(best_loss)
    fitted["condition"] = condition

    logger.info("Fit %s: loss=%.6f params=%s", condition,
                best_loss, {k: f"{v:.4f}" for k, v in fitted.items() if k not in ("condition",)})
    return fitted


def cross_validate_prediction(
    fit_params: dict,
    target_trajectories: dict[str, np.ndarray],
) -> dict:
    """
    Validate: fit on one condition, predict others.
    Key test: fit on A -> predict C phase transition point.
    """
    results = {}

    for test_cond, target in target_trajectories.items():
        T = float(len(target))
        rho0 = target[0] if target[0] > 0.001 else 0.05

        # use fitted params but adjust beta2 based on condition
        params = fit_params.copy()
        if test_cond in ("A", "B"):
            model = LLMContagionModel(beta1=params["beta1"], beta2=0.0,
                                       mu=params["mu"], lam=0, C=1)
        else:
            model = LLMContagionModel(**{k: params[k] for k in ["beta1", "beta2", "mu", "lam", "C"]})

        t_sim, rho_sim = model.simulate(T=T, rho0=rho0, dt=1.0)

        if len(rho_sim) != len(target):
            rho_sim = np.interp(
                np.linspace(0, 1, len(target)),
                np.linspace(0, 1, len(rho_sim)),
                rho_sim,
            )

        mse = float(np.mean((rho_sim - target) ** 2))
        results[test_cond] = {
            "mse": mse,
            "predicted_final": float(rho_sim[-1]),
            "actual_final": float(target[-1]),
        }

        logger.info("  Predict %s: MSE=%.6f (pred=%.3f, actual=%.3f)",
                    test_cond, mse, rho_sim[-1], target[-1])

    return results


def run_fitting(source: str = "abm") -> dict:
    """Run full parameter fitting pipeline."""
    if source == "abm":
        trajs = load_abm_trajectories()
    else:
        trajs = load_llm_trajectories()

    if not trajs:
        logger.error("No trajectories to fit")
        return {}

    results = {"source": source, "fits": {}, "predictions": {}}

    # fit each condition
    for cond, traj in trajs.items():
        fitted = fit_to_trajectory(traj, condition=cond, n_restarts=5)
        results["fits"][cond] = fitted

    # cross-validation: fit on C -> predict others
    if "C" in results["fits"]:
        logger.info("\nCross-validation: predict from Condition C fit")
        predictions = cross_validate_prediction(results["fits"]["C"], trajs)
        results["predictions"]["from_C"] = predictions

    # save
    outdir = ROOT / "results" / "model_fitting"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"fitting_{source}.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    logger.info("Results saved to %s", outdir)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Parameter Fitting ===")

    # try ABM first
    abm_results = run_fitting("abm")

    # then LLM if available
    llm_results = run_fitting("llm")

    logger.info("=== Done ===")
