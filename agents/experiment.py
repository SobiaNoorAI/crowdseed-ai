"""
experiment.py — runs the Social vs Independent multi-agent experiment.

This is the direct AI-agent analogue of the core manipulation:
  - SOCIAL agents can see the top designs found by the whole group so
    far each generation (paper: a panel of 13 other users' designs).
  - INDEPENDENT agents only ever see their own design history (paper:
    the control group saw only their own past designs).

Everything else (physics scoring, hill-climbing the controller,
feature/latent-factor extraction) is identical between conditions --
visibility into the group is the only manipulated variable, exactly
as in the paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

from sim.physics import hill_climb
from sim.features import compute_features, features_to_vector
from agents.llm_agent import AgentMemory, propose_design

Condition = Literal["social", "independent"]

# How many of the group's current-best designs a social agent gets to
# see each generation. The paper used 13; we default lower to keep
# prompts small and runs fast, but it's tunable.
DEFAULT_VISIBLE_DESIGNS = 5

# Physics/hill-climb settings tuned for demo speed rather than the
# paper's full fidelity -- see README for the tradeoff.
DEFAULT_SIM_DURATION = 6.0
DEFAULT_HILLCLIMB_ITERATIONS = 5


@dataclass
class GenerationLog:
    condition: Condition
    generation: int
    agent_id: str
    edges: list
    distance: float
    features: dict
    latent_factor: float | None = None  # filled in after the full run, once SVD is computed


def _top_designs(pool: list[dict], n: int) -> list[dict]:
    return sorted(pool, key=lambda d: d["distance"], reverse=True)[:n]


def run_condition(
    condition: Condition,
    n_agents: int,
    n_generations: int,
    visible_designs: int = DEFAULT_VISIBLE_DESIGNS,
    sim_duration: float = DEFAULT_SIM_DURATION,
    hillclimb_iterations: int = DEFAULT_HILLCLIMB_ITERATIONS,
    on_generation: Callable[[int, list[GenerationLog]], None] | None = None,
) -> pd.DataFrame:
    """Run one condition end to end. `on_generation` is an optional
    callback invoked after each generation with (generation_index,
    logs_so_far) -- handy for Streamlit's live-updating charts."""
    agents = [AgentMemory(agent_id=f"{condition}-{i}") for i in range(n_agents)]
    group_pool: list[dict] = []  # every design+distance found so far in this condition
    all_logs: list[GenerationLog] = []

    for generation in range(n_generations):
        visible = _top_designs(group_pool, visible_designs) if condition == "social" else None

        for agent in agents:
            edges = propose_design(agent, visible_designs=visible)

            best_distance, best_phase, _ = hill_climb(
                edges, n_iterations=hillclimb_iterations, duration=sim_duration
            )
            features = compute_features(edges)

            agent.history.append({"edges": edges, "distance": best_distance})
            group_pool.append({"edges": edges, "distance": best_distance})

            all_logs.append(
                GenerationLog(
                    condition=condition,
                    generation=generation,
                    agent_id=agent.agent_id,
                    edges=edges,
                    distance=best_distance,
                    features=features,
                )
            )

        if on_generation:
            on_generation(generation, all_logs)

    return _logs_to_dataframe(all_logs)


def _logs_to_dataframe(logs: list[GenerationLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        row = {
            "condition": log.condition,
            "generation": log.generation,
            "agent_id": log.agent_id,
            "distance": log.distance,
            "edges": log.edges,
        }
        row.update(log.features)
        rows.append(row)
    return pd.DataFrame(rows)


def run_full_experiment(
    n_agents: int = 4,
    n_generations: int = 6,
    visible_designs: int = DEFAULT_VISIBLE_DESIGNS,
    sim_duration: float = DEFAULT_SIM_DURATION,
    hillclimb_iterations: int = DEFAULT_HILLCLIMB_ITERATIONS,
    on_generation: Callable[[str, int, pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Run both conditions with matched settings, then attach a
    latent-factor column computed via SVD over the *combined* dataset
    (so both conditions are projected onto the same axis, making the
    comparison meaningful -- matches the paper's approach of deriving
    one shared latent factor for both groups)."""
    from sim.features import FEATURE_NAMES, compute_latent_factors

    def wrap(cond):
        def callback(gen, logs):
            if on_generation:
                on_generation(cond, gen, _logs_to_dataframe(logs))
        return callback

    social_df = run_condition(
        "social", n_agents, n_generations, visible_designs, sim_duration,
        hillclimb_iterations, on_generation=wrap("social"),
    )
    independent_df = run_condition(
        "independent", n_agents, n_generations, visible_designs, sim_duration,
        hillclimb_iterations, on_generation=wrap("independent"),
    )

    combined = pd.concat([social_df, independent_df], ignore_index=True)
    feature_matrix = combined[FEATURE_NAMES].to_numpy(dtype=float)
    latent_values, weights = compute_latent_factors(feature_matrix)
    combined["latent_factor"] = latent_values
    combined.attrs["latent_factor_weights"] = dict(zip(FEATURE_NAMES, weights))

    return combined


def save_results(df: pd.DataFrame, path: str = "data/results.csv") -> None:
    df.to_csv(path, index=False)