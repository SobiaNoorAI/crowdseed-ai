"""
smoke_test.py — quick end-to-end sanity check that doesn't require an
ANTHROPIC_API_KEY. Runs the full Social vs Independent pipeline using
the non-LLM fallback mutator in place of real agent calls, so you can
verify the sim/physics/features/experiment code all work together
before spending API budget on real runs.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.experiment as experiment
from agents.llm_agent import _fallback_mutation


def fake_propose_design(memory, visible_designs=None):
    return _fallback_mutation(memory)


def main():
    experiment.propose_design = fake_propose_design

    print("Running smoke test (fallback agent, no API calls)...")
    t0 = time.time()
    df = experiment.run_full_experiment(
        n_agents=3, n_generations=4, sim_duration=3.0, hillclimb_iterations=3
    )
    elapsed = time.time() - t0

    print(f"\nCompleted {len(df)} designs across both conditions in {elapsed:.1f}s\n")
    print(df.groupby("condition")["distance"].agg(["count", "mean", "max"]))

    os.makedirs("data", exist_ok=True)
    experiment.save_results(df, "data/smoke_test_results.csv")
    print("\nSaved to data/smoke_test_results.csv")
    print("\nIf this ran without errors, the core pipeline is healthy.")
    print("Set ANTHROPIC_API_KEY and use the Streamlit app for real agent runs.")


if __name__ == "__main__":
    main()