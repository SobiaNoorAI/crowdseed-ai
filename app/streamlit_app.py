"""
streamlit_app.py — CrowdSeed AI demo.

Lets a judge/visitor run the Social vs Independent multi-agent
experiment live (or load a pre-computed results file if API calls are
too slow for a smooth demo) and watch the paper's core finding play
out with AI agents instead of humans: does the socially-connected
crowd of agents out-design the isolated one?
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from sim.grid import render_grid
from agents import experiment

st.set_page_config(page_title="CrowdSeed AI", layout="wide")

st.title("🐜 CrowdSeed AI")
st.caption(
    "Replacing a human crowd with an AI-agent crowd —  "
    "social visibility improves collaborative robot design."
)

with st.expander("About this experiment", expanded=False):
    st.markdown(
        """
        In a crowd of humans designed robot bodies on a
        5x5 grid. One group (**Social**) could see other users' designs while
        working; a control group (**Independent**) could only see their own past
        designs. The social group's robots traveled significantly farther, and a
        latent "design factor" (derived via SVD from geometric features, dominated
        by **symmetry**) was more prevalent in their designs.

        **CrowdSeed AI** reproduces this exact experiment, but with LLM agents
        standing in for the human crowd. Same design space (5x5 grid), same
        physics-based scoring, same hill-climbing controller optimization, same
        SVD latent-factor analysis — the only thing that changes is *who* is
        designing.
        """
    )

st.sidebar.header("Experiment settings")
n_agents = st.sidebar.slider("Agents per condition", 2, 8, 4)
n_generations = st.sidebar.slider("Generations", 2, 15, 6)
sim_duration = st.sidebar.slider("Simulated seconds per trial", 3.0, 15.0, 6.0)
hillclimb_iters = st.sidebar.slider("Hill-climb iterations per design", 2, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "⚠️ Live runs make real LLM API calls per agent per generation. "
    "For a smoother demo, use a smaller agent/generation count, or load "
    "pre-computed results below."
)

run_live = st.sidebar.button("▶ Run experiment live")
load_saved = st.sidebar.button("📂 Load saved results (data/results.csv)")


def render_results(df: pd.DataFrame):
    left, right = st.columns(2)

    with left:
        st.subheader("Best distance per generation")
        best_by_gen = (
            df.groupby(["condition", "generation"])["distance"].max().reset_index()
        )
        pivot = best_by_gen.pivot(index="generation", columns="condition", values="distance")
        st.line_chart(pivot)

    with right:
        st.subheader("Mean latent factor per generation")
        if "latent_factor" in df.columns:
            lf_by_gen = (
                df.groupby(["condition", "generation"])["latent_factor"].mean().reset_index()
            )
            pivot_lf = lf_by_gen.pivot(index="generation", columns="condition", values="latent_factor")
            st.line_chart(pivot_lf)
        else:
            st.info("Latent factor not available for this dataset.")

    st.subheader("Top designs found")
    top_n = df.sort_values("distance", ascending=False).head(6)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_n.iterrows()):
        with cols[i % 3]:
            png = render_grid(row["edges"], title=f"{row['condition']} · d={row['distance']:.1f}")
            st.image(png)
            if "symmetry" in row:
                st.caption(f"symmetry={row['symmetry']:.2f}, legs={int(row.get('number_of_legs', 0))}")

    st.subheader("Summary statistics")
    summary = df.groupby("condition")["distance"].agg(["count", "mean", "max"])
    st.dataframe(summary)

    if "latent_factor" in df.columns:
        social_lf = df[df.condition == "social"]["latent_factor"]
        indep_lf = df[df.condition == "independent"]["latent_factor"]
        try:
            from scipy import stats

            ks_dist = stats.ks_2samp(
                df[df.condition == "social"]["distance"],
                df[df.condition == "independent"]["distance"],
            )
            ks_latent = stats.ks_2samp(social_lf, indep_lf)
            st.markdown(
                f"**Distance, Social vs Independent** — KS test: "
                f"D={ks_dist.statistic:.3f}, p={ks_dist.pvalue:.4f}"
            )
            st.markdown(
                f"**Latent factor, Social vs Independent** — KS test: "
                f"D={ks_latent.statistic:.3f}, p={ks_latent.pvalue:.4f}"
            )
        except ImportError:
            pass


if run_live:
    status = st.empty()
    progress = st.progress(0.0)
    total_steps = n_generations * 2  # two conditions

    def on_generation(condition, generation, partial_df):
        step = generation + 1 + (0 if condition == "social" else n_generations)
        progress.progress(min(step / total_steps, 1.0))
        status.text(f"Running {condition} condition — generation {generation + 1}/{n_generations}")

    t0 = time.time()
    df = experiment.run_full_experiment(
        n_agents=n_agents,
        n_generations=n_generations,
        sim_duration=sim_duration,
        hillclimb_iterations=hillclimb_iters,
        on_generation=on_generation,
    )
    status.text(f"Done in {time.time() - t0:.1f}s")
    os.makedirs("data", exist_ok=True)
    experiment.save_results(df, "data/results.csv")
    st.session_state["results_df"] = df

elif load_saved:
    try:
        st.session_state["results_df"] = pd.read_csv("data/results.csv")
        st.success("Loaded saved results.")
    except FileNotFoundError:
        st.error("No saved results found at data/results.csv yet — run the experiment live first.")

if "results_df" in st.session_state:
    df = st.session_state["results_df"]
    if "edges" in df.columns and isinstance(df["edges"].iloc[0], str):
        import ast
        df["edges"] = df["edges"].apply(ast.literal_eval)
    render_results(df)
else:
    st.info("Choose experiment settings in the sidebar, then click **Run experiment live** to begin.")