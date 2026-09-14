# crowdseed-ai# CrowdSeed AI

**Multi-agent evolutionary robot design, with LLM agents standing in for a human crowd.**

Built for the [AI Infra Summit Hackathon](https://lablab.ai) (online track), Sep 2026.

## What this is

A reproduction of the core experiment from:

> A research paper

In the original paper, human participants designed 2D robot bodies on a 5x5
grid of dots. One group (**Social**) could see other users' designs while
working; a control group (**Independent**) could only see their own past
attempts. The social group's robots traveled significantly farther, and a
latent "design factor" — distilled via SVD from the robots' geometric
features, and dominated by **symmetry** — was measurably more common in
their designs.

**CrowdSeed AI reruns this exact experiment with LLM agents in place of the
human crowd**, using the same design space, the same hill-climbing
controller-optimization method, and the same SVD-based latent-factor
analysis, to see whether the same social synergy emerges.

## How it works

```
sim/
  grid.py       5x5 dot-grid design representation, validity checks, rendering
  physics.py    pymunk-based physics scoring + hill-climbing controller search
  features.py   geometric feature extraction + SVD latent-factor computation
agents/
  llm_agent.py  LLM-backed design proposer/mutator (with a non-LLM fallback)
  experiment.py Social vs Independent experiment loop
app/
  streamlit_app.py   interactive demo
```

- Robots are represented as edge lists on a 5x5 grid, exactly as in the paper.
- Distance is scored with a lightweight **2D pymunk physics simulation**
  (a deliberate simplification of the paper's 3D web-embedded engine, chosen
  to keep the pipeline fast and dependency-light for a hackathon timeline).
- Each design's controller (joint phase configuration) is optimized with the
  same hill-climbing method described in the paper.
- Geometric features (degree stats, symmetry, number of legs, etc.) are
  computed per design and reduced to a single latent factor via SVD on the
  combined Social + Independent dataset, so both conditions are compared on
  the same axis.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # required for live LLM agent runs
streamlit run app/streamlit_app.py
```

Open the printed local URL, choose your agent/generation settings in the
sidebar, and click **Run experiment live**. Results are also saved to
`data/results.csv` so they can be reloaded without re-running the agents.

## Notes on scope

This was built in a 3-4 day hackathon window. Deliberate simplifications
worth knowing about:

- **2D physics, not 3D.** pymunk instead of the paper's WebGL/3D engine —
  much faster to get right under time pressure, and sufficient to test
  whether a design can locomote at all.
- **Small agent/generation counts by default**, to keep live demo runs fast
  and LLM API costs low. The pipeline itself has no hardcoded limit — turn
  the sliders up for a more statistically convincing (but slower) run.
- **A non-LLM fallback mutator** kicks in automatically if an LLM response
  is invalid or the API call fails, so a single bad response never stalls
  the whole experiment.

## Team

Built by [Your Name], Ahmed, and Syed for the AI Infra Summit Hackathon.