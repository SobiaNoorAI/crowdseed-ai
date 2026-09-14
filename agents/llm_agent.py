"""
llm_agent.py — an LLM-backed "designer" agent that proposes or mutates
5x5 grid robot designs, replacing the human participants from Wagy &
Bongard (2016).

Each agent call gets:
  - its own design history (what it tried before, and how far it went)
  - in the SOCIAL condition only: a handful of the group's current
    best designs (mirrors the paper's panel of 13 other users' designs)

The agent is asked to propose ONE new design, expressed as an edit to
an existing design (add/remove a single edge) so the model is doing
small, tractable design steps rather than inventing geometry from
scratch every call — this keeps designs valid more often and keeps
API costs/latency low.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from sim.grid import Edge, is_valid_design, clean_design, GRID_SIZE, T_SHAPE

MODEL_NAME = "claude-sonnet-4-6"
MAX_RETRIES = 1


SYSTEM_PROMPT = """You are one of many designers collaborating on a robot-design task.

You are designing a simple walking robot on a 5x5 grid of dots (rows and \
columns numbered 0-4). A design is a list of straight line segments, \
each connecting two grid-adjacent dots (horizontally or vertically \
neighboring only -- no diagonals). Each segment becomes a rigid body \
part of the robot; every dot touched by a segment becomes a joint or \
end effector. The robot is judged on how far it can travel in 15 \
seconds once a walking motion is applied to its joints.

Known helpful patterns: designs with more limbs (dead-end segments) \
tend to have more ways to push against the ground, and symmetric \
designs (mirror-matched left/right, top/bottom, or diagonally) tend \
to walk in a straighter, more effective line.

Respond with ONLY a JSON object of this exact form, no other text:
{"edges": [[[r1, c1], [r2, c2]], [[r3, c3], [r4, c4]], ...]}

Each [r, c] must satisfy 0 <= r <= 4 and 0 <= c <= 4, and each edge's \
two dots must be grid-adjacent (differ by 1 in exactly one coordinate).
"""


def _get_api_key() -> str | None:
    """Resolve the API key from, in order: a local .env file (via
    python-dotenv, for local development), the environment (works
    locally and matches how most hosts inject secrets), or Streamlit's
    secrets manager (how Streamlit Community Cloud exposes secrets set
    in its dashboard). Checking all three means the same code works
    unchanged on your laptop and once deployed."""
    try:
        from dotenv import load_dotenv, find_dotenv

        # find_dotenv(usecwd=True) searches starting from the current
        # working directory rather than the calling module's file
        # location. Without usecwd=True, the default search can fail
        # to find a .env sitting at the repo root when this code is
        # invoked from a subfolder module (like agents/llm_agent.py)
        # -- a known source of silent "key not found" bugs.
        load_dotenv(find_dotenv(usecwd=True))  # no-op if no .env file found
    except ImportError:
        pass

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _client():
    """Lazily construct the Anthropic client so importing this module
    doesn't require an API key to be set (useful for tests)."""
    import anthropic

    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found. Set it in a .env file, as an "
            "environment variable, or (when deployed) in Streamlit Cloud's "
            "app secrets."
        )
    return anthropic.Anthropic(api_key=api_key)


@dataclass
class AgentMemory:
    agent_id: str
    history: list[dict] = field(default_factory=list)  # [{"edges":..., "distance":...}]

    def best_design(self):
        if not self.history:
            return None
        return max(self.history, key=lambda h: h["distance"])


def _build_user_prompt(memory: AgentMemory, visible_designs: list[dict] | None) -> str:
    parts = []

    if memory.history:
        recent = memory.history[-3:]
        parts.append("Your recent designs and how far they traveled (grid cells):")
        for h in recent:
            parts.append(f"  edges={h['edges']} -> distance={h['distance']:.2f}")
        best = memory.best_design()
        parts.append(f"Your best so far: edges={best['edges']} -> distance={best['distance']:.2f}")
        base_design = best["edges"]
    else:
        parts.append("You have no designs yet. Here is a simple starting design to build from:")
        parts.append(f"  edges={T_SHAPE}")
        base_design = T_SHAPE

    if visible_designs:
        parts.append("\nOther designers in your group have found these top designs:")
        for d in visible_designs:
            parts.append(f"  edges={d['edges']} -> distance={d['distance']:.2f}")
        parts.append(
            "You may borrow ideas from these, combine them with your own, or "
            "keep exploring your own direction -- your choice."
        )

    parts.append(
        "\nPropose ONE new design: either a small mutation (add or remove one "
        "edge) of your best design, an idea inspired by a visible design above, "
        "or a fresh idea if you think it's promising. Return only the JSON object."
    )
    parts.append(f"\nYour current base design for reference: {base_design}")
    return "\n".join(parts)


def _parse_response(text: str) -> list | None:
    text = text.strip()
    # Be forgiving of accidental markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        return data.get("edges")
    except (json.JSONDecodeError, AttributeError):
        return None


def propose_design(memory: AgentMemory, visible_designs: list[dict] | None = None) -> list[Edge]:
    """Ask the LLM for one new design. Falls back to a random mutation
    of the agent's best-known design (or a seed design) if the LLM
    response is missing/invalid after one retry, so a bad API call
    never stalls the experiment loop."""
    client = _client()
    user_prompt = _build_user_prompt(memory, visible_designs)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
            edges = _parse_response(raw_text)
            if edges is not None:
                cleaned = clean_design(edges)
                if is_valid_design(cleaned):
                    return cleaned
        except Exception:
            pass  # fall through to retry / fallback

        if attempt < MAX_RETRIES:
            user_prompt += "\n\nYour previous response was invalid or unparsable. Try again, returning ONLY the JSON object."

    return _fallback_mutation(memory)


def _fallback_mutation(memory: AgentMemory) -> list[Edge]:
    """Random add/remove-one-edge mutation, used when the LLM can't
    produce a valid design. Guarantees the experiment loop never
    stalls on a single bad API response."""
    base = memory.best_design()["edges"] if memory.history else T_SHAPE
    edges = set(tuple(sorted((tuple(a), tuple(b)))) for a, b in base)

    all_possible = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if r + 1 < GRID_SIZE:
                all_possible.append(((r, c), (r + 1, c)))
            if c + 1 < GRID_SIZE:
                all_possible.append(((r, c), (r, c + 1)))

    if random.random() < 0.5:
        candidates = [e for e in all_possible if e not in edges]
        if candidates:
            edges.add(random.choice(candidates))
    else:
        if len(edges) > 1:
            edges.remove(random.choice(list(edges)))

    return clean_design(list(edges))
