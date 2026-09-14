"""
grid.py — 5x5 dot-grid robot design representation.

A design is a list of edges, where each edge is a pair of adjacent
dot coordinates: [[(r1, c1), (r2, c2)], ...].
"""

from __future__ import annotations

import io
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_SIZE = 5
Dot = tuple[int, int]
Edge = tuple[Dot, Dot]


def normalize_edge(edge: Iterable[Iterable[int]]) -> Edge:
    """Convert a raw edge (possibly lists, possibly unordered) into a
    canonical, hashable (sorted) tuple-of-tuples form."""
    (r1, c1), (r2, c2) = edge
    a, b = (int(r1), int(c1)), (int(r2), int(c2))
    return (a, b) if a <= b else (b, a)


def _in_bounds(dot: Dot) -> bool:
    r, c = dot
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def _is_adjacent(a: Dot, b: Dot) -> bool:
    """Adjacent means horizontally or vertically neighboring dots
    (no diagonals, matches the paper's drawing constraint)."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr, dc) in ((0, 1), (1, 0))


def is_valid_design(edges: list) -> bool:
    """A design is valid if it's non-empty, every edge connects two
    distinct, in-bounds, grid-adjacent dots, and there are no
    duplicate edges."""
    if not edges:
        return False
    seen = set()
    for raw_edge in edges:
        try:
            edge = normalize_edge(raw_edge)
        except (ValueError, TypeError):
            return False
        a, b = edge
        if a == b:
            return False
        if not (_in_bounds(a) and _in_bounds(b)):
            return False
        if not _is_adjacent(a, b):
            return False
        if edge in seen:
            return False
        seen.add(edge)
    return True


def clean_design(edges: list) -> list[Edge]:
    """Normalize + de-duplicate an edge list, dropping invalid edges
    individually rather than rejecting the whole design. Useful for
    salvaging a mostly-valid LLM response."""
    cleaned = []
    seen = set()
    for raw_edge in edges:
        try:
            edge = normalize_edge(raw_edge)
        except (ValueError, TypeError):
            continue
        a, b = edge
        if a == b or not (_in_bounds(a) and _in_bounds(b)):
            continue
        if not _is_adjacent(a, b):
            continue
        if edge in seen:
            continue
        seen.add(edge)
        cleaned.append(edge)
    return cleaned


def dots_used(edges: list[Edge]) -> list[Dot]:
    """All dots touched by at least one edge, in stable sorted order."""
    dots = set()
    for a, b in edges:
        dots.add(a)
        dots.add(b)
    return sorted(dots)


def grid_to_adjacency(edges: list[Edge]):
    """Build a NxN adjacency matrix (N = number of dots actually used)
    plus the dot-index mapping, for use by features.py."""
    import numpy as np

    dots = dots_used(edges)
    index = {dot: i for i, dot in enumerate(dots)}
    n = len(dots)
    adj = np.zeros((n, n), dtype=int)
    for a, b in edges:
        i, j = index[a], index[b]
        adj[i, j] = 1
        adj[j, i] = 1
    return adj, dots


def render_grid(edges: list[Edge], title: str = "", highlight_color: str = "#2563eb"):
    """Render a design as a matplotlib figure: light grey grid dots,
    plus solid line segments for the drawn edges. Returns a PNG image
    (bytes) suitable for st.image() or saving to disk."""
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ax.plot(c, -r, "o", color="#d1d5db", markersize=6, zorder=1)

    for (r1, c1), (r2, c2) in edges:
        ax.plot([c1, c2], [-r1, -r2], "-", color=highlight_color, linewidth=4, zorder=2)

    for dot in dots_used(edges):
        r, c = dot
        ax.plot(c, -r, "o", color=highlight_color, markersize=9, zorder=3)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-(GRID_SIZE - 0.5), 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --- A few reference designs, useful for testing and as LLM seed examples ---

T_SHAPE = [
    [(0, 2), (1, 2)],
    [(1, 2), (2, 2)],
    [(2, 2), (3, 2)],
    [(3, 2), (4, 2)],
    [(2, 1), (2, 2)],
    [(2, 2), (2, 3)],
]

PLUS_SHAPE = [
    [(1, 2), (2, 2)],
    [(2, 2), (3, 2)],
    [(2, 1), (2, 2)],
    [(2, 2), (2, 3)],
]

LINE_SHAPE = [
    [(2, 0), (2, 1)],
    [(2, 1), (2, 2)],
    [(2, 2), (2, 3)],
    [(2, 3), (2, 4)],
]