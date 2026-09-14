"""
features.py — geometric measures + SVD latent factor:

  "These geometric measures included minimum, average and maximum
   degree measures; maximal matching; length of the shortest path;
   node connectivity; number of legs; number of segments; radius;
   transitivity; number of cliques; indicators of bipartiteness,
   regularity, whether the network is a tree and biconnectedness;
   and symmetry ..."
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from sim.grid import Edge, dots_used, GRID_SIZE, clean_design

FEATURE_NAMES = [
    "min_degree",
    "avg_degree",
    "max_degree",
    "maximal_matching_size",
    "shortest_path_length",
    "node_connectivity",
    "number_of_legs",
    "number_of_segments",
    "radius",
    "transitivity",
    "number_of_cliques",
    "is_bipartite",
    "is_regular",
    "is_tree",
    "is_biconnected",
    "symmetry",
]


def _to_graph(edges: list[Edge]) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def number_of_legs(edges: list[Edge]) -> int:
    """A 'leg' is an endpoint connected at only one end (degree-1 node
    in the design graph) — matches the paper's description: designs
    whose segments are all connected at both ends have 0 legs."""
    g = _to_graph(clean_design(edges))
    return sum(1 for _, d in g.degree() if d == 1)


def symmetry(edges: list[Edge]) -> float:
    """Maximum proportion of segments that are matched with another
    segment when the design is reflected across the horizontal,
    vertical, or either diagonal axis of the 5x5 grid.

    We reflect every edge across each candidate axis and measure what
    fraction of reflected edges land exactly on another edge in the
    original design (including possibly itself, for edges that lie on
    the axis)."""
    edges = clean_design(edges)
    if not edges:
        return 0.0

    edge_set = set(edges)
    center = (GRID_SIZE - 1) / 2.0  # =2.0 for a 5x5 grid

    def reflect_horizontal(dot):
        r, c = dot
        return (int(2 * center - r), c)

    def reflect_vertical(dot):
        r, c = dot
        return (r, int(2 * center - c))

    def reflect_diag_main(dot):
        r, c = dot
        return (c, r)

    def reflect_diag_anti(dot):
        r, c = dot
        return (GRID_SIZE - 1 - c, GRID_SIZE - 1 - r)

    axes = [reflect_horizontal, reflect_vertical, reflect_diag_main, reflect_diag_anti]

    best_ratio = 0.0
    for reflect in axes:
        matched = 0
        for a, b in edges:
            ra, rb = reflect(a), reflect(b)
            reflected_edge = (ra, rb) if ra <= rb else (rb, ra)
            if reflected_edge in edge_set:
                matched += 1
        ratio = matched / len(edges)
        best_ratio = max(best_ratio, ratio)
    return best_ratio


def compute_features(edges: list[Edge]) -> dict:
    """Compute the full geometric feature vector for one design.
    Falls back to 0 for features that are undefined for a given graph
    shape (e.g., radius/connectivity on a disconnected graph)."""
    edges = clean_design(edges)
    g = _to_graph(edges)
    degrees = [d for _, d in g.degree()]

    def safe(fn, default=0.0):
        try:
            return fn()
        except Exception:
            return default

    features = {
        "min_degree": min(degrees) if degrees else 0,
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "maximal_matching_size": len(safe(lambda: nx.maximal_matching(g), set())),
        "shortest_path_length": safe(lambda: nx.average_shortest_path_length(g)),
        "node_connectivity": safe(lambda: nx.node_connectivity(g)),
        "number_of_legs": number_of_legs(edges),
        "number_of_segments": len(edges),
        "radius": safe(lambda: nx.radius(g)),
        "transitivity": safe(lambda: nx.transitivity(g)),
        "number_of_cliques": safe(lambda: sum(1 for _ in nx.find_cliques(g))),
        "is_bipartite": float(safe(lambda: nx.is_bipartite(g), False)),
        "is_regular": float(safe(lambda: nx.is_regular(g), False)),
        "is_tree": float(safe(lambda: nx.is_tree(g), False)),
        "is_biconnected": float(safe(lambda: nx.is_biconnected(g), False)),
        "symmetry": symmetry(edges),
    }
    return features


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)


def compute_latent_factors(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run SVD on a (n_designs x n_features) matrix of z-scored
    features and project onto the first singular vector, exactly as
    the paper describes. Returns (latent_values, component_weights)
    where component_weights lets you see which features actually
    drove the factor for *this* dataset (compare to the paper's
    symmetry=0.99 / legs=0.01 finding)."""
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0  # avoid div-by-zero for constant columns
    normalized = (feature_matrix - mean) / std

    U, S, Vt = np.linalg.svd(normalized, full_matrices=False)
    first_component = Vt[0]  # weights of each feature in the first singular vector
    latent_values = U[:, 0] * S[0]

    return latent_values, first_component