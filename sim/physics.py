"""
physics.py — 2D physics scoring for grid robot designs, using pymunk.

This is a deliberately lightweight stand-in for a 3D
web-embedded physics engine. Each dot adjacent to at least one edge
becomes a small rigid body ("cube" in the paper, a box here). Each
edge becomes a motorized pivot joint ("hinge") between the two dot
bodies, driven by a sinusoidal target angle — same amplitude and
frequency for every joint, phase in {0, 180} degrees per joint,
exactly as described in the paper's Methods section.

Distance traveled = net displacement of the robot's centroid over a
fixed simulation duration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pymunk

from sim.grid import Edge, dots_used

CELL_SIZE = 20.0          # pixels per grid cell (spacing between dots)
BODY_SIZE = 14.0          # box side length for each dot's body
JOINT_AMPLITUDE = math.radians(45)   # +/-45 degrees, matches the paper
JOINT_FREQUENCY_HZ = 1.5             # matches the paper
SIM_DT = 1.0 / 120.0
GROUND_Y = 0.0
GRAVITY = -900.0


PhaseConfig = dict[int, float]  # edge_index -> phase in radians (0 or pi)


def random_phase_config(n_edges: int) -> PhaseConfig:
    return {i: random.choice([0.0, math.pi]) for i in range(n_edges)}


def mutate_phase_config(phase_config: PhaseConfig, flip_prob: float = 0.15) -> PhaseConfig:
    """Copy a phase config, flipping each joint's phase independently
    with probability flip_prob — this is the hill-climber's mutation
    operator from the paper (a 'slightly altered version' of the
    original configuration)."""
    new_config = {}
    for i, phase in phase_config.items():
        if random.random() < flip_prob:
            new_config[i] = math.pi if phase == 0.0 else 0.0
        else:
            new_config[i] = phase
    return new_config


@dataclass
class SimResult:
    distance: float
    start_x: float
    end_x: float
    valid: bool = True
    error: str | None = None


def _build_world(edges: list[Edge], phase_config: PhaseConfig):
    """Construct the pymunk space, ground, and robot bodies/joints
    for one design. Returns (space, bodies_by_dot, motors)."""
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)

    # Ground: a static segment robots rest and push against.
    ground = pymunk.Segment(space.static_body, (-5000, GROUND_Y), (5000, GROUND_Y), 5)
    ground.friction = 1.0
    space.add(ground)

    dots = dots_used(edges)
    dot_index = {dot: i for i, dot in enumerate(dots)}

    # Place the design so its bounding box sits just above the ground,
    # centered horizontally at x=0.
    rows = [r for r, _ in dots]
    cols = [c for _, c in dots]
    min_r, max_r = min(rows), max(rows)
    mid_c = (min(cols) + max(cols)) / 2.0

    bodies = []
    for (r, c) in dots:
        x = (c - mid_c) * CELL_SIZE
        y = (max_r - r) * CELL_SIZE + BODY_SIZE  # flip so higher grid row = higher y, lift off ground
        mass = 1.0
        moment = pymunk.moment_for_box(mass, (BODY_SIZE, BODY_SIZE))
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        shape = pymunk.Poly.create_box(body, (BODY_SIZE, BODY_SIZE))
        shape.friction = 0.9
        shape.elasticity = 0.0
        space.add(body, shape)
        bodies.append(body)

    motors = []  # (motor_constraint, phase)
    for edge_idx, (a, b) in enumerate(edges):
        body_a = bodies[dot_index[a]]
        body_b = bodies[dot_index[b]]
        pivot_point = body_a.position.interpolate_to(body_b.position, 0.5)

        pivot = pymunk.PivotJoint(body_a, body_b, pivot_point)
        pivot.collide_bodies = False
        space.add(pivot)

        motor = pymunk.SimpleMotor(body_a, body_b, 0.0)
        space.add(motor)

        phase = phase_config.get(edge_idx, 0.0)
        motors.append((motor, phase))

    return space, bodies, motors


def simulate(edges: list[Edge], phase_config: PhaseConfig, duration: float = 15.0) -> SimResult:
    """Run the physics sim for `duration` seconds and return the net
    horizontal distance traveled by the robot's centroid."""
    if not edges:
        return SimResult(distance=0.0, start_x=0.0, end_x=0.0, valid=False, error="empty design")

    try:
        space, bodies, motors = _build_world(edges, phase_config)
    except Exception as exc:  # keep the experiment loop alive on malformed designs
        return SimResult(distance=0.0, start_x=0.0, end_x=0.0, valid=False, error=str(exc))

    def centroid_x():
        return sum(b.position.x for b in bodies) / len(bodies)

    start_x = centroid_x()
    t = 0.0
    steps = int(duration / SIM_DT)

    try:
        for _ in range(steps):
            # Sinusoidal, displacement-controlled signal: drive each
            # motor's angular target via its rate so the joint angle
            # itself tracks a sine wave of the given amplitude/phase.
            omega = 2 * math.pi * JOINT_FREQUENCY_HZ
            for motor, phase in motors:
                target_angle = JOINT_AMPLITUDE * math.sin(omega * t + phase)
                # Simple proportional drive toward the target angle via rate.
                current_angle = motor.a.angle - motor.b.angle
                error = target_angle - current_angle
                motor.rate = max(-6.0, min(6.0, error * 8.0))
            space.step(SIM_DT)
            t += SIM_DT
    except Exception as exc:
        return SimResult(distance=0.0, start_x=start_x, end_x=start_x, valid=False, error=str(exc))

    end_x = centroid_x()
    distance = abs(end_x - start_x) / CELL_SIZE  # normalize to "grid cells traveled"
    return SimResult(distance=distance, start_x=start_x, end_x=end_x, valid=True)


def hill_climb(edges: list[Edge], n_iterations: int = 8, duration: float = 15.0):
    """Reproduce the paper's per-design hill-climber over phase
    configurations: start random, repeatedly mutate, keep the
    mutation only if it travels farther. Returns (best_distance,
    best_phase_config, history) where history is a list of the
    distance achieved at each iteration."""
    n_edges = len(edges)
    best_phase = random_phase_config(n_edges)
    best_result = simulate(edges, best_phase, duration)
    best_distance = best_result.distance
    history = [best_distance]

    for _ in range(n_iterations - 1):
        candidate_phase = mutate_phase_config(best_phase)
        candidate_result = simulate(edges, candidate_phase, duration)
        if candidate_result.valid and candidate_result.distance > best_distance:
            best_phase = candidate_phase
            best_distance = candidate_result.distance
        history.append(best_distance)

    return best_distance, best_phase, history