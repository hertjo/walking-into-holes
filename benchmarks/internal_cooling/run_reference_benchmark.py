#!/usr/bin/env python3
"""Seeded 3D internal-cooling reference benchmark for Walking into Holes.

The reference implementation uses analytic box/sphere distance queries so the
statistical and optimization protocol can be tested without a mesh or voxel
discretization. The API is intended to drive a later C++/Embree kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from scipy.stats import qmc

Array = np.ndarray


@dataclass(frozen=True)
class Config:
    seed: int = 20260912
    half_extents: tuple[float, float, float] = (1.20, 0.80, 0.45)
    cavity_radius: float = 0.115
    outer_clearance: float = 0.035
    cavity_clearance: float = 0.035
    insertions: int = 7
    candidate_count: int = 2048
    scout_walks: int = 16
    shortlist: int = 24
    validation_walks: int = 192
    evaluation_walks: int = 64
    evaluation_probes: int = 768
    slice_walks: int = 96
    termination_epsilon: float = 2.0e-5
    max_wos_steps: int = 512
    confidence_z: float = 1.96
    voxel_shape: tuple[int, int, int] = (49, 33, 19)
    source_x_slope: float = 0.45
    source_y_slope: float = -0.25


@dataclass
class Selection:
    center: Array
    scout_score: float
    validation_score: float
    validation_se: float
    lower_confidence: float
    feasible_count: int
    mean_walk_steps: float


def source(points: Array, config: Config) -> Array:
    extents = np.asarray(config.half_extents)
    return (
        1.0
        + config.source_x_slope * points[:, 0] / extents[0]
        + config.source_y_slope * points[:, 1] / extents[1]
    )


def inside_cavities(points: Array, centers: Array, radius: float) -> Array:
    if centers.size == 0:
        return np.zeros(len(points), dtype=bool)
    squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.any(squared <= radius**2, axis=1)


def boundary_distance(
    points: Array, centers: Array, radius: float, half_extents: Array
) -> Array:
    distance = np.min(half_extents[None, :] - np.abs(points), axis=1)
    if centers.size:
        sphere_distance = np.min(
            np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            - radius,
            axis=1,
        )
        distance = np.minimum(distance, sphere_distance)
    return np.maximum(distance, 0.0)


def random_unit_vectors(rng: np.random.Generator, count: int) -> Array:
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def wos_samples(
    points: Array,
    sample_count: int,
    centers: Array,
    config: Config,
    rng: np.random.Generator,
    chunk_path_count: int = 120_000,
) -> tuple[Array, float]:
    """Estimate -Delta u=f, u=0 on the box and cavity boundaries.

    The affine source is harmonic. Its exact Green contribution on a WoS ball
    of radius R centered at x is f(x) R^2/(2d), with d=3.
    """
    points = np.asarray(points, dtype=float)
    half_extents = np.asarray(config.half_extents)
    output = np.empty((len(points), sample_count), dtype=float)
    steps_total = 0
    paths_total = len(points) * sample_count
    point_chunk = max(1, chunk_path_count // sample_count)

    for begin in range(0, len(points), point_chunk):
        end = min(len(points), begin + point_chunk)
        count = end - begin
        positions = np.repeat(points[begin:end], sample_count, axis=0)
        accumulated = np.zeros(len(positions), dtype=float)
        path_steps = np.zeros(len(positions), dtype=np.int32)
        for _ in range(config.max_wos_steps):
            radii = boundary_distance(
                positions, centers, config.cavity_radius, half_extents
            )
            active = radii > config.termination_epsilon
            if not np.any(active):
                break
            indices = np.flatnonzero(active)
            active_positions = positions[indices]
            active_radii = radii[indices]
            accumulated[indices] += (
                source(active_positions, config) * active_radii**2 / 6.0
            )
            positions[indices] += active_radii[:, None] * random_unit_vectors(
                rng, len(indices)
            )
            path_steps[indices] += 1
        else:
            remaining = boundary_distance(
                positions, centers, config.cavity_radius, half_extents
            )
            if np.any(remaining > config.termination_epsilon):
                raise RuntimeError("WoS maximum-step limit reached")
        output[begin:end] = accumulated.reshape(count, sample_count)
        steps_total += int(path_steps.sum())
    return output, steps_total / max(paths_total, 1)


def crossfit_square(samples: Array) -> tuple[Array, Array]:
    """Leave-diagonal-out estimate of E[X]^2 and jackknife standard error."""
    n = samples.shape[1]
    if n < 4:
        raise ValueError("cross-fitting requires at least four samples")
    sums = np.sum(samples, axis=1)
    sums2 = np.sum(samples**2, axis=1)
    estimate = (sums**2 - sums2) / (n * (n - 1))
    leave_sums = sums[:, None] - samples
    leave_sums2 = sums2[:, None] - samples**2
    leave_estimates = (leave_sums**2 - leave_sums2) / ((n - 1) * (n - 2))
    leave_mean = np.mean(leave_estimates, axis=1, keepdims=True)
    standard_error = np.sqrt(
        (n - 1) / n * np.sum((leave_estimates - leave_mean) ** 2, axis=1)
    )
    return estimate, standard_error


def sobol_points(config: Config, count: int, margin: float, seed: int) -> Array:
    exponent = int(math.ceil(math.log2(count)))
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)[:count]
    extents = np.asarray(config.half_extents) - margin
    return qmc.scale(unit, -extents, extents)


def feasible_mask(candidates: Array, centers: Array, config: Config) -> Array:
    required_outer = config.cavity_radius + config.outer_clearance
    extents = np.asarray(config.half_extents)
    mask = np.min(extents[None, :] - np.abs(candidates), axis=1) >= required_outer
    if centers.size:
        required_pair = 2.0 * config.cavity_radius + config.cavity_clearance
        distances = np.linalg.norm(
            candidates[:, None, :] - centers[None, :, :], axis=2
        )
        mask &= np.min(distances, axis=1) >= required_pair
    return mask


def select_twos(
    candidates: Array,
    centers: Array,
    config: Config,
    rng: np.random.Generator,
    fresh_validation: bool,
) -> Selection:
    feasible = candidates[feasible_mask(candidates, centers, config)]
    if len(feasible) < config.shortlist:
        raise RuntimeError("too few feasible candidate centers")
    scout_samples, scout_steps = wos_samples(
        feasible, config.scout_walks, centers, config, rng
    )
    scout_score, scout_se = crossfit_square(scout_samples)

    if not fresh_validation:
        chosen = int(np.argmax(scout_score))
        return Selection(
            center=feasible[chosen],
            scout_score=float(scout_score[chosen]),
            validation_score=float(scout_score[chosen]),
            validation_se=float(scout_se[chosen]),
            lower_confidence=float(
                scout_score[chosen] - config.confidence_z * scout_se[chosen]
            ),
            feasible_count=len(feasible),
            mean_walk_steps=scout_steps,
        )

    shortlist_count = min(config.shortlist, len(feasible))
    shortlist_indices = np.argpartition(scout_score, -shortlist_count)[
        -shortlist_count:
    ]
    shortlisted = feasible[shortlist_indices]
    validation_samples, validation_steps = wos_samples(
        shortlisted, config.validation_walks, centers, config, rng
    )
    validation_score, validation_se = crossfit_square(validation_samples)
    lower = validation_score - config.confidence_z * validation_se
    chosen_short = int(np.argmax(lower))
    chosen_full = int(shortlist_indices[chosen_short])
    return Selection(
        center=shortlisted[chosen_short],
        scout_score=float(scout_score[chosen_full]),
        validation_score=float(validation_score[chosen_short]),
        validation_se=float(validation_se[chosen_short]),
        lower_confidence=float(lower[chosen_short]),
        feasible_count=len(feasible),
        mean_walk_steps=(scout_steps + validation_steps) / 2.0,
    )


def solve_voxel(
    centers: Array, config: Config
) -> tuple[tuple[Array, Array, Array], Array, dict[str, float]]:
    """Structured-grid Dirichlet reference used as a discretized TD baseline."""
    shape = config.voxel_shape
    extents = np.asarray(config.half_extents)
    axes = tuple(np.linspace(-extents[i], extents[i], shape[i]) for i in range(3))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    interior = np.ones(shape, dtype=bool)
    interior[[0, -1], :, :] = False
    interior[:, [0, -1], :] = False
    interior[:, :, [0, -1]] = False
    if centers.size:
        cavity = inside_cavities(coordinates, centers, config.cavity_radius).reshape(
            shape
        )
        interior &= ~cavity

    node_id = np.full(shape, -1, dtype=np.int64)
    active_flat = np.flatnonzero(interior.ravel())
    node_id.ravel()[active_flat] = np.arange(len(active_flat))
    spacings = [axis[1] - axis[0] for axis in axes]
    inverse_h2 = [1.0 / h**2 for h in spacings]
    diagonal = 2.0 * sum(inverse_h2)
    rows = [np.arange(len(active_flat), dtype=np.int64)]
    cols = [np.arange(len(active_flat), dtype=np.int64)]
    data = [np.full(len(active_flat), diagonal)]
    active_indices = np.column_stack(np.unravel_index(active_flat, shape))
    for dimension, weight in enumerate(inverse_h2):
        for direction in (-1, 1):
            neighbors = active_indices.copy()
            neighbors[:, dimension] += direction
            neighbor_ids = node_id[tuple(neighbors.T)]
            valid = neighbor_ids >= 0
            rows.append(np.arange(len(active_flat), dtype=np.int64)[valid])
            cols.append(neighbor_ids[valid])
            data.append(np.full(np.count_nonzero(valid), -weight))
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(active_flat), len(active_flat)),
    ).tocsr()
    rhs = source(coordinates[active_flat], config)
    started = time.perf_counter()
    solution, info = cg(matrix, rhs, rtol=1.0e-10, atol=0.0, maxiter=4000)
    elapsed = time.perf_counter() - started
    if info != 0:
        raise RuntimeError(f"voxel conjugate-gradient solve failed with info={info}")
    field = np.zeros(shape, dtype=float)
    field.ravel()[active_flat] = solution
    stats = {
        "active_nodes": float(len(active_flat)),
        "nonzeros": float(matrix.nnz),
        "solve_seconds": elapsed,
        "matrix_megabytes": float(
            (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
            / 2**20
        ),
    }
    return axes, field, stats


def select_voxel(
    candidates: Array, centers: Array, config: Config
) -> tuple[Selection, dict[str, float]]:
    axes, field, stats = solve_voxel(centers, config)
    feasible = candidates[feasible_mask(candidates, centers, config)]
    score = RegularGridInterpolator(
        axes, field, bounds_error=False, fill_value=0.0
    )(feasible) ** 2
    chosen = int(np.argmax(score))
    return Selection(
        center=feasible[chosen],
        scout_score=float(score[chosen]),
        validation_score=float(score[chosen]),
        validation_se=0.0,
        lower_confidence=float(score[chosen]),
        feasible_count=len(feasible),
        mean_walk_steps=0.0,
    ), stats


def select_random(
    candidates: Array, centers: Array, config: Config, rng: np.random.Generator
) -> Selection:
    feasible = candidates[feasible_mask(candidates, centers, config)]
    chosen = int(rng.integers(len(feasible)))
    return Selection(
        center=feasible[chosen],
        scout_score=float("nan"),
        validation_score=float("nan"),
        validation_se=float("nan"),
        lower_confidence=float("nan"),
        feasible_count=len(feasible),
        mean_walk_steps=0.0,
    )


def estimate_objective(
    centers: Array,
    probes: Array,
    config: Config,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    outside = ~inside_cavities(probes, centers, config.cavity_radius)
    point_values = np.zeros(len(probes), dtype=float)
    mean_steps = 0.0
    if np.any(outside):
        samples, mean_steps = wos_samples(
            probes[outside], config.evaluation_walks, centers, config, rng
        )
        point_values[outside] = source(probes[outside], config) * np.mean(
            samples, axis=1
        )
    return (
        float(np.mean(point_values)),
        float(np.std(point_values, ddof=1) / math.sqrt(len(probes))),
        mean_steps,
    )


def run_design(
    name: str,
    selector: Callable[[Array], tuple[Selection, dict[str, float]]],
    candidates: Array,
    probes: Array,
    config: Config,
    evaluation_seed: int,
) -> tuple[list[dict[str, float | int | str]], Array, list[dict[str, float]]]:
    centers = np.empty((0, 3), dtype=float)
    rows: list[dict[str, float | int | str]] = []
    solver_stats: list[dict[str, float]] = []
    for iteration in range(config.insertions + 1):
        objective, objective_se, evaluation_steps = estimate_objective(
            centers,
            probes,
            config,
            np.random.default_rng(evaluation_seed + iteration),
        )
        row: dict[str, float | int | str] = {
            "method": name,
            "iteration": iteration,
            "objective": objective,
            "objective_se": objective_se,
            "cavity_count": len(centers),
            "evaluation_mean_steps": evaluation_steps,
            "selected_x": float("nan"),
            "selected_y": float("nan"),
            "selected_z": float("nan"),
            "scout_score": float("nan"),
            "validation_score": float("nan"),
            "validation_se": float("nan"),
            "lower_confidence": float("nan"),
            "feasible_count": 0,
            "selection_mean_steps": float("nan"),
        }
        if iteration < config.insertions:
            selection, stats = selector(centers)
            row.update(
                {
                    "selected_x": float(selection.center[0]),
                    "selected_y": float(selection.center[1]),
                    "selected_z": float(selection.center[2]),
                    "scout_score": selection.scout_score,
                    "validation_score": selection.validation_score,
                    "validation_se": selection.validation_se,
                    "lower_confidence": selection.lower_confidence,
                    "feasible_count": selection.feasible_count,
                    "selection_mean_steps": selection.mean_walk_steps,
                }
            )
            centers = np.vstack((centers, selection.center))
            solver_stats.append(stats)
        rows.append(row)
    return rows, centers, solver_stats


def evaluate_slice(
    centers: Array, config: Config, seed: int
) -> tuple[Array, Array, Array, Array]:
    extents = np.asarray(config.half_extents)
    x = np.linspace(-extents[0] + 0.01, extents[0] - 0.01, 73)
    y = np.linspace(-extents[1] + 0.01, extents[1] - 0.01, 49)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    outside = ~inside_cavities(points, centers, config.cavity_radius)
    temperature = np.full(len(points), np.nan)
    score = np.full(len(points), np.nan)
    samples, _ = wos_samples(
        points[outside], config.slice_walks, centers, config, np.random.default_rng(seed)
    )
    temperature[outside] = np.mean(samples, axis=1)
    score[outside] = crossfit_square(samples)[0]
    return x, y, temperature.reshape(xx.shape), score.reshape(xx.shape)


def initial_field_agreement(
    candidates: Array, config: Config, seed: int
) -> tuple[float, float]:
    subset = candidates[::8]
    samples, _ = wos_samples(
        subset,
        max(config.validation_walks, 256),
        np.empty((0, 3)),
        config,
        np.random.default_rng(seed),
    )
    twos = crossfit_square(samples)[0]
    axes, voxel, _ = solve_voxel(np.empty((0, 3)), config)
    voxel_values = RegularGridInterpolator(axes, voxel)(subset) ** 2
    correlation = float(np.corrcoef(twos, voxel_values)[0, 1])
    relative_rmse = float(
        np.sqrt(np.mean((twos - voxel_values) ** 2))
        / max(np.sqrt(np.mean(voxel_values**2)), 1.0e-12)
    )
    return correlation, relative_rmse


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_block(ax: plt.Axes, extents: Array) -> None:
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    vertices = signs * extents
    for i, first in enumerate(signs):
        for j, second in enumerate(signs):
            if j <= i or np.count_nonzero(first != second) != 1:
                continue
            edge = vertices[[i, j]]
            ax.plot(*edge.T, color="#555d66", lw=0.75, alpha=0.7)


def draw_spheres(ax: plt.Axes, centers: Array, radius: float) -> None:
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 15)
    palette = mpl.colormaps["winter"](np.linspace(0.18, 0.88, max(len(centers), 1)))
    for index, center in enumerate(centers):
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(
            x, y, z, color=palette[index], linewidth=0, alpha=0.94, shade=True
        )


def make_figure(
    output_dir: Path,
    config: Config,
    rows: list[dict[str, float | int | str]],
    centers_by_method: dict[str, Array],
    initial_slice: tuple[Array, Array, Array, Array],
    final_slice: tuple[Array, Array, Array, Array],
) -> None:
    configure_plotting()
    colors = {
        "TWoS + fresh validation": "#168a83",
        "TWoS + reused scout": "#d85b53",
        "Voxel adjoint TD": "#6676c8",
        "Random feasible": "#9ca3a8",
    }
    fig = plt.figure(figsize=(12.2, 7.25))
    grid = fig.add_gridspec(
        2, 3, left=0.055, right=0.98, bottom=0.13, top=0.95,
        wspace=0.27, hspace=0.33,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2], projection="3d"),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes
    x, y, initial_temp, initial_score = initial_slice
    _, _, final_temp, _ = final_slice
    extent = [x[0], x[-1], y[0], y[-1]]
    temp_norm = Normalize(vmin=0.0, vmax=float(np.nanmax(initial_temp)))
    score_norm = Normalize(vmin=0.0, vmax=float(np.nanmax(initial_score)))

    image_a = ax_a.imshow(
        initial_temp, origin="lower", extent=extent, cmap="inferno",
        norm=temp_norm, interpolation="bilinear", aspect="equal",
    )
    ax_a.contour(x, y, initial_temp, levels=7, colors="white", alpha=0.34, linewidths=0.55)
    ax_a.set_title(r"a   Initial temperature $u$")
    ax_a.set(xlabel="$x$", ylabel="$y$")
    fig.colorbar(image_a, ax=ax_a, fraction=0.046, pad=0.035)

    image_b = ax_b.imshow(
        initial_score, origin="lower", extent=extent, cmap="magma",
        norm=score_norm, interpolation="bilinear", aspect="equal",
    )
    ours = centers_by_method["TWoS + fresh validation"]
    ax_b.scatter(
        ours[:, 0], ours[:, 1], c=np.arange(1, len(ours) + 1), cmap="winter",
        s=34, edgecolor="white", linewidth=0.8, zorder=3,
    )
    for index, center in enumerate(ours):
        ax_b.text(
            center[0], center[1], str(index + 1), ha="center", va="center",
            fontsize=6.8, color="white", fontweight="bold", zorder=4,
        )
    ax_b.set_title(r"b   Direct score $-D_T=u^2$")
    ax_b.set(xlabel="$x$", ylabel="$y$")
    fig.colorbar(image_b, ax=ax_b, fraction=0.046, pad=0.035)

    extents = np.asarray(config.half_extents)
    draw_block(ax_c, extents)
    draw_spheres(ax_c, ours, config.cavity_radius)
    ax_c.set_xlim(-extents[0], extents[0])
    ax_c.set_ylim(-extents[1], extents[1])
    ax_c.set_zlim(-extents[2], extents[2])
    ax_c.set_box_aspect(2 * extents)
    ax_c.view_init(elev=23, azim=-55)
    ax_c.set_title("c   Grid-free cooling cavities", pad=2)
    ax_c.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    ax_c.tick_params(labelsize=6, pad=-2)
    ax_c.grid(False)
    ax_c.xaxis.pane.set_alpha(0)
    ax_c.yaxis.pane.set_alpha(0)
    ax_c.zaxis.pane.set_alpha(0)

    methods = list(colors)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        iterations = np.array([int(row["iteration"]) for row in method_rows])
        objectives = np.array([float(row["objective"]) for row in method_rows])
        errors = np.array([float(row["objective_se"]) for row in method_rows])
        ax_d.plot(
            iterations, objectives / objectives[0], marker="o", ms=3.8,
            lw=1.8, color=colors[method], label=method,
        )
        ax_d.fill_between(
            iterations, (objectives - errors) / objectives[0],
            (objectives + errors) / objectives[0], color=colors[method],
            alpha=0.10, linewidth=0,
        )
    ax_d.set_title("d   Independently evaluated cooling")
    ax_d.set(xlabel="inserted cavities", ylabel="normalized compliance")
    ax_d.set_xticks(range(config.insertions + 1))
    ax_d.grid(color="#d7dadd", lw=0.6)
    ax_d.legend(frameon=False, loc="best")

    improvements = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        first = float(method_rows[0]["objective"])
        last = float(method_rows[-1]["objective"])
        improvements.append(100.0 * (first - last) / first)
    bars = ax_e.barh(
        np.arange(len(methods)), improvements, color=[colors[m] for m in methods],
        edgecolor="none", height=0.62,
    )
    ax_e.set_yticks(
        np.arange(len(methods)), ["TWoS + fresh", "TWoS + reuse", "voxel TD", "random"]
    )
    ax_e.invert_yaxis()
    ax_e.set_xlabel("compliance reduction (%)")
    ax_e.set_title(f"e   Improvement after {config.insertions} cavities")
    ax_e.grid(axis="x", color="#d7dadd", lw=0.6)
    ax_e.set_axisbelow(True)
    for bar, value in zip(bars, improvements):
        ax_e.text(
            value + 0.35, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
            va="center", fontsize=8, fontweight="bold",
        )
    ax_e.set_xlim(0, max(improvements) * 1.20)

    image_f = ax_f.imshow(
        final_temp, origin="lower", extent=extent, cmap="inferno",
        norm=temp_norm, interpolation="bilinear", aspect="equal",
    )
    ax_f.contour(x, y, final_temp, levels=7, colors="white", alpha=0.30, linewidths=0.55)
    for center in ours:
        if abs(center[2]) < config.cavity_radius:
            section_radius = math.sqrt(config.cavity_radius**2 - center[2] ** 2)
            ax_f.add_patch(
                plt.Circle(
                    (center[0], center[1]), section_radius, fill=False,
                    color="#22c4bd", lw=1.4,
                )
            )
    ax_f.set_title(r"f   Final temperature $u$")
    ax_f.set(xlabel="$x$", ylabel="$y$")
    fig.colorbar(image_f, ax=ax_f, fraction=0.046, pad=0.035)

    fig.text(
        0.5, 0.025,
        f"Affine heat source • {config.candidate_count:,} candidate sites • "
        f"{config.scout_walks} scout / {config.validation_walks} fresh walks • "
        f"fixed radius {config.cavity_radius:.3f}",
        ha="center", va="bottom", color="#747b80", fontsize=9,
    )
    fig.savefig(output_dir / "internal-cooling-benchmark.pdf", bbox_inches="tight")
    temporary = output_dir / "internal-cooling-benchmark.tmp.png"
    fig.savefig(temporary, dpi=210, bbox_inches="tight")
    temporary.replace(output_dir / "internal-cooling-benchmark.png")
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, config: Config) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    margin = config.cavity_radius + config.outer_clearance
    candidates = sobol_points(config, config.candidate_count, margin, config.seed + 1)
    probes = sobol_points(config, config.evaluation_probes, 0.0, config.seed + 2)
    fresh_rng = np.random.default_rng(config.seed + 10)
    reuse_rng = np.random.default_rng(config.seed + 20)
    random_rng = np.random.default_rng(config.seed + 30)
    selectors: dict[str, Callable[[Array], tuple[Selection, dict[str, float]]]] = {
        "TWoS + fresh validation": lambda centers: (
            select_twos(candidates, centers, config, fresh_rng, True), {}
        ),
        "TWoS + reused scout": lambda centers: (
            select_twos(candidates, centers, config, reuse_rng, False), {}
        ),
        "Voxel adjoint TD": lambda centers: select_voxel(candidates, centers, config),
        "Random feasible": lambda centers: (
            select_random(candidates, centers, config, random_rng), {}
        ),
    }

    all_rows: list[dict[str, float | int | str]] = []
    centers_by_method: dict[str, Array] = {}
    voxel_stats: list[dict[str, float]] = []
    method_runtimes: dict[str, float] = {}
    for name, selector in selectors.items():
        method_started = time.perf_counter()
        rows, centers, stats = run_design(
            name, selector, candidates, probes, config,
            evaluation_seed=config.seed + 100,
        )
        method_runtimes[name] = time.perf_counter() - method_started
        all_rows.extend(rows)
        centers_by_method[name] = centers
        if name == "Voxel adjoint TD":
            voxel_stats = stats

    initial_slice = evaluate_slice(np.empty((0, 3)), config, config.seed + 200)
    final_slice = evaluate_slice(
        centers_by_method["TWoS + fresh validation"], config, config.seed + 201
    )
    correlation, relative_rmse = initial_field_agreement(
        candidates, config, config.seed + 300
    )
    make_figure(
        output_dir, config, all_rows, centers_by_method, initial_slice, final_slice
    )
    write_rows(output_dir / "design-trajectories.csv", all_rows)
    center_rows: list[dict[str, float | int | str]] = []
    for method, centers in centers_by_method.items():
        for order, center in enumerate(centers, start=1):
            center_rows.append(
                {"method": method, "order": order, "x": float(center[0]),
                 "y": float(center[1]), "z": float(center[2])}
            )
    write_rows(output_dir / "selected-centers.csv", center_rows)

    methods_summary: dict[str, dict[str, float]] = {}
    for method in selectors:
        method_rows = [row for row in all_rows if row["method"] == method]
        initial = float(method_rows[0]["objective"])
        final = float(method_rows[-1]["objective"])
        methods_summary[method] = {
            "initial_objective": initial,
            "final_objective": final,
            "relative_reduction_percent": 100.0 * (initial - final) / initial,
            "final_objective_se": float(method_rows[-1]["objective_se"]),
            "python_runtime_seconds": method_runtimes[method],
        }

    summary: dict[str, object] = {
        "scope": "3D analytic-geometry reference benchmark; not final equal-time mesh result",
        "configuration": asdict(config),
        "methods": methods_summary,
        "initial_field_agreement": {
            "twos_voxel_pearson": correlation,
            "relative_rmse": relative_rmse,
        },
        "voxel_baseline": {
            "mean_active_nodes": float(np.mean([s["active_nodes"] for s in voxel_stats])),
            "mean_matrix_megabytes": float(np.mean([s["matrix_megabytes"] for s in voxel_stats])),
            "mean_solve_seconds": float(np.mean([s["solve_seconds"] for s in voxel_stats])),
        },
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": mpl.__version__,
        },
    }
    fresh = methods_summary["TWoS + fresh validation"]
    random_result = methods_summary["Random feasible"]
    if not fresh["final_objective"] < fresh["initial_objective"]:
        raise AssertionError("fresh-validation design did not improve compliance")
    if not fresh["final_objective"] < random_result["final_objective"]:
        raise AssertionError("fresh-validation design did not beat random insertion")
    if correlation < 0.90:
        raise AssertionError("TWoS and voxel TD fields disagree unexpectedly")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, Config()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
