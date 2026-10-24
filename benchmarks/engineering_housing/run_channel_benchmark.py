#!/usr/bin/env python3
"""Connected cooling-channel growth and Differential WoS refinement benchmark."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import igl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import t as student_t
from channel_geometry import strictly_overlapping_chain

Array = np.ndarray
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20260912


def load_housing_module():
    path = HERE / "run_housing_benchmark.py"
    spec = importlib.util.spec_from_file_location("housing_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dense_candidates(housing, count: int, clearance: float) -> Array:
    samples = housing.sobol_box(
        16, np.array([-0.92, -0.48, -0.18]),
        np.array([0.92, 0.48, 0.18]), SEED + 41,
    )
    feasible = samples[housing.housing_sdf(samples) < -clearance]
    if len(feasible) < count:
        raise RuntimeError(f"only {len(feasible)} connected candidates survived")
    return feasible[:count]


def select_terminals(points: Array, scores: Array) -> tuple[int, int]:
    order = np.argsort(scores)[::-1]
    first = int(order[0])
    opposite = [
        int(index) for index in order
        if points[index, 0] * points[first, 0] < 0.0
        and np.linalg.norm(points[index] - points[first]) >= 0.68
    ]
    if not opposite:
        opposite = [
            int(index) for index in order
            if np.linalg.norm(points[index] - points[first]) >= 0.68
        ]
    if not opposite:
        raise RuntimeError("could not find separated channel terminals")
    return first, opposite[0]


def topology_weighted_path(
    housing, points: Array, scores: Array, start: int, target: int,
    clearance: float, neighbors: int,
) -> tuple[Array, dict[str, float]]:
    lower, upper = np.quantile(scores, [0.05, 0.995])
    normalized = np.clip((scores - lower) / max(upper - lower, 1e-15), 0.0, 1.0)
    tree = cKDTree(points)
    build_start = time.perf_counter()
    for neighbor_count in (neighbors, neighbors + 8, neighbors + 16):
        distances, indices = tree.query(points, k=neighbor_count + 1)
        rows: list[int] = []
        columns: list[int] = []
        weights: list[float] = []
        for first in range(len(points)):
            for distance, second_value in zip(distances[first, 1:], indices[first, 1:]):
                second = int(second_value)
                if second == first or not np.isfinite(distance):
                    continue
                midpoint = 0.5 * (points[first] + points[second])
                if housing.housing_sdf(midpoint[None, :])[0] >= -clearance:
                    continue
                penalty = 1.0 + 2.2 * (
                    1.0 - 0.5 * (normalized[first] + normalized[second])
                )
                rows.extend((first, second))
                columns.extend((second, first))
                weights.extend((float(distance * penalty),) * 2)
        graph = csr_matrix((weights, (rows, columns)), shape=(len(points), len(points)))
        distances_from_start, predecessor = dijkstra(
            graph, directed=False, indices=start, return_predecessors=True,
        )
        if np.isfinite(distances_from_start[target]):
            break
    else:
        raise RuntimeError("candidate graph did not connect the terminals")

    path_indices = [target]
    while path_indices[-1] != start:
        previous = int(predecessor[path_indices[-1]])
        if previous < 0:
            raise RuntimeError("invalid predecessor in connected path")
        path_indices.append(previous)
    path_indices.reverse()
    path = points[np.asarray(path_indices)]
    return path, {
        "graph_build_and_search_seconds": time.perf_counter() - build_start,
        "graph_vertices": float(len(points)),
        "graph_edges": float(graph.nnz // 2),
        "weighted_path_cost": float(distances_from_start[target]),
    }


def resample_polyline(points: Array, count: int) -> Array:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0.0:
        raise RuntimeError("degenerate channel path")
    targets = np.linspace(0.0, cumulative[-1], count)
    result = np.empty((count, 3))
    for index, target in enumerate(targets):
        segment = min(int(np.searchsorted(cumulative, target, side="right") - 1),
                      len(points) - 2)
        segment = max(segment, 0)
        denominator = max(cumulative[segment + 1] - cumulative[segment], 1e-15)
        weight = (target - cumulative[segment]) / denominator
        result[index] = (1.0 - weight) * points[segment] + weight * points[segment + 1]
    return result


def smooth_controls(housing, controls: Array, clearance: float, rounds: int = 4) -> Array:
    smoothed = controls.copy()
    for _ in range(rounds):
        proposal = smoothed.copy()
        proposal[1:-1] = (
            0.25 * smoothed[:-2] + 0.50 * smoothed[1:-1]
            + 0.25 * smoothed[2:]
        )
        feasible = housing.housing_sdf(proposal[1:-1]) < -clearance
        smoothed[1:-1][feasible] = proposal[1:-1][feasible]
    return smoothed


def subdivision_counts(controls: Array, spacing: float) -> Array:
    return np.maximum(
        1, np.ceil(np.linalg.norm(np.diff(controls, axis=0), axis=1) / spacing)
    ).astype(int)


def dense_polyline(controls: Array, subdivisions: Array) -> tuple[Array, Array]:
    points: list[Array] = []
    weights: list[Array] = []
    count = len(controls)
    for segment, pieces in enumerate(subdivisions):
        for local in range(int(pieces)):
            t = local / float(pieces)
            points.append((1.0 - t) * controls[segment] + t * controls[segment + 1])
            row = np.zeros(count)
            row[segment] = 1.0 - t
            row[segment + 1] = t
            weights.append(row)
    points.append(controls[-1].copy())
    row = np.zeros(count)
    row[-1] = 1.0
    weights.append(row)
    return np.asarray(points), np.asarray(weights)


def minimum_sphere_distance(points: Array, centers: Array) -> Array:
    if not len(centers):
        return np.full(len(points), np.inf)
    distances, _ = cKDTree(centers).query(points, k=1, workers=1)
    return np.asarray(distances)


def volume_samples(housing, power: int) -> tuple[Array, float, int]:
    low = np.array([-1.20, -0.77, -0.40])
    high = np.array([1.20, 0.77, 0.77])
    samples = housing.sobol_box(power, low, high, SEED + 52)
    interior = housing.housing_sdf(samples) < 0.0
    return samples[interior], float(np.prod(high - low)), len(samples)


def calibrated_radius(
    interior_samples: Array, box_volume: float, total_samples: int,
    centers: Array, target_volume: float,
) -> float:
    distances = np.sort(minimum_sphere_distance(interior_samples, centers))
    rank = int(round(target_volume / box_volume * total_samples))
    rank = min(max(rank, 1), len(distances) - 1)
    return float(0.5 * (distances[rank - 1] + distances[rank]))


def removed_volume(
    interior_samples: Array, box_volume: float, total_samples: int,
    centers: Array, radius: float,
) -> float:
    removed = np.count_nonzero(
        minimum_sphere_distance(interior_samples, centers) <= radius
    )
    return box_volume * removed / total_samples


def objective_values(
    housing, executable: Path, mesh: Path, output: Path, tag: str,
    probes: Array, centers: Array, radius: float, volume: float,
    walks: int, seeds: list[int], threads: int,
) -> tuple[Array, float, float]:
    values = []
    total_seconds = 0.0
    peak_memory = 0.0
    for repeat, seed in enumerate(seeds):
        value, summary = housing.objective(
            executable, mesh, output, f"{tag}-{repeat}", probes, centers,
            radius, volume, walks, seed, threads,
        )
        values.append(value)
        total_seconds += float(summary["mesh_build_seconds"]) + float(
            summary["runtime_seconds_median"]
        )
        peak_memory = max(peak_memory, float(summary["peak_rss_megabytes"]))
    return np.asarray(values), total_seconds, peak_memory


def run_diffwos(
    housing, executable: Path, mesh: Path, output: Path, tag: str,
    probes: Array, centers: Array, radius: float, volume: float,
    walks: int, seed: int, threads: int, offset: float,
) -> tuple[Array, Array, dict[str, object]]:
    distances = np.linalg.norm(probes[:, None, :] - centers[None, :, :], axis=2)
    active = np.all(distances > radius + 0.0015, axis=1)
    active_points = probes[active]
    points_path = output / f"points-{tag}.csv"
    cavities_path = output / f"cavities-{tag}.csv"
    field_path = output / f"field-{tag}.csv"
    summary_path = output / f"summary-{tag}.json"
    housing.write_points(points_path, active_points)
    housing.write_points(cavities_path, centers)
    command = [
        str(executable), "--mode", "diffwos", "--mesh", str(mesh),
        "--points", str(points_path), "--output", str(field_path),
        "--summary", str(summary_path), "--source-mode", "housing",
        "--walks", str(walks), "--seed", str(seed), "--threads", str(threads),
        "--repeats", "1", "--epsilon", "0.001", "--max-steps", "512",
        "--cavities", str(cavities_path), "--cavity-radius", str(radius),
        "--diff-offset", str(offset),
    ]
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    })
    subprocess.run(command, check=True, env=environment)
    rows = housing.read_rows(field_path)
    center_count = len(centers)
    derivative = np.asarray([
        [[float(row[f"dc{index}_{axis}"]) for axis in "xyz"]
         for index in range(center_count)]
        for row in rows
    ])
    derivative_se = np.asarray([
        [[float(row[f"dc{index}_se_{axis}"]) for axis in "xyz"]
         for index in range(center_count)]
        for row in rows
    ])
    factors = housing.source(active_points)[:, None, None]
    gradient = volume / len(probes) * np.sum(factors * derivative, axis=0)
    gradient_se = volume / len(probes) * np.sqrt(
        np.sum((factors * derivative_se) ** 2, axis=0)
    )
    return gradient, gradient_se, json.loads(summary_path.read_text())


def length_gradient(controls: Array) -> Array:
    gradient = np.zeros_like(controls)
    for index in range(len(controls) - 1):
        edge = controls[index + 1] - controls[index]
        length = np.linalg.norm(edge)
        if length <= 1e-15:
            continue
        direction = edge / length
        gradient[index] -= direction
        gradient[index + 1] += direction
    return gradient


def projected_direction(controls: Array, gradient: Array) -> Array:
    direction = -gradient.copy()
    direction[1:-1] = (
        0.20 * direction[:-2] + 0.60 * direction[1:-1]
        + 0.20 * direction[2:]
    )
    direction[[0, -1]] = 0.0
    constraint = length_gradient(controls)
    constraint[[0, -1]] = 0.0
    denominator = float(np.sum(constraint * constraint))
    if denominator > 1e-15:
        direction -= constraint * (
            float(np.sum(constraint * direction)) / denominator
        )
    direction[[0, -1]] = 0.0
    maximum = float(np.max(np.linalg.norm(direction, axis=1)))
    if maximum <= 1e-15:
        return direction
    return direction / maximum


def feasible_channel(housing, controls: Array, subdivisions: Array,
                     radius: float, clearance: float) -> bool:
    dense, _ = dense_polyline(controls, subdivisions)
    return (strictly_overlapping_chain(dense, radius)
            and bool(np.all(housing.housing_sdf(dense) < -(radius + clearance))))


def refine_channel(
    housing, executable: Path, mesh: Path, output: Path, probes: Array,
    controls: Array, subdivisions: Array, radius: float, volume: float,
    initial_objective: float, volume_points: Array, box_volume: float,
    total_volume_samples: int, target_volume: float, args: argparse.Namespace,
) -> tuple[Array, float, list[dict[str, object]], float, float]:
    current = controls.copy()
    current_radius = radius
    trajectory: list[dict[str, object]] = []
    total_seconds = 0.0
    peak_memory = 0.0
    step_fraction = args.refine_step_fraction
    line_multiplier = float(student_t.ppf(
        1.0 - args.line_family_alpha / (3.0 * args.refine_iterations),
        df=args.line_repeats - 1,
    ))

    initial_centers, _ = dense_polyline(current, subdivisions)
    values, seconds, memory = objective_values(
        housing, executable, mesh, output, "channel-refine-initial", probes,
        initial_centers, radius, volume, args.line_walks,
        [SEED + 15000 + index for index in range(args.line_repeats)], args.threads,
    )
    total_seconds += seconds
    peak_memory = max(peak_memory, memory)
    trajectory.append({
        "iteration": 0, "accepted_scale": 0.0,
        "objective": float(np.mean(values)),
        "objective_se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
        "reduction_percent": 100.0 * (1.0 - float(np.mean(values)) / initial_objective),
        "radius": current_radius,
        "removed_volume": removed_volume(
            volume_points, box_volume, total_volume_samples,
            initial_centers, current_radius,
        ),
    })

    for iteration in range(1, args.refine_iterations + 1):
        centers, weights = dense_polyline(current, subdivisions)
        sphere_gradient, sphere_gradient_se, diff_summary = run_diffwos(
            housing, executable, mesh, output, f"channel-diff-{iteration:02d}",
            probes, centers, current_radius, volume, args.gradient_walks,
            SEED + 16000 + iteration, args.threads, args.diff_offset,
        )
        total_seconds += float(diff_summary["mesh_build_seconds"]) + float(
            diff_summary["runtime_seconds_median"]
        )
        peak_memory = max(peak_memory, float(diff_summary["peak_rss_megabytes"]))
        control_gradient = weights.T @ sphere_gradient
        control_gradient_se = np.sqrt((weights * weights).T @ (sphere_gradient_se ** 2))
        direction = projected_direction(current, control_gradient)
        if not np.any(direction):
            break

        line_seeds = [
            SEED + 17000 + 100 * iteration + index
            for index in range(args.line_repeats)
        ]
        current_centers, _ = dense_polyline(current, subdivisions)
        current_values, seconds, memory = objective_values(
            housing, executable, mesh, output,
            f"channel-line-{iteration:02d}-scale-0", probes, current_centers,
            current_radius, volume, args.line_walks, line_seeds, args.threads,
        )
        total_seconds += seconds
        peak_memory = max(peak_memory, memory)
        best_controls = current
        best_values = current_values
        best_scale = 0.0
        best_radius = current_radius
        best_difference_upper = 0.0
        reported_scale = 0.0
        reported_difference = 0.0
        reported_difference_se = 0.0
        reported_difference_upper = math.inf
        for scale in (1.0, 0.5, 0.25):
            trial = current + scale * step_fraction * current_radius * direction
            trial_centers, _ = dense_polyline(trial, subdivisions)
            trial_radius = calibrated_radius(
                volume_points, box_volume, total_volume_samples,
                trial_centers, target_volume,
            )
            if not feasible_channel(
                housing, trial, subdivisions, trial_radius,
                args.surface_clearance,
            ):
                continue
            trial_values, seconds, memory = objective_values(
                housing, executable, mesh, output,
                f"channel-line-{iteration:02d}-scale-{scale:g}", probes,
                trial_centers, trial_radius, volume, args.line_walks, line_seeds,
                args.threads,
            )
            total_seconds += seconds
            peak_memory = max(peak_memory, memory)
            differences = trial_values - current_values
            difference_se = float(
                np.std(differences, ddof=1) / math.sqrt(len(differences))
            )
            difference_upper = float(
                np.mean(differences) + line_multiplier * difference_se
            )
            if difference_upper < reported_difference_upper:
                reported_scale = scale
                reported_difference = float(np.mean(differences))
                reported_difference_se = difference_se
                reported_difference_upper = difference_upper
            if difference_upper < best_difference_upper:
                best_controls = trial
                best_values = trial_values
                best_scale = scale
                best_radius = trial_radius
                best_difference_upper = difference_upper

        current = best_controls.copy()
        current_radius = best_radius
        current_centers, _ = dense_polyline(current, subdivisions)
        objective_mean = float(np.mean(best_values))
        trajectory.append({
            "iteration": iteration,
            "accepted_scale": best_scale,
            "objective": objective_mean,
            "objective_se": float(
                np.std(best_values, ddof=1) / math.sqrt(len(best_values))
            ),
            "reduction_percent": 100.0 * (1.0 - objective_mean / initial_objective),
            "gradient_norm": float(np.linalg.norm(control_gradient)),
            "gradient_se_norm": float(np.linalg.norm(control_gradient_se)),
            "tested_scale": reported_scale,
            "paired_change": reported_difference,
            "paired_change_se": reported_difference_se,
            "directional_upper_change": reported_difference_upper,
            "familywise_t_multiplier": line_multiplier,
            "diffwos_seconds": float(diff_summary["runtime_seconds_median"]),
            "diffwos_parameters": int(diff_summary["gradient_parameters"]),
            "radius": current_radius,
            "removed_volume": removed_volume(
                volume_points, box_volume, total_volume_samples,
                current_centers, current_radius,
            ),
        })
        if best_scale == 0.0:
            step_fraction *= 0.5
        if step_fraction < 0.04:
            break
    return current, current_radius, trajectory, total_seconds, peak_memory


def gradient_validation(
    housing, executable: Path, mesh: Path, output: Path, probes: Array,
    controls: Array, subdivisions: Array, radius: float, volume: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    centers, weights = dense_polyline(controls, subdivisions)
    sphere_gradient, _, summary = run_diffwos(
        housing, executable, mesh, output, "channel-gradient-reference", probes,
        centers, radius, volume, args.validation_gradient_walks, SEED + 21000,
        args.threads, args.diff_offset,
    )
    control_gradient = weights.T @ sphere_gradient
    rng = np.random.default_rng(SEED + 22000)
    candidates: list[tuple[float, Array]] = []
    descent = projected_direction(controls, control_gradient)
    descent_norm = float(np.linalg.norm(descent))
    if descent_norm > 1e-15:
        descent /= descent_norm
        candidates.append((abs(float(np.sum(control_gradient * descent))), descent))
    random_candidates: list[tuple[float, Array]] = []
    for _ in range(512):
        random_gradient = rng.normal(size=controls.shape)
        direction = projected_direction(controls, -random_gradient)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-15:
            continue
        direction /= norm
        if not feasible_channel(
            housing, controls + args.validation_step * direction,
            subdivisions, radius, args.surface_clearance,
        ) or not feasible_channel(
            housing, controls - args.validation_step * direction,
            subdivisions, radius, args.surface_clearance,
        ):
            continue
        signal = abs(float(np.sum(control_gradient * direction)))
        random_candidates.append((signal, direction))
    random_candidates.sort(key=lambda item: item[0], reverse=True)
    high_signal_count = max(
        args.validation_directions - 1,
        int(math.ceil(0.30 * len(random_candidates))),
    )
    high_signal = random_candidates[:high_signal_count]
    needed = args.validation_directions - len(candidates)
    if needed > 0:
        indices = np.linspace(0, len(high_signal) - 1, needed).round().astype(int)
        candidates.extend(high_signal[int(index)] for index in indices)

    rows: list[dict[str, float]] = []
    for direction_index, (signal, selected_direction) in enumerate(candidates):
        direction = selected_direction.copy()
        predicted = float(np.sum(control_gradient * direction))
        desired_sign = -1.0 if direction_index % 2 == 0 else 1.0
        if predicted * desired_sign < 0.0:
            direction *= -1.0
            predicted *= -1.0
        estimates = []
        for repeat in range(args.validation_repeats):
            seed = SEED + 23000 + 100 * direction_index + repeat
            pair = []
            for sign in (-1.0, 1.0):
                perturbed = controls + sign * args.validation_step * direction
                perturbed_centers, _ = dense_polyline(perturbed, subdivisions)
                value, _ = housing.objective(
                    executable, mesh, output,
                    f"channel-gradient-fd-{direction_index}-{repeat}-{sign:+g}",
                    probes, perturbed_centers, radius, volume,
                    args.validation_fd_walks, seed, args.threads,
                )
                pair.append(value)
            estimates.append((pair[1] - pair[0]) / (2.0 * args.validation_step))
        measured = float(np.mean(estimates))
        measured_se = float(
            np.std(estimates, ddof=1) / math.sqrt(len(estimates))
        ) if len(estimates) > 1 else 0.0
        rows.append({
            "direction": float(direction_index),
            "diffwos_directional_derivative": predicted,
            "finite_difference_directional_derivative": measured,
            "finite_difference_se": measured_se,
            "absolute_predicted_signal": signal,
        })
    predicted = np.array([row["diffwos_directional_derivative"] for row in rows])
    measured = np.array([row["finite_difference_directional_derivative"] for row in rows])
    correlation = float(np.corrcoef(predicted, measured)[0, 1])
    cosine = float(np.dot(predicted, measured) /
                   max(np.linalg.norm(predicted) * np.linalg.norm(measured), 1e-15))
    slope = float(np.dot(predicted, measured) / max(np.dot(predicted, predicted), 1e-15))
    return rows, {
        "pearson": correlation, "cosine": cosine, "slope_fd_on_diffwos": slope,
        "diffwos_seconds": float(summary["runtime_seconds_median"]),
        "gradient_parameters": float(summary["gradient_parameters"]),
        "stage": "pre-refinement TD graph channel",
        "selection": "descent direction plus seeded top-30%-signal feasible tangents",
        "finite_difference_step": args.validation_step,
    }


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp.csv")
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows[1:] for key in row if key not in fieldnames
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def channel_surface(centers: Array, radius: float) -> tuple[Array, Array]:
    low = centers.min(axis=0) - 1.15 * radius
    high = centers.max(axis=0) + 1.15 * radius
    span = high - low
    longest = 120
    shape = np.maximum(28, np.ceil(longest * span / span.max()).astype(int))
    axes = [np.linspace(low[d], high[d], int(shape[d])) for d in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack((
        xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")
    ))
    values = minimum_sphere_distance(grid, centers) - radius
    vertices, faces, _ = igl.marching_cubes(
        np.ascontiguousarray(values), np.ascontiguousarray(grid),
        int(shape[0]), int(shape[1]), int(shape[2]), 0.0,
    )
    return vertices, faces


def add_channel(ax, housing, centers: Array, controls: Array, radius: float,
                color: str, alpha: float = 0.96) -> None:
    vertices, faces = channel_surface(centers, radius)
    ax.add_collection3d(housing.polygon_collection(
        vertices[faces], color, alpha, rasterized=True
    ))
    ax.plot(controls[:, 0], controls[:, 1], controls[:, 2], "o-",
            color="#183f49", lw=1.1, ms=2.8, alpha=0.88)


def make_figure(
    housing, output: Path, vertices: Array, faces: Array, candidates: Array,
    scores: Array, straight: tuple[Array, Array, float],
    topology: tuple[Array, Array, float], refined: tuple[Array, Array, float],
    trajectory: list[dict[str, object]], validation_rows: list[dict[str, float]],
    methods: dict[str, dict[str, object]], initial_objective: float,
) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.7,
        "axes.titlesize": 10.1, "axes.titleweight": "bold",
        "axes.labelsize": 8.8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.75, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 420,
    })
    render_vertices, render_faces = vertices, faces
    if len(faces) > 18000:
        render_vertices, render_faces, _, _ = igl.decimate(
            np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)),
            18000,
        )
    fig = plt.figure(figsize=(12.4, 7.35), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, left=0.045, right=0.985, bottom=0.105, top=0.955,
        wspace=0.27, hspace=0.34,
    )
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    topology_controls, topology_centers, topology_radius = topology
    refined_controls, refined_centers, refined_radius = refined
    straight_controls, straight_centers, straight_radius = straight

    housing.add_cutaway(ax_a, render_vertices, render_faces)
    visible = candidates[:, 1] >= 0.0
    vmax = float(np.quantile(scores, 0.992))
    scatter = ax_a.scatter(
        candidates[visible, 0], candidates[visible, 1], candidates[visible, 2],
        c=scores[visible], cmap="inferno", norm=Normalize(0.0, vmax), s=5.5,
        alpha=0.72, linewidth=0.0, depthshade=False,
    )
    ax_a.plot(topology_controls[:, 0], topology_controls[:, 1],
              topology_controls[:, 2], "o-", color="#34d3c5", lw=2.2, ms=4.0)
    ax_a.scatter(topology_controls[[0, -1], 0], topology_controls[[0, -1], 1],
                 topology_controls[[0, -1], 2], s=35, c="#f6c85f",
                 edgecolor="#29363b", linewidth=0.6, depthshade=False)
    housing.style_3d(ax_a, render_vertices, cutaway=True)
    ax_a.set_title(r"a   TD-weighted connection graph", pad=-2)
    colorbar = fig.colorbar(scatter, ax=ax_a, fraction=0.033, pad=0.003, shrink=0.68)
    colorbar.ax.tick_params(labelsize=6.7)

    housing.add_cutaway(ax_b, render_vertices, render_faces)
    add_channel(ax_b, housing, topology_centers, topology_controls,
                topology_radius, "#2a9d8f")
    housing.style_3d(ax_b, render_vertices, cutaway=True)
    ax_b.set_title("b   Connected ball-union channel", pad=-2)
    ax_b.text2D(0.5, 0.015, f"{len(topology_centers)} overlapping balls • equal volume",
                transform=ax_b.transAxes, ha="center", color="#657075", fontsize=7.5)

    housing.add_cutaway(ax_c, render_vertices, render_faces)
    add_channel(ax_c, housing, refined_centers, refined_controls,
                refined_radius, "#277da1")
    ax_c.plot(topology_controls[:, 0], topology_controls[:, 1],
              topology_controls[:, 2], "--", color="#d55e57", lw=1.2,
              alpha=0.9, label="before")
    displacement = refined_controls - topology_controls
    ax_c.quiver(
        topology_controls[1:-1, 0], topology_controls[1:-1, 1],
        topology_controls[1:-1, 2], displacement[1:-1, 0],
        displacement[1:-1, 1], displacement[1:-1, 2], color="#f3b65d",
        linewidth=1.5, arrow_length_ratio=0.30,
    )
    housing.style_3d(ax_c, render_vertices, cutaway=True)
    ax_c.set_title("c   Differential WoS refinement", pad=-2)

    refinement_rows = trajectory[1:]
    iterations = np.array([int(row["iteration"]) for row in refinement_rows])
    changes = 1.0e4 * np.array([
        float(row["paired_change"]) for row in refinement_rows
    ])
    change_errors = 1.0e4 * np.array([
        float(row["familywise_t_multiplier"]) * float(row["paired_change_se"])
        for row in refinement_rows
    ])
    accepted = np.array([
        float(row["accepted_scale"]) > 0.0 for row in refinement_rows
    ])
    ax_d.axhline(0.0, color="#6f7d82", ls="--", lw=1.0)
    ax_d.errorbar(iterations[accepted], changes[accepted],
                  yerr=change_errors[accepted], fmt="o", color="#277da1",
                  ecolor="#70a9bf", ms=5.2, capsize=3.0, label="accepted")
    ax_d.errorbar(iterations[~accepted], changes[~accepted],
                  yerr=change_errors[~accepted], fmt="o", color="#d55e57",
                  ecolor="#df9d98", ms=5.2, capsize=3.0, label="rejected")
    ax_d.set(title="d   Fresh-sample acceptance test", xlabel="shape iteration",
             ylabel=r"paired change in $J$ ($\times 10^{-4}$)")
    ax_d.grid(True, color="#d9dde0", lw=0.55)
    ax_d.legend(frameon=False, loc="lower right")

    predicted = np.array([row["diffwos_directional_derivative"]
                          for row in validation_rows])
    measured = np.array([row["finite_difference_directional_derivative"]
                         for row in validation_rows])
    measured_se = np.array([row["finite_difference_se"] for row in validation_rows])
    ax_e.errorbar(predicted, measured, yerr=measured_se, fmt="o", ms=5.0,
                  color="#663f8c", ecolor="#9a86b0", capsize=2.2)
    limits = np.array([predicted.min(), predicted.max(), measured.min(), measured.max()])
    padding = 0.08 * max(limits.max() - limits.min(), 1e-8)
    low, high = limits.min() - padding, limits.max() + padding
    ax_e.plot([low, high], [low, high], color="#69757a", ls="--", lw=1.0)
    ax_e.set_xlim(low, high)
    ax_e.set_ylim(low, high)
    ax_e.set(title="e   Directional-derivative check",
             xlabel="Differential WoS", ylabel="central finite difference")
    ax_e.grid(True, color="#d9dde0", lw=0.55)
    pearson = float(np.corrcoef(predicted, measured)[0, 1])
    slope = float(np.dot(predicted, measured) /
                  max(np.dot(predicted, predicted), 1e-15))
    ax_e.text(0.04, 0.94, rf"$r={pearson:.3f}$   slope $={slope:.2f}$",
              transform=ax_e.transAxes, ha="left", va="top", fontsize=7.8,
              color="#4f3a63")

    names = list(methods)
    colors = ["#8f9ba1", "#e29e45", "#2a9d8f", "#277da1"]
    values = np.array([methods[name]["reduction_percent"] for name in names])
    standard_errors = np.array([methods[name]["reduction_se"] for name in names])
    positions = np.arange(len(names))
    for position, value, standard_error, color in zip(
        positions, values, standard_errors, colors
    ):
        ax_f.errorbar(position, value, yerr=standard_error, fmt="o", ms=7.0,
                      color=color, ecolor=color, capsize=3.0, lw=1.5)
        ax_f.text(position, value + 0.65, f"{value:.1f}%", ha="center",
                  fontsize=7.3)
    labels = ["disjoint\nTWoS", "straight\nchannel", "TD graph\nchannel",
              "TD + DiffWoS"]
    ax_f.set_xticks(positions, labels)
    ax_f.set(title="f   Equal-volume design quality",
             ylabel="compliance reduction (%)")
    ax_f.grid(True, axis="y", color="#d9dde0", lw=0.55)
    ax_f.set_ylim(float(values.min() - 1.2), float(values.max() + 2.0))
    connected_gain = values[3] - values[2]
    ax_f.annotate(
        f"+{connected_gain:.2f} pp", xy=(3, values[3]),
        xytext=(2.42, values[3] + 3.0), ha="center", fontsize=7.3,
        arrowprops={"arrowstyle": "->", "color": "#52646b", "lw": 0.9},
        color="#52646b",
    )

    fig.text(
        0.5, 0.022,
        "two TD-selected terminals • grid-free candidate graph • fixed-volume "
        "connected channel • independent objective streams",
        ha="center", fontsize=8.4, color="#687176",
    )
    pdf = output / "engineering-housing-channels.pdf"
    png = output / "engineering-housing-channels.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=420, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    housing = load_housing_module()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mesh = HERE / "assets" / "stylized-finned-housing.ply"
    vertices, faces, checks = housing.generate_mesh(mesh)
    executable = housing.load_compiled_module().build_cpp()
    probes = housing.probe_points(args.probes)
    volume = housing.volume_estimate()
    baseline_summary = json.loads((HERE / "output" / "summary.json").read_text())
    initial_objective = float(baseline_summary["initial_objective"])
    disconnected_centers = np.asarray(
        baseline_summary["designs"]["TWoS + validation"]["cavities"], dtype=float
    )
    disconnected_radius = float(baseline_summary["problem"]["cavity_radius"])
    target_volume = len(disconnected_centers) * 4.0 * math.pi * disconnected_radius ** 3 / 3.0

    candidates = dense_candidates(
        housing, args.candidates, args.provisional_radius + args.surface_clearance
    )
    score_start = time.perf_counter()
    scores, _, score_summary = housing.run_solver(
        executable, mesh, output, "channel-topological-field", candidates,
        np.empty((0, 3)), args.provisional_radius, "twos", args.scout_walks,
        SEED + 12000, args.threads,
    )
    scoring_seconds = time.perf_counter() - score_start
    first, second = select_terminals(candidates, scores)
    raw_path, graph_stats = topology_weighted_path(
        housing, candidates, scores, first, second,
        args.provisional_radius + args.surface_clearance, args.graph_neighbors,
    )
    topology_controls = resample_polyline(raw_path, args.controls)
    topology_controls = smooth_controls(
        housing, topology_controls,
        args.provisional_radius + args.surface_clearance,
    )
    straight_controls = np.linspace(
        topology_controls[0], topology_controls[-1], args.controls
    )

    volume_points, box_volume, total_volume_samples = volume_samples(
        housing, args.volume_power
    )
    provisional_subdivisions = subdivision_counts(
        topology_controls, args.provisional_radius * 0.72
    )
    provisional_centers, _ = dense_polyline(
        topology_controls, provisional_subdivisions
    )
    topology_radius = calibrated_radius(
        volume_points, box_volume, total_volume_samples, provisional_centers,
        target_volume,
    )
    topology_subdivisions = subdivision_counts(
        topology_controls, topology_radius * 0.72
    )
    topology_centers, _ = dense_polyline(topology_controls, topology_subdivisions)
    topology_radius = calibrated_radius(
        volume_points, box_volume, total_volume_samples, topology_centers,
        target_volume,
    )
    straight_subdivisions = subdivision_counts(straight_controls, topology_radius * 0.72)
    straight_centers, _ = dense_polyline(straight_controls, straight_subdivisions)
    straight_radius = calibrated_radius(
        volume_points, box_volume, total_volume_samples, straight_centers,
        target_volume,
    )

    refined_controls, refined_radius, trajectory, refinement_seconds, refinement_memory = refine_channel(
        housing, executable, mesh, output, probes, topology_controls,
        topology_subdivisions, topology_radius, volume, initial_objective,
        volume_points, box_volume, total_volume_samples, target_volume, args,
    )
    refined_centers, _ = dense_polyline(refined_controls, topology_subdivisions)

    method_geometry = {
        "Disconnected TWoS": (disconnected_centers, disconnected_radius),
        "Straight channel": (straight_centers, straight_radius),
        "TD graph channel": (topology_centers, topology_radius),
        "TD + DiffWoS": (refined_centers, refined_radius),
    }
    methods: dict[str, dict[str, object]] = {}
    method_samples: dict[str, Array] = {}
    final_seeds = [SEED + 25000 + index for index in range(args.final_repeats)]
    for method_index, (name, (centers, radius)) in enumerate(method_geometry.items()):
        values, seconds, memory = objective_values(
            housing, executable, mesh, output, f"channel-final-{method_index}",
            probes, centers, radius, volume, args.final_walks, final_seeds,
            args.threads,
        )
        mean = float(np.mean(values))
        method_samples[name] = values
        standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        methods[name] = {
            "final_objective": mean, "final_objective_se": standard_error,
            "reduction_percent": 100.0 * (1.0 - mean / initial_objective),
            "reduction_se": 100.0 * standard_error / initial_objective,
            "evaluation_seconds": seconds, "peak_rss_megabytes": memory,
            "radius": radius,
            "removed_volume": removed_volume(
                volume_points, box_volume, total_volume_samples, centers, radius
            ),
            "sphere_count": float(len(centers)),
            "objective_samples": values.tolist(),
        }

    comparisons: dict[str, dict[str, float]] = {}
    for label, baseline_name, method_name in (
        ("TD graph vs straight channel", "Straight channel", "TD graph channel"),
        ("DiffWoS refinement vs TD graph", "TD graph channel", "TD + DiffWoS"),
    ):
        differences = method_samples[method_name] - method_samples[baseline_name]
        mean_difference = float(np.mean(differences))
        difference_se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
        baseline_mean = float(np.mean(method_samples[baseline_name]))
        comparisons[label] = {
            "paired_objective_change": mean_difference,
            "paired_objective_change_se": difference_se,
            "relative_objective_improvement_percent":
                -100.0 * mean_difference / baseline_mean,
        }

    validation_rows, validation_summary = gradient_validation(
        housing, executable, mesh, output, probes, topology_controls,
        topology_subdivisions, topology_radius, volume, args,
    )

    control_rows: list[dict[str, object]] = []
    for stage, controls in (("topology", topology_controls),
                            ("refined", refined_controls)):
        for index, point in enumerate(controls):
            control_rows.append({
                "stage": stage, "control": index,
                "x": point[0], "y": point[1], "z": point[2],
            })
    atomic_csv(output / "channel-controls.csv", control_rows)
    sphere_rows: list[dict[str, object]] = []
    for stage, centers, radius in (
        ("straight", straight_centers, straight_radius),
        ("topology", topology_centers, topology_radius),
        ("refined", refined_centers, refined_radius),
    ):
        for index, point in enumerate(centers):
            sphere_rows.append({
                "stage": stage, "sphere": index,
                "x": point[0], "y": point[1], "z": point[2], "radius": radius,
            })
    atomic_csv(output / "channel-spheres.csv", sphere_rows)
    atomic_csv(output / "channel-refinement-trajectory.csv", trajectory)
    atomic_csv(output / "channel-gradient-validation.csv", validation_rows)

    summary = {
        "seed": SEED,
        "geometry": checks,
        "problem": {
            "description": "equal-volume connected cooling channel in the finned housing",
            "target_removed_volume": target_volume,
            "candidate_count": len(candidates), "control_count": len(topology_controls),
            "dense_ball_count": len(topology_centers),
            "segment_subdivisions": topology_subdivisions.tolist(),
            "volume_qmc_samples": total_volume_samples,
        },
        "topology_stage": {
            "terminal_indices": [first, second],
            "terminal_points": [candidates[first].tolist(), candidates[second].tolist()],
            "scoring_seconds": scoring_seconds,
            "solver_seconds": float(score_summary["mesh_build_seconds"]) + float(
                score_summary["runtime_seconds_median"]
            ),
            **graph_stats,
        },
        "refinement": {
            "iterations_requested": args.refine_iterations,
            "iterations_completed": len(trajectory) - 1,
            "gradient_walks_per_probe": args.gradient_walks,
            "normal_offset": args.diff_offset,
            "line_family_alpha": args.line_family_alpha,
            "line_repeats": args.line_repeats,
            "total_seconds": refinement_seconds,
            "peak_rss_megabytes": refinement_memory,
            "trajectory": trajectory,
        },
        "gradient_validation": validation_summary,
        "methods": methods,
        "paired_comparisons": comparisons,
        "scope": (
            "The connected channel is a dense union of overlapping equal-radius balls. "
            "Differential WoS estimates the fixed-radius center derivatives; endpoints "
            "are fixed and directions are projected to preserve centerline length to "
            "first order. Every trial radius is QMC-recalibrated to the target removed "
            "volume. Fresh common-seed objective streams drive backtracking, and final "
            "values use separate streams."
        ),
    }
    housing.atomic_json(output / "channel-summary.json", summary)
    make_figure(
        housing, output, vertices, faces, candidates, scores,
        (straight_controls, straight_centers, straight_radius),
        (topology_controls, topology_centers, topology_radius),
        (refined_controls, refined_centers, refined_radius),
        trajectory, validation_rows, methods, initial_objective,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "output" / "channels"))
    parser.add_argument("--candidates", type=int, default=3072)
    parser.add_argument("--probes", type=int, default=384)
    parser.add_argument("--controls", type=int, default=7)
    parser.add_argument("--provisional-radius", type=float, default=0.095)
    parser.add_argument("--surface-clearance", type=float, default=0.006)
    parser.add_argument("--scout-walks", type=int, default=48)
    parser.add_argument("--graph-neighbors", type=int, default=14)
    parser.add_argument("--volume-power", type=int, default=20)
    parser.add_argument("--refine-iterations", type=int, default=6)
    parser.add_argument("--refine-step-fraction", type=float, default=0.32)
    parser.add_argument("--gradient-walks", type=int, default=4096)
    parser.add_argument("--diff-offset", type=float, default=0.004)
    parser.add_argument("--line-walks", type=int, default=320)
    parser.add_argument("--line-repeats", type=int, default=8)
    parser.add_argument("--line-family-alpha", type=float, default=0.05)
    parser.add_argument("--final-walks", type=int, default=768)
    parser.add_argument("--final-repeats", type=int, default=8)
    parser.add_argument("--validation-gradient-walks", type=int, default=8192)
    parser.add_argument("--validation-directions", type=int, default=6)
    parser.add_argument("--validation-step", type=float, default=0.012)
    parser.add_argument("--validation-fd-walks", type=int, default=4096)
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
