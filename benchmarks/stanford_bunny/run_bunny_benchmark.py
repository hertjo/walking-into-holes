#!/usr/bin/env python3
"""Stanford Bunny curved-mesh stress test for Walking into Holes."""

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

import igl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from scipy.stats import qmc

Array = np.ndarray


@dataclass(frozen=True)
class Config:
    seed: int = 20260912
    cavity_radius: float = 0.050
    outer_clearance: float = 0.012
    cavity_clearance: float = 0.014
    insertions: int = 6
    candidate_count: int = 1024
    scout_walks: int = 12
    shortlist: int = 20
    validation_walks: int = 128
    evaluation_probes: int = 384
    evaluation_walks: int = 32
    field_walks: int = 48
    termination_epsilon: float = 5.0e-4
    max_wos_steps: int = 512
    confidence_z: float = 1.96
    voxel_shape: tuple[int, int, int] = (54, 64, 48)
    rendering_faces: int = 7000
    source_y_slope: float = 0.65


@dataclass
class Selection:
    center: Array
    scout_score: float
    validation_score: float
    validation_se: float
    lower_confidence: float
    feasible_count: int
    mean_walk_steps: float


@dataclass
class BunnyMesh:
    vertices: Array
    faces: Array
    render_vertices: Array
    render_faces: Array
    tree: object
    bounds_min: Array
    bounds_max: Array
    repair_stats: dict[str, object]

    def unsigned_distance(self, points: Array) -> Array:
        squared, _, _ = self.tree.squared_distance(
            self.vertices, self.faces, np.asarray(points, dtype=float)
        )
        return np.sqrt(np.maximum(squared, 0.0))

    def signed_distance(self, points: Array) -> Array:
        return igl.signed_distance(
            np.asarray(points, dtype=float),
            self.vertices,
            self.faces,
            igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        )[0]


def signed_volume(vertices: Array, faces: Array) -> float:
    triangles = vertices[faces]
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        / 6.0
    )


def load_and_repair_mesh(source_path: Path, output_path: Path, config: Config) -> BunnyMesh:
    vertices_raw, faces_raw = igl.read_triangle_mesh(str(source_path))
    vertices, faces, _, _ = igl.remove_unreferenced(vertices_raw, faces_raw)
    faces, _ = igl.bfs_orient(faces)
    loops = igl.boundary_loop_all(faces)
    if len(loops) != 5:
        raise RuntimeError(f"expected five canonical Bunny holes, found {len(loops)}")

    vertices_parts = [vertices]
    faces_parts = [faces]
    next_vertex = len(vertices)
    for loop in loops:
        indices = np.asarray(loop, dtype=np.int64)
        vertices_parts.append(np.mean(vertices[indices], axis=0, keepdims=True))
        faces_parts.append(
            np.column_stack(
                (
                    np.roll(indices, -1),
                    indices,
                    np.full(len(indices), next_vertex, dtype=np.int64),
                )
            )
        )
        next_vertex += 1
    vertices = np.vstack(vertices_parts)
    faces = np.vstack(faces_parts)
    faces, _ = igl.bfs_orient(faces)
    if signed_volume(vertices, faces) < 0.0:
        faces = faces[:, [0, 2, 1]]

    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    height = float(vertices[:, 1].max() - vertices[:, 1].min())
    vertices = (vertices - center) / height
    volume = signed_volume(vertices, faces)
    edge_manifold = bool(igl.is_edge_manifold(faces)[0])
    vertex_manifold = bool(np.all(igl.is_vertex_manifold(faces)))
    remaining_loops = igl.boundary_loop_all(faces)
    euler = int(igl.euler_characteristic(faces))
    if not edge_manifold or not vertex_manifold or remaining_loops or euler != 2:
        raise RuntimeError("Bunny repair did not produce a closed genus-zero manifold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    igl.write_triangle_mesh(str(output_path), vertices, faces)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("failed to write repaired Bunny mesh")

    render_vertices, render_faces, _, _ = igl.decimate(
        np.asfortranarray(vertices),
        np.asfortranarray(faces.astype(np.int32)),
        config.rendering_faces,
    )
    tree = igl.AABB()
    tree.init(vertices, faces)
    stats: dict[str, object] = {
        "source_vertices": int(len(vertices_raw)),
        "source_faces": int(len(faces_raw)),
        "unreferenced_vertices_removed": int(len(vertices_raw) - (len(vertices) - 5)),
        "boundary_loops_capped": int(len(loops)),
        "boundary_loop_lengths": [int(len(loop)) for loop in loops],
        "repaired_vertices": int(len(vertices)),
        "repaired_faces": int(len(faces)),
        "euler_characteristic": euler,
        "edge_manifold": edge_manifold,
        "vertex_manifold": vertex_manifold,
        "normalized_volume": volume,
    }
    return BunnyMesh(
        vertices=vertices,
        faces=faces,
        render_vertices=render_vertices,
        render_faces=render_faces,
        tree=tree,
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
        repair_stats=stats,
    )


def source(points: Array, config: Config) -> Array:
    return 1.0 + config.source_y_slope * (points[:, 1] + 0.5)


def inside_cavities(points: Array, centers: Array, radius: float) -> Array:
    if centers.size == 0:
        return np.zeros(len(points), dtype=bool)
    distances2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.any(distances2 <= radius**2, axis=1)


def boundary_distance(points: Array, centers: Array, mesh: BunnyMesh, config: Config) -> Array:
    distance = mesh.unsigned_distance(points)
    if centers.size:
        sphere_distance = np.min(
            np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            - config.cavity_radius,
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
    mesh: BunnyMesh,
    config: Config,
    rng: np.random.Generator,
    chunk_path_count: int = 50_000,
) -> tuple[Array, float]:
    points = np.asarray(points, dtype=float)
    output = np.empty((len(points), sample_count), dtype=float)
    path_total = len(points) * sample_count
    step_total = 0
    point_chunk = max(1, chunk_path_count // sample_count)
    for begin in range(0, len(points), point_chunk):
        end = min(len(points), begin + point_chunk)
        count = end - begin
        positions = np.repeat(points[begin:end], sample_count, axis=0)
        accumulated = np.zeros(len(positions), dtype=float)
        steps = np.zeros(len(positions), dtype=np.int32)
        for _ in range(config.max_wos_steps):
            radii = boundary_distance(positions, centers, mesh, config)
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
            steps[indices] += 1
        else:
            if np.any(
                boundary_distance(positions, centers, mesh, config)
                > config.termination_epsilon
            ):
                raise RuntimeError("WoS maximum-step limit reached")
        output[begin:end] = accumulated.reshape(count, sample_count)
        step_total += int(steps.sum())
    return output, step_total / max(path_total, 1)


def crossfit_square(samples: Array) -> tuple[Array, Array]:
    n = samples.shape[1]
    sums = samples.sum(axis=1)
    sums2 = np.sum(samples**2, axis=1)
    estimate = (sums**2 - sums2) / (n * (n - 1))
    leave_sums = sums[:, None] - samples
    leave_sums2 = sums2[:, None] - samples**2
    leave = (leave_sums**2 - leave_sums2) / ((n - 1) * (n - 2))
    leave_mean = leave.mean(axis=1, keepdims=True)
    standard_error = np.sqrt(
        (n - 1) / n * np.sum((leave - leave_mean) ** 2, axis=1)
    )
    return estimate, standard_error


def sobol_interior(
    mesh: BunnyMesh,
    count: int,
    clearance: float,
    seed: int,
) -> Array:
    exponent = 14
    while exponent <= 20:
        unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)
        points = qmc.scale(unit, mesh.bounds_min, mesh.bounds_max)
        signed = mesh.signed_distance(points)
        accepted = points[signed <= -clearance]
        if len(accepted) >= count:
            return accepted[:count]
        exponent += 1
    raise RuntimeError(f"could not sample {count} interior Bunny points")


def feasible_mask(candidates: Array, centers: Array, mesh: BunnyMesh, config: Config) -> Array:
    required_outer = config.cavity_radius + config.outer_clearance
    mask = mesh.unsigned_distance(candidates) >= required_outer
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
    mesh: BunnyMesh,
    config: Config,
    rng: np.random.Generator,
    fresh: bool,
) -> Selection:
    feasible = candidates[feasible_mask(candidates, centers, mesh, config)]
    scout_samples, scout_steps = wos_samples(
        feasible, config.scout_walks, centers, mesh, config, rng
    )
    scout, scout_se = crossfit_square(scout_samples)
    if not fresh:
        chosen = int(np.argmax(scout))
        return Selection(
            feasible[chosen], float(scout[chosen]), float(scout[chosen]),
            float(scout_se[chosen]),
            float(scout[chosen] - config.confidence_z * scout_se[chosen]),
            len(feasible), scout_steps,
        )
    shortlist_count = min(config.shortlist, len(feasible))
    shortlist_indices = np.argpartition(scout, -shortlist_count)[-shortlist_count:]
    validation_samples, validation_steps = wos_samples(
        feasible[shortlist_indices], config.validation_walks, centers,
        mesh, config, rng,
    )
    validation, validation_se = crossfit_square(validation_samples)
    lower = validation - config.confidence_z * validation_se
    local = int(np.argmax(lower))
    original = int(shortlist_indices[local])
    return Selection(
        feasible[original], float(scout[original]), float(validation[local]),
        float(validation_se[local]), float(lower[local]), len(feasible),
        0.5 * (scout_steps + validation_steps),
    )


def solve_voxel(
    mesh: BunnyMesh, centers: Array, config: Config
) -> tuple[tuple[Array, Array, Array], Array, dict[str, float]]:
    axes = tuple(
        np.linspace(mesh.bounds_min[i], mesh.bounds_max[i], config.voxel_shape[i])
        for i in range(3)
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    active = (mesh.signed_distance(points) < 0.0).reshape(config.voxel_shape)
    if centers.size:
        active &= ~inside_cavities(points, centers, config.cavity_radius).reshape(
            config.voxel_shape
        )
    node_id = np.full(config.voxel_shape, -1, dtype=np.int64)
    active_flat = np.flatnonzero(active.ravel())
    node_id.ravel()[active_flat] = np.arange(len(active_flat))
    indices = np.column_stack(np.unravel_index(active_flat, config.voxel_shape))
    inverse_h2 = [1.0 / (axis[1] - axis[0]) ** 2 for axis in axes]
    rows = [np.arange(len(active_flat), dtype=np.int64)]
    cols = [np.arange(len(active_flat), dtype=np.int64)]
    data = [np.full(len(active_flat), 2.0 * sum(inverse_h2))]
    for dimension, weight in enumerate(inverse_h2):
        for direction in (-1, 1):
            neighbors = indices.copy()
            neighbors[:, dimension] += direction
            in_bounds = (
                (neighbors[:, dimension] >= 0)
                & (neighbors[:, dimension] < config.voxel_shape[dimension])
            )
            neighbor_ids = np.full(len(indices), -1, dtype=np.int64)
            valid_indices = np.flatnonzero(in_bounds)
            neighbor_ids[valid_indices] = node_id[tuple(neighbors[valid_indices].T)]
            linked = neighbor_ids >= 0
            rows.append(np.arange(len(active_flat), dtype=np.int64)[linked])
            cols.append(neighbor_ids[linked])
            data.append(np.full(np.count_nonzero(linked), -weight))
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(active_flat), len(active_flat)),
    ).tocsr()
    rhs = source(points[active_flat], config)
    started = time.perf_counter()
    solution, info = cg(matrix, rhs, rtol=1.0e-9, atol=0.0, maxiter=5000)
    elapsed = time.perf_counter() - started
    if info != 0:
        raise RuntimeError(f"voxel solve failed with info={info}")
    field = np.zeros(config.voxel_shape, dtype=float)
    field.ravel()[active_flat] = solution
    return axes, field, {
        "active_nodes": float(len(active_flat)),
        "matrix_megabytes": float(
            (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
            / 2**20
        ),
        "solve_seconds": elapsed,
    }


def select_voxel(
    candidates: Array, centers: Array, mesh: BunnyMesh, config: Config
) -> tuple[Selection, dict[str, float]]:
    axes, field, stats = solve_voxel(mesh, centers, config)
    feasible = candidates[feasible_mask(candidates, centers, mesh, config)]
    score = RegularGridInterpolator(
        axes, field, bounds_error=False, fill_value=0.0
    )(feasible) ** 2
    chosen = int(np.argmax(score))
    return Selection(
        feasible[chosen], float(score[chosen]), float(score[chosen]), 0.0,
        float(score[chosen]), len(feasible), 0.0,
    ), stats


def select_random(
    candidates: Array,
    centers: Array,
    mesh: BunnyMesh,
    config: Config,
    rng: np.random.Generator,
) -> Selection:
    feasible = candidates[feasible_mask(candidates, centers, mesh, config)]
    chosen = int(rng.integers(len(feasible)))
    return Selection(
        feasible[chosen], math.nan, math.nan, math.nan, math.nan,
        len(feasible), 0.0,
    )


def estimate_objective(
    probes: Array,
    centers: Array,
    mesh: BunnyMesh,
    config: Config,
    seed: int,
) -> tuple[float, float, float]:
    outside = ~inside_cavities(probes, centers, config.cavity_radius)
    values = np.zeros(len(probes), dtype=float)
    samples, steps = wos_samples(
        probes[outside], config.evaluation_walks, centers, mesh, config,
        np.random.default_rng(seed),
    )
    values[outside] = source(probes[outside], config) * samples.mean(axis=1)
    return (
        float(values.mean()),
        float(values.std(ddof=1) / math.sqrt(len(values))),
        steps,
    )


def run_design(
    name: str,
    selector: Callable[[Array], tuple[Selection, dict[str, float]]],
    probes: Array,
    mesh: BunnyMesh,
    config: Config,
) -> tuple[list[dict[str, float | int | str]], Array, list[dict[str, float]]]:
    centers = np.empty((0, 3), dtype=float)
    rows: list[dict[str, float | int | str]] = []
    solver_stats: list[dict[str, float]] = []
    for iteration in range(config.insertions + 1):
        objective, objective_se, evaluation_steps = estimate_objective(
            probes, centers, mesh, config, config.seed + 100 + iteration
        )
        row: dict[str, float | int | str] = {
            "method": name, "iteration": iteration, "objective": objective,
            "objective_se": objective_se, "cavity_count": len(centers),
            "evaluation_mean_steps": evaluation_steps,
            "selected_x": math.nan, "selected_y": math.nan, "selected_z": math.nan,
            "scout_score": math.nan, "validation_score": math.nan,
            "validation_se": math.nan, "lower_confidence": math.nan,
            "feasible_count": 0, "selection_mean_steps": math.nan,
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


def field_agreement(
    candidates: Array, mesh: BunnyMesh, config: Config
) -> tuple[float, float, Array]:
    samples, _ = wos_samples(
        candidates, config.field_walks, np.empty((0, 3)), mesh, config,
        np.random.default_rng(config.seed + 300),
    )
    twos = crossfit_square(samples)[0]
    axes, voxel, _ = solve_voxel(mesh, np.empty((0, 3)), config)
    voxel_score = RegularGridInterpolator(axes, voxel)(candidates) ** 2
    correlation = float(np.corrcoef(twos, voxel_score)[0, 1])
    relative_rmse = float(
        np.sqrt(np.mean((twos - voxel_score) ** 2))
        / max(np.sqrt(np.mean(voxel_score**2)), 1.0e-12)
    )
    return correlation, relative_rmse, twos


def evaluate_slice(
    mesh: BunnyMesh, centers: Array, config: Config
) -> tuple[Array, Array, Array]:
    x = np.linspace(mesh.bounds_min[0], mesh.bounds_max[0], 92)
    y = np.linspace(mesh.bounds_min[1], mesh.bounds_max[1], 112)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    inside = mesh.signed_distance(points) < 0.0
    outside_cavities = ~inside_cavities(points, centers, config.cavity_radius)
    active = inside & outside_cavities
    temperature = np.full(len(points), np.nan)
    samples, _ = wos_samples(
        points[active], config.field_walks, centers, mesh, config,
        np.random.default_rng(config.seed + 400),
    )
    temperature[active] = samples.mean(axis=1)
    return x, y, temperature.reshape(xx.shape)


def display_vertices(vertices: Array) -> Array:
    return vertices[:, [0, 2, 1]]


def surface_collection(mesh: BunnyMesh, alpha: float, color: str) -> Poly3DCollection:
    vertices = display_vertices(mesh.render_vertices)
    triangles = vertices[mesh.render_faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    light = np.array([-0.45, -0.70, 0.65])
    light /= np.linalg.norm(light)
    intensity = 0.55 + 0.45 * np.clip(normals @ light, 0.0, 1.0)
    base = np.asarray(mpl.colors.to_rgb(color))
    facecolors = np.clip(intensity[:, None] * base[None, :], 0.0, 1.0)
    collection = Poly3DCollection(
        triangles, facecolors=facecolors, edgecolors="none", alpha=alpha,
        rasterized=True,
    )
    return collection


def style_bunny_axis(ax: plt.Axes, mesh: BunnyMesh) -> None:
    vertices = display_vertices(mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    middle = 0.5 * (low + high)
    radius = 0.53 * max(high - low)
    ax.set_xlim(middle[0] - radius, middle[0] + radius)
    ax.set_ylim(middle[1] - radius, middle[1] + radius)
    ax.set_zlim(middle[2] - radius, middle[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-68)
    ax.set_axis_off()


def draw_spheres(ax: plt.Axes, centers: Array, radius: float) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 16)
    colors = mpl.colormaps["winter"](np.linspace(0.18, 0.92, max(len(centers), 1)))
    for index, physical_center in enumerate(centers):
        center = physical_center[[0, 2, 1]]
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=colors[index], linewidth=0, shade=True)


def make_figure(
    output_dir: Path,
    mesh: BunnyMesh,
    config: Config,
    candidates: Array,
    initial_scores: Array,
    rows: list[dict[str, float | int | str]],
    centers_by_method: dict[str, Array],
    final_slice: tuple[Array, Array, Array],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9.0,
            "axes.titlesize": 11.0, "axes.titleweight": "bold",
            "axes.labelsize": 9.5, "legend.fontsize": 8.0,
            "axes.linewidth": 0.8, "pdf.fonttype": 42,
        }
    )
    colors = {
        "TWoS + fresh validation": "#168a83",
        "TWoS + reused scout": "#d85b53",
        "Voxel adjoint TD": "#6676c8",
        "Random feasible": "#9ca3a8",
    }
    fig = plt.figure(figsize=(12.2, 7.1))
    grid = fig.add_gridspec(
        2, 3, left=0.04, right=0.985, bottom=0.12, top=0.95,
        wspace=0.25, hspace=0.30,
    )
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    ax_a.add_collection3d(surface_collection(mesh, 1.0, "#c7d0d3"))
    style_bunny_axis(ax_a, mesh)
    ax_a.set_title("a   Canonical mesh, repaired", pad=-2)
    ax_a.text2D(
        0.5, 0.05,
        f"{len(mesh.vertices):,} vertices • {len(mesh.faces):,} faces • 5 caps",
        transform=ax_a.transAxes, ha="center", fontsize=8, color="#5e666b",
    )

    ax_b.add_collection3d(surface_collection(mesh, 0.12, "#9eb4b8"))
    shown = slice(None, None, 2)
    display_candidates = display_vertices(candidates[shown])
    scatter = ax_b.scatter(
        display_candidates[:, 0], display_candidates[:, 1], display_candidates[:, 2],
        c=initial_scores[shown], cmap="magma", s=8, alpha=0.92,
        depthshade=False,
    )
    style_bunny_axis(ax_b, mesh)
    ax_b.set_title(r"b   Interior direct score $-D_T=u^2$", pad=-2)
    fig.colorbar(scatter, ax=ax_b, fraction=0.033, pad=-0.01, shrink=0.72)

    ours = centers_by_method["TWoS + fresh validation"]
    ax_c.add_collection3d(surface_collection(mesh, 0.16, "#9eb4b8"))
    draw_spheres(ax_c, ours, config.cavity_radius)
    style_bunny_axis(ax_c, mesh)
    ax_c.set_title("c   Grid-free cavity insertion", pad=-2)

    methods = list(colors)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        iterations = np.array([int(row["iteration"]) for row in method_rows])
        objectives = np.array([float(row["objective"]) for row in method_rows])
        errors = np.array([float(row["objective_se"]) for row in method_rows])
        ax_d.plot(
            iterations, objectives / objectives[0], marker="o", ms=3.7,
            lw=1.8, color=colors[method], label=method,
        )
        ax_d.fill_between(
            iterations, (objectives - errors) / objectives[0],
            (objectives + errors) / objectives[0], color=colors[method],
            alpha=0.10, linewidth=0,
        )
    ax_d.set_title("d   Fresh-path compliance")
    ax_d.set(xlabel="inserted cavities", ylabel="normalized compliance")
    ax_d.set_xticks(range(config.insertions + 1))
    ax_d.grid(color="#d7dadd", lw=0.6)
    ax_d.legend(frameon=False)

    improvements = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        first = float(method_rows[0]["objective"])
        last = float(method_rows[-1]["objective"])
        improvements.append(100.0 * (first - last) / first)
    bars = ax_e.barh(
        np.arange(len(methods)), improvements,
        color=[colors[method] for method in methods], height=0.62,
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
            value + 0.25, bar.get_y() + 0.5 * bar.get_height(), f"{value:.1f}%",
            va="center", fontsize=8, fontweight="bold",
        )
    ax_e.set_xlim(0.0, max(improvements) * 1.20)

    x, y, temperature = final_slice
    image = ax_f.imshow(
        temperature, origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
        cmap="inferno", interpolation="bilinear", aspect="equal",
    )
    finite = np.isfinite(temperature)
    if np.any(finite):
        levels = np.linspace(np.nanmin(temperature), np.nanmax(temperature), 8)[1:-1]
        ax_f.contour(x, y, temperature, levels=levels, colors="white", alpha=0.32, linewidths=0.55)
    for center in ours:
        if abs(center[2]) < config.cavity_radius:
            section_radius = math.sqrt(config.cavity_radius**2 - center[2] ** 2)
            ax_f.add_patch(
                plt.Circle(
                    (center[0], center[1]), section_radius, fill=False,
                    color="#22c4bd", lw=1.4,
                )
            )
    ax_f.set_title(r"f   Final central slice $u$")
    ax_f.set(xlabel="$x$", ylabel="$y$")
    fig.colorbar(image, ax=ax_f, fraction=0.046, pad=0.035)

    fig.text(
        0.5, 0.024,
        f"Stanford Bunny • {config.candidate_count:,} interior candidates • "
        f"{config.scout_walks} scout / {config.validation_walks} fresh walks • "
        f"radius {config.cavity_radius:.3f}",
        ha="center", color="#737b80", fontsize=9,
    )
    fig.savefig(output_dir / "stanford-bunny-benchmark.pdf", bbox_inches="tight")
    temporary = output_dir / "stanford-bunny-benchmark.tmp.png"
    fig.savefig(temporary, dpi=210, bbox_inches="tight")
    temporary.replace(output_dir / "stanford-bunny-benchmark.png")
    plt.close(fig)


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source_mesh: Path, output_dir: Path, config: Config) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh = load_and_repair_mesh(
        source_mesh, output_dir / "stanford-bunny-watertight.ply", config
    )
    candidates = sobol_interior(
        mesh, config.candidate_count,
        config.cavity_radius + config.outer_clearance,
        config.seed + 1,
    )
    probes = sobol_interior(mesh, config.evaluation_probes, 0.0, config.seed + 2)
    fresh_rng = np.random.default_rng(config.seed + 10)
    reuse_rng = np.random.default_rng(config.seed + 20)
    random_rng = np.random.default_rng(config.seed + 30)
    selectors: dict[str, Callable[[Array], tuple[Selection, dict[str, float]]]] = {
        "TWoS + fresh validation": lambda centers: (
            select_twos(candidates, centers, mesh, config, fresh_rng, True), {}
        ),
        "TWoS + reused scout": lambda centers: (
            select_twos(candidates, centers, mesh, config, reuse_rng, False), {}
        ),
        "Voxel adjoint TD": lambda centers: select_voxel(
            candidates, centers, mesh, config
        ),
        "Random feasible": lambda centers: (
            select_random(candidates, centers, mesh, config, random_rng), {}
        ),
    }
    all_rows: list[dict[str, float | int | str]] = []
    centers_by_method: dict[str, Array] = {}
    voxel_stats: list[dict[str, float]] = []
    runtimes: dict[str, float] = {}
    for name, selector in selectors.items():
        method_started = time.perf_counter()
        rows, centers, stats = run_design(name, selector, probes, mesh, config)
        runtimes[name] = time.perf_counter() - method_started
        all_rows.extend(rows)
        centers_by_method[name] = centers
        if name == "Voxel adjoint TD":
            voxel_stats = stats

    correlation, relative_rmse, initial_scores = field_agreement(
        candidates, mesh, config
    )
    final_slice = evaluate_slice(
        mesh, centers_by_method["TWoS + fresh validation"], config
    )
    make_figure(
        output_dir, mesh, config, candidates, initial_scores, all_rows,
        centers_by_method, final_slice,
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

    method_summary: dict[str, dict[str, float]] = {}
    for method in selectors:
        method_rows = [row for row in all_rows if row["method"] == method]
        initial = float(method_rows[0]["objective"])
        final = float(method_rows[-1]["objective"])
        method_summary[method] = {
            "initial_objective": initial,
            "final_objective": final,
            "final_objective_se": float(method_rows[-1]["objective_se"]),
            "relative_reduction_percent": 100.0 * (initial - final) / initial,
            "python_runtime_seconds": runtimes[method],
        }
    summary: dict[str, object] = {
        "scope": "curved-mesh geometry/application stress test; not final equal-time result",
        "configuration": asdict(config),
        "mesh_repair": mesh.repair_stats,
        "initial_field_agreement": {
            "twos_voxel_pearson": correlation,
            "relative_rmse": relative_rmse,
        },
        "methods": method_summary,
        "voxel_baseline": {
            "mean_active_nodes": float(np.mean([s["active_nodes"] for s in voxel_stats])),
            "mean_matrix_megabytes": float(np.mean([s["matrix_megabytes"] for s in voxel_stats])),
            "mean_solve_seconds": float(np.mean([s["solve_seconds"] for s in voxel_stats])),
        },
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "matplotlib": mpl.__version__,
            "libigl": "2.6.3",
        },
        "data_attribution": "Stanford University Computer Graphics Laboratory",
    }
    if correlation < 0.90:
        raise AssertionError("TWoS and voxel fields disagree unexpectedly")
    if not (
        method_summary["TWoS + fresh validation"]["final_objective"]
        < method_summary["Random feasible"]["final_objective"]
    ):
        raise AssertionError("validated TWoS did not beat random placement")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    benchmark_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-mesh", type=Path,
        default=benchmark_root / "assets" / "bun_zipper.ply",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=benchmark_root / "output"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.source_mesh, args.output_dir, Config()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
