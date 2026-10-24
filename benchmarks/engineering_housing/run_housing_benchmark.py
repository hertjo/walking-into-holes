#!/usr/bin/env python3
"""Compiled topology-design benchmark on a procedural finned housing."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import igl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.stats import qmc

Array = np.ndarray
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
COMPILED = ROOT / "benchmarks" / "compiled_bunny"
SEED = 20260912


def load_compiled_module():
    path = COMPILED / "run_compiled_benchmark.py"
    spec = importlib.util.spec_from_file_location("compiled_backend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rounded_box_sdf(points: Array, center: Array, half: Array, radius: float) -> Array:
    q = np.abs(points - center) - (half - radius)
    return (
        np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        + np.minimum(np.max(q, axis=-1), 0.0)
        - radius
    )


def capped_cylinder_z_sdf(
    points: Array, center: Array, radius: float, half_height: float
) -> Array:
    q = np.column_stack((
        np.linalg.norm(points[:, :2] - center[:2], axis=1) - radius,
        np.abs(points[:, 2] - center[2]) - half_height,
    ))
    return (
        np.linalg.norm(np.maximum(q, 0.0), axis=1)
        + np.minimum(np.max(q, axis=1), 0.0)
    )


def housing_sdf(points: Array) -> Array:
    points = np.asarray(points).reshape(-1, 3)
    fields = [rounded_box_sdf(
        points, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.58, 0.28]), 0.09
    )]
    for x_position in np.linspace(-0.82, 0.82, 9):
        fields.append(rounded_box_sdf(
            points, np.array([x_position, 0.0, 0.43]),
            np.array([0.046, 0.49, 0.27]), 0.025,
        ))
    for x_position in (-0.98, 0.98):
        for y_position in (-0.57, 0.57):
            fields.append(capped_cylinder_z_sdf(
                points, np.array([x_position, y_position, -0.13]), 0.155, 0.17
            ))
    return np.min(np.vstack(fields), axis=0)


def source(points: Array) -> Array:
    points = np.asarray(points).reshape(-1, 3)
    chip_a = np.exp(
        -((points[:, 0] + 0.48) / 0.24) ** 2
        -((points[:, 1] + 0.18) / 0.22) ** 2
        -((points[:, 2] + 0.08) / 0.16) ** 2
    )
    chip_b = np.exp(
        -((points[:, 0] - 0.43) / 0.21) ** 2
        -((points[:, 1] - 0.19) / 0.20) ** 2
        -((points[:, 2] + 0.07) / 0.15) ** 2
    )
    regulator = np.exp(
        -((points[:, 0] - 0.02) / 0.30) ** 2
        -((points[:, 1] + 0.31) / 0.16) ** 2
        -((points[:, 2] + 0.02) / 0.19) ** 2
    )
    return 0.08 + 5.2 * chip_a + 4.6 * chip_b + 2.0 * regulator


def write_ascii_ply(path: Path, vertices: Array, faces: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.ply")
    with temporary.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(handle, vertices, fmt="%.9g")
        np.savetxt(handle, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d")
    temporary.replace(path)


def mesh_checks(vertices: Array, faces: Array) -> dict[str, object]:
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    component_count, _ = igl.facet_components(faces)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "edges": int(len(unique_edges)),
        "euler_characteristic": int(len(vertices) - len(unique_edges) + len(faces)),
        "edge_manifold": bool(np.all(counts == 2)),
        "vertex_manifold": bool(np.all(igl.is_vertex_manifold(faces))),
        "connected_components": int(component_count),
    }


def generate_mesh(path: Path) -> tuple[Array, Array, dict[str, object]]:
    shape = (116, 78, 72)
    axes = (
        np.linspace(-1.20, 1.20, shape[0]),
        np.linspace(-0.77, 0.77, shape[1]),
        np.linspace(-0.40, 0.77, shape[2]),
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack((
        xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")
    ))
    values = housing_sdf(grid)
    vertices, faces, _ = igl.marching_cubes(
        np.ascontiguousarray(values), np.ascontiguousarray(grid), *shape, 0.0
    )
    checks = mesh_checks(vertices, faces)
    if not checks["edge_manifold"] or not checks["vertex_manifold"]:
        raise RuntimeError(f"generated housing is not manifold: {checks}")
    if checks["connected_components"] != 1:
        raise RuntimeError(f"generated housing is disconnected: {checks}")
    write_ascii_ply(path, vertices, faces)
    return vertices, faces, checks


def sobol_box(count_power: int, low: Array, high: Array, seed: int) -> Array:
    samples = qmc.Sobol(3, scramble=True, seed=seed).random_base2(count_power)
    return low + samples * (high - low)


def candidate_points(count: int) -> Array:
    samples = sobol_box(
        13, np.array([-0.84, -0.41, -0.135]),
        np.array([0.84, 0.41, 0.135]), SEED + 10,
    )
    feasible = samples[housing_sdf(samples) < -0.118]
    if len(feasible) < count:
        raise RuntimeError(f"only {len(feasible)} cavity candidates survived")
    return feasible[:count]


def probe_points(count: int) -> Array:
    samples = sobol_box(
        16, np.array([-1.20, -0.77, -0.40]),
        np.array([1.20, 0.77, 0.77]), SEED + 20,
    )
    interior = samples[housing_sdf(samples) < -0.004]
    if len(interior) < count:
        raise RuntimeError("insufficient interior probes")
    return interior[:count]


def volume_estimate() -> float:
    low = np.array([-1.20, -0.77, -0.40])
    high = np.array([1.20, 0.77, 0.77])
    samples = sobol_box(18, low, high, SEED + 30)
    return float(np.prod(high - low) * np.mean(housing_sdf(samples) < 0.0))


def write_points(path: Path, points: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(temporary, points, delimiter=",", fmt="%.17g")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_solver(
    executable: Path,
    mesh: Path,
    output_dir: Path,
    tag: str,
    points: Array,
    cavities: Array,
    radius: float,
    mode: str,
    walks: int,
    seed: int,
    threads: int,
    grid: tuple[int, int, int] = (96, 64, 54),
    mg_tolerance: float = 1.0e-6,
) -> tuple[Array, Array | None, dict[str, object]]:
    points_path = output_dir / f"points-{tag}.csv"
    field_path = output_dir / f"field-{tag}.csv"
    summary_path = output_dir / f"summary-{tag}.json"
    write_points(points_path, points)
    command = [
        str(executable), "--mode", mode, "--mesh", str(mesh),
        "--points", str(points_path), "--output", str(field_path),
        "--summary", str(summary_path), "--source-mode", "housing",
        "--walks", str(walks), "--seed", str(seed), "--threads", str(threads),
        "--repeats", "1", "--epsilon", "0.001", "--max-steps", "512",
    ]
    if len(cavities):
        cavity_path = output_dir / f"cavities-{tag}.csv"
        write_points(cavity_path, cavities)
        command += ["--cavities", str(cavity_path), "--cavity-radius", str(radius)]
    if mode in ("voxel", "mg"):
        command += ["--grid", ",".join(map(str, grid))]
    if mode == "mg":
        command += ["--mg-tolerance", str(mg_tolerance)]
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    })
    subprocess.run(command, check=True, env=environment)
    rows = read_rows(field_path)
    scores = np.array([float(row["score"]) for row in rows])
    state = None if mode in ("voxel", "mg") else np.array(
        [float(row["u"]) for row in rows]
    )
    return scores, state, json.loads(summary_path.read_text())


def feasible_candidates(candidates: Array, cavities: Array, radius: float) -> Array:
    if not len(cavities):
        return np.arange(len(candidates))
    distances = np.linalg.norm(
        candidates[:, None, :] - cavities[None, :, :], axis=2
    )
    return np.flatnonzero(np.all(distances >= 2.0 * radius + 0.025, axis=1))


def objective(
    executable: Path,
    mesh: Path,
    output_dir: Path,
    tag: str,
    probes: Array,
    cavities: Array,
    radius: float,
    volume: float,
    walks: int,
    seed: int,
    threads: int,
) -> tuple[float, dict[str, object]]:
    active = np.ones(len(probes), dtype=bool)
    if len(cavities):
        distances = np.linalg.norm(
            probes[:, None, :] - cavities[None, :, :], axis=2
        )
        active = np.all(distances > radius + 0.0015, axis=1)
    _, state, summary = run_solver(
        executable, mesh, output_dir, tag, probes[active], cavities, radius,
        "twos", walks, seed, threads,
    )
    assert state is not None
    integrand = np.zeros(len(probes))
    integrand[active] = source(probes[active]) * state
    return volume * float(np.mean(integrand)), summary


def display_vertices(vertices: Array) -> Array:
    return vertices


def polygon_collection(
    polygons: list[Array] | Array, color: str, alpha: float,
    rasterized: bool = True,
) -> Poly3DCollection:
    shown = [display_vertices(np.asarray(polygon)) for polygon in polygons]
    normals = np.array([
        np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        for polygon in shown
    ])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.35, -0.75, 0.72])
    light /= np.linalg.norm(light)
    intensity = 0.33 + 0.67 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray(mpl.colors.to_rgb(color))
    colors = np.clip(intensity[:, None] * base[None, :], 0.0, 1.0)
    return Poly3DCollection(
        shown, facecolors=colors, edgecolors="none", alpha=alpha,
        rasterized=rasterized,
    )


def clip_triangle(triangle: Array, cut_y: float) -> list[Array]:
    polygon = [vertex.copy() for vertex in triangle]
    clipped: list[Array] = []
    for current, following in zip(polygon, polygon[1:] + polygon[:1]):
        current_inside = current[1] >= cut_y
        following_inside = following[1] >= cut_y
        if current_inside:
            clipped.append(current)
        if current_inside != following_inside:
            weight = (cut_y - current[1]) / (following[1] - current[1])
            clipped.append(current + weight * (following - current))
    return clipped if len(clipped) >= 3 else []


def section_segments(vertices: Array, faces: Array, cut_y: float) -> list[Array]:
    segments: list[Array] = []
    for triangle in vertices[faces]:
        hits: list[Array] = []
        for first, second in zip(triangle, np.roll(triangle, -1, axis=0)):
            first_side = first[1] - cut_y
            second_side = second[1] - cut_y
            if first_side * second_side < 0.0:
                weight = -first_side / (second_side - first_side)
                hits.append(first + weight * (second - first))
        if len(hits) == 2:
            segments.append(np.vstack(hits))
    return segments


def add_cutaway(ax, vertices: Array, faces: Array, cut_y: float = 0.0) -> None:
    ax.add_collection3d(polygon_collection(vertices[faces], "#b8c3c7", 0.032))
    rear = [
        polygon for triangle in vertices[faces]
        if (polygon := clip_triangle(triangle, cut_y))
    ]
    ax.add_collection3d(polygon_collection(rear, "#7f959d", 0.58))
    rim = section_segments(vertices, faces, cut_y)
    if rim:
        ax.add_collection3d(Line3DCollection(
            rim, colors="#0f7376", linewidths=0.85, alpha=0.94,
            rasterized=True,
        ))


def style_3d(ax, vertices: Array, cutaway: bool = False) -> None:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    middle = 0.5 * (low + high)
    span = high - low
    ax.set_xlim(low[0] - 0.03, high[0] + 0.03)
    ax.set_ylim(low[1] - 0.03, high[1] + 0.03)
    ax.set_zlim(low[2] - 0.03, high[2] + 0.03)
    ax.set_box_aspect(span)
    ax.view_init(elev=20, azim=-58 if not cutaway else -50)
    ax.set_axis_off()


def add_cavity_spheres(ax, cavities: Array, radius: float) -> None:
    latitude = np.linspace(0.0, np.pi, 18)
    longitude = np.linspace(0.0, 2.0 * np.pi, 32)
    colors = mpl.colormaps["viridis"](np.linspace(0.22, 0.88, max(len(cavities), 1)))
    for index, (center, color) in enumerate(zip(cavities, colors), start=1):
        x = center[0] + radius * np.outer(np.sin(latitude), np.cos(longitude))
        y = center[1] + radius * np.outer(np.sin(latitude), np.sin(longitude))
        z = center[2] + radius * np.outer(np.cos(latitude), np.ones_like(longitude))
        if center[1] >= 0.0:
            ax.plot_surface(x, y, z, color=color, alpha=0.92, linewidth=0,
                            antialiased=True, shade=True)
        else:
            ax.plot_wireframe(x, y, z, color=color, alpha=0.52,
                              linewidth=0.45, rstride=2, cstride=3)
        ax.text(center[0], center[1], center[2] + 1.25 * radius, str(index),
                ha="center", va="bottom", fontsize=6.8, color="#273238")


def make_figure(
    output_dir: Path,
    vertices: Array,
    faces: Array,
    candidates: Array,
    visual_scores: Array,
    slice_points: Array,
    slice_state: Array,
    slice_grid: tuple[Array, Array],
    designs: dict[str, dict[str, object]],
    initial_objective: float,
    radius: float,
    threads: int,
) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.7,
        "axes.titlesize": 10.2, "axes.titleweight": "bold",
        "axes.labelsize": 9.0, "legend.fontsize": 7.7,
        "axes.linewidth": 0.75, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 420,
    })
    colors = {
        "TWoS + validation": "#008f88",
        "TWoS equal-time": "#005f73",
        "geometric multigrid": "#2878b5",
        "voxel FD/CG": "#686ecf",
        "random feasible": "#df7a3e",
    }
    render_vertices, render_faces = vertices, faces
    if len(faces) > 18000:
        render_vertices, render_faces, _, _ = igl.decimate(
            np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)),
            18000,
        )

    fig = plt.figure(figsize=(12.4, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, left=0.045, right=0.985, bottom=0.105, top=0.955,
        wspace=0.25, hspace=0.32,
    )
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    ax_d = fig.add_subplot(grid[1, 0], projection="3d")
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    ax_a.add_collection3d(polygon_collection(
        render_vertices[render_faces], "#8299a1", 0.96
    ))
    style_3d(ax_a, render_vertices)
    ax_a.set_title("a   Finned conduction housing", pad=-2)
    ax_a.text2D(
        0.5, 0.02, f"closed procedural mesh • {len(faces):,} triangles",
        transform=ax_a.transAxes, ha="center", color="#657075", fontsize=7.8,
    )

    sx, sz = slice_grid
    slice_image = np.full(sx.shape, np.nan)
    slice_image.ravel()[np.flatnonzero(housing_sdf(np.column_stack((
        sx.ravel(), np.zeros(sx.size), sz.ravel()
    ))) < -0.003)] = slice_state
    levels = np.linspace(0.0, np.nanquantile(slice_image, 0.99), 18)
    temperature = ax_b.contourf(sx, sz, slice_image, levels=levels,
                                cmap="inferno", extend="max")
    ax_b.contour(sx, sz, housing_sdf(np.column_stack((
        sx.ravel(), np.zeros(sx.size), sz.ravel()
    ))).reshape(sx.shape), levels=[0.0], colors="#2f454c", linewidths=1.05)
    ax_b.scatter([-0.48, 0.43], [-0.08, -0.07], marker="x", s=26,
                 color="#5ee2d5", linewidth=1.2, zorder=3)
    ax_b.set_title("b   Initial temperature • central cut")
    ax_b.set(xlabel="$x$", ylabel="$z$")
    ax_b.set_aspect("equal")
    ax_b.grid(False)
    colorbar_b = fig.colorbar(temperature, ax=ax_b, fraction=0.046, pad=0.02)
    colorbar_b.ax.tick_params(labelsize=7)

    add_cutaway(ax_c, render_vertices, render_faces)
    visible = candidates[:, 1] >= 0.0
    vmax = float(np.quantile(visual_scores, 0.985))
    score_scatter = ax_c.scatter(
        candidates[visible, 0], candidates[visible, 1], candidates[visible, 2],
        c=visual_scores[visible], cmap="inferno", norm=Normalize(0.0, vmax),
        s=9.5, alpha=0.97, edgecolor="#251b2c", linewidth=0.12,
        depthshade=False,
    )
    style_3d(ax_c, render_vertices, cutaway=True)
    ax_c.set_title(r"c   Grid-free score $-D_T=u^2$", pad=-2)
    colorbar_c = fig.colorbar(score_scatter, ax=ax_c, fraction=0.035,
                              pad=0.005, shrink=0.70)
    colorbar_c.ax.tick_params(labelsize=7)

    final_cavities = np.asarray(designs["TWoS + validation"]["cavities"])
    add_cutaway(ax_d, render_vertices, render_faces)
    add_cavity_spheres(ax_d, final_cavities, radius)
    style_3d(ax_d, render_vertices, cutaway=True)
    ax_d.set_title("d   Validated grid-free design", pad=-2)
    ax_d.text2D(
        0.5, 0.02, "solid rear cavities • wireframe cavities in removed half",
        transform=ax_d.transAxes, ha="center", color="#657075", fontsize=7.5,
    )

    display_names = (
        "TWoS + validation", "TWoS equal-time",
        "geometric multigrid", "random feasible",
    )
    steps = np.arange(len(designs["TWoS + validation"]["objectives"]))
    for name in display_names:
        design = designs[name]
        remaining = 100.0 * np.asarray(design["objectives"]) / initial_objective
        ax_e.plot(steps, remaining, "o-", color=colors[name], lw=1.9,
                  ms=4.6, label=name)
    ax_e.set_title("e   Independently evaluated design")
    ax_e.set(xlabel="inserted cavities", ylabel="compliance (% of initial)")
    ax_e.set_xticks(steps)
    ax_e.grid(True, color="#d9dde0", lw=0.55)
    ax_e.legend(frameon=False)

    names = list(display_names)
    reductions = np.array([
        100.0 * (1.0 - float(designs[name]["aggregate_final_mean"]) /
                 initial_objective)
        for name in names
    ])
    errors = np.array([
        100.0 * float(designs[name]["aggregate_final_se"]) /
        initial_objective
        for name in names
    ])
    bars = ax_f.bar(
        np.arange(len(names)), reductions, yerr=errors,
        color=[colors[name] for name in names], width=0.68, capsize=3,
        edgecolor="white", linewidth=0.8,
    )
    for bar, name in zip(bars, names):
        seconds = float(designs[name]["selection_seconds_mean"])
        ax_f.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + errors[names.index(name)] + 1.0,
            f"{bar.get_height():.1f}%\n{seconds:.2f} s select",
            ha="center", va="bottom", fontsize=7.2, color="#354147",
        )
    ax_f.set_title("f   Final reduction • independent designs")
    ax_f.set(ylabel="compliance reduction (%)")
    tick_labels = {
        "TWoS + validation": "TWoS +\nvalidation",
        "TWoS equal-time": "TWoS\nequal-time",
        "geometric multigrid": "geometric\nmultigrid",
        "voxel FD/CG": "voxel\nFD/CG",
        "random feasible": "random\nfeasible",
    }
    ax_f.set_xticks(np.arange(len(names)), [tick_labels[name] for name in names])
    ax_f.grid(True, axis="y", color="#d9dde0", lw=0.55)
    ax_f.set_ylim(0.0, max(reductions + errors) * 1.23)

    fig.text(
        0.5, 0.022,
        f"stylized engineering geometry • localized volumetric heating • "
        f"six radius-{radius:.3f} cavities • {threads} CPU threads",
        ha="center", fontsize=8.5, color="#687176",
    )
    pdf = output_dir / "engineering-housing-benchmark.pdf"
    png = output_dir / "engineering-housing-benchmark.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=360, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def make_tradeoff_figure(
    output_dir: Path,
    sweep_rows: list[dict[str, object]],
    designs: dict[str, dict[str, object]],
    initial_objective: float,
) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 420,
    })
    colors = {
        "TWoS + validation": "#008f88",
        "TWoS equal-time": "#005f73",
        "geometric multigrid": "#2878b5",
        "voxel FD/CG": "#686ecf",
    }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.35, 3.05))
    ordered = sorted(sweep_rows, key=lambda row: int(row["grid_cells"]))
    cells = np.array([float(row["grid_cells"]) / 1000.0 for row in ordered])
    errors = np.array([float(row["relative_rmse"]) for row in ordered])
    ax_a.plot(cells, errors, color="#99a9ae", lw=1.25, zorder=1)
    for row, cell_value, error_value in zip(ordered, cells, errors):
        selected = bool(row["selected"])
        ax_a.scatter(
            cell_value, error_value, s=74 if selected else 35,
            marker="*" if selected else "o",
            color="#008f88" if selected else "#667b84",
            edgecolor="white", linewidth=0.65, zorder=3,
        )
        label = str(row["grid"]).replace("x", r"$\!\times\!$")
        label_offsets = {
            "93x61x47": (4, 4),
            "105x69x53": (4, 5),
            "113x73x57": (5, -15),
            "121x77x61": (5, 14),
            "133x85x67": (-5, -13),
        }
        ax_a.annotate(
            label, (cell_value, error_value),
            xytext=label_offsets[str(row["grid"])],
            textcoords="offset points", fontsize=6.5, color="#45545a",
            ha="right" if str(row["grid"]) == "133x85x67" else "left",
        )
    ax_a.axhline(0.09, color="#d56d4b", ls="--", lw=1.0)
    ax_a.text(
        0.98, 0.09, "0.09 accuracy target", transform=ax_a.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=6.7, color="#b75638",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6,
              "alpha": 0.92},
    )
    ax_a.set(
        title="a   Predeclared multigrid tuning sweep",
        xlabel="fine-grid cells (thousands)",
        ylabel="relative RMSE vs. 512-walk reference",
    )
    ax_a.grid(True, color="#dce1e3", lw=0.5)

    shown = (
        "TWoS + validation", "TWoS equal-time",
        "geometric multigrid", "voxel FD/CG",
    )
    for name in shown:
        design = designs[name]
        seconds = float(design["selection_seconds_mean"])
        reduction = 100.0 * (
            1.0 - float(design["aggregate_final_mean"]) / initial_objective
        )
        reduction_se = 100.0 * float(
            design["aggregate_final_se"]
        ) / initial_objective
        memory = float(design["peak_rss_megabytes"])
        ax_b.errorbar(
            seconds, reduction, yerr=reduction_se, fmt="none",
            ecolor=colors[name], elinewidth=1.1, capsize=2.5, zorder=2,
        )
        ax_b.scatter(
            seconds, reduction, s=22.0 + 1.15 * memory,
            color=colors[name], edgecolor="white", linewidth=0.75, zorder=3,
        )
        short = {
            "TWoS + validation": "TWoS sparse",
            "TWoS equal-time": "TWoS equal-time",
            "geometric multigrid": "multigrid",
            "voxel FD/CG": "FD/CG",
        }[name]
        offsets = {
            "TWoS + validation": (5, -12),
            "TWoS equal-time": (-63, -13),
            "geometric multigrid": (-55, -5),
            "voxel FD/CG": (5, -25),
        }[name]
        ax_b.annotate(
            f"{short}\n{memory:.0f} MiB", (seconds, reduction),
            xytext=offsets, textcoords="offset points", fontsize=6.8,
            color="#354147",
        )
    ax_b.set(
        title="b   Quality-time-memory tradeoff",
        xlabel="six-round selection time (s)",
        ylabel="compliance reduction (%)",
    )
    ax_b.grid(True, color="#dce1e3", lw=0.5)
    ax_b.set_xlim(0.25, 2.10)
    ax_b.set_ylim(67.9, 70.62)
    fig.tight_layout(w_pad=2.0)
    pdf = output_dir / "engineering-housing-tradeoffs.pdf"
    png = output_dir / "engineering-housing-tradeoffs.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=420, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = HERE / "assets" / "stylized-finned-housing.ply"
    vertices, faces, checks = generate_mesh(mesh_path)
    candidates = candidate_points(args.candidates)
    probes = probe_points(args.probes)
    volume = volume_estimate()
    compiled = load_compiled_module()
    executable = compiled.build_cpp()
    radius = args.radius

    initial_objective, initial_eval = objective(
        executable, mesh_path, output, "objective-initial", probes,
        np.empty((0, 3)), radius, volume, args.objective_walks,
        SEED + 100, args.threads,
    )

    method_names = (
        "TWoS + validation", "TWoS equal-time", "geometric multigrid",
        "voxel FD/CG", "random feasible",
    )
    designs: dict[str, dict[str, object]] = {
        name: {
            "cavities": [], "objectives": [initial_objective],
            "selection_seconds": 0.0, "peak_rss_megabytes": 0.0,
        }
        for name in method_names
    }
    random_generator = np.random.default_rng(SEED + 200)
    first_voxel_scores: Array | None = None
    first_multigrid_scores: Array | None = None
    first_multigrid_summary: dict[str, object] | None = None

    for step in range(1, args.holes + 1):
        for method_index, name in enumerate(method_names):
            design = designs[name]
            cavities = np.asarray(design["cavities"], dtype=float).reshape(-1, 3)
            indices = feasible_candidates(candidates, cavities, radius)
            if len(indices) == 0:
                raise RuntimeError(f"no feasible candidates for {name} at step {step}")
            available = candidates[indices]
            if name in ("TWoS + validation", "TWoS equal-time"):
                scout_walks = (
                    args.scout_walks if name == "TWoS + validation"
                    else args.equal_scout_walks
                )
                validation_walks = (
                    args.validation_walks if name == "TWoS + validation"
                    else args.equal_validation_walks
                )
                tag_prefix = "twos" if name == "TWoS + validation" else "twos-equal"
                scout_scores, _, scout_summary = run_solver(
                    executable, mesh_path, output,
                    f"{tag_prefix}-scout-{step:02d}", available, cavities,
                    radius, "twos", scout_walks,
                    SEED + 1000 + step, args.threads,
                )
                shortlist_local = np.argsort(scout_scores)[-min(args.shortlist, len(indices)):]
                shortlist = available[shortlist_local]
                validation_scores, _, validation_summary = run_solver(
                    executable, mesh_path, output,
                    f"{tag_prefix}-validate-{step:02d}", shortlist, cavities,
                    radius, "twos", validation_walks,
                    SEED + 2000 + step, args.threads,
                )
                selected = shortlist[int(np.argmax(validation_scores))]
                selection_seconds = sum(
                    float(summary["mesh_build_seconds"])
                    + float(summary["runtime_seconds_median"])
                    for summary in (scout_summary, validation_summary)
                )
                peak_memory = max(float(scout_summary["peak_rss_megabytes"]),
                                  float(validation_summary["peak_rss_megabytes"]))
            elif name == "geometric multigrid":
                multigrid_scores, _, multigrid_summary = run_solver(
                    executable, mesh_path, output,
                    f"multigrid-select-{step:02d}", available, cavities,
                    radius, "mg", 16, SEED + 3000 + step, args.threads,
                    grid=tuple(args.mg_grid), mg_tolerance=args.mg_tolerance,
                )
                if step == 1:
                    first_multigrid_scores = multigrid_scores.copy()
                    first_multigrid_summary = multigrid_summary.copy()
                selected = available[int(np.argmax(multigrid_scores))]
                selection_seconds = float(multigrid_summary["total_seconds"])
                peak_memory = float(multigrid_summary["peak_rss_megabytes"])
            elif name == "voxel FD/CG":
                voxel_scores, _, voxel_summary = run_solver(
                    executable, mesh_path, output, f"voxel-select-{step:02d}",
                    available, cavities, radius, "voxel", 16,
                    SEED + 3000 + step, args.threads,
                )
                if step == 1:
                    first_voxel_scores = voxel_scores.copy()
                selected = available[int(np.argmax(voxel_scores))]
                selection_seconds = float(voxel_summary["total_seconds"])
                peak_memory = float(voxel_summary["peak_rss_megabytes"])
            else:
                selected = available[random_generator.integers(len(available))]
                selection_seconds = 0.0
                peak_memory = 0.0

            design["cavities"].append(selected.tolist())
            design["selection_seconds"] = float(design["selection_seconds"]) + selection_seconds
            design["peak_rss_megabytes"] = max(
                float(design["peak_rss_megabytes"]), peak_memory
            )
            updated = np.asarray(design["cavities"], dtype=float)
            measured, _ = objective(
                executable, mesh_path, output,
                f"objective-{method_index}-{step:02d}", probes, updated,
                radius, volume, args.objective_walks,
                SEED + 4000 + 100 * method_index + step, args.threads,
            )
            design["objectives"].append(measured)

    final_repeats = args.final_repeats
    for method_index, name in enumerate(method_names):
        design = designs[name]
        cavities = np.asarray(design["cavities"], dtype=float)
        reevaluations = []
        for repeat in range(final_repeats):
            measured, _ = objective(
                executable, mesh_path, output,
                f"final-{method_index}-{repeat}", probes, cavities, radius,
                volume, args.final_walks,
                SEED + 7000 + 100 * method_index + repeat, args.threads,
            )
            reevaluations.append(measured)
        design["final_reevaluations"] = reevaluations
        design["final_mean"] = float(np.mean(reevaluations))
        design["final_se"] = float(np.std(reevaluations, ddof=1) / math.sqrt(final_repeats))

    stochastic_names = (
        "TWoS + validation", "TWoS equal-time", "random feasible",
    )
    for name, design in designs.items():
        design["design_replicates"] = [{
            "replicate": 0,
            "seed": SEED,
            "cavities": design["cavities"],
            "selection_seconds": design["selection_seconds"],
            "final_mean": design["final_mean"],
            "final_se": design["final_se"],
        }]

    for replicate in range(1, args.design_repeats):
        for name in stochastic_names:
            method_index = method_names.index(name)
            replicate_seed = SEED + 100000 * replicate
            if name == "random feasible":
                replicate_seed += 10000 * method_index
            cavities_list: list[list[float]] = []
            selection_seconds = 0.0
            peak_memory = 0.0
            random_replicate = np.random.default_rng(replicate_seed + 3000)
            for step in range(1, args.holes + 1):
                cavities = np.asarray(cavities_list, dtype=float).reshape(-1, 3)
                indices = feasible_candidates(candidates, cavities, radius)
                available = candidates[indices]
                if name == "random feasible":
                    selected = available[random_replicate.integers(len(available))]
                else:
                    scout_walks = (
                        args.scout_walks if name == "TWoS + validation"
                        else args.equal_scout_walks
                    )
                    validation_walks = (
                        args.validation_walks if name == "TWoS + validation"
                        else args.equal_validation_walks
                    )
                    tag_prefix = (
                        "twos" if name == "TWoS + validation" else "twos-equal"
                    )
                    scout_scores, _, scout_summary = run_solver(
                        executable, mesh_path, output,
                        f"rep-{replicate}-{tag_prefix}-scout-{step:02d}",
                        available, cavities, radius, "twos", scout_walks,
                        replicate_seed + 1000 + step, args.threads,
                    )
                    shortlist_local = np.argsort(scout_scores)[
                        -min(args.shortlist, len(indices)):
                    ]
                    shortlist = available[shortlist_local]
                    validation_scores, _, validation_summary = run_solver(
                        executable, mesh_path, output,
                        f"rep-{replicate}-{tag_prefix}-validate-{step:02d}",
                        shortlist, cavities, radius, "twos", validation_walks,
                        replicate_seed + 2000 + step, args.threads,
                    )
                    selected = shortlist[int(np.argmax(validation_scores))]
                    selection_seconds += sum(
                        float(summary["mesh_build_seconds"]) +
                        float(summary["runtime_seconds_median"])
                        for summary in (scout_summary, validation_summary)
                    )
                    peak_memory = max(
                        peak_memory,
                        float(scout_summary["peak_rss_megabytes"]),
                        float(validation_summary["peak_rss_megabytes"]),
                    )
                cavities_list.append(selected.tolist())

            final_values = []
            cavities = np.asarray(cavities_list, dtype=float)
            for evaluation in range(args.design_eval_repeats):
                measured, _ = objective(
                    executable, mesh_path, output,
                    f"rep-{replicate}-final-{method_index}-{evaluation}",
                    probes, cavities, radius, volume, args.final_walks,
                    replicate_seed + 5000 + evaluation, args.threads,
                )
                final_values.append(measured)
            replicate_record = {
                "replicate": replicate,
                "seed": replicate_seed,
                "cavities": cavities_list,
                "selection_seconds": selection_seconds,
                "peak_rss_megabytes": peak_memory,
                "final_reevaluations": final_values,
                "final_mean": float(np.mean(final_values)),
                "final_se": float(np.std(final_values, ddof=1) /
                                  math.sqrt(len(final_values))),
            }
            designs[name]["design_replicates"].append(replicate_record)

    replicate_rows: list[dict[str, object]] = []
    for name, design in designs.items():
        replicates = design["design_replicates"]
        replicate_means = np.array(
            [float(record["final_mean"]) for record in replicates]
        )
        selection_times = np.array(
            [float(record["selection_seconds"]) for record in replicates]
        )
        if name in stochastic_names:
            design["aggregate_final_mean"] = float(np.mean(replicate_means))
            design["aggregate_final_se"] = float(
                np.std(replicate_means, ddof=1) / math.sqrt(len(replicate_means))
            )
        else:
            design["aggregate_final_mean"] = design["final_mean"]
            design["aggregate_final_se"] = design["final_se"]
        design["selection_seconds_mean"] = float(np.mean(selection_times))
        design["selection_seconds_se"] = (
            float(np.std(selection_times, ddof=1) /
                  math.sqrt(len(selection_times)))
            if len(selection_times) > 1 else 0.0
        )
        for record in replicates:
            replicate_rows.append({
                "method": name,
                "replicate": record["replicate"],
                "seed": record["seed"],
                "selection_seconds": record["selection_seconds"],
                "final_objective": record["final_mean"],
                "compliance_reduction_percent": 100.0 * (
                    1.0 - float(record["final_mean"]) / initial_objective
                ),
            })
    replicate_path = output / "design-replicates.csv"
    with replicate_path.with_suffix(".tmp.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replicate_rows[0]))
        writer.writeheader()
        writer.writerows(replicate_rows)
    replicate_path.with_suffix(".tmp.csv").replace(replicate_path)

    visual_scores, _, visual_summary = run_solver(
        executable, mesh_path, output, "visual-score", candidates,
        np.empty((0, 3)), radius, "twos", 128, SEED + 8100, args.threads,
    )
    reference_scores, _, reference_summary = run_solver(
        executable, mesh_path, output, "reference-score", candidates,
        np.empty((0, 3)), radius, "twos", 512, SEED + 8200, args.threads,
    )
    if (first_voxel_scores is None or first_multigrid_scores is None or
            first_multigrid_summary is None):
        raise RuntimeError("initial deterministic fields were not retained")

    def relative_rmse(values: Array, reference: Array) -> float:
        return float(np.sqrt(np.mean((values - reference) ** 2)) /
                     np.sqrt(np.mean(reference ** 2)))

    sweep_grids = [
        (93, 61, 47), (105, 69, 53), tuple(args.mg_grid),
        (121, 77, 61), (133, 85, 67),
    ]
    sweep_rows: list[dict[str, object]] = []
    for grid in sweep_grids:
        grid_tag = "x".join(map(str, grid))
        summaries: list[dict[str, object]] = []
        grid_scores: Array | None = None
        for repeat in range(args.mg_sweep_repeats):
            if grid == tuple(args.mg_grid) and repeat == 0:
                scores = first_multigrid_scores
                solver_summary = first_multigrid_summary
            else:
                scores, _, solver_summary = run_solver(
                    executable, mesh_path, output,
                    f"multigrid-sweep-{grid_tag}-{repeat}", candidates,
                    np.empty((0, 3)), radius, "mg", 16,
                    SEED + 9000 + repeat, args.threads, grid=grid,
                    mg_tolerance=args.mg_tolerance,
                )
            if grid_scores is None:
                grid_scores = scores.copy()
            summaries.append(solver_summary)
        assert grid_scores is not None
        sweep_rows.append({
            "grid": grid_tag,
            "grid_cells": int(summaries[0]["grid_cells"]),
            "active_nodes": int(summaries[0]["active_nodes"]),
            "levels": int(summaries[0]["levels"]),
            "relative_rmse": relative_rmse(grid_scores, reference_scores),
            "pearson": float(np.corrcoef(grid_scores, reference_scores)[0, 1]),
            "median_seconds": float(np.median([
                float(item["total_seconds"]) for item in summaries
            ])),
            "peak_rss_megabytes": float(max(
                float(item["peak_rss_megabytes"]) for item in summaries
            )),
            "median_v_cycles": float(np.median([
                int(item["v_cycles"]) for item in summaries
            ])),
            "selected": grid == tuple(args.mg_grid),
        })
    eligible = [
        row for row in sorted(sweep_rows, key=lambda row: int(row["grid_cells"]))
        if float(row["relative_rmse"]) <= args.mg_target_rrmse
    ]
    if not eligible or eligible[0]["grid"] != "x".join(map(str, args.mg_grid)):
        raise RuntimeError(
            "configured multigrid grid is not the smallest sweep grid meeting "
            "the predeclared relative-RMSE target"
        )
    sweep_path = output / "multigrid-resolution-sweep.csv"
    with sweep_path.with_suffix(".tmp.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    sweep_path.with_suffix(".tmp.csv").replace(sweep_path)

    slice_x = np.linspace(-1.14, 1.14, 72)
    slice_z = np.linspace(-0.34, 0.73, 56)
    sx, sz = np.meshgrid(slice_x, slice_z)
    all_slice = np.column_stack((sx.ravel(), np.zeros(sx.size), sz.ravel()))
    slice_mask = housing_sdf(all_slice) < -0.003
    slice_points = all_slice[slice_mask]
    _, slice_state, _ = run_solver(
        executable, mesh_path, output, "slice-initial", slice_points,
        np.empty((0, 3)), radius, "twos", 128, SEED + 8300, args.threads,
    )
    assert slice_state is not None

    trajectory_rows: list[dict[str, object]] = []
    for name, design in designs.items():
        for step, objective_value in enumerate(design["objectives"]):
            row: dict[str, object] = {
                "method": name, "step": step, "objective": objective_value,
            }
            if step:
                center = design["cavities"][step - 1]
                row.update({"x": center[0], "y": center[1], "z": center[2]})
            trajectory_rows.append(row)
    trajectory_path = output / "design-trajectories.csv"
    with trajectory_path.with_suffix(".tmp.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "step", "objective", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(trajectory_rows)
    trajectory_path.with_suffix(".tmp.csv").replace(trajectory_path)

    summary = {
        "seed": SEED,
        "geometry": {
            "description": "procedural stylized finned conduction housing",
            **checks, "volume_estimate": volume,
        },
        "problem": {
            "pde": "-Laplacian(u)=f with zero Dirichlet exterior and cavity boundaries",
            "objective": "integral f*u over remaining material",
            "cavity_radius": radius, "cavities": args.holes,
            "candidates": len(candidates), "objective_probes": len(probes),
            "stochastic_design_repeats": args.design_repeats,
            "fresh_evaluations_per_additional_design": args.design_eval_repeats,
            "fresh_evaluations_canonical_and_deterministic": args.final_repeats,
        },
        "initial_objective": initial_objective,
        "initial_objective_solver": initial_eval,
        "field_comparison": {
            "twos_128_relative_rmse_vs_512_walk_reference": relative_rmse(
                visual_scores, reference_scores
            ),
            "voxel_relative_rmse_vs_512_walk_reference": relative_rmse(
                first_voxel_scores, reference_scores
            ),
            "multigrid_relative_rmse_vs_512_walk_reference": relative_rmse(
                first_multigrid_scores, reference_scores
            ),
            "twos_voxel_pearson": float(np.corrcoef(
                visual_scores, first_voxel_scores
            )[0, 1]),
            "twos_multigrid_pearson": float(np.corrcoef(
                visual_scores, first_multigrid_scores
            )[0, 1]),
            "twos_summary": visual_summary,
            "reference_summary": reference_summary,
        },
        "multigrid_tuning": {
            "selection_rule": (
                "smallest tested grid with relative RMSE at most "
                f"{args.mg_target_rrmse} versus the independent 512-walk field"
            ),
            "repeats_per_grid": args.mg_sweep_repeats,
            "rows": sweep_rows,
        },
        "designs": designs,
        "scope": (
            "Measured CPU study on a license-free procedural engineering geometry; "
            "includes matrix-free geometric multigrid and sparse FD/CG, but not "
            "industrial FEM, BEM, or GPU topology-optimization systems."
        ),
    }
    atomic_json(output / "summary.json", summary)
    make_figure(
        output, vertices, faces, candidates, visual_scores, slice_points,
        slice_state, (sx, sz), designs, initial_objective, radius, args.threads,
    )
    make_tradeoff_figure(output, sweep_rows, designs, initial_objective)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "output"))
    parser.add_argument("--candidates", type=int, default=768)
    parser.add_argument("--probes", type=int, default=384)
    parser.add_argument("--holes", type=int, default=6)
    parser.add_argument("--radius", type=float, default=0.095)
    parser.add_argument("--scout-walks", type=int, default=12)
    parser.add_argument("--shortlist", type=int, default=24)
    parser.add_argument("--validation-walks", type=int, default=192)
    parser.add_argument("--equal-scout-walks", type=int, default=96)
    parser.add_argument("--equal-validation-walks", type=int, default=768)
    parser.add_argument("--objective-walks", type=int, default=160)
    parser.add_argument("--final-walks", type=int, default=384)
    parser.add_argument("--final-repeats", type=int, default=10)
    parser.add_argument("--design-repeats", type=int, default=5)
    parser.add_argument("--design-eval-repeats", type=int, default=5)
    parser.add_argument("--mg-grid", type=int, nargs=3, default=(113, 73, 57))
    parser.add_argument("--mg-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--mg-target-rrmse", type=float, default=0.09)
    parser.add_argument("--mg-sweep-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
