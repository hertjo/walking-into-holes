#!/usr/bin/env python3
"""Measured C++/Embree scaling study for the Stanford Bunny TWoS field."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import igl
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from scipy.stats import qmc

Array = np.ndarray
ROOT = Path(__file__).resolve().parents[2]
BUNNY_DIR = ROOT / "benchmarks" / "stanford_bunny"
HERE = Path(__file__).resolve().parent


def load_reference_module():
    path = BUNNY_DIR / "run_bunny_benchmark.py"
    spec = importlib.util.spec_from_file_location("bunny_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_xyz(path: Path, points: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(temporary, points, delimiter=",", fmt="%.17g")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def embree_cmake_dir() -> Path:
    explicit = os.environ.get("EMBREE_DIR")
    if explicit:
        return Path(explicit)
    roots = {
        Path(sysconfig.get_paths()["purelib"]),
        Path(site.getusersitepackages()),
    }
    candidates = sorted(
        candidate
        for root in roots
        for candidate in (root / "lib64" / "cmake").glob("embree-*")
    )
    if not candidates:
        raise RuntimeError(
            "Embree 4 CMake package not found. Install embreex==4.4.0 or set EMBREE_DIR."
        )
    return candidates[-1]


def cmake_binary() -> str:
    found = shutil.which("cmake")
    if found:
        return found
    candidates = (
        Path(sysconfig.get_paths()["scripts"]) / "cmake",
        Path(site.getuserbase()) / "bin" / "cmake",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("CMake not found. Install cmake==4.1.0.")


def build_cpp() -> Path:
    # Keep compiler products outside the synced manuscript tree. Some hosted
    # workspaces normalize generated binaries while a benchmark is still running.
    build = Path(tempfile.gettempdir()) / "walking-into-holes-embree-build"
    executable = build / "wih_twos"
    configure = [
        cmake_binary(), "-S", str(HERE), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", f"-Dembree_DIR={embree_cmake_dir()}",
    ]
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake_binary(), "--build", str(build), "--parallel"], check=True
    )
    if not executable.is_file():
        raise RuntimeError("C++ build completed without wih_twos executable")
    return executable


def run_cpp(
    executable: Path,
    mesh: Path,
    points: Path,
    output: Path,
    summary: Path,
    walks: int,
    seed: int,
    repeats: int,
    threads: int,
    trace: Path | None = None,
    trace_sites: int = 0,
) -> dict[str, object]:
    command = [
        str(executable), "--mesh", str(mesh), "--points", str(points),
        "--output", str(output), "--summary", str(summary),
        "--walks", str(walks), "--seed", str(seed),
        "--repeats", str(repeats), "--threads", str(threads),
        "--epsilon", "0.0005", "--max-steps", "512",
        "--source-y-slope", "0.65",
    ]
    if trace is not None:
        command += ["--trace-output", str(trace), "--trace-sites", str(trace_sites)]
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    subprocess.run(command, check=True, env=environment)
    return json.loads(summary.read_text())


def run_cpp_voxel(
    executable: Path,
    mesh: Path,
    points: Path,
    output: Path,
    summary_path: Path,
    shape: tuple[int, int, int],
    repeats: int,
    threads: int,
) -> dict[str, object]:
    command = [
        str(executable), "--mode", "voxel", "--mesh", str(mesh),
        "--points", str(points), "--grid", ",".join(map(str, shape)),
        "--output", str(output), "--summary", str(summary_path),
        "--threads", str(threads),
    ]
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
    })
    trials: list[dict[str, object]] = []
    for _ in range(repeats):
        subprocess.run(command, check=True, env=environment)
        trials.append(json.loads(summary_path.read_text()))
    summary = dict(trials[0])
    summary["timings_seconds"] = [float(trial["total_seconds"]) for trial in trials]
    summary["total_seconds"] = float(np.median(summary["timings_seconds"]))
    summary["peak_rss_megabytes"] = float(max(
        float(trial["peak_rss_megabytes"]) for trial in trials
    ))
    atomic_json(summary_path, summary)
    return summary


def voxel_solve(mesh_path: Path, points_path: Path, shape: tuple[int, int, int]):
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    points = np.loadtxt(points_path, delimiter=",").reshape(-1, 3)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    axes = tuple(
        np.linspace(bounds_min[i], bounds_max[i], shape[i]) for i in range(3)
    )
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    total_start = time.perf_counter()
    start = time.perf_counter()
    signed = igl.signed_distance(
        grid_points, vertices, faces,
        igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
    )[0]
    active = (signed < 0.0).reshape(shape)
    classification_seconds = time.perf_counter() - start

    start = time.perf_counter()
    node_id = np.full(shape, -1, dtype=np.int64)
    active_flat = np.flatnonzero(active.ravel())
    node_id.ravel()[active_flat] = np.arange(len(active_flat))
    indices = np.column_stack(np.unravel_index(active_flat, shape))
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
                & (neighbors[:, dimension] < shape[dimension])
            )
            neighbor_ids = np.full(len(indices), -1, dtype=np.int64)
            valid = np.flatnonzero(in_bounds)
            neighbor_ids[valid] = node_id[tuple(neighbors[valid].T)]
            linked = neighbor_ids >= 0
            rows.append(np.arange(len(active_flat), dtype=np.int64)[linked])
            cols.append(neighbor_ids[linked])
            data.append(np.full(np.count_nonzero(linked), -weight))
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(active_flat), len(active_flat)),
    ).tocsr()
    rhs = 1.0 + 0.65 * (grid_points[active_flat, 1] + 0.5)
    assembly_seconds = time.perf_counter() - start

    start = time.perf_counter()
    solution, info = cg(matrix, rhs, rtol=1.0e-9, atol=0.0, maxiter=8000)
    solve_seconds = time.perf_counter() - start
    if info != 0:
        raise RuntimeError(f"voxel CG failed for {shape}, info={info}")
    field = np.zeros(shape, dtype=float)
    field.ravel()[active_flat] = solution

    start = time.perf_counter()
    values = RegularGridInterpolator(
        axes, field, bounds_error=False, fill_value=0.0
    )(points)
    scores = values**2
    interpolation_seconds = time.perf_counter() - start
    total_seconds = time.perf_counter() - total_start
    matrix_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    array_bytes = (
        grid_points.nbytes + signed.nbytes + active.nbytes + node_id.nbytes
        + indices.nbytes + field.nbytes + rhs.nbytes + solution.nbytes
    )
    return scores, {
        "shape": list(shape),
        "grid_cells": int(np.prod(shape)),
        "active_nodes": int(len(active_flat)),
        "classification_seconds": classification_seconds,
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
        "interpolation_seconds": interpolation_seconds,
        "total_seconds": total_seconds,
        "matrix_megabytes": matrix_bytes / 2**20,
        "accounted_working_set_megabytes": (matrix_bytes + array_bytes) / 2**20,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "max_spacing": float(max(axis[1] - axis[0] for axis in axes)),
    }


def voxel_worker(args: argparse.Namespace) -> None:
    shape = tuple(int(value) for value in args.shape.split(","))
    scores, summary = voxel_solve(Path(args.mesh), Path(args.points), shape)
    rows = [{"site": i, "score": float(value)} for i, value in enumerate(scores)]
    write_csv(Path(args.field_output), rows)
    atomic_json(Path(args.summary), summary)


def decimated_meshes(vertices: Array, faces: Array, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for target in (1000, 4000, 16000):
        out = output_dir / f"bunny-{target:05d}-faces.ply"
        if not out.is_file():
            dec_v, dec_f, _, _ = igl.decimate(
                np.asfortranarray(vertices),
                np.asfortranarray(faces.astype(np.int32)), target,
            )
            igl.write_triangle_mesh(str(out), dec_v, dec_f)
        paths.append(out)
    full = output_dir / "bunny-69674-faces.ply"
    if not full.is_file():
        igl.write_triangle_mesh(str(full), vertices, faces)
    paths.append(full)
    return paths


def relative_rmse(estimate: Array, reference: Array) -> float:
    return float(
        np.sqrt(np.mean((estimate - reference) ** 2))
        / np.sqrt(np.mean(reference**2))
    )


def display_vertices(vertices: Array) -> Array:
    return vertices[:, [0, 2, 1]]


def polygon_collection(
    polygons: list[Array] | Array,
    base_color: str,
    alpha: float,
    rasterized: bool = True,
):
    shown = [display_vertices(np.asarray(polygon)) for polygon in polygons]
    normals = np.array([
        np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        for polygon in shown
    ])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.45, -0.70, 0.65])
    light /= np.linalg.norm(light)
    intensity = 0.38 + 0.62 * np.clip(np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray(mpl.colors.to_rgb(base_color))
    colors = np.clip(intensity[:, None] * base[None, :], 0.0, 1.0)
    return Poly3DCollection(
        shown, facecolors=colors, edgecolors="none", alpha=alpha,
        rasterized=rasterized,
    )


def mesh_collection(vertices: Array, faces: Array, alpha: float = 0.16):
    return polygon_collection(vertices[faces], "#9fb0b5", alpha)


def clip_triangle_to_rear(triangle: Array, cut_z: float) -> list[Array]:
    """Clip a physical-space triangle to z >= cut_z."""
    polygon = [vertex.copy() for vertex in triangle]
    clipped: list[Array] = []
    for current, following in zip(polygon, polygon[1:] + polygon[:1]):
        current_inside = current[2] >= cut_z
        following_inside = following[2] >= cut_z
        if current_inside:
            clipped.append(current)
        if current_inside != following_inside:
            weight = (cut_z - current[2]) / (following[2] - current[2])
            clipped.append(current + weight * (following - current))
    return clipped if len(clipped) >= 3 else []


def section_segments(vertices: Array, faces: Array, cut_z: float) -> list[Array]:
    segments: list[Array] = []
    for triangle in vertices[faces]:
        intersections: list[Array] = []
        for first, second in zip(triangle, np.roll(triangle, -1, axis=0)):
            first_side = first[2] - cut_z
            second_side = second[2] - cut_z
            if first_side * second_side < 0.0:
                weight = -first_side / (second_side - first_side)
                intersections.append(first + weight * (second - first))
            elif abs(first_side) < 1.0e-12:
                intersections.append(first.copy())
        unique: list[Array] = []
        for point in intersections:
            if not any(np.linalg.norm(point - previous) < 1.0e-10 for previous in unique):
                unique.append(point)
        if len(unique) == 2:
            segments.append(display_vertices(np.vstack(unique)))
    return segments


def add_cutaway_shell(
    ax,
    vertices: Array,
    faces: Array,
    cut_z: float,
    rear_alpha: float = 0.62,
) -> None:
    """Render the rear half as a solid shell and the removed half as a ghost."""
    ax.add_collection3d(
        polygon_collection(vertices[faces], "#b9c4c8", 0.035)
    )
    rear_polygons = [
        polygon
        for triangle in vertices[faces]
        if (polygon := clip_triangle_to_rear(triangle, cut_z))
    ]
    ax.add_collection3d(
        polygon_collection(rear_polygons, "#81979e", rear_alpha)
    )
    rim = section_segments(vertices, faces, cut_z)
    if rim:
        ax.add_collection3d(Line3DCollection(
            rim, colors="#176d73", linewidths=1.05, alpha=0.90,
            rasterized=True,
        ))


def style_3d(ax, vertices: Array) -> None:
    shown = display_vertices(vertices)
    low, high = shown.min(axis=0), shown.max(axis=0)
    middle = 0.5 * (low + high)
    radius = 0.53 * max(high - low)
    ax.set_xlim(middle[0] - radius, middle[0] + radius)
    ax.set_ylim(middle[1] - radius, middle[1] + radius)
    ax.set_zlim(middle[2] - radius, middle[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-68)
    ax.set_axis_off()


def plot_paths(
    ax,
    trace_rows: list[dict[str, str]],
    vertices: Array,
    faces: Array,
    cut_z: float,
):
    add_cutaway_shell(ax, vertices, faces, cut_z)
    groups: dict[int, list[Array]] = {}
    for row in trace_rows:
        groups.setdefault(int(row["path"]), []).append(
            np.array([float(row["x"]), float(row["y"]), float(row["z"])])
        )
    palette = mpl.colormaps["turbo"](np.linspace(0.08, 0.90, max(len(groups), 1)))
    longest = sorted(groups, key=lambda key: len(groups[key]), reverse=True)[:6]
    for color, key in zip(palette, longest):
        path = display_vertices(np.vstack(groups[key]))
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, lw=1.0, alpha=0.85)
        ax.scatter(
            path[:: max(1, len(path) // 8), 0],
            path[:: max(1, len(path) // 8), 1],
            path[:: max(1, len(path) // 8), 2],
            color=color, s=8, depthshade=False,
        )
    style_3d(ax, vertices)


def make_figure(
    output_dir: Path,
    vertices: Array,
    faces: Array,
    points: Array,
    trace_rows: list[dict[str, str]],
    twos_field: Array,
    voxel_field: Array,
    twos_rows: list[dict[str, object]],
    voxel_rows: list[dict[str, object]],
    output_rows: list[dict[str, object]],
    threads: int,
) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.7,
        "axes.titlesize": 10.3,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.1,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 420,
    })
    teal = "#008f88"
    purple = "#686ecf"
    orange = "#df7a3e"
    grid_color = "#d9dde0"
    fig = plt.figure(figsize=(12.4, 7.15), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, left=0.042, right=0.985, bottom=0.115, top=0.955,
        wspace=0.26, hspace=0.31,
    )
    ax_a = fig.add_subplot(grid[0, 0], projection="3d")
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[0, 2], projection="3d")
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    cut_z = float(np.quantile(vertices[:, 2], 0.46))
    plot_paths(ax_a, trace_rows, vertices, faces, cut_z)
    ax_a.set_title("a   Embree-accelerated sphere walks", pad=-1)
    ax_a.text2D(
        0.5, 0.035, "cutaway shell • actual C++ paths • no volume grid",
        transform=ax_a.transAxes, ha="center", color="#626b70", fontsize=8,
    )

    visible = points[:, 2] >= cut_z
    display_points = display_vertices(points[visible])
    vmax = float(np.quantile(np.r_[twos_field, voxel_field], 0.985))
    normalizer = Normalize(vmin=0.0, vmax=vmax)
    for ax, values, title in (
        (ax_b, twos_field, r"b   Direct TWoS score $-D_T$"),
        (ax_c, voxel_field, r"c   Voxel-adjoint score $-D_T$"),
    ):
        add_cutaway_shell(ax, vertices, faces, cut_z)
        scatter = ax.scatter(
            display_points[:, 0], display_points[:, 1], display_points[:, 2],
            c=values[visible], cmap="inferno", norm=normalizer, s=13, alpha=0.98,
            linewidth=0.18, edgecolor="#2c1b34", depthshade=False,
        )
        style_3d(ax, vertices)
        ax.set_title(title, pad=-1)
    colorbar = fig.colorbar(scatter, ax=[ax_b, ax_c], fraction=0.023, pad=0.005, shrink=0.72)
    colorbar.ax.tick_params(labelsize=7)

    twos_time = np.array([float(row["wall_seconds"]) for row in twos_rows])
    twos_error = np.array([float(row["relative_rmse"]) for row in twos_rows])
    voxel_time = np.array([float(row["wall_seconds"]) for row in voxel_rows])
    voxel_error = np.array([float(row["relative_rmse"]) for row in voxel_rows])
    ax_d.loglog(
        twos_time, twos_error, "o-", color=teal, lw=1.9, ms=4.8,
        label="C++ TWoS",
    )
    ax_d.loglog(
        voxel_time, voxel_error, "s-", color=purple, lw=1.9, ms=4.6,
        label="C++ voxel FD/CG",
    )
    for row in twos_rows[::2]:
        ax_d.annotate(
            f"{int(row['walks'])}w", (float(row["wall_seconds"]), float(row["relative_rmse"])),
            xytext=(4, -9), textcoords="offset points", color=teal, fontsize=7,
        )
    for row in voxel_rows[::2]:
        shape = row["shape"]
        if isinstance(shape, str):
            shape = json.loads(shape)
        ax_d.annotate(
            f"{shape[0]}x{shape[1]}x{shape[2]}",
            (float(row["wall_seconds"]), float(row["relative_rmse"])),
            xytext=(4, 4), textcoords="offset points", color=purple, fontsize=7,
        )
    ax_d.set_title("d   Accuracy versus wall time")
    ax_d.set(xlabel="end-to-end field time (s)", ylabel="relative field RMSE")
    ax_d.grid(True, which="both", color=grid_color, lw=0.55)
    ax_d.legend(frameon=False, loc="best")

    twos_memory = np.array([float(row["peak_rss_megabytes"]) for row in twos_rows])
    voxel_memory = np.array([float(row["peak_rss_megabytes"]) for row in voxel_rows])
    ax_e.semilogx(
        twos_error, twos_memory, "o-", color=teal, lw=1.9,
        ms=4.8, label="C++ TWoS",
    )
    ax_e.semilogx(
        voxel_error, voxel_memory, "s-", color=purple, lw=1.9,
        ms=4.6, label="C++ voxel FD/CG",
    )
    ax_e.invert_xaxis()
    ax_e.set_title("e   Accuracy versus memory")
    ax_e.set(xlabel="relative field RMSE (lower is better)", ylabel="peak RSS (MiB)")
    ax_e.grid(True, which="both", color=grid_color, lw=0.55)
    ax_e.legend(frameon=False)
    ax_e.text(
        0.97, 0.93, "TWoS stores no volumetric operator",
        transform=ax_e.transAxes, ha="right", va="top", fontsize=7.7,
        color="#596267",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    output_sites = np.array([int(row["sites"]) for row in output_rows])
    output_twos = np.array([float(row["twos_seconds"]) for row in output_rows])
    output_voxel = np.array([float(row["voxel_seconds"]) for row in output_rows])
    ax_f.loglog(
        output_sites, output_twos, "o-", color=teal, lw=1.9, ms=4.8,
        label="C++ TWoS, 512 walks/site",
    )
    ax_f.loglog(
        output_sites, output_voxel, "s-", color=purple, lw=1.9, ms=4.6,
        label="C++ voxel, 96x112x84",
    )
    faster = output_sites[output_twos < output_voxel]
    if len(faster):
        crossover = int(output_sites[np.argmin(np.abs(np.log(output_twos / output_voxel)))])
        ax_f.axvspan(output_sites.min(), faster.max(), color=teal, alpha=0.075, lw=0)
        ax_f.text(
            0.04, 0.92, f"timing crossover near {crossover} outputs",
            transform=ax_f.transAxes, va="top", color="#087a75", fontsize=7.5,
        )
    ax_f.set_title("f   Output-sensitive scaling")
    ax_f.set(xlabel="requested derivative sites", ylabel="end-to-end field time (s)")
    ax_f.grid(True, which="both", color=grid_color, lw=0.55)
    ax_f.legend(frameon=False, loc="lower right")

    fig.text(
        0.5, 0.025,
        f"Stanford Bunny • {len(points)} common interior sites • {threads} CPU threads • "
        "independent 2,048-walk reference",
        ha="center", fontsize=8.6, color="#687176",
    )
    pdf = output_dir / "compiled-bunny-scaling.pdf"
    png = output_dir / "compiled-bunny-scaling.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=360, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def make_surface_figure(output_dir: Path, face_rows: list[dict[str, object]]) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.3, "axes.linewidth": 0.75,
        "pdf.fonttype": 42, "savefig.dpi": 420,
    })
    orange = "#df7a3e"
    slate = "#697277"
    grid_color = "#d9dde0"
    face_count = np.array([int(row["faces"]) for row in face_rows])
    queries = np.array([
        float(row["distance_queries_per_second"]) for row in face_rows
    ]) / 1e6
    build = np.array([float(row["mesh_build_seconds"]) for row in face_rows]) * 1e3
    memory = np.array([float(row["peak_rss_megabytes"]) for row in face_rows])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15))
    axes[0].semilogx(face_count, queries, "o-", color=orange, lw=2.0, ms=5.2)
    axes[0].set_title("a   Closest-boundary throughput")
    axes[0].set(xlabel="boundary triangles", ylabel="distance queries / s (million)")
    axes[1].semilogx(face_count, build, "D-", color=slate, lw=1.8, ms=4.7,
                    label="BVH build")
    axes[1].set_title("b   Embree acceleration structure")
    axes[1].set(xlabel="boundary triangles", ylabel="BVH build (ms)")
    twin = axes[1].twinx()
    twin.semilogx(face_count, memory, "o--", color="#008f88", lw=1.4, ms=4.2,
                  label="peak RSS")
    twin.set_ylabel("peak RSS (MiB)", color="#087a75")
    twin.tick_params(axis="y", colors="#087a75")
    for axis in axes:
        axis.grid(True, which="both", color=grid_color, lw=0.55)
    fig.text(
        0.5, 0.015,
        "384 sites • 32 walks/site • 8 CPU threads • identical physical domain",
        ha="center", fontsize=8.2, color="#687176",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    pdf = output_dir / "compiled-bunny-surface-scaling.pdf"
    png = output_dir / "compiled-bunny-surface-scaling.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=360, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def run_full(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_reference_module()
    config = reference.Config(rendering_faces=7000)
    repaired = BUNNY_DIR / "output" / "stanford-bunny-watertight.ply"
    mesh = reference.load_and_repair_mesh(
        BUNNY_DIR / "assets" / "bun_zipper.ply", repaired, config
    )
    points = reference.sobol_interior(
        mesh, args.sites, 0.065, config.seed + 710
    )
    points_path = output / "evaluation-points.csv"
    write_xyz(points_path, points)
    executable = build_cpp()

    reference_field_path = output / "field-reference.csv"
    reference_summary_path = output / "reference-summary.json"
    reference_summary = run_cpp(
        executable, repaired, points_path, reference_field_path,
        reference_summary_path, args.reference_walks, config.seed + 720,
        1, args.threads, output / "walk-traces.csv", 12,
    )
    reference_rows = read_csv(reference_field_path)
    continuum_reference = np.array([float(row["score"]) for row in reference_rows])

    twos_results: list[dict[str, object]] = []
    retained_field: Array | None = None
    for walks in (4, 8, 16, 32, 64, 128, 256, 384, 512):
        field_path = output / f"field-twos-{walks:04d}.csv"
        summary_path = output / f"summary-twos-{walks:04d}.json"
        summary = run_cpp(
            executable, repaired, points_path, field_path, summary_path,
            walks, config.seed + 1000 + walks, args.repeats, args.threads,
        )
        field = np.array([float(row["score"]) for row in read_csv(field_path)])
        if walks == 32:
            retained_field = field
        twos_results.append({
            "method": "C++ TWoS", "walks": walks,
            "wall_seconds": float(summary["runtime_seconds_median"])
                + float(summary["mesh_build_seconds"]),
            "kernel_seconds": float(summary["runtime_seconds_median"]),
            "relative_rmse": relative_rmse(field, continuum_reference),
            "pearson": float(np.corrcoef(field, continuum_reference)[0, 1]),
            "peak_rss_megabytes": float(summary["peak_rss_megabytes"]),
            "paths_per_second": float(summary["paths_per_second"]),
            "distance_queries_per_second": float(summary["distance_queries_per_second"]),
            "mean_steps": float(summary["mean_steps"]),
        })
    if retained_field is None:
        raise RuntimeError("32-walk field was not retained")

    voxel_results: list[dict[str, object]] = []
    voxel_field: Array | None = None
    shapes = ((24, 28, 20), (32, 38, 28), (42, 50, 36),
              (54, 64, 48), (68, 80, 60), (82, 96, 72),
              (96, 112, 84), (110, 128, 96))
    for shape in shapes:
        tag = "x".join(map(str, shape))
        field_path = output / f"field-voxel-{tag}.csv"
        summary_path = output / f"summary-voxel-{tag}.json"
        summary = run_cpp_voxel(
            executable, repaired, points_path, field_path, summary_path,
            shape, args.repeats, args.threads,
        )
        field = np.array([float(row["score"]) for row in read_csv(field_path)])
        if shape == (54, 64, 48):
            voxel_field = field
        voxel_results.append({
            "method": "C++ voxel FD/CG", **summary,
            "wall_seconds": float(summary["total_seconds"]),
            "relative_rmse": relative_rmse(field, continuum_reference),
            "pearson": float(np.corrcoef(field, continuum_reference)[0, 1]),
        })
    if voxel_field is None:
        raise RuntimeError("54x64x48 voxel field was not retained")

    python_voxel_field, python_voxel_summary = voxel_solve(
        repaired, points_path, shapes[0]
    )
    cpp_voxel_validation = {
        "shape": list(shapes[0]),
        "active_nodes_cpp": int(voxel_results[0]["active_nodes"]),
        "active_nodes_python": int(python_voxel_summary["active_nodes"]),
        "score_relative_rmse": relative_rmse(
            np.array([
                float(row["score"])
                for row in read_csv(output / "field-voxel-24x28x20.csv")
            ]),
            python_voxel_field,
        ),
    }

    vertices, faces = igl.read_triangle_mesh(str(repaired))
    face_results: list[dict[str, object]] = []
    for mesh_path in decimated_meshes(vertices, faces, output):
        tag = mesh_path.stem
        summary = run_cpp(
            executable, mesh_path, points_path,
            output / f"field-{tag}.csv", output / f"summary-{tag}.json",
            32, config.seed + 1800, args.repeats, args.threads,
        )
        face_results.append(summary)

    output_results: list[dict[str, object]] = []
    output_shape = (96, 112, 84)
    for site_count in (8, 16, 32, 64, 96, 128, 192, 256, 384):
        if site_count > len(points):
            continue
        subset_path = output / f"evaluation-points-{site_count:04d}.csv"
        write_xyz(subset_path, points[:site_count])
        twos_field_path = output / f"field-output-twos-{site_count:04d}.csv"
        twos_summary_path = output / f"summary-output-twos-{site_count:04d}.json"
        twos_summary = run_cpp(
            executable, repaired, subset_path, twos_field_path,
            twos_summary_path, 512, config.seed + 3200,
            args.repeats, args.threads,
        )
        voxel_field_path = output / f"field-output-voxel-{site_count:04d}.csv"
        voxel_summary_path = output / f"summary-output-voxel-{site_count:04d}.json"
        voxel_summary = run_cpp_voxel(
            executable, repaired, subset_path, voxel_field_path,
            voxel_summary_path, output_shape, args.repeats, args.threads,
        )
        twos_subset = np.array([
            float(row["score"]) for row in read_csv(twos_field_path)
        ])
        voxel_subset = np.array([
            float(row["score"]) for row in read_csv(voxel_field_path)
        ])
        twos_seconds = float(twos_summary["runtime_seconds_median"]) + float(
            twos_summary["mesh_build_seconds"]
        )
        voxel_seconds = float(voxel_summary["total_seconds"])
        output_results.append({
            "sites": site_count,
            "twos_walks_per_site": 512,
            "voxel_shape": list(output_shape),
            "twos_seconds": twos_seconds,
            "voxel_seconds": voxel_seconds,
            "voxel_over_twos_speedup": voxel_seconds / twos_seconds,
            "twos_relative_rmse": relative_rmse(
                twos_subset, continuum_reference[:site_count]
            ),
            "voxel_relative_rmse": relative_rmse(
                voxel_subset, continuum_reference[:site_count]
            ),
            "twos_peak_rss_megabytes": float(twos_summary["peak_rss_megabytes"]),
            "voxel_peak_rss_megabytes": float(voxel_summary["peak_rss_megabytes"]),
        })

    trace_rows = read_csv(output / "walk-traces.csv")
    trace_initial = {
        int(row["path"]): float(row["radius"])
        for row in trace_rows if int(row["step"]) == 0
    }
    trace_indices = [index * len(points) // len(trace_initial) for index in range(len(trace_initial))]
    libigl_distance = np.sqrt(
        mesh.tree.squared_distance(mesh.vertices, mesh.faces, points[trace_indices])[0]
    )
    embree_distance = np.array([trace_initial[index] for index in sorted(trace_initial)])
    maximum_distance_error = float(np.max(np.abs(libigl_distance - embree_distance)))
    if maximum_distance_error > 2.0e-6:
        raise RuntimeError(f"Embree/libigl distance mismatch: {maximum_distance_error}")

    validation_count = min(48, len(points))
    validation_started = time.perf_counter()
    python_samples, _ = reference.wos_samples(
        points[:validation_count], 512, np.empty((0, 3)), mesh, config,
        np.random.default_rng(config.seed + 2710),
    )
    python_scores = reference.crossfit_square(python_samples)[0]
    python_seconds = time.perf_counter() - validation_started
    cpp_python_validation = {
        "sites": validation_count,
        "python_walks_per_site": 512,
        "cpp_walks_per_site": args.reference_walks,
        "relative_rmse": relative_rmse(
            python_scores, continuum_reference[:validation_count]
        ),
        "pearson": float(np.corrcoef(
            python_scores, continuum_reference[:validation_count]
        )[0, 1]),
        "python_seconds": python_seconds,
    }

    write_csv(output / "twos-scaling.csv", twos_results)
    write_csv(output / "voxel-scaling.csv", voxel_results)
    write_csv(output / "surface-scaling.csv", face_results)
    write_csv(output / "output-scaling.csv", output_results)
    render_v, render_f, _, _ = igl.decimate(
        np.asfortranarray(vertices), np.asfortranarray(faces.astype(np.int32)), 7000
    )
    make_figure(
        output, render_v, render_f, points, trace_rows, retained_field,
        voxel_field, twos_results, voxel_results, output_results, args.threads,
    )
    make_surface_figure(output, face_results)
    best_twos = min(twos_results, key=lambda row: float(row["relative_rmse"]))
    summary = {
        "seed": config.seed,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "benchmark_threads": args.threads,
        },
        "reference": reference_summary,
        "reference_walks": args.reference_walks,
        "evaluation_sites": args.sites,
        "maximum_embree_libigl_distance_error": maximum_distance_error,
        "cpp_python_field_validation": cpp_python_validation,
        "cpp_python_voxel_validation": cpp_voxel_validation,
        "twos": twos_results,
        "voxel": voxel_results,
        "surface_scaling": face_results,
        "output_scaling": output_results,
        "best_twos_measured": best_twos,
        "scope": (
            "Matched C++ CPU reference study; this is not yet a claim against "
            "tuned multigrid, FEM, BEM, or GPU solvers."
        ),
    }
    atomic_json(output / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "output"))
    parser.add_argument("--sites", type=int, default=384)
    parser.add_argument("--reference-walks", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--voxel-worker", action="store_true")
    parser.add_argument("--mesh")
    parser.add_argument("--points")
    parser.add_argument("--shape")
    parser.add_argument("--field-output")
    parser.add_argument("--summary")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.voxel_worker:
        voxel_worker(arguments)
    else:
        run_full(arguments)
