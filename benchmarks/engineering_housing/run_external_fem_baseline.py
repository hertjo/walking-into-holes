#!/usr/bin/env python3
"""Independent tetrahedral FEM audit of the housing topological field.

This script deliberately lives outside the compiled TWoS implementation.  It
uses scikit-fem P1 tetrahedra, removes tetrahedra by the analytic housing SDF,
solves the same Poisson problem, and compares the compliance ranking density
u^2 at the identical 768 candidate sites.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from skfem import (Basis, BilinearForm, ElementTetP1, LinearForm, MeshTet,
                   asm, condense, solve)
from skfem.helpers import dot, grad

Array = np.ndarray
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_housing_module():
    path = HERE / "run_housing_benchmark.py"
    spec = importlib.util.spec_from_file_location("housing_fem_geometry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_points(path: Path) -> Array:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def read_scores(path: Path) -> Array:
    with path.open(newline="") as handle:
        return np.asarray([float(row["score"]) for row in csv.DictReader(handle)])


def relative_rmse(values: Array, reference: Array) -> float:
    return float(np.linalg.norm(values - reference) / np.linalg.norm(reference))


def cosine(values: Array, reference: Array) -> float:
    return float(np.dot(values, reference) /
                 (np.linalg.norm(values) * np.linalg.norm(reference)))


def top_overlap(values: Array, reference: Array, count: int) -> float:
    first = set(np.argsort(values)[-count:])
    second = set(np.argsort(reference)[-count:])
    return len(first & second) / count


def interpolate_in_chunks(basis: Basis, state: Array, points: Array,
                          chunk_size: int = 48) -> Array:
    """Bound scikit-fem's point-location working set without changing values."""
    values = np.empty(len(points))
    interpolator = basis.interpolator(state)
    for first in range(0, len(points), chunk_size):
        last = min(first + chunk_size, len(points))
        values[first:last] = interpolator(points[first:last].T)
    return values


def parse_resolution(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().replace("x", ",").split(","))
    if len(values) != 3 or min(values) < 5:
        raise argparse.ArgumentTypeError("resolution must be NX,NY,NZ with entries >= 5")
    return values


def solve_fem(housing, resolution: tuple[int, int, int], candidates: Array):
    low = np.array([-1.20, -0.77, -0.40])
    high = np.array([1.20, 0.77, 0.77])
    axes = [np.linspace(low[i], high[i], resolution[i]) for i in range(3)]

    started = time.perf_counter()
    background = MeshTet.init_tensor(*axes)
    centroids = background.p[:, background.t].mean(axis=1).T
    outside = np.flatnonzero(housing.housing_sdf(centroids) >= 0.0)
    mesh = background.remove_elements(outside)
    mesh_seconds = time.perf_counter() - started

    basis = Basis(mesh, ElementTetP1())

    @BilinearForm
    def laplace(u, v, _):
        return dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        shape = w.x.shape[1:]
        values = housing.source(w.x.reshape(3, -1).T).reshape(shape)
        return values * v

    started = time.perf_counter()
    stiffness = asm(laplace, basis)
    load_vector = asm(load, basis)
    boundary = basis.get_dofs().all()
    assembly_seconds = time.perf_counter() - started

    started = time.perf_counter()
    state = solve(*condense(stiffness, load_vector, D=boundary))
    solve_seconds = time.perf_counter() - started

    started = time.perf_counter()
    candidate_state = interpolate_in_chunks(basis, state, candidates)
    scores = np.square(candidate_state)
    interpolation_seconds = time.perf_counter() - started

    residual = stiffness @ state - load_vector
    free = np.setdiff1d(np.arange(len(state)), boundary, assume_unique=True)
    relative_residual = float(
        np.linalg.norm(residual[free]) / max(np.linalg.norm(load_vector[free]), 1e-300)
    )
    summary = {
        "resolution": list(resolution),
        "vertices": int(mesh.nvertices),
        "tetrahedra": int(mesh.nelements),
        "boundary_dofs": int(len(boundary)),
        "stiffness_nonzeros": int(stiffness.nnz),
        "mesh_seconds": mesh_seconds,
        "assembly_seconds": assembly_seconds,
        "solve_seconds": solve_seconds,
        "interpolation_seconds": interpolation_seconds,
        "total_seconds": mesh_seconds + assembly_seconds + solve_seconds + interpolation_seconds,
        "relative_algebraic_residual": relative_residual,
        "compliance": float(np.dot(load_vector, state)),
        "peak_rss_megabytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    return mesh, basis, state, scores, summary


def cutaway_facets(mesh: MeshTet, max_facets: int = 18000) -> Array:
    facets = mesh.facets[:, mesh.boundary_facets()].T
    triangles = mesh.p[:, facets].transpose(1, 2, 0)
    # Remove the viewer-facing half to reveal the volumetric mesh and hot sites.
    triangles = triangles[triangles[:, :, 1].mean(axis=1) >= -0.015]
    if len(triangles) > max_facets:
        stride = int(np.ceil(len(triangles) / max_facets))
        triangles = triangles[::stride]
    return triangles


def make_figure(housing, output: Path, mesh: MeshTet, basis: Basis, state: Array,
                candidates: Array, fem_scores: Array, twos_scores: Array,
                records: list[dict[str, object]]) -> tuple[Path, Path]:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9.3, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "axes.linewidth": 0.65, "pdf.fonttype": 42,
    })
    blue, orange, ink, grid = "#2563a6", "#e56b2f", "#20252b", "#d9dee5"
    figure = plt.figure(figsize=(10.8, 5.5), constrained_layout=True)
    layout = figure.add_gridspec(2, 4, height_ratios=[1.0, 0.92])

    ax_a = figure.add_subplot(layout[:, :2], projection="3d")
    triangles = cutaway_facets(mesh)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    collection = Poly3DCollection(
        triangles, facecolors="#dce3e9", edgecolors="#7d8b98",
        linewidths=0.12, alpha=0.42, rasterized=True,
    )
    ax_a.add_collection3d(collection)
    hottest = np.argsort(fem_scores)[-36:]
    scatter = ax_a.scatter(candidates[hottest, 0], candidates[hottest, 1],
                           candidates[hottest, 2], c=fem_scores[hottest], s=19,
                           cmap="inferno", edgecolor="white", linewidth=0.25,
                           depthshade=False)
    ax_a.set_xlim(-1.2, 1.2); ax_a.set_ylim(-0.77, 0.77); ax_a.set_zlim(-0.4, 0.77)
    ax_a.set_box_aspect((2.4, 1.54, 1.17)); ax_a.view_init(22, -58)
    ax_a.set_axis_off()
    ax_a.set_title("a   Independent tetrahedral FEM audit", pad=-3)
    bar = figure.colorbar(scatter, ax=ax_a, fraction=0.03, pad=-0.02, shrink=0.52)
    bar.set_label(r"FEM ranking density $u_h^2$")

    ax_b = figure.add_subplot(layout[0, 2])
    x = np.linspace(-1.08, 1.08, 220)
    z = np.linspace(-0.30, 0.34, 110)
    xx, zz = np.meshgrid(x, z)
    points = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
    inside = housing.housing_sdf(points) < -0.025
    values = np.full(len(points), np.nan)
    if np.any(inside):
        values[inside] = np.square(interpolate_in_chunks(
            basis, state, points[inside], chunk_size=96))
    image = ax_b.imshow(values.reshape(xx.shape), origin="lower",
                        extent=(x[0], x[-1], z[0], z[-1]), cmap="inferno",
                        interpolation="bilinear", aspect="auto")
    ax_b.contour(xx, zz, housing.housing_sdf(points).reshape(xx.shape),
                 levels=[0.0], colors="white", linewidths=0.55)
    ax_b.set_xlabel("$x$"); ax_b.set_ylabel("$z$")
    ax_b.set_title("b   Central-slice field")
    figure.colorbar(image, ax=ax_b, fraction=0.05, pad=0.025)

    ax_c = figure.add_subplot(layout[0, 3])
    slope = float(np.dot(fem_scores, twos_scores) / np.dot(fem_scores, fem_scores))
    scaled = slope * fem_scores
    ax_c.scatter(scaled, twos_scores, c=candidates[:, 0], cmap="coolwarm", s=8,
                 alpha=0.72, linewidth=0, rasterized=True)
    limit = 1.04 * max(float(np.max(scaled)), float(np.max(twos_scores)))
    ax_c.plot([0, limit], [0, limit], color=ink, lw=0.8, ls="--")
    ax_c.set_xlim(0, limit); ax_c.set_ylim(0, limit)
    ax_c.set_xlabel("scaled FEM $u_h^2$"); ax_c.set_ylabel("TWoS $u^2$")
    ax_c.set_title("c   Identical 768 sites")
    rho = pearsonr(fem_scores, twos_scores).statistic
    ax_c.text(0.05, 0.92, rf"Pearson $r={rho:.3f}$", transform=ax_c.transAxes,
              color=ink, va="top")

    ax_d = figure.add_subplot(layout[1, 2:])
    times = np.asarray([float(row["total_seconds"]) for row in records])
    errors = np.asarray([float(row["relative_rmse_vs_finest_fem"]) for row in records])
    positive = errors[errors > 0.0]
    floor = float(np.min(positive) / 2.7)
    nonzero = np.where(errors > 0.0, errors, floor)
    memory = np.asarray([float(row["peak_rss_megabytes"]) for row in records])
    normalized_memory = (np.log(memory) - np.log(memory.min())) / max(
        np.log(memory.max()) - np.log(memory.min()), 1e-12)
    sizes = 35.0 + 65.0 * normalized_memory
    ax_d.plot(times[:-1], nonzero[:-1], color=blue, lw=1.2, zorder=1)
    ax_d.scatter(times[:-1], nonzero[:-1], s=sizes[:-1], color=blue,
                 edgecolor="white", linewidth=0.6, zorder=2)
    ax_d.scatter(times[-1], nonzero[-1], s=sizes[-1], marker="v", color=orange,
                 edgecolor="white", linewidth=0.6, zorder=3)
    for row, tx, error in zip(records, times, nonzero):
        label = "×".join(str(v) for v in row["resolution"])
        suffix = " (reference)" if error == nonzero[-1] and tx == times[-1] else ""
        ax_d.annotate(label + suffix, (tx, error), xytext=(4, 5), textcoords="offset points",
                      fontsize=7, color=ink)
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_xlabel("end-to-end FEM time (s)")
    ax_d.set_ylabel("relative RMSE vs. finest FEM")
    ax_d.set_title("d   Mesh convergence and resource cost")
    ax_d.grid(True, which="both", color=grid, lw=0.45)
    ax_d.text(0.98, 0.06, "marker area encodes peak RSS; triangle is reference",
              transform=ax_d.transAxes,
              ha="right", color="#59636f", fontsize=7.2)

    for axis in (ax_b, ax_c, ax_d):
        axis.spines[["top", "right"]].set_visible(False)
    pdf = output / "external-fem-audit.pdf"
    png = output / "external-fem-audit.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def run(args: argparse.Namespace) -> None:
    housing = load_housing_module()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_output = HERE / "output"
    candidates = read_points(source_output / "points-reference-score.csv")
    twos_128_scores = read_scores(source_output / "field-visual-score.csv")
    twos_scores = read_scores(source_output / "field-reference-score.csv")
    mg_scores = read_scores(source_output / "field-multigrid-sweep-133x85x67-0.csv")
    if not (len(candidates) == len(twos_128_scores) == len(twos_scores) == len(mg_scores)):
        raise RuntimeError("candidate and reference-field lengths differ")

    records: list[dict[str, object]] = []
    solutions = []
    for resolution in args.resolution:
        mesh, basis, state, scores, record = solve_fem(housing, resolution, candidates)
        solutions.append((mesh, basis, state, scores))
        records.append(record)
        np.savetxt(output / ("fem-scores-" + "x".join(map(str, resolution)) + ".csv"),
                   np.column_stack((np.arange(len(scores)), candidates, scores)),
                   delimiter=",", header="site,x,y,z,score", comments="")
        print(f"FEM {resolution}: {record['vertices']} vertices, "
              f"{record['tetrahedra']} tetrahedra, {record['total_seconds']:.3f} s")

    finest = solutions[-1][3]
    for record, (_, _, _, scores) in zip(records, solutions):
        record["relative_rmse_vs_finest_fem"] = relative_rmse(scores, finest)
        record["pearson_vs_finest_fem"] = float(pearsonr(scores, finest).statistic)

    comparisons = {}
    for name, scores in (("TWoS-128", twos_128_scores),
                         ("TWoS-512", twos_scores),
                         ("multigrid-133x85x67", mg_scores)):
        comparisons[name] = {
            "pearson": float(pearsonr(scores, finest).statistic),
            "spearman": float(spearmanr(scores, finest).statistic),
            "cosine": cosine(scores, finest),
            "relative_rmse_after_origin_fit": relative_rmse(
                scores, finest * float(np.dot(finest, scores) / np.dot(finest, finest))),
            "top_20_overlap": top_overlap(scores, finest, 20),
            "top_50_overlap": top_overlap(scores, finest, 50),
        }

    mesh, basis, state, _, = solutions[-1]
    pdf, png = make_figure(housing, output, mesh, basis, state, candidates,
                           finest, twos_scores, records)
    summary = {
        "engine": "scikit-fem 11.0.0 P1 tetrahedral FEM + SciPy SuperLU",
        "independence": "separate Python implementation; shared analytic SDF, load, and candidate CSV only",
        "pde": "-Laplacian(u)=f; homogeneous Dirichlet boundary",
        "topological_ranking_density": "u_h^2 (self-adjoint compliance)",
        "candidates": len(candidates),
        "resolutions": records,
        "comparisons_against_finest_fem": comparisons,
        "outputs": {"figure_pdf": str(pdf), "figure_png": str(png)},
        "limitations": [
            "background tetrahedra are retained by centroid classification, so the curved boundary is faceted",
            "this is an independent forward/adjoint field audit, not a production level-set optimizer",
            "runtime includes Python mesh construction and sparse direct solution",
        ],
    }
    (output / "external-fem-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "output" / "external_fem"))
    parser.add_argument("--resolution", action="append", type=parse_resolution,
                        default=None, help="repeatable NX,NY,NZ")
    args = parser.parse_args()
    if args.resolution is None:
        args.resolution = [(41, 27, 23), (61, 39, 33), (81, 51, 43)]
    run(args)


if __name__ == "__main__":
    main()
