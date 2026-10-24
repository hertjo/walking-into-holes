#!/usr/bin/env python3
"""Frozen-channel Differential WoS normal-offset and seam-strip audit.

The experiment freezes the pre-refinement seven-control, 22-ball channel from
``run_channel_benchmark.py``.  A separate pilot pass selects six feasible,
high-signal length-tangent directions.  Central finite differences are then
estimated once, while matched independent Differential WoS streams sweep the
normal offset.  The compiled estimator reports the harmonic exit mass removed
by its one-offset-width seam strip and by any additional unsafe normal segment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20260912


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows[1:] for key in row if key not in fieldnames
    )
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sample_se(values: Array) -> float:
    return (float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0)


def frozen_geometry(channel, source: Path) -> tuple[Array, Array, Array, float, dict]:
    summary = json.loads((source / "channel-summary.json").read_text())
    with (source / "channel-controls.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == "topology"]
    rows.sort(key=lambda row: int(row["control"]))
    controls = np.asarray([
        [float(row["x"]), float(row["y"]), float(row["z"])] for row in rows
    ])
    subdivisions = np.asarray(
        summary["problem"]["segment_subdivisions"], dtype=int
    )
    centers, weights = channel.dense_polyline(controls, subdivisions)
    radius = float(summary["methods"]["TD graph channel"]["radius"])

    with (source / "channel-spheres.csv").open(newline="") as handle:
        sphere_rows = [
            row for row in csv.DictReader(handle) if row["stage"] == "topology"
        ]
    sphere_rows.sort(key=lambda row: int(row["sphere"]))
    recorded = np.asarray([
        [float(row["x"]), float(row["y"]), float(row["z"])]
        for row in sphere_rows
    ])
    if recorded.shape != centers.shape or not np.allclose(
        recorded, centers, rtol=0.0, atol=2.0e-15
    ):
        raise RuntimeError("frozen controls do not reproduce recorded dense centers")
    if len(controls) != 7 or len(centers) != 22:
        raise RuntimeError("expected the frozen seven-control, 22-ball channel")
    return controls, centers, weights, radius, summary


def gradient_from_field(
    housing, field: Path, probes: Array, centers: Array,
    radius: float, volume: float,
) -> tuple[Array, Array]:
    distances = np.linalg.norm(probes[:, None, :] - centers[None, :, :], axis=2)
    active = np.all(distances > radius + 0.0015, axis=1)
    active_points = probes[active]
    rows = housing.read_rows(field)
    if len(rows) != len(active_points):
        raise RuntimeError(f"cached field {field} has the wrong number of sites")
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
    return gradient, gradient_se


def cached_diffwos(
    channel, housing, executable: Path, mesh: Path, work: Path, tag: str,
    probes: Array, centers: Array, radius: float, volume: float,
    walks: int, seed: int, threads: int, offset: float, force: bool,
) -> tuple[Array, Array, dict[str, object]]:
    field = work / f"field-{tag}.csv"
    summary_path = work / f"summary-{tag}.json"
    if not force and field.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        valid = (
            "unsafe_offset_fraction" in summary
            and int(summary.get("seed", -1)) == seed
            and int(summary.get("walks_per_site", -1)) == walks
            and math.isclose(float(summary.get("normal_offset", -1.0)), offset)
            and math.isclose(float(summary.get("cavity_radius", -1.0)), radius)
        )
        if valid:
            gradient, gradient_se = gradient_from_field(
                housing, field, probes, centers, radius, volume
            )
            print(f"reuse {tag}")
            return gradient, gradient_se, summary
    return channel.run_diffwos(
        housing, executable, mesh, work, tag, probes, centers, radius, volume,
        walks, seed, threads, offset,
    )


def objective_from_field(
    housing, field: Path, probes: Array, centers: Array,
    radius: float, volume: float,
) -> float:
    active = np.ones(len(probes), dtype=bool)
    distances = np.linalg.norm(probes[:, None, :] - centers[None, :, :], axis=2)
    active = np.all(distances > radius + 0.0015, axis=1)
    rows = housing.read_rows(field)
    if len(rows) != int(np.count_nonzero(active)):
        raise RuntimeError(f"cached field {field} has the wrong number of sites")
    state = np.asarray([float(row["u"]) for row in rows])
    integrand = np.zeros(len(probes))
    integrand[active] = housing.source(probes[active]) * state
    return volume * float(np.mean(integrand))


def cached_objective(
    housing, executable: Path, mesh: Path, work: Path, tag: str,
    probes: Array, centers: Array, radius: float, volume: float,
    walks: int, seed: int, threads: int, force: bool,
) -> float:
    field = work / f"field-{tag}.csv"
    summary_path = work / f"summary-{tag}.json"
    if not force and field.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (int(summary.get("walks_per_site", -1)) == walks
                and math.isclose(float(summary.get("cavity_radius", -1.0)), radius)):
            print(f"reuse {tag}")
            return objective_from_field(
                housing, field, probes, centers, radius, volume
            )
    value, _ = housing.objective(
        executable, mesh, work, tag, probes, centers, radius, volume,
        walks, seed, threads,
    )
    return value


def select_directions(
    channel, housing, controls: Array, subdivisions: Array,
    radius: float, control_gradient: Array, direction_count: int,
    step: float, clearance: float,
) -> tuple[list[Array], list[float]]:
    rng = np.random.default_rng(SEED + 42000)
    candidates: list[tuple[float, Array]] = []
    descent = channel.projected_direction(controls, control_gradient)
    descent_norm = float(np.linalg.norm(descent))
    if descent_norm <= 1.0e-15:
        raise RuntimeError("pilot descent direction vanished")
    descent /= descent_norm
    candidates.append((abs(float(np.sum(control_gradient * descent))), descent))

    random_candidates: list[tuple[float, Array]] = []
    for _ in range(512):
        random_gradient = rng.normal(size=controls.shape)
        direction = channel.projected_direction(controls, -random_gradient)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-15:
            continue
        direction /= direction_norm
        if not channel.feasible_channel(
            housing, controls + step * direction, subdivisions,
            radius, clearance,
        ) or not channel.feasible_channel(
            housing, controls - step * direction, subdivisions,
            radius, clearance,
        ):
            continue
        signal = abs(float(np.sum(control_gradient * direction)))
        random_candidates.append((signal, direction))
    random_candidates.sort(key=lambda item: item[0], reverse=True)
    high_signal_count = max(
        direction_count - 1, int(math.ceil(0.30 * len(random_candidates)))
    )
    high_signal = random_candidates[:high_signal_count]
    needed = direction_count - 1
    if len(high_signal) < needed:
        raise RuntimeError("not enough feasible high-signal tangent directions")
    indices = np.linspace(0, len(high_signal) - 1, needed).round().astype(int)
    candidates.extend(high_signal[int(index)] for index in indices)

    directions: list[Array] = []
    signals: list[float] = []
    for index, (signal, selected) in enumerate(candidates):
        direction = selected.copy()
        predicted = float(np.sum(control_gradient * direction))
        desired_sign = -1.0 if index % 2 == 0 else 1.0
        if predicted * desired_sign < 0.0:
            direction *= -1.0
        directions.append(direction)
        signals.append(signal)
    return directions, signals


def fit_loglog(x: Array, y: Array) -> tuple[float, float, float]:
    positive = y > 0.0
    coefficients = np.polyfit(np.log(x[positive]), np.log(y[positive]), 1)
    fitted = np.polyval(coefficients, np.log(x[positive]))
    residual = float(np.sum((np.log(y[positive]) - fitted) ** 2))
    total = float(np.sum((np.log(y[positive]) - np.mean(np.log(y[positive]))) ** 2))
    r_squared = 1.0 - residual / max(total, 1.0e-30)
    return float(coefficients[0]), float(np.exp(coefficients[1])), r_squared


def make_figure(
    output: Path, radius: float, offsets: Array,
    directional_rows: list[dict[str, object]], fd_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]], run_rows: list[dict[str, object]],
    unsafe_exponent: float, total_segment_unsafe: int, total_paths: int,
) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = ["#0B7285", "#E07A1F", "#6A51A3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.48))

    fd_mean = {}
    fd_se = {}
    for direction in sorted({int(row["direction"]) for row in fd_rows}):
        values = np.asarray([
            float(row["finite_difference"]) for row in fd_rows
            if int(row["direction"]) == direction
        ])
        fd_mean[direction] = float(np.mean(values))
        fd_se[direction] = sample_se(values)
    shown_offsets = [offsets[0], offsets[np.argmin(abs(offsets - 0.004))], offsets[-1]]
    ax = axes[0]
    all_values: list[float] = []
    for color, offset in zip(colors, shown_offsets):
        x_values, x_errors, y_values, y_errors = [], [], [], []
        for direction in sorted(fd_mean):
            values = np.asarray([
                float(row["diffwos_directional_derivative"])
                for row in directional_rows
                if int(row["direction"]) == direction
                and math.isclose(float(row["offset"]), float(offset))
            ])
            x_values.append(1.0e3 * float(np.mean(values)))
            x_errors.append(1.0e3 * sample_se(values))
            y_values.append(1.0e3 * fd_mean[direction])
            y_errors.append(1.0e3 * fd_se[direction])
        all_values.extend(x_values + y_values)
        ax.errorbar(
            x_values, y_values, xerr=x_errors, yerr=y_errors,
            fmt="o", ms=3.8, capsize=1.8, lw=0.75, color=color,
            ecolor=color, alpha=0.90,
            label=rf"$\ell/r={offset / radius:.3f}$",
        )
    bound = 1.10 * max(abs(min(all_values)), abs(max(all_values)))
    ax.plot([-bound, bound], [-bound, bound], color="#687176", lw=0.85,
            ls="--", zorder=0)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound),
           xlabel=r"Differential WoS  ($\times10^{-3}$)",
           ylabel=r"central difference  ($\times10^{-3}$)",
           title="a   Frozen directional audit")
    ax.grid(True, color="#e1e5e7", lw=0.45)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.25)

    ax = axes[1]
    normalized = np.asarray([float(row["ell_over_radius"]) for row in metric_rows])
    slopes = np.asarray([float(row["slope_fd_on_diffwos"]) for row in metric_rows])
    cosines = np.asarray([float(row["cosine_similarity"]) for row in metric_rows])
    ax.semilogx(normalized, slopes, "o-", color="#0B7285", lw=1.35, ms=4.1,
                label="FD-on-DiffWoS slope")
    ax.axhline(1.0, color="#687176", lw=0.80, ls="--")
    ax.set(xlabel=r"normalized offset $\ell/r$", ylabel="FD-on-estimate slope",
           ylim=(0.90, max(1.10, float(np.max(slopes)) + 0.12)),
           title="b   Offset convergence")
    twin = ax.twinx()
    twin.semilogx(normalized, cosines, "s-", color="#2A9D55", lw=1.20,
                  ms=3.7, label="cosine similarity")
    twin.tick_params(axis="y", colors="#2A9D55")
    twin.set_ylim(min(0.975, float(np.min(cosines)) - 0.003), 1.002)
    ticks = [0.025, 0.05, 0.10, 0.20]
    tick_labels = ["0.025", "0.05", "0.10", "0.20"]
    ax.set_xticks(ticks, tick_labels)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.grid(True, which="both", color="#e1e5e7", lw=0.45)
    handles, labels = ax.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, labels + twin_labels,
              frameon=False, loc="upper left")

    ax = axes[2]
    unsafe_means, unsafe_ses, conditional_means, conditional_ses = [], [], [], []
    for offset in offsets:
        rows = [row for row in run_rows
                if math.isclose(float(row["offset"]), float(offset))]
        unsafe = np.asarray([float(row["unsafe_offset_fraction"]) for row in rows])
        conditional = np.asarray([
            float(row["unsafe_given_cavity_fraction"]) for row in rows
        ])
        unsafe_means.append(float(np.mean(unsafe)))
        unsafe_ses.append(sample_se(unsafe))
        conditional_means.append(float(np.mean(conditional)))
        conditional_ses.append(sample_se(conditional))
    unsafe_means_array = np.asarray(unsafe_means)
    conditional_means_array = np.asarray(conditional_means)
    ax.errorbar(normalized, unsafe_means_array, yerr=unsafe_ses, fmt="o-",
                color="#C23B22", lw=1.35, ms=4.0, capsize=1.7,
                label="all exits")
    ax.errorbar(normalized, conditional_means_array, yerr=conditional_ses,
                fmt="s-", color="#6A51A3", lw=1.20, ms=3.7, capsize=1.7,
                label="given cavity exit")
    guide = unsafe_means_array[0] * normalized / normalized[0]
    ax.loglog(normalized, guide, ls="--", lw=0.85, color="#687176",
              label=r"$O(\ell)$ guide")
    ax.set_xticks(ticks, tick_labels)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set(xlabel=r"normalized offset $\ell/r$", ylabel="discarded exit mass",
           title=rf"c   Safe-strip rate  ($\widehat p={unsafe_exponent:.2f}$)")
    ax.grid(True, which="both", color="#e1e5e7", lw=0.45)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.04, 0.95,
            f"extra segment failures: {total_segment_unsafe} / {total_paths / 1e6:.1f}M",
            transform=ax.transAxes, va="top", color="#4f5b60", fontsize=6.2)

    fig.text(
        0.5, 0.006,
        "frozen pre-refinement housing channel • 22 overlapping balls • "
        "matched streams across offsets • independent pilot and finite differences",
        ha="center", fontsize=6.6, color="#687176",
    )
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.25, top=0.86, wspace=0.52)
    pdf = output / "offset-seam-sweep.pdf"
    png = output / "offset-seam-sweep.png"
    fig.savefig(pdf, bbox_inches="tight")
    temporary = png.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=420, bbox_inches="tight")
    temporary.replace(png)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    channel = load_module(HERE / "run_channel_benchmark.py", "channel_benchmark")
    housing = channel.load_housing_module()
    source = Path(args.channel_output)
    output = Path(args.output)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)

    controls, centers, weights, radius, channel_summary = frozen_geometry(
        channel, source
    )
    subdivisions = np.asarray(
        channel_summary["problem"]["segment_subdivisions"], dtype=int
    )
    mesh = HERE / "assets" / "stylized-finned-housing.ply"
    if not mesh.is_file():
        housing.generate_mesh(mesh)
    executable = housing.load_compiled_module().build_cpp()
    probes = housing.probe_points(args.probes)
    volume = housing.volume_estimate()
    offsets = np.asarray(args.offsets, dtype=float)
    if np.any(offsets <= 0.001):
        raise ValueError("all offsets must exceed the fixed WoS shell epsilon 0.001")

    pilot_gradient, _, pilot_summary = cached_diffwos(
        channel, housing, executable, mesh, work, "offset-pilot",
        probes, centers, radius, volume, args.pilot_walks, SEED + 41000,
        args.threads, args.pilot_offset, args.force,
    )
    pilot_control_gradient = weights.T @ pilot_gradient
    directions, signals = select_directions(
        channel, housing, controls, subdivisions, radius,
        pilot_control_gradient, args.directions, args.fd_step,
        args.surface_clearance,
    )

    direction_rows: list[dict[str, object]] = []
    for direction_index, (direction, signal) in enumerate(zip(directions, signals)):
        for control_index, vector in enumerate(direction):
            direction_rows.append({
                "direction": direction_index,
                "control": control_index,
                "dx": vector[0], "dy": vector[1], "dz": vector[2],
                "pilot_absolute_signal": signal,
            })
    atomic_csv(output / "offset-sweep-directions.csv", direction_rows)

    fd_rows: list[dict[str, object]] = []
    for direction_index, direction in enumerate(directions):
        for repeat in range(args.fd_repeats):
            seed = SEED + 43000 + 100 * direction_index + repeat
            values = []
            for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                perturbed = controls + sign * args.fd_step * direction
                perturbed_centers, _ = channel.dense_polyline(
                    perturbed, subdivisions
                )
                value = cached_objective(
                    housing, executable, mesh, work,
                    f"offset-fd-d{direction_index:02d}-r{repeat:02d}-{sign_name}",
                    probes, perturbed_centers, radius, volume,
                    args.fd_walks, seed, args.threads, args.force,
                )
                values.append(value)
            fd_rows.append({
                "direction": direction_index, "replicate": repeat,
                "seed": seed, "step": args.fd_step,
                "objective_minus": values[0], "objective_plus": values[1],
                "finite_difference": (values[1] - values[0]) /
                    (2.0 * args.fd_step),
            })
        atomic_csv(output / "offset-sweep-finite-differences.csv", fd_rows)

    run_rows: list[dict[str, object]] = []
    directional_rows: list[dict[str, object]] = []
    for offset_index, offset in enumerate(offsets):
        for repeat in range(args.repeats):
            seed = SEED + 45000 + repeat
            tag = f"offset-e{int(round(offset * 1.0e6)):05d}-r{repeat:02d}"
            sphere_gradient, _, summary = cached_diffwos(
                channel, housing, executable, mesh, work, tag,
                probes, centers, radius, volume, args.walks, seed,
                args.threads, float(offset), args.force,
            )
            control_gradient = weights.T @ sphere_gradient
            predictions = [
                float(np.sum(control_gradient * direction))
                for direction in directions
            ]
            run_rows.append({
                "offset": offset, "ell_over_radius": offset / radius,
                "replicate": repeat, "seed": seed,
                "sites": summary["sites"],
                "walks_per_site": summary["walks_per_site"],
                "outer_paths": int(summary["sites"]) *
                    int(summary["walks_per_site"]),
                "runtime_seconds": summary["runtime_seconds_median"],
                "cavity_termination_fraction": summary["cavity_termination_fraction"],
                "near_seam_fraction": summary["near_seam_fraction"],
                "segment_unsafe_fraction": summary["segment_unsafe_fraction"],
                "unsafe_offset_fraction": summary["unsafe_offset_fraction"],
                "unsafe_given_cavity_fraction":
                    summary["unsafe_given_cavity_fraction"],
                "near_seam_terminations": summary["near_seam_terminations"],
                "segment_unsafe_terminations":
                    summary["segment_unsafe_terminations"],
                "unsafe_offset_terminations":
                    summary["unsafe_offset_terminations"],
            })
            for direction_index, prediction in enumerate(predictions):
                directional_rows.append({
                    "offset": offset, "ell_over_radius": offset / radius,
                    "replicate": repeat, "seed": seed,
                    "direction": direction_index,
                    "diffwos_directional_derivative": prediction,
                })
            atomic_csv(output / "offset-sweep-runs.csv", run_rows)
            atomic_csv(output / "offset-sweep-directional.csv", directional_rows)

    fd_means = np.asarray([
        np.mean([float(row["finite_difference"]) for row in fd_rows
                 if int(row["direction"]) == direction])
        for direction in range(args.directions)
    ])
    predicted_means = np.empty((len(offsets), args.directions))
    predicted_ses = np.empty_like(predicted_means)
    for offset_index, offset in enumerate(offsets):
        for direction in range(args.directions):
            values = np.asarray([
                float(row["diffwos_directional_derivative"])
                for row in directional_rows
                if int(row["direction"]) == direction
                and math.isclose(float(row["offset"]), float(offset))
            ])
            predicted_means[offset_index, direction] = float(np.mean(values))
            predicted_ses[offset_index, direction] = sample_se(values)

    fit_mask = offsets <= args.linear_fit_max_offset + 1.0e-15
    design = np.column_stack((np.ones(np.count_nonzero(fit_mask)), offsets[fit_mask]))
    extrapolated = np.empty(args.directions)
    linear_slopes = np.empty(args.directions)
    for direction in range(args.directions):
        coefficients, *_ = np.linalg.lstsq(
            design, predicted_means[fit_mask, direction], rcond=None
        )
        extrapolated[direction], linear_slopes[direction] = coefficients

    metric_rows: list[dict[str, object]] = []
    for offset_index, offset in enumerate(offsets):
        predicted = predicted_means[offset_index]
        slope = float(np.dot(predicted, fd_means) /
                      max(np.dot(predicted, predicted), 1.0e-30))
        cosine = float(np.dot(predicted, fd_means) /
                       max(np.linalg.norm(predicted) * np.linalg.norm(fd_means),
                           1.0e-30))
        metric_rows.append({
            "offset": offset, "ell_over_radius": offset / radius,
            "pearson": float(np.corrcoef(predicted, fd_means)[0, 1]),
            "cosine_similarity": cosine,
            "slope_fd_on_diffwos": slope,
            "relative_l2_error": float(
                np.linalg.norm(predicted - fd_means) /
                max(np.linalg.norm(fd_means), 1.0e-30)
            ),
            "scale_corrected_relative_l2_error": float(
                np.linalg.norm(slope * predicted - fd_means) /
                max(np.linalg.norm(fd_means), 1.0e-30)
            ),
            "relative_distance_to_linear_extrapolate": float(
                np.linalg.norm(predicted - extrapolated) /
                max(np.linalg.norm(extrapolated), 1.0e-30)
            ),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(fd_means))),
        })
    atomic_csv(output / "offset-sweep-metrics.csv", metric_rows)

    unsafe_means = np.asarray([
        np.mean([float(row["unsafe_offset_fraction"]) for row in run_rows
                 if math.isclose(float(row["offset"]), float(offset))])
        for offset in offsets
    ])
    unsafe_exponent, unsafe_prefactor, unsafe_r_squared = fit_loglog(
        offsets / radius, unsafe_means
    )
    conditional_means = np.asarray([
        np.mean([float(row["unsafe_given_cavity_fraction"]) for row in run_rows
                 if math.isclose(float(row["offset"]), float(offset))])
        for offset in offsets
    ])
    conditional_exponent, _, conditional_r_squared = fit_loglog(
        offsets / radius, conditional_means
    )
    fd_extrapolated_slope = float(
        np.dot(extrapolated, fd_means) /
        max(np.dot(extrapolated, extrapolated), 1.0e-30)
    )
    fd_extrapolated_cosine = float(
        np.dot(extrapolated, fd_means) /
        max(np.linalg.norm(extrapolated) * np.linalg.norm(fd_means), 1.0e-30)
    )
    total_segment_unsafe = int(sum(
        int(row["segment_unsafe_terminations"]) for row in run_rows
    ))
    total_paths = int(sum(int(row["outer_paths"]) for row in run_rows))
    source_hash = hashlib.sha256(
        (ROOT / "benchmarks" / "compiled_bunny" / "src" / "wih_twos.cpp")
        .read_bytes()
    ).hexdigest()
    summary = {
        "seed": SEED,
        "stage": "frozen pre-refinement TD graph channel",
        "geometry": {
            "controls": len(controls), "dense_balls": len(centers),
            "radius": radius, "subdivisions": subdivisions.tolist(),
        },
        "protocol": {
            "probes": args.probes, "offsets": offsets.tolist(),
            "normalized_offsets": (offsets / radius).tolist(),
            "diffwos_walks_per_probe": args.walks,
            "diffwos_independent_replicates": args.repeats,
            "matched_seeds_across_offsets": True,
            "pilot_offset": args.pilot_offset,
            "pilot_walks_per_probe": args.pilot_walks,
            "pilot_seed": SEED + 41000,
            "direction_count": args.directions,
            "direction_selection": (
                "separate pilot descent plus seeded feasible top-30%-signal "
                "length-tangent directions"
            ),
            "finite_difference_step": args.fd_step,
            "finite_difference_walks_per_probe": args.fd_walks,
            "finite_difference_paired_replicates": args.fd_repeats,
            "wos_epsilon": 0.001,
            "seam_strip_multiplier": 1.0,
            "linear_extrapolation_offsets": offsets[fit_mask].tolist(),
        },
        "results": {
            "unsafe_exit_loglog_exponent": unsafe_exponent,
            "unsafe_exit_loglog_prefactor_normalized_offset": unsafe_prefactor,
            "unsafe_exit_loglog_r_squared": unsafe_r_squared,
            "conditional_unsafe_loglog_exponent": conditional_exponent,
            "conditional_unsafe_loglog_r_squared": conditional_r_squared,
            "total_sweep_outer_paths": total_paths,
            "total_near_seam_terminations": int(sum(
                int(row["near_seam_terminations"]) for row in run_rows
            )),
            "total_segment_unsafe_terminations": total_segment_unsafe,
            "total_unsafe_offset_terminations": int(sum(
                int(row["unsafe_offset_terminations"]) for row in run_rows
            )),
            "fd_on_linear_extrapolate_slope": fd_extrapolated_slope,
            "fd_linear_extrapolate_cosine": fd_extrapolated_cosine,
            "linear_directional_slopes": linear_slopes.tolist(),
            "linear_extrapolated_directional_derivatives": extrapolated.tolist(),
            "metrics_by_offset": metric_rows,
        },
        "pilot_summary": pilot_summary,
        "compiled_source_sha256": source_hash,
        "scope": (
            "The measured seam-strip exponent is an empirical audit of the "
            "frozen channel and is not a proof of the harmonic-measure hypothesis. "
            "Central differences have finite step and Monte Carlo error."
        ),
    }
    housing.atomic_json(output / "offset-sweep-summary.json", summary)
    make_figure(
        output, radius, offsets, directional_rows, fd_rows, metric_rows,
        run_rows, unsafe_exponent, total_segment_unsafe, total_paths,
    )
    print(json.dumps(summary["results"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-output", default=str(HERE / "output" / "channels")
    )
    parser.add_argument(
        "--output", default=str(HERE / "output" / "channels" / "offset_sweep")
    )
    parser.add_argument(
        "--offsets", type=float, nargs="+",
        default=[0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016],
    )
    parser.add_argument("--probes", type=int, default=384)
    parser.add_argument("--walks", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pilot-offset", type=float, default=0.004)
    parser.add_argument("--pilot-walks", type=int, default=8192)
    parser.add_argument("--directions", type=int, default=6)
    parser.add_argument("--fd-step", type=float, default=0.012)
    parser.add_argument("--fd-walks", type=int, default=4096)
    parser.add_argument("--fd-repeats", type=int, default=3)
    parser.add_argument("--linear-fit-max-offset", type=float, default=0.008)
    parser.add_argument("--surface-clearance", type=float, default=0.006)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
