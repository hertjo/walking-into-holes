#!/usr/bin/env python3
"""Reproducible controlled experiments for Walking into Holes.

The experiments in this file are intentionally diagnostic rather than a
substitute for the final 3D applications.  They test four precise claims:

1. A natural coupled finite-cavity quotient has a rare-event variance barrier.
2. Leave-diagonal-out cross-fitting removes arbitrary within-pair covariance
   bias without requiring independent primal and adjoint batches.
3. Fresh validation controls false insertions after adaptive spatial search.
4. The cross-fitted estimator reconstructs a manufactured harmonic-pair
   product using actual Walk-on-Spheres exit samples (not a PDE adjoint).

Every figure is generated from raw CSV files emitted alongside a JSON summary.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "output"
FIGURES = ROOT / "figures"
SEED = 20260912

INK = "#242323"
MUTED = "#74787a"
PRIMAL = "#efa657"
ADJOINT = "#788fe4"
TOPOLOGY = "#d75d56"
MINT = "#4aa388"
GRAY = "#9da1a3"
PALE = "#ececea"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "grid.color": "#d7d8d6",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_png_atomic(fig: mpl.figure.Figure, path: Path, **kwargs: object) -> None:
    """Write previews through a temporary inode so open viewers cannot race."""
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, format="png", **kwargs)
    temporary.replace(path)


def crossfit_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Leave-diagonal-out estimate of -E[A]E[B] along the last axis."""
    n = a.shape[-1]
    if n < 2:
        raise ValueError("cross-fitting requires at least two pairs")
    sum_a = np.sum(a, axis=-1)
    sum_b = np.sum(b, axis=-1)
    sum_ab = np.sum(a * b, axis=-1)
    return -(sum_a * sum_b - sum_ab) / (n * (n - 1))


def bootstrap_variance_interval(
    estimates: np.ndarray, rng: np.random.Generator, draws: int = 400
) -> tuple[float, float]:
    """Percentile bootstrap interval for variance across independent runs."""
    n = estimates.size
    indices = rng.integers(0, n, size=(draws, n))
    boot = np.var(estimates[indices], axis=1, ddof=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval for a Bernoulli rate."""
    z = norm.ppf(0.975)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return float(center - radius), float(center + radius)


def disk_harmonic_exit(
    start_points: np.ndarray,
    samples_per_point: int,
    rng: np.random.Generator,
    tolerance: float = 1.0e-7,
    max_steps: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unit-disk exit points with a vectorized Walk on Spheres.

    The largest centered ball has radius 1-|x|.  Walks stop in a thin shell and
    are projected radially to the exact boundary.  The chosen shell makes the
    resulting deterministic termination error far smaller than sampling error
    in the displayed experiment.
    """
    starts = np.asarray(start_points, dtype=np.complex128)
    points = np.repeat(starts, samples_per_point)
    steps = np.zeros(points.size, dtype=np.int16)

    for _ in range(max_steps):
        distance = 1.0 - np.abs(points)
        active = distance > tolerance
        if not np.any(active):
            break
        radii = distance[active]
        angles = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(active)))
        points[active] += radii * np.exp(1j * angles)
        steps[active] += 1
    else:
        raise RuntimeError("Walk on Spheres did not reach the termination shell")

    points /= np.abs(points)
    shape = (starts.size, samples_per_point)
    return points.reshape(shape), steps.reshape(shape)


def run_variance_experiment(
    rng: np.random.Generator, bootstrap_rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Exact annulus benchmark for the rare-event variance theorem."""
    outer_radius = 1.0
    start_radius = 0.35
    n = 256
    replicates = 3000
    epsilons = np.power(10.0, -np.linspace(1.0, 24.0, 12))
    log_start = np.log(outer_radius / start_radius)
    true_td = -log_start / (2.0 * np.pi)
    p_mean = -true_td / start_radius
    p_fluctuation = 0.45

    # The direct estimator has no epsilon input.  Estimate its distribution
    # once, then place that same measured variance against every capacity.
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(replicates, n))
    w = np.exp(1j * phi)
    exit_points = (start_radius + w) / (1.0 + start_radius * w)
    cosine = np.real(exit_points)
    u_samples = cosine
    p_samples = p_mean + p_fluctuation * (cosine - start_radius)
    direct = crossfit_product(u_samples, p_samples)
    direct_variance = float(np.var(direct, ddof=1))
    direct_ci = bootstrap_variance_interval(direct, bootstrap_rng)

    rows: list[dict[str, float]] = []
    estimates_by_method: dict[str, list[np.ndarray]] = {
        "Direct TWoS": [],
        "Coupled finite cavity": [],
        "Independent finite cavity": [],
    }

    for epsilon in epsilons:
        capacity = 2.0 * np.pi / np.log(outer_radius / epsilon)
        hit_probability = log_start / np.log(outer_radius / epsilon)

        # A Brownian path hits the inner disk before the outer boundary with
        # this exact probability.  The coupled payoff is nonzero only then.
        hit = rng.random(size=(replicates, n)) < hit_probability
        coupled = -np.mean(hit, axis=1) / capacity

        # Independent solves retain O(1) per-path objective noise that common
        # random numbers would have cancelled.
        base_eps = rng.uniform(-0.5, 0.5, size=(replicates, n))
        base_zero = rng.uniform(-0.5, 0.5, size=(replicates, n))
        independent = np.mean(base_eps - base_zero - hit, axis=1) / capacity

        for method, values in (
            ("Direct TWoS", direct),
            ("Coupled finite cavity", coupled),
            ("Independent finite cavity", independent),
        ):
            if method == "Direct TWoS":
                variance = direct_variance
                ci_low, ci_high = direct_ci
            else:
                variance = float(np.var(values, ddof=1))
                ci_low, ci_high = bootstrap_variance_interval(values, bootstrap_rng)
            rows.append(
                {
                    "method": method,
                    "epsilon": float(epsilon),
                    "capacity": float(capacity),
                    "hit_probability": float(hit_probability),
                    "true_td": float(true_td),
                    "mean_estimate": float(np.mean(values)),
                    "variance": variance,
                    "variance_ci_low": ci_low,
                    "variance_ci_high": ci_high,
                    "replicates": replicates,
                    "samples_per_replicate": n,
                }
            )
            estimates_by_method[method].append(values)

    frame = pd.DataFrame(rows)
    small_capacity = frame[
        frame["capacity"] <= float(np.median(frame["capacity"]))
    ]
    slopes: dict[str, float] = {}
    for method in estimates_by_method:
        subset = small_capacity[small_capacity["method"] == method]
        slopes[method] = float(
            np.polyfit(np.log(subset["capacity"]), np.log(subset["variance"]), 1)[0]
        )

    direct_rows = frame[frame["method"] == "Direct TWoS"]
    summary = {
        "true_td": float(true_td),
        "direct_loglog_slope": slopes["Direct TWoS"],
        "coupled_fd_loglog_slope": slopes["Coupled finite cavity"],
        "independent_fd_loglog_slope": slopes["Independent finite cavity"],
        "direct_variance_max_to_min": float(
            direct_rows["variance"].max() / direct_rows["variance"].min()
        ),
    }
    return frame, summary


def run_crossfit_experiment(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Bias under arbitrary within-pair correlation."""
    n = 32
    replicates = 30000
    mu_a, mu_b = 0.8, 0.65
    sigma_a, sigma_b = 0.5, 0.6
    true_td = -mu_a * mu_b
    correlations = np.linspace(-0.9, 0.9, 10)
    rows: list[dict[str, float]] = []

    for correlation in correlations:
        z1 = rng.standard_normal(size=(replicates, n))
        z2 = rng.standard_normal(size=(replicates, n))
        z3 = rng.standard_normal(size=(replicates, n))
        a = mu_a + sigma_a * z1
        b = mu_b + sigma_b * (
            correlation * z1 + np.sqrt(1.0 - correlation * correlation) * z2
        )
        b_independent = mu_b + sigma_b * z3
        covariance = correlation * sigma_a * sigma_b

        estimates = {
            "Same-seed product": -np.mean(a * b, axis=1),
            "Product of means": -np.mean(a, axis=1) * np.mean(b, axis=1),
            "Cross-fit TWoS": crossfit_product(a, b),
            "Independent batches": -np.mean(a, axis=1)
            * np.mean(b_independent, axis=1),
        }
        theoretical_bias = {
            "Same-seed product": -covariance,
            "Product of means": -covariance / n,
            "Cross-fit TWoS": 0.0,
            "Independent batches": 0.0,
        }

        for method, values in estimates.items():
            error = values - true_td
            rows.append(
                {
                    "method": method,
                    "correlation": float(correlation),
                    "true_td": float(true_td),
                    "mean_estimate": float(np.mean(values)),
                    "bias": float(np.mean(error)),
                    "bias_se": float(np.std(error, ddof=1) / np.sqrt(replicates)),
                    "theoretical_bias": float(theoretical_bias[method]),
                    "rmse": float(np.sqrt(np.mean(error * error))),
                    "replicates": replicates,
                    "pairs_per_replicate": n,
                }
            )

    frame = pd.DataFrame(rows)
    high_corr = frame[np.isclose(frame["correlation"], 0.9)]
    summary = {
        "true_td": float(true_td),
        "same_seed_bias_at_corr_0.9": float(
            high_corr.loc[high_corr["method"] == "Same-seed product", "bias"].iloc[0]
        ),
        "plugin_bias_at_corr_0.9": float(
            high_corr.loc[high_corr["method"] == "Product of means", "bias"].iloc[0]
        ),
        "crossfit_bias_at_corr_0.9": float(
            high_corr.loc[high_corr["method"] == "Cross-fit TWoS", "bias"].iloc[0]
        ),
        "independent_bias_at_corr_0.9": float(
            high_corr.loc[high_corr["method"] == "Independent batches", "bias"].iloc[0]
        ),
    }
    return frame, summary


def run_selection_experiment(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """False certification after choosing the noisiest of K null sites."""
    alpha = 0.05
    n = 64
    sigma = 1.0
    standard_error = sigma / np.sqrt(n)
    replicates = 40000
    candidate_counts = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    rows: list[dict[str, float]] = []

    for k in candidate_counts:
        scout = standard_error * rng.standard_normal(size=(replicates, int(k)))
        selected = np.min(scout, axis=1)
        pointwise_cutoff = norm.ppf(1.0 - alpha) * standard_error
        bonferroni_cutoff = norm.ppf(1.0 - alpha / k) * standard_error
        fresh = standard_error * rng.standard_normal(size=replicates)

        decisions = {
            "Reuse selected samples": selected + pointwise_cutoff < 0.0,
            "Bonferroni scout": selected + bonferroni_cutoff < 0.0,
            "Fresh validation": fresh + pointwise_cutoff < 0.0,
        }

        for method, accepted in decisions.items():
            successes = int(np.sum(accepted))
            ci_low, ci_high = wilson_interval(successes, replicates)
            rows.append(
                {
                    "method": method,
                    "candidates": int(k),
                    "false_accept_rate": successes / replicates,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_alpha": alpha,
                    "replicates": replicates,
                    "samples_per_candidate": n,
                }
            )

    frame = pd.DataFrame(rows)
    at_256 = frame[frame["candidates"] == 256]
    summary = {
        "nominal_alpha": alpha,
        "naive_false_accept_at_k256": float(
            at_256.loc[
                at_256["method"] == "Reuse selected samples", "false_accept_rate"
            ].iloc[0]
        ),
        "bonferroni_false_accept_at_k256": float(
            at_256.loc[
                at_256["method"] == "Bonferroni scout", "false_accept_rate"
            ].iloc[0]
        ),
        "fresh_false_accept_at_k256": float(
            at_256.loc[
                at_256["method"] == "Fresh validation", "false_accept_rate"
            ].iloc[0]
        ),
    }
    return frame, summary


def harmonic_u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.28 + 0.58 * x - 0.30 * y + 0.22 * (x * x - y * y)


def harmonic_p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.46 - 0.38 * x + 0.34 * y + 0.20 * (2.0 * x * y)


def run_spatial_experiment(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    """Manufactured harmonic-pair product diagnostic queried by actual WoS."""
    grid_size = 91
    candidate_radius = 0.88
    samples_per_point = 96
    coordinates = np.linspace(-candidate_radius, candidate_radius, grid_size)
    xx, yy = np.meshgrid(coordinates, coordinates)
    mask = xx * xx + yy * yy <= candidate_radius * candidate_radius
    x = xx[mask]
    y = yy[mask]
    starts = x + 1j * y

    exits, steps = disk_harmonic_exit(starts, samples_per_point, rng)
    boundary_x = np.real(exits)
    boundary_y = np.imag(exits)
    u_samples = harmonic_u(boundary_x, boundary_y)
    p_samples = harmonic_p(boundary_x, boundary_y)

    true_u = harmonic_u(x, y)
    true_p = harmonic_p(x, y)
    true_td = -true_u * true_p
    same_seed = -np.mean(u_samples * p_samples, axis=1)
    product_means = -np.mean(u_samples, axis=1) * np.mean(p_samples, axis=1)
    crossfit = crossfit_product(u_samples, p_samples)

    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "true_u": true_u,
            "true_p": true_p,
            "true_td": true_td,
            "same_seed_td": same_seed,
            "product_means_td": product_means,
            "crossfit_td": crossfit,
            "mean_wos_steps": np.mean(steps, axis=1),
        }
    )
    for method in ("same_seed", "product_means", "crossfit"):
        frame[f"{method}_error"] = frame[f"{method}_td"] - frame["true_td"]

    def metrics(error: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.sqrt(np.mean(error * error))),
            float(np.mean(error)),
            float(np.quantile(np.abs(error), 0.95)),
        )

    same_metrics = metrics(same_seed - true_td)
    plugin_metrics = metrics(product_means - true_td)
    crossfit_metrics = metrics(crossfit - true_td)
    summary = {
        "grid_size": grid_size,
        "candidate_count": int(x.size),
        "samples_per_candidate": samples_per_point,
        "mean_wos_steps": float(np.mean(steps)),
        "same_seed_rmse": same_metrics[0],
        "same_seed_mean_bias": same_metrics[1],
        "same_seed_p95_abs_error": same_metrics[2],
        "product_means_rmse": plugin_metrics[0],
        "product_means_mean_bias": plugin_metrics[1],
        "product_means_p95_abs_error": plugin_metrics[2],
        "crossfit_rmse": crossfit_metrics[0],
        "crossfit_mean_bias": crossfit_metrics[1],
        "crossfit_p95_abs_error": crossfit_metrics[2],
    }
    arrays = {
        "xx": xx,
        "yy": yy,
        "mask": mask,
        "true_u": true_u,
        "true_p": true_p,
        "true_td": true_td,
        "same_seed_error": same_seed - true_td,
        "product_means_error": product_means - true_td,
        "crossfit_error": crossfit - true_td,
    }
    return frame, summary, arrays


def plot_controlled_summary(
    variance: pd.DataFrame,
    crossfit: pd.DataFrame,
    selection: pd.DataFrame,
    variance_summary: dict[str, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35))

    # (a) Radius scaling.
    ax = axes[0]
    styles = {
        "Direct TWoS": (ADJOINT, "o", "-"),
        "Coupled finite cavity": (TOPOLOGY, "s", "-"),
        "Independent finite cavity": (GRAY, "^", "-"),
    }
    for method, (color, marker, line) in styles.items():
        data = variance[variance["method"] == method].sort_values("capacity")
        ax.loglog(
            data["capacity"],
            data["variance"],
            marker=marker,
            linestyle=line,
            color=color,
            markersize=3.2,
            linewidth=1.35,
            label=method,
        )
        ax.fill_between(
            data["capacity"].to_numpy(),
            data["variance_ci_low"].to_numpy(),
            data["variance_ci_high"].to_numpy(),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.set_title("a  Vanishing-cavity variance", loc="left")
    ax.set_xlabel(r"relative capacity $\rho(\varepsilon)$")
    ax.set_ylabel("empirical variance")
    ax.grid(True, which="both")
    ax.legend(loc="lower left", handlelength=1.5)
    ax.text(
        0.97,
        0.97,
        "fitted slopes\n"
        + f"direct {variance_summary['direct_loglog_slope']:+.2f}\n"
        + f"coupled {variance_summary['coupled_fd_loglog_slope']:+.2f}\n"
        + f"indep. {variance_summary['independent_fd_loglog_slope']:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=6.7,
    )

    # (b) Correlation bias.
    ax = axes[1]
    styles_bias = {
        "Same-seed product": (TOPOLOGY, "o", "-"),
        "Product of means": (PRIMAL, "s", "-"),
        "Cross-fit TWoS": (MINT, "D", "-"),
        "Independent batches": (GRAY, "^", "--"),
    }
    for method, (color, marker, line) in styles_bias.items():
        data = crossfit[crossfit["method"] == method].sort_values("correlation")
        ax.plot(
            data["correlation"],
            data["bias"],
            marker=marker,
            linestyle=line,
            color=color,
            markersize=3.0,
            linewidth=1.25,
            label=method,
        )
        if method in ("Same-seed product", "Product of means"):
            ax.plot(
                data["correlation"],
                data["theoretical_bias"],
                color=color,
                linewidth=0.7,
                alpha=0.45,
                linestyle=":",
            )
    ax.axhline(0.0, color=INK, linewidth=0.75)
    ax.set_title("b  Bias from shared randomness", loc="left")
    ax.set_xlabel("within-pair correlation")
    ax.set_ylabel(r"bias in $\widehat D_T$")
    ax.grid(True)
    ax.legend(loc="lower left", handlelength=1.5)

    # (c) Post-selection false acceptance.
    ax = axes[2]
    styles_selection = {
        "Reuse selected samples": (TOPOLOGY, "o", "-"),
        "Bonferroni scout": (GRAY, "^", "--"),
        "Fresh validation": (MINT, "D", "-"),
    }
    for method, (color, marker, line) in styles_selection.items():
        data = selection[selection["method"] == method].sort_values("candidates")
        ax.plot(
            data["candidates"],
            100.0 * data["false_accept_rate"],
            marker=marker,
            linestyle=line,
            color=color,
            markersize=3.2,
            linewidth=1.3,
            label=method,
        )
        ax.fill_between(
            data["candidates"].to_numpy(),
            100.0 * data["ci_low"].to_numpy(),
            100.0 * data["ci_high"].to_numpy(),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(5.0, color=INK, linestyle=":", linewidth=0.9, label="5% target")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-1.0, 103.0)
    ax.set_title("c  Adaptive search needs fresh data", loc="left")
    ax.set_xlabel("candidate sites $K$")
    ax.set_ylabel("false insertions (%)")
    ax.grid(True)
    ax.legend(loc="center right", handlelength=1.5)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.91, wspace=0.39)
    fig.savefig(FIGURES / "controlled-experiments.pdf", bbox_inches="tight")
    save_png_atomic(
        fig, OUTPUT / "controlled-experiments.png", dpi=320, bbox_inches="tight"
    )
    plt.close(fig)


def plot_spatial_field(
    arrays: dict[str, np.ndarray], summary: dict[str, float]
) -> None:
    xx = arrays["xx"]
    yy = arrays["yy"]
    mask = arrays["mask"]

    def packed(values: np.ndarray) -> np.ndarray:
        image = np.full(xx.shape, np.nan)
        image[mask] = values
        return image

    maps = [
        ("a  Harmonic field $u$", packed(arrays["true_u"]), "coolwarm", None),
        ("b  Harmonic field $p$", packed(arrays["true_p"]), "coolwarm", None),
        ("c  Exact product $-up$", packed(arrays["true_td"]), "coolwarm", None),
        (
            "d  Same-seed product error",
            packed(arrays["same_seed_error"]),
            "RdBu_r",
            summary["same_seed_rmse"],
        ),
        (
            "e  Product-of-means error",
            packed(arrays["product_means_error"]),
            "RdBu_r",
            summary["product_means_rmse"],
        ),
        (
            "f  Cross-fit TWoS error",
            packed(arrays["crossfit_error"]),
            "RdBu_r",
            summary["crossfit_rmse"],
        ),
    ]
    error_limit = float(
        np.quantile(
            np.abs(
                np.concatenate(
                    [
                        arrays["same_seed_error"],
                        arrays["product_means_error"],
                        arrays["crossfit_error"],
                    ]
                )
            ),
            0.985,
        )
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.35))
    for index, (title, values, cmap, rmse) in enumerate(maps):
        ax = axes.flat[index]
        if index >= 3:
            vmin, vmax = -error_limit, error_limit
        else:
            finite = values[np.isfinite(values)]
            if index == 2:
                limit = float(np.max(np.abs(finite)))
                vmin, vmax = -limit, limit
            else:
                vmin, vmax = float(np.min(finite)), float(np.max(finite))
        image = ax.imshow(
            values,
            origin="lower",
            extent=(-0.88, 0.88, -0.88, 0.88),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        domain_boundary = plt.Circle(
            (0.0, 0.0), 1.0, fill=False, color=INK, linewidth=1.15
        )
        candidate_boundary = plt.Circle(
            (0.0, 0.0),
            0.88,
            fill=False,
            color=MUTED,
            linewidth=0.55,
            linestyle=(0, (2.2, 2.2)),
        )
        ax.add_patch(domain_boundary)
        ax.add_patch(candidate_boundary)
        ax.contour(xx, yy, values, levels=7, colors=INK, linewidths=0.26, alpha=0.35)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlim(-1.04, 1.04)
        ax.set_ylim(-1.04, 1.04)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
        colorbar.ax.tick_params(labelsize=5.7, length=2)
        colorbar.outline.set_linewidth(0.45)
        if rmse is not None:
            ax.text(
                0.04,
                0.04,
                f"RMSE {rmse:.3f}",
                transform=ax.transAxes,
                fontsize=6.7,
                fontweight="bold",
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

    fig.text(
        0.5,
        0.015,
        f"Unit-disk WoS • {summary['candidate_count']:,} candidates • "
        f"{summary['samples_per_candidate']} paired walks per candidate • "
        f"mean {summary['mean_wos_steps']:.1f} spheres per walk",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.065, top=0.95, wspace=0.14, hspace=0.18)
    fig.savefig(FIGURES / "manufactured-field.pdf", bbox_inches="tight")
    save_png_atomic(
        fig, OUTPUT / "manufactured-field.png", dpi=320, bbox_inches="tight"
    )
    plt.close(fig)


def validate_results(
    variance: dict[str, float],
    crossfit: dict[str, float],
    selection: dict[str, float],
    spatial: dict[str, float],
) -> None:
    """Fail loudly if a seeded run no longer supports the intended claims."""
    checks = {
        "direct slope is radius-independent": abs(variance["direct_loglog_slope"])
        <= 0.05,
        "coupled finite-cavity slope is near -1": -1.15
        <= variance["coupled_fd_loglog_slope"]
        <= -0.85,
        "independent finite-cavity slope is near -2": -2.20
        <= variance["independent_fd_loglog_slope"]
        <= -1.60,
        "cross-fit bias stays negligible at correlation 0.9": abs(
            crossfit["crossfit_bias_at_corr_0.9"]
        )
        <= 0.004,
        "naive adaptive reuse fails at many candidates": selection[
            "naive_false_accept_at_k256"
        ]
        >= 0.95,
        "fresh validation remains calibrated": abs(
            selection["fresh_false_accept_at_k256"]
            - selection["nominal_alpha"]
        )
        <= 0.006,
        "Bonferroni scout remains calibrated": selection[
            "bonferroni_false_accept_at_k256"
        ]
        <= selection["nominal_alpha"] + 0.006,
        "spatial cross-fit removes the same-seed error floor": spatial[
            "crossfit_rmse"
        ]
        <= spatial["same_seed_rmse"] / 3.0,
        "spatial cross-fit has negligible signed bias": abs(
            spatial["crossfit_mean_bias"]
        )
        <= 0.004,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError("failed experimental checks: " + "; ".join(failures))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    seeds = np.random.SeedSequence(SEED).spawn(5)
    variance, variance_summary = run_variance_experiment(
        np.random.default_rng(seeds[0]), np.random.default_rng(seeds[1])
    )
    crossfit, crossfit_summary = run_crossfit_experiment(
        np.random.default_rng(seeds[2])
    )
    selection, selection_summary = run_selection_experiment(
        np.random.default_rng(seeds[3])
    )
    spatial, spatial_summary, spatial_arrays = run_spatial_experiment(
        np.random.default_rng(seeds[4])
    )

    validate_results(
        variance_summary, crossfit_summary, selection_summary, spatial_summary
    )

    variance.to_csv(OUTPUT / "variance_scaling.csv", index=False)
    crossfit.to_csv(OUTPUT / "crossfit_bias.csv", index=False)
    selection.to_csv(OUTPUT / "selection_coverage.csv", index=False)
    spatial.to_csv(OUTPUT / "manufactured_field.csv", index=False)

    plot_controlled_summary(variance, crossfit, selection, variance_summary)
    plot_spatial_field(spatial_arrays, spatial_summary)

    summary = {
        "scope": "controlled diagnostics; not final 3D application results",
        "seed": SEED,
        "variance": variance_summary,
        "crossfit": crossfit_summary,
        "selection": selection_summary,
        "spatial": spatial_summary,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
