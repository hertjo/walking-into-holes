"""Manufactured finite-benefit check; not an application benchmark.

Run from any directory. The PDE is -Delta u = 1 in the unit ball with u=0
on its boundary. Sampling, inequalities and acceptance follow Appendix
Minimum-Benefit Certification. Floating-point checks are numerical evidence,
not formal verification of real-arithmetic oracles or confidence coverage.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cooling_lower(radius, containing_radius, state_lower, source_lower=1., source_upper=1.):
    r, R = np.asarray(radius), containing_radius
    if not (R > 0 and np.all((r > 0) & (r < R))):
        raise ValueError("Require 0 < cavity radius < containing radius.")
    if not (0 <= source_lower <= source_upper):
        raise ValueError("Require valid nonnegative source bounds.")
    ratio = r / R
    harmonic = (1-ratio)/(1+ratio)**2
    floor = source_lower*(R*R-r*r)/6 + harmonic*max(0., state_lower-source_upper*R*R/6)
    return 4*math.pi*r*floor**2


def exact_benefit(r):
    r = np.asarray(r)
    return math.pi*r*(1-r)*(1+r)**2/9 + 4*math.pi*r**5/45


def unit_ball_walks(n, steps, radius_cap, seed, shell=1e-6):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    total = np.zeros(n)
    active = np.ones(n, dtype=bool)
    counts = np.zeros(n, dtype=int)
    for _ in range(steps):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        distance = 1-np.linalg.norm(x[ids], axis=1)
        assert np.all(distance > 0)
        stop = distance <= shell
        active[ids[stop]] = False
        ids, distance = ids[~stop], distance[~stop]
        radii = np.minimum(radius_cap, .95*distance)
        direction = rng.normal(size=(len(ids), 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        # Source is exactly one, so its Green integral is deterministic.
        total[ids] += radii**2/6
        x[ids] += radii[:, None]*direction
        counts[ids] += 1
    residual = (1-np.sum(x*x, axis=1))/6
    assert np.all(residual >= 0)
    return total, residual, counts


def main():
    seed, n, steps, cap, delta = 2026091407, 65536, 32, .35, .01
    R, chosen_r, target_fraction = .9, .1, .10
    J0, center = 4*math.pi/45, 1/6
    W, residual, counts = unit_ball_walks(n, steps, cap, seed)
    A = steps*cap**2/6
    margin = A*math.sqrt(math.log(1/delta)/(2*n))
    L = max(0., float(W.mean())-margin)
    assert np.all((0 <= W) & (W <= A))
    grid = np.linspace(.0001, R-.0001, 2001)
    exact = exact_benefit(grid)
    oracle_bound = cooling_lower(grid, R, center)
    floor_bound = math.pi*grid*(1-grid**2)**2/9
    source_only = cooling_lower(grid, R, 0.)
    assert np.all(source_only <= oracle_bound+1e-14)
    assert np.all(oracle_bound <= floor_bound+1e-14)
    assert np.all(floor_bound <= exact+1e-14)
    # Independent quadrature of J0 - integral(v), not the energy expression.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    errors = []
    for r in [.001, .03, .1, .25, .5, .8, .95]:
        s = (1-r)*(nodes+1)/2+r
        v = (1-s*s)/6-r*(1+r)/6*(1/s-1)
        Jr = 4*math.pi*(1-r)/2*np.sum(weights*s*s*v)
        errors.append(abs(J0-Jr-float(exact_benefit(r))))
    assert max(errors) < 1e-13
    # The telescoping equality is a diagnostic with sampling uncertainty.
    telescope = W+residual
    se = float(telescope.std(ddof=1)/math.sqrt(n))
    lower = float(cooling_lower(chosen_r, R, L))
    actual = float(exact_benefit(chosen_r))
    source = float(cooling_lower(chosen_r, R, 0.))
    target = target_fraction*J0
    report = {
        "scope": "Manufactured unit-ball numerical check; not a production-geometry certificate or coverage experiment.",
        "oracle_assumptions": "Mathematical guarantees assume exact contained-ball geometry, conditional sampling laws and bounded source contributions; this implementation uses floating-point arithmetic and a pseudorandom generator.",
        "seed": seed, "n_independent_walks": n, "max_steps": steps,
        "radius_cap": cap, "distance_fraction": .95, "shell": 1e-6,
        "delta": delta, "source": 1., "containing_radius": R,
        "selected_cavity_radius": chosen_r, "fixed_target_fraction": target_fraction,
        "sample_range_upper": A, "sample_mean": float(W.mean()),
        "sampling_margin": margin, "state_lower": L, "state_exact": center,
        "mean_terminal_state": float(residual.mean()),
        "telescope_mean": float(telescope.mean()), "telescope_standard_error": se,
        "telescope_z_score": float((telescope.mean()-center)/se),
        "mean_steps": float(counts.mean()), "baseline_compliance": J0,
        "source_only_benefit_lower": source, "mc_benefit_lower": lower,
        "exact_benefit": actual, "target_absolute": target,
        "source_only_accepts": source >= target, "mc_accepts": lower >= target,
        "state_coverage_on_this_seed": L <= center,
        "certified_fraction": lower/J0, "actual_fraction": actual/J0,
        "radius_grid_count": len(grid), "max_quadrature_identity_error": max(errors),
        "analytical_grid_inequalities_pass": True,
    }
    out = ROOT/'experiments/output/minimum-benefit-verification.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    np.savetxt(ROOT/'experiments/output/minimum-benefit-radius-check.csv',
               np.column_stack((grid, exact, floor_bound, oracle_bound, source_only)),
               delimiter=',', header='radius,exact_benefit,exact_floor_capacity_bound,center_oracle_bound,source_only_bound', comments='')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
