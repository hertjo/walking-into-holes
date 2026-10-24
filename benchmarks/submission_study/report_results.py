"""A compact, data-derived reading guide to the completed campaign."""
import json
import pandas as pd
from run_study import OUT, SPECS

NAMES = {'twos':'TWoS','mg':'Multigrid TD (65)','mg49':'Multigrid TD (49)',
         'nearest':'Nearest sensor','hot':'Hottest point','random':'Random'}

def report():
    g = pd.read_csv(OUT/'group-results.csv')
    e = pd.read_csv(OUT/'estimator-results.csv')
    lines = ['# Matched experiment results', '',
        'All entries below are generated from the saved measurements. Point estimates do not establish a winner, equivalence, or a guarantee on another geometry.', '',
        'Design intervals jointly resample five paired design seeds and four shared evaluation blocks. SD describes the observed spread across five evaluated designs, including evaluation noise. Intervals are approximate, pointwise, and do not cover numerical bias.', '']
    for study in ['pipeline','channels']:
        lines += ['## '+study.title(), '',
            '| Geometry | Method | Mean cooling (%) | Seed SD (pp) | 95% interval | Complete time (s) |',
            '|---|---|---:|---:|---:|---:|']
        for name in SPECS:
            for method in NAMES:
                state = 'initial' if study == 'channels' else ('refine16' if method in ['hot','random'] else 'refine8')
                r = g.query('study==@study and geometry==@name and method==@method and state==@state').iloc[0]
                lines.append(f'| {name} | {NAMES[method]} | {r.reduction_pct:.3f} | {r.seed_sd_pp:.3f} | [{r.cooling_ci95_low:.3f}, {r.cooling_ci95_high:.3f}] | {r.seconds:.3f} |')
        lines.append('')
    lines += ['## Actual-mesh estimator variance', '',
        'All entries use 2,048 paths per query site, eight fixed sites, and sixteen repetitions. Ratios compare measured sums of site variances. The finite-cavity estimators target finite differences and retain radius bias relative to the infinitesimal coefficient. These ratios are descriptive point estimates, not confidence bounds.', '',
        '| Geometry | Trial radius | CRN variance / TWoS variance | Harmonic-correction variance / TWoS variance | Observed correction hit fraction |',
        '|---|---:|---:|---:|---:|']
    for name in SPECS:
        direct = e.query('geometry==@name and method=="twos" and budget==2048').iloc[0]
        raw = pd.DataFrame(json.loads((OUT/'estimators'/name/'results.json').read_text())['rows'])
        for radius in [.08,.04,.02,.01]:
            a = e.query('geometry==@name and method=="coupled-fd" and budget==2048 and radius==@radius').iloc[0]
            b = e.query('geometry==@name and method=="hit-correction" and budget==2048 and radius==@radius').iloc[0]
            hits = raw.query('method=="hit-correction" and walks==2048 and radius==@radius').hits.mean()/2048
            ratio = 'unresolved (all zero)' if b.variance_status=='unresolved_all_zero' else f'{b.normalized_variance/direct.normalized_variance:.3g}'
            lines.append(f'| {name} | {radius:.2f} | {a.normalized_variance/direct.normalized_variance:.3g} | {ratio} | {hits:.5g} |')
    lines += ['', '## Interpretation limits', '',
        '- The shape-refinement component is a specialized Differential WoS implementation shared by all initializations; it is not a reproduction of the authors\' full application.',
        '- Sensor proximity and tuned multigrid are informed controls. Improvements over hottest-point placement alone do not establish superiority over these controls.',
        '- Channel heat input is fixed by protected compact source supports; removed volume is calibrated and independently audited numerically. No fluid flow is solved.',
        '- Numerical references and rendering fields are approximate. See `reference-audit.csv` and `finite-grid-audit.csv`; sampling intervals do not enclose their discretization or termination errors.',
        '- Query timings replay direct TWoS with chunk size one to use all four workers at eight sites. Component fields match the original outputs bitwise. Complete-design timings retain their original scheduling.',
        '- Runs were serial on a managed shared machine. Timing intervals include algorithmic and machine variation.',
        '- Renderings show the first predetermined seed; statistics use all five. Older hero bands remain observed ranges over three designs, not 95% confidence intervals.', '']
    (OUT/'results-summary.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    report()
