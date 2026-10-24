from pathlib import Path
import json
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'output'
NAMES={'housing':'Housing','branch':'Branch','ring':'Ring','bunny':'Bunny','bunny_compliance':'Bunny compliance'}
def fmt(x,d=3):return '--' if pd.isna(x) else f'{x:.{d}f}'
def main():
 s=pd.read_csv(O/'selected.csv');c=pd.read_csv(O/'reference_checks.csv');d=pd.read_csv(O/'runs.csv')
 primary=s[s.target==.2]
 lines=['# Matched-accuracy query study — September 17, 2026','',
 'The final fixed-budget confirmation supports a narrow result: eight original-domain Bunny compliance queries take 89.9 ms for source-integrated TWoS versus 113.9 ms for MG49 (1.27x; pointwise 95% bootstrap interval 1.05–1.50), with 9.0% mean reference-relative error. After insertion MG is faster in the mean. This is not a general SOTA or full-optimization speed claim.', '',
 'The original sweep was frozen before results. The analytic-source and incremental-grid extensions were added after the first housing result; their budgets and gates were fixed before their measurements. These extensions are disclosed rather than retrospectively described as part of the initial protocol.','',
 'Errors are relative to MG257, with MG193 refinement and independent Monte Carlo spot checks. They are not certified continuum errors. The analytic-source variant is specialized to the prescribed radial loads. A result is the best among tested implementations and budgets, not proof of universal state of the art.','',
 '| Object | State | Queries | Reference resolved | Original TWoS seconds | Analytic TWoS seconds | Best MG seconds | MG / analytic TWoS time |','|---|---|---:|:---:|---:|---:|---:|---:|']
 for _,x in primary.iterrows():
  lines.append(f"| {x.geometry} | {x.state} | {x.queries} | {x.reference_resolved} | {fmt(x.get('twos_seconds',np.nan))} | {fmt(x.get('twos_rb_seconds',np.nan))} | {fmt(x.get('best_mg_seconds',np.nan))} | {fmt(x.get('twos_rb_speedup',np.nan),2)} |")
 lines+=['','A dash means no tested budget meets the error rule, or the reference is unresolved; it is not an infinite speedup. Ratios above one favor analytic-source TWoS. All timing intervals, per-seed results, budget choices, reference diagnostics, and 10%/40% sensitivity targets are in the CSV files.','',
 '## Scientific interpretation','',
 'The original source-sampled TWoS estimator is retained as an ablation. Any advantage of analytically integrated TWoS must be stated with that source restriction. The variance reduction proof is an application of classical conditional expectation, not a novelty claim for Rao–Blackwellization. Multigrid gets both solved-field reuse and a separately measured outer-mask/warm-start variant.','',
 '## Reproducibility','',
 f"Measured query-prefix records: {len(d)}. Four geometries and five paired seeds. See output/audit.json, integral-audit.json, protocol.json, and the three analysis/extension protocol files. No optimized geometry was chosen from these query results; the cavity edit is fixed before scoring."]
 confirmations=[]
 for stage,filename in [('Selected budgets','holdout-summary.csv'),('Fixed 256 follow-up','conservative-holdout-summary.csv')]:
  path=O/filename
  if path.exists():
   for _,v in pd.read_csv(path).iterrows():confirmations.append((stage,v))
 lines += ['', '## Fresh-seed confirmations', '', 'Each row uses 20 fresh paired seeds. The first confirmation freezes calibration-selected budgets. A separately declared follow-up fixes 256 paths for both eight-site compliance states after the 64-path accuracy failure. All rows are retained; no further budget search is used.', '', '| Stage | Workload | State | Sites | Paths | TWoS ms | MG ms | Mean error / upper endpoint | Speed ratio [95% interval] | Both rules pass |', '|---|---|---|---:|---:|---:|---:|---:|---:|:---:|']
 for stage,v in confirmations:
  lines.append(f"| {stage} | {v.geometry} | {v.state} | {int(v.queries)} | {int(v.twos_budget)} | {1000*v.twos_seconds:.1f} | {1000*v.mg_seconds:.1f} | {100*v.mean_error:.1f}% / {100*v.error_hi:.1f}% | {v.speedup:.2f} [{v.speedup_lo:.2f}, {v.speedup_hi:.2f}] | {v.verified} |")
 lines += ['', 'The four calibration-selected apparent wins did not pass both fresh-seed rules: three compliance cases failed the accuracy upper-endpoint rule; housing failed the speed rule. The follow-up is exploratory confirmation of a fixed conservative budget, not a retrospectively preregistered success. All speed ratios exclude common geometry loading and concern query time only.', '', 'The audit retains the stream-identity correction at 4096 paths, charges all replay work, and checks clean replays. Numerical reference gates and bootstrap intervals do not certify continuum accuracy or familywise coverage.']
 (O/'report.md').write_text('\n'.join(lines)+'\n')
 tex=[r'\paragraph{Measured results.}',
 r'Table~\ref{tab:matched-primary} reports the predeclared 20\% target. A dash means an unresolved reference or no eligible tested budget; it never denotes an infinite speedup. The analytic-source and incremental-grid extensions were introduced after the first housing result and frozen before their own measurements. Their results are therefore separate diagnostic extensions, not retroactively part of the initial protocol.',
 r'\begin{table*}[t]\centering\small',r'\caption{Calibration results at the 20\% reference-relative accuracy target. Times include every batch through the listed query count. MG is the fastest eligible tested resolution and reuse variant. RB denotes exact-source integration in ideal arithmetic. All quantities are empirical; source restrictions and numerical-reference gates apply.}',r'\label{tab:matched-primary}',r'\begin{tabular}{llrrrrrr}\toprule',r'Object & State & Queries & TWoS (s) & TWoS--RB (s) & MG (s) & MG/RB & RB paths\\\midrule']
 for _,x in primary[primary.queries.isin([8,512])].iterrows():
  tex.append(f"{NAMES[x.geometry]} & {x.state} & {int(x.queries)} & {fmt(x.get('twos_seconds',np.nan))} & {fmt(x.get('twos_rb_seconds',np.nan))} & {fmt(x.get('best_mg_seconds',np.nan))} & {fmt(x.get('twos_rb_speedup',np.nan),2)} & {fmt(x.get('twos_rb_budget',np.nan),0)}"+r'\\')
 tex += [r'\bottomrule\end{tabular}\end{table*}',
 r'\begin{table*}[t]\centering\small',r'\caption{Independent reference diagnostics. Refinement discrepancies are shown at 8 and 512 sites; the 32-site Monte Carlo diagnostic is common to each state. Values are percentages. These quantities are not rigorous error bounds.}',r'\begin{tabular}{llrrrr}\toprule Object & State & $\delta_8$ & $\delta_{512}$ & MC SE norm & MC--grid discrepancy\\\midrule']
 for (name,state),g in c.groupby(['geometry','state']):
  v=g.set_index('queries');x=v.loc[8]
  tex.append(f"{NAMES[name]} & {state} & {100*x.grid_delta:.2f} & {100*v.loc[512].grid_delta:.2f} & {100*x.spot_se:.2f} & {100*x.spot_delta:.2f}"+r'\\')
 tex += [r'\bottomrule\end{tabular}\end{table*}']
 for state in ['original','edited']:
  tex += [r'\begin{figure*}[p]\centering',r'\includegraphics[width=\textwidth]{figures/matched-accuracy-'+state+'.pdf}',r'\caption{All measured budgets on the '+state+r' domain, at 8 sites (top) and 512 sites (bottom). Bars are pointwise 95\% bootstrap intervals over five seeds; they exclude numerical-reference bias. Dashed horizontal lines mark the 20\% target before applying the independent reference gates. Numbers label path counts or longest-axis grid counts. Original and source-integrated TWoS and both grid reuse variants are retained.}\end{figure*}']
 tex += [r'\begin{figure*}[p]\centering',r'\includegraphics[width=\textwidth]{figures/matched-accuracy-scaling.pdf}',r'\caption{Measured query-count scaling at fixed budgets in the initially frozen sweep. Shaded bands show pointwise 95\% bootstrap intervals for mean time. Original-source TWoS scales with the number of sites, whereas solved-field multigrid amortizes its setup. These fixed-budget curves do not assert matched accuracy.}\end{figure*}']
 tex += [r'\paragraph{Fresh-seed confirmation and its failures.}',
 r'The four apparent 20\%-target wins selected from calibration did not pass both fresh-seed rules. The three compliance cases exceeded the mean-error upper-endpoint threshold; the housing timing interval included no advantage. We then declared a separate, fixed 256-path follow-up for both eight-site compliance states, with 20 further fresh seeds and no subsequent budget search. All results are retained below. These pointwise bootstrap checks are not simultaneous confidence guarantees.',
 r'\begin{table*}[t]\centering\small',r'\caption{Fresh-seed query confirmations. Times exclude common geometry loading. Error upper endpoints and speed intervals use 10,000 paired bootstrap resamples of 20 seeds. Passing requires error upper endpoint $\leq20\%$ and speed-ratio lower endpoint $>1$. Every row uses MG49.}',r'\begin{tabular}{llrrrrrr}\toprule Workload / state & Stage & Sites & Paths & TWoS / MG (ms) & Error upper & MG/TWoS interval & Pass\\\midrule']
 for stage,v in confirmations:
  tex.append(f"{NAMES[v.geometry]} / {v.state} & {'Selected' if stage=='Selected budgets' else 'Fixed 256'} & {int(v.queries)} & {int(v.twos_budget)} & {1000*v.twos_seconds:.1f} / {1000*v.mg_seconds:.1f} & {100*v.error_hi:.1f}"+r'\% & '+f"[{v.speedup_lo:.2f}, {v.speedup_hi:.2f}] & {'yes' if v.verified else 'no'}"+r'\\')
 tex += [r'\bottomrule\end{tabular}\end{table*}']
 (R/'sections/22_matched_accuracy_results.tex').write_text('\n'.join(tex)+'\n')
 print('Wrote report and supplementary result tables.')
if __name__=='__main__':main()
