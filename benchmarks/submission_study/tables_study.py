"""Generate manuscript tables directly from audited numerical outputs."""
from pathlib import Path
import pandas as pd
from run_study import HERE,OUT,SPECS
N={'nearest':'Nearest sensor','twos':'TWoS','mg':'Multigrid TD (65)','mg49':'Multigrid TD (49)','hot':'Hottest point','random':'Random'}

def tables():
 g=pd.read_csv(OUT/'group-results.csv');parts=[]
 for study in ['pipeline','channels']:
  title='Complete cavity-design pipelines' if study=='pipeline' else 'Matched-volume accessible channels'
  parts += [r'\begin{table*}[t]',r'\centering\small',r'\caption{\textbf{'+title+r'.} Means over five paired design seeds; SD is the sample standard deviation of their evaluated cooling reductions in percentage points, including evaluation noise. Cooling intervals are pointwise 95\% bootstrap intervals jointly resampling seeds and independent evaluation blocks. Backend RSS is process high-water memory, not total driver memory; a dash denotes no PDE backend call. Times include all design stages but exclude independent evaluation. '+('Hot and random use sixteen refinement rounds; TWoS, multigrid, and nearest-sensor use eight.' if study=='pipeline' else 'Independent removed-volume audits are reported separately; every compact heat support is preserved.')+'}',r'\label{tab:submission'+study+'}',r'\begin{tabular}{llrrrrr}',r'\toprule',r'Geometry & Method & Cooling (\%) & SD (pp) & 95\% interval & Time (s) & RSS (MiB) \\',r'\midrule']
  for i,name in enumerate(SPECS):
   if i:parts.append(r'\midrule')
   for j,method in enumerate(['twos','mg49','mg','nearest','hot','random']):
    state='initial' if study=='channels' else ('refine16' if method in ['hot','random'] else 'refine8')
    row=g.query('study==@study and geometry==@name and method==@method and state==@state').iloc[0]
    label={'housing':'Finned housing','branch':'Branching manifold','ring':'Annular spreader'}[name] if j==0 else ''
    rss=f'{row.backend_peak_rss_mb:.1f}' if row.backend_peak_rss_mb else r'---'
    parts.append(f'{label} & {N[method]} & {row.reduction_pct:.2f} & {row.seed_sd_pp:.2f} & [{row.cooling_ci95_low:.2f}, {row.cooling_ci95_high:.2f}] & {row.seconds:.2f} & {rss} '+r'\\')
  parts += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
 ref=pd.read_csv(OUT/'reference-audit.csv')
 parts += [r'\begin{table*}[t]',r'\centering\small',r'\caption{\textbf{Independent resolution check.} First predeclared seed, before sphere refinement. Ranges span all six methods on each geometry. The grid columns give longest-axis node counts. Differences are percentage points of objective reduction. These comparisons test reference sensitivity; neither numerical method is exact ground truth.}',r'\label{tab:submissionreference}',r'\begin{tabular}{llrr}',r'\toprule',r'Study & Geometry & Grid 129 minus 97 (pp) & MC minus grid 129 (pp) \\',r'\midrule']
 for (study,name),a in ref.groupby(['study','geometry'],sort=False):
  parts.append(f'{study.title()} & '+{'housing':'Finned housing','branch':'Branching manifold','ring':'Annular spreader'}[name]+f' & [{a.grid_change_pp.min():.2f}, {a.grid_change_pp.max():.2f}] & [{a.MC_minus_grid129_pp.min():.2f}, {a.MC_minus_grid129_pp.max():.2f}] '+r'\\')
 parts += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
 if 'fine_grid_change_pp' in ref:
  parts += [r'\begin{table*}[t]',r'\centering\small',r'\caption{\textbf{Finer reference checks for flagged cases.} A case is refined when any method changes by more than 1 pp from grid 97 to 129, or differs by more than 2 pp between MC and grid 129. Every method in that case is checked. Ranges span all six methods; sphere states are before refinement.}',r'\label{tab:submissionfinereference}',r'\begin{tabular}{llrr}',r'\toprule',r'Study & Geometry & Grid 257 minus 193 (pp) & MC minus grid 257 (pp) \\',r'\midrule']
  for (study,name),a in ref.dropna(subset=['fine_grid_change_pp']).groupby(['study','geometry'],sort=False):
   parts.append(f'{study.title()} & '+{'housing':'Finned housing','branch':'Branching manifold','ring':'Annular spreader'}[name]+f' & [{a.fine_grid_change_pp.min():.2f}, {a.fine_grid_change_pp.max():.2f}] & [{a.MC_minus_grid257_pp.min():.2f}, {a.MC_minus_grid257_pp.max():.2f}] '+r'\\')
  parts += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
 dest=HERE.parents[1]/'sections/18_submission_tables.tex';dest.write_text('\n'.join(parts)+'\n');print(dest)
if __name__=='__main__':tables()
