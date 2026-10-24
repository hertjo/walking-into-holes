"""Same geometry, two objectives: query accuracy and matched-target scaling."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot import COL,curve,style
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'output'
d=pd.read_csv(O/'summary.csv');s=pd.read_csv(O/'selected.csv')
fig,axs=plt.subplots(2,2,figsize=(7.1,3.4),layout='constrained')
for i,(name,title) in enumerate([('bunny','Localized sensor objective'),('bunny_compliance','Distributed-heating compliance')]):
 ax=axs[i,0];style(ax);ax.set_title(title+' · 8 queries',fontweight='bold')
 for m in ['twos','twos_rb','mg_update']:
  g=d[(d.geometry==name)&(d.state=='original')&(d.queries==8)&(d.method.isin(['mg','mg_update']) if m=='mg_update' else d.method==m)]
  if m=='mg_update' and len(g):g=g.loc[g.groupby('budget').seconds.idxmin()]
  if len(g):curve(ax,g,m)
 if i==1:
  ax=axs[i,1];ax.set_title('Fresh seeds · 256 paths versus MG49',fontweight='bold')
  ax.grid(True,axis='x',color='#e8e8e8',lw=.5);ax.set_axisbelow(True)
  h=pd.read_csv(O/'conservative-holdout-summary.csv')
  rng=np.random.default_rng(17092028);idx=rng.integers(0,20,(10000,20))
  for j,state in enumerate(['original','edited']):
   v=h[h.state==state].iloc[0]
   folder=O/'holdout'/f'bunny_compliance-{state}-q8-T256-M49-mg_update'
   for m,offset,color in [('twos_rb',-.16,COL['twos_rb']),('mg',.16,COL['mg_update'])]:
    import json
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob('seed*/'+m+'/result.json'))]
    vals=np.array([r['seconds'] for r in rows])*1000;mean=vals.mean();lo,hi=np.quantile(vals[idx].mean(1),[.025,.975])
    ax.barh(j+offset,mean,height=.27,color=color,zorder=2)
    ax.errorbar(mean,j+offset,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='none',ecolor='#333333',capsize=2,lw=.8,zorder=3)
    ax.text(hi+3,j+offset,f'{mean:.0f} ms',va='center',fontsize=6.5)
  ax.set_yticks([0,1],['Original','After cavity edit']);ax.invert_yaxis();ax.set_xlabel('Query time (ms); lower is better');ax.set_xlim(0,170)
  continue
 ax=axs[i,1];ax.set_title('Localized objective · calibration, 20% target',fontweight='bold')
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,color='#e8e8e8',lw=.5);ax.set_axisbelow(True)
 g=s[(s.geometry==name)&(s.state=='original')&(s.target==.2)].sort_values('queries')
 for key,color in [('twos_rb',COL['twos_rb']),('best_mg',COL['mg_update'])]:
  v=g.dropna(subset=[key+'_seconds'])
  ax.plot(v.queries,v[key+'_seconds'],color=color,marker='o',ms=3.5,lw=1)
  pale='#eee6f3' if key=='twos_rb' else '#dce9ed'
  ax.fill_between(v.queries,v[key+'_seconds_lo'],v[key+'_seconds_hi'],color=pale,lw=0,zorder=0)
 missing=g[g['twos_rb_seconds'].isna()]
 if len(missing):ax.text(.03,.97,'Missing purple points:\nno eligible TWoS budget',transform=ax.transAxes,va='top',fontsize=6.5)
 ax.set_xticks([8,32,128,512],['8','32','128','512']);ax.set_xlabel('Query sites');ax.set_ylabel('Best tested eligible time (s)')
fig.legend(handles=[Line2D([0],[0],color=COL[m],marker='o',lw=1,label=label) for m,label in [('twos','Sampled-source TWoS'),('twos_rb','Analytically integrated TWoS'),('mg_update','Reusable multigrid')]],loc='outside upper center',ncol=3,frameon=False)
fig.savefig(R/'figures/matched-accuracy-bunny.pdf');fig.savefig(R/'figures/matched-accuracy-bunny.png',dpi=450)
