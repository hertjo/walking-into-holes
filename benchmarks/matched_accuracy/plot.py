"""Publication plots from retained measurements; no hand-entered numbers."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'output';F=R/'figures'
COL={'twos':'#969696','twos_rb':'#9363AE','mg':'#418C9E','mg_update':'#235D70'}
LABEL={'housing':'Finned housing','branch':'Branching manifold','ring':'Annular spreader','bunny':'Stanford bunny'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.8,'axes.titlesize':7.5,'axes.labelsize':6.8,'legend.fontsize':6.6,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#aaaaaa','axes.linewidth':.6,'xtick.color':'#454545','ytick.color':'#454545'})
def style(ax):
 ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='major',color='#e8e8e8',lw=.5);ax.set_axisbelow(True)
 ax.axhline(.2,color='#aaa',ls=':',lw=.8);ax.set_ylabel('Relative score error');ax.set_xlabel('Cumulative query time (s)')
def curve(ax,d,method,ls='-',label=None):
 d=d.sort_values('budget')
 ax.errorbar(d.seconds,d.error,xerr=np.maximum(0,np.array([d.seconds-d.seconds_lo,d.seconds_hi-d.seconds])),yerr=np.maximum(0,np.array([d.error-d.error_lo,d.error_hi-d.error])),fmt='o',markersize=3.8,color=COL[method],elinewidth=.8,capsize=1.5,ls=ls,lw=1,label=label)
 for _,r in d.iterrows():ax.annotate(str(int(r.budget)),(r.seconds,r.error),xytext=(4,4),textcoords='offset points',fontsize=6,color=COL[method])
def main():
 d=pd.read_csv(O/'summary.csv');sel=pd.read_csv(O/'selected.csv');check=pd.read_csv(O/'reference_checks.csv')
 fig,axs=plt.subplots(1,4,figsize=(7.1,2.3),layout='constrained')
 for ax,(name,title) in zip(axs,LABEL.items()):
  ax.set_title(title,fontweight='bold');style(ax)
  for method in ['twos','twos_rb','mg']:curve(ax,d[(d.geometry==name)&(d.state=='original')&(d.queries==8)&(d.method==method)],method)
  row=sel[(sel.geometry==name)&(sel.state=='original')&(sel.queries==8)&(sel.target==.2)].iloc[0]
  note='Reference unresolved' if not row.reference_resolved else '20% target: '
  if row.reference_resolved:
   if pd.notna(row.get('twos_rb_speedup',np.nan)):note+=f"{row.twos_rb_speedup:.2f}× MG / RB"
   else:note='20%: no eligible pair'
  ax.text(.02,.02,note,transform=ax.transAxes,fontsize=6.5,va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=1))
 fig.legend(handles=[Line2D([0],[0],color=COL[m],marker='o',lw=1,label={'twos':'Sampled-source TWoS','twos_rb':'Analytic-source TWoS','mg':'MG, field reuse','mg_update':'MG, incremental'}[m]) for m in ['twos','twos_rb','mg']],loc='outside upper center',ncol=2,frameon=False)
 fig.savefig(F/'matched-accuracy-sparse.pdf');fig.savefig(F/'matched-accuracy-sparse.png',dpi=400);plt.close(fig)
 for state in ['original','edited']:
  fig,axs=plt.subplots(2,4,figsize=(7.1,4.8),layout='constrained')
  for j,(name,title) in enumerate(LABEL.items()):
   for i,q in enumerate([8,512]):
    ax=axs[i,j];style(ax);ax.set_title(f'{title} · {q} queries',fontweight='bold')
    for method in COL:curve(ax,d[(d.geometry==name)&(d.state==state)&(d.queries==q)&(d.method==method)],method)
  fig.legend(handles=[Line2D([0],[0],color=COL[m],marker='o',lw=1,label={'twos':'Sampled-source TWoS','twos_rb':'Analytic-source TWoS','mg':'MG, field reuse','mg_update':'MG, incremental'}[m]) for m in COL],loc='outside upper center',ncol=2,frameon=False)
  fig.savefig(F/f'matched-accuracy-{state}.pdf');fig.savefig(F/f'matched-accuracy-{state}.png',dpi=350);plt.close(fig)
 # Query scaling at each fixed budget: bands show paired seed variability.
 fig,axs=plt.subplots(2,4,figsize=(7.1,4.8),layout='constrained')
 for j,(name,title) in enumerate(LABEL.items()):
  for i,state in enumerate(['original','edited']):
   ax=axs[i,j];ax.set_title(f'{title} · '+('original' if i==0 else 'after insertion'),fontweight='bold')
   ax.set_xscale('log',base=2);ax.set_yscale('log');ax.grid(True,color='#e8e8e8',lw=.5)
   for method,budget,ls in [('twos',256,'-'),('twos',4096,'--'),('mg',49,'-'),('mg',129,'--')]:
    g=d[(d.geometry==name)&(d.state==state)&(d.method==method)&(d.budget==budget)].sort_values('queries')
    ax.plot(g.queries,g.seconds,color=COL[method],ls=ls,lw=1,label=f'{method} {budget}')
    ax.fill_between(g.queries,g.seconds_lo,g.seconds_hi,color={'twos':'#e6e6e6','mg':'#dcebef'}[method],lw=0)
   ax.set_xticks([8,32,128,512],['8','32','128','512']);ax.set_xlabel('Query sites');ax.set_ylabel('Cumulative time (s)')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
 fig.savefig(F/'matched-accuracy-scaling.pdf');fig.savefig(F/'matched-accuracy-scaling.png',dpi=350);plt.close(fig)
if __name__=='__main__':main()
