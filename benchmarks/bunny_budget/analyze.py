#!/usr/bin/env python3
"""Recompute all budget-study summaries and publication plots from retained data."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'output'
ROOT=HERE.parents[1]
COLORS=dict(twos='#9365a8',voxel='#6597aa',random='#99999e')
LABELS=dict(twos='TWoS',voxel='Voxel adjoint',random='Random feasible')

def cost_envelope(counts,objectives):
    counts=np.r_[0,counts];q=np.r_[1.,objectives]
    cuts={0.,.20}
    for i in range(len(q)):
        for j in range(i):
            x=(q[i]-q[j])/(counts[j]-counts[i])
            if 0<x<.20:cuts.add(float(x))
    cuts=sorted(cuts);segments=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        index=int(np.argmin(q+(lo+hi)/2*counts));n=int(counts[index])
        if segments and segments[-1]['count']==n:segments[-1]['upper_cost']=hi
        else:segments.append(dict(lower_cost=lo,upper_cost=hi,count=n))
    return segments

def main():
    raw=json.loads((OUT/'results.json').read_text())['results'];df=pd.read_csv(OUT/'results.csv')
    assert len(df)==72 and len(df[['volume_units','count','seed','method']].drop_duplicates())==72
    volume=json.loads((ROOT/'benchmarks/stanford_bunny/output/summary.json').read_text())['mesh_repair']['normalized_volume']
    groups=[]
    for (units,n,m),g in df.groupby(['volume_units','count','method']):
        records=[r for r in raw if (r['volume_units'],r['count'],r['method'])==(units,n,m)]
        block=np.mean([r['block_cooling_percent'] for r in records],axis=0)
        groups.append(dict(volume_units=int(units),count=int(n),method=m,
            radius=float(g.radius.iloc[0]),removed_fraction=float(g.removed_volume.iloc[0]/volume),
            cooling_mean=float(g.cooling_percent.mean()),cooling_min=float(g.cooling_percent.min()),
            cooling_max=float(g.cooling_percent.max()),design_standard_deviation=float(g.cooling_percent.std(ddof=1)),
            conditional_block_se=float(block.std(ddof=1)/2),
            surface_area=float(g.surface_area.iloc[0])))
    summary=pd.DataFrame(groups);summary.to_csv(OUT/'summary.csv',index=False)
    full=summary[(summary.volume_units==6)&(summary.method=='twos')].sort_values('count')
    envelope=cost_envelope(full['count'].values,1-full.cooling_mean.values/100)
    (OUT/'cost-envelope.json').write_text(json.dumps(envelope,indent=2)+'\n')
    comparisons=[]
    for units in [3,6]:
        for n in [2,4,6,12]:
            g=df[(df.volume_units==units)&(df['count']==n)].pivot(index='seed',columns='method',values='cooling_percent')
            for other in ['random','voxel']:
                delta=g.twos-g[other]
                comparisons.append(dict(volume_units=units,count=n,comparison='twos-minus-'+other,
                    mean_pp=float(delta.mean()),min_pp=float(delta.min()),max_pp=float(delta.max())))
    pd.DataFrame(comparisons).to_csv(OUT/'paired-comparisons.csv',index=False)
    mpl.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix',
      'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,
      'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.55),gridspec_kw={'width_ratios':[1,1,1.05]})
    fig.subplots_adjust(left=.068,right=.986,bottom=.245,top=.81,wspace=.4)
    for ax,units,letter in zip(axes[:2],[3,6],['a','b']):
        group=summary[summary.volume_units==units]
        for m in ['random','voxel','twos']:
            g=group[group.method==m].sort_values('count')
            ax.plot(g['count'],g.cooling_mean,'-o',c=COLORS[m],label=LABELS[m],lw=1.4,ms=3.6)
            ax.fill_between(g['count'],g.cooling_min,g.cooling_max,color=COLORS[m],alpha=.13,lw=0)
        ax.set(xlim=(1,13),ylim=(15,70),xticks=[2,4,6,12],xlabel='Cavity count $n$')
        ax.set_title(f'{letter}  {100*group.removed_fraction.iloc[0]:.2f}% removed volume',loc='left',pad=8,fontweight='bold')
        ax.grid(axis='y',ls=':',lw=.5,color='#ccc')
        if units==3:ax.set_ylabel('Heat-objective reduction (%)')
        radii=.05*(units/np.array([2,4,6,12]))**(1/3)
        ax.text(.5,-.35,'Radii: '+', '.join(f'{r:.3f}' for r in radii),transform=ax.transAxes,ha='center',fontsize=7.3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.40,1.025),ncol=3,frameon=False,fontsize=8.5)
    ax=axes[2]
    for seg in envelope:
        ax.plot(np.array([seg['lower_cost'],seg['upper_cost']])*100,[seg['count']]*2,color=COLORS['twos'],lw=2)
    for a,b in zip(envelope[:-1],envelope[1:]):
        ax.plot([a['upper_cost']*100]*2,[a['count'],b['count']],color=COLORS['twos'],ls=':',lw=.8)
    ax.set(xlim=(0,4),ylim=(-.5,13),yticks=[0,2,4,6,12],xticks=[0,1,2,3,4],
        xlabel=r'Assumed cost per cavity (% of $J_0$)',ylabel='Preferred tested count')
    ax.set_title('c  A cost-dependent choice',loc='left',pad=8,fontweight='bold')
    zero=next(s['lower_cost'] for s in envelope if s['count']==0)
    ax.text(.67,.78,f'Above {100*zero:.1f}%: no cavities',transform=ax.transAxes,
            ha='center',fontsize=7.3,color='#69626e')
    ax.text(.5,-.35,r'Minimize $J/J_0+\kappa n$; includes no holes',transform=ax.transAxes,ha='center',fontsize=7.3)
    ax.grid(axis='y',ls=':',lw=.5,color='#ccc')
    fig.savefig(OUT/'bunny-budget.pdf');fig.savefig(OUT/'bunny-budget.png',dpi=600);plt.close(fig)
    validation=dict(designs=len(df),all_finite=bool(np.isfinite(df[['radius','objective','cooling_percent']]).all().all()),
      minimum_outer_gap=float(df.minimum_outer_gap.min()),minimum_pair_gap=float(df.minimum_pair_gap.min()),
      maximum_volume_error=float(max(abs(r['removed_volume']-r['volume_units']*4*math.pi*.05**3/3) for r in raw)),
      raw_results='72 complete designs, three seeds, four randomized evaluation blocks each',
      uncertainty='Shading is design-seed min/max, not a confidence interval. Block SE is conditional on saved designs and omits termination bias.',
      interpretation='Ideal zero-temperature cavities; exact volume control, unequal cooling area; exploratory finite choices, no global optimum or speed claim.')
    save=OUT/'validation.json';save.write_text(json.dumps(validation,indent=2)+'\n')
    tolerance=json.loads((OUT/'termination-check.json').read_text())
    def g(v,n,m='twos'):
        return summary[(summary.volume_units==v)&(summary['count']==n)&(summary.method==m)].iloc[0]
    text=(r'At the full volume budget, increasing the TWoS count from two to twelve '
      rf'raises mean cooling from ${g(6,2).cooling_mean:.2f}\%$ to ${g(6,12).cooling_mean:.2f}\%$, '
      rf'while the radius decreases from ${g(6,2).radius:.4f}$ to ${g(6,12).radius:.4f}$. '
      rf'At twelve cavities, the voxel and random means are ${g(6,12,"voxel").cooling_mean:.2f}\%$ '
      rf'and ${g(6,12,"random").cooling_mean:.2f}\%$, respectively. '
      r'The most useful allocation comparison is across budgets: twelve small TWoS cavities '
      rf'at half volume yield ${g(3,12).cooling_mean:.2f}\%$ cooling '
      rf'(design range ${g(3,12).cooling_min:.2f}$--${g(3,12).cooling_max:.2f}\%$), '
      rf'versus ${g(6,6).cooling_mean:.2f}\%$ '
      rf'(${g(6,6).cooling_min:.2f}$--${g(6,6).cooling_max:.2f}\%$) '
      r'for six larger cavities at full volume. The former also has $20.6\%$ less '
      r'cavity surface area, by exact sphere geometry. This is similar measured cooling '
      r'with less removal, not a proved equivalence or an advantage unique to TWoS. '
      r'The voxel selector shows a similar allocation trend. '
      r'A fourfold thinner WoS termination shell, $1.25\times10^{-4}$, changes '
      r'cooling by at most '
      rf'${tolerance["maximum_absolute_change_pp"]:.3f}$ percentage points '
      r'for the pre-specified first-seed full-budget endpoint designs from both TD selectors. '
      r'This sensitivity check does not bound the remaining bias.'+'\n')
    if 'allocation_check' in tolerance:
        alloc=tolerance['allocation_check']
        text+=('An additional first-seed check of the half-volume comparison, chosen after '
          'inspecting the allocation results, gives '
          rf'${alloc[0]["fine_cooling_percent"]:.2f}\%$ versus ${alloc[1]["fine_cooling_percent"]:.2f}\%$ '
          'cooling with the thinner shell, respectively.\n')
    (OUT/'measured-results.tex').write_text(text)
    print(summary.to_string(index=False));print('Cost envelope:',envelope)

if __name__=='__main__':main()
