"""Produce data-driven vector and high-resolution regional benchmark figures."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,MaxNLocator,FuncFormatter,NullFormatter
from matplotlib.patches import Circle
from scipy.stats import t
import run_benchmark as b

LABEL={'twos_sparse':'TWoS sparse','twos_dense':'TWoS larger budget',
       'wos_hot':'WoS hottest','mg_td':'MG TD (113)','mg_hot':'MG hottest',
       'random':'Random','mg57':'MG TD (57)','mg85':'MG TD (85)','mg169':'MG TD (169)'}
COLOR={'twos_sparse':'#9653b4','twos_dense':'#6c2e8b','wos_hot':'#d18a45',
       'mg_hot':'#a35b20','random':'#989898','mg_td':'#42849b',
       'mg57':'#42849b','mg85':'#42849b','mg169':'#42849b'}

def main():
    data=pd.concat([pd.read_csv(b.OUT/'results.csv'),pd.read_csv(b.OUT/'grid_sweep.csv')],ignore_index=True)
    assert len(data)==90 and data.groupby(['side','method']).size().eq(5).all()
    common=[]
    for _,row in data.iterrows():
        setup=json.loads((b.OUT/f'side{int(row.side)}/seed{int(row.seed)}/candidate_setup.json').read_text())
        common.append(setup['seconds'])
    data['complete_design_seconds']=data.driver_seconds+common
    data['candidate_setup_seconds']=common
    data.to_csv(b.OUT/'complete_results.csv',index=False)
    rows=[]
    for (side,method),g in data.groupby(['side','method']):
        rows.append({'side':int(side),'method':method,'n':len(g),
            'reduction_percent':100*g.reduction.mean(),'design_se_pp':100*g.reduction.std(ddof=1)/np.sqrt(len(g)),
            'grid_reduction_percent':100*g.grid_reduction.mean(),
            'wall_seconds':g.complete_design_seconds.mean(),'kernel_seconds':g.kernel_seconds.mean(),
            'peak_rss_mib':g.peak_rss_megabytes.mean()})
    summary=pd.DataFrame(rows);summary.to_csv(b.OUT/'summary_table.csv',index=False)
    gaps=[]
    for side in [-1,1]:
        pivot=data[data.side==side].pivot(index='seed',columns='method',values='reduction')
        for ours in ['twos_sparse','twos_dense']:
            for baseline in ['wos_hot','mg_hot','mg_td','mg57','mg85','mg169']:
                diff=100*(pivot[ours]-pivot[baseline]);half=t.ppf(.975,4)*diff.std(ddof=1)/np.sqrt(5)
                gaps.append({'side':side,'method':ours,'baseline':baseline,'mean_pp':diff.mean(),
                    'paired_t_lower_pp':diff.mean()-half,'paired_t_upper_pp':diff.mean()+half})
    sensitivity=pd.read_csv(b.OUT/'reference_sensitivity.csv')
    first=data[(data.seed==b.CONFIG['seeds'][0])&data.method.isin(b.METHODS)]
    check=first.merge(sensitivity,on=['side','method'])
    fieldchecks=[]
    for side in [-1,1]:
        root=b.OUT/f'side{side}/field_audit'
        w=pd.read_csv(root/'twos4096/field.csv');fine=pd.read_csv(root/'mg225/field.csv')
        for name in ['twos4096','mg113','mg169']:
            x=pd.read_csv(root/name/'field.csv')
            fieldchecks.append({'side':side,'method':name,
                'score_relative_l2':float(np.linalg.norm(x.score-fine.score)/np.linalg.norm(fine.score)),
                'score_correlation':float(np.corrcoef(x.score,fine.score)[0,1])})
    report={'table':rows,'paired_differences':gaps,'field_checks':fieldchecks,
       'max_shell_sensitivity_pp':float(100*np.max(np.abs(check.reduction-check.tight_reduction))),
       'max_grid_sensitivity_pp':float(100*np.max(np.abs(check.grid_reduction-check.fine_grid_reduction))),
       'max_reference_disagreement_pp':float(100*np.max(np.abs(data.reduction-data.grid_reduction))),
       'uncertainty':'paired t intervals across five designs conditional on fixed reference probes; not generalization guarantees'}
    persistent=pd.read_csv(b.HERE/'persistent_output/results.csv')
    assert len(persistent)==70
    eq=json.loads((b.HERE/'persistent_output/geometry_equivalence.json').read_text())
    assert len(eq)==70 and all(x['all_six_sequential_centers_identical'] for x in eq)
    pm=persistent.groupby('method').mean(numeric_only=True)
    report['persistent_means']=pm.reset_index().to_dict(orient='records')
    report['persistent_designs_verified']=70
    b.save(b.OUT/'summary.json',report)
    table=summary.set_index(['side','method'])
    value=lambda side,m:table.loc[(side,m),'reduction_percent']
    pooled=data.groupby('method').mean(numeric_only=True)
    fastest=min(['mg57','mg85','mg_td','mg169'],key=lambda m:pooled.loc[m,'complete_design_seconds'])
    paragraph=(
      f"Sparse \\method{{}} reduces the left/right regional objectives by "
      f"${value(-1,'twos_sparse'):.2f}\\%$/${value(1,'twos_sparse'):.2f}\\%$, compared with "
      f"${value(-1,'mg_hot'):.2f}\\%$/${value(1,'mg_hot'):.2f}\\%$ for deterministic hottest-point placement. "
      f"The primary multigrid topological selector gives ${value(-1,'mg_td'):.2f}\\%$/${value(1,'mg_td'):.2f}\\%$. "
      f"Averaged across both cases, complete sparse \\method{{}} design takes "
      f"${pooled.loc['twos_sparse','complete_design_seconds']:.2f}$\\,s, versus "
      f"${pooled.loc['mg_td','complete_design_seconds']:.2f}$\\,s for the primary multigrid setting. "
      f"The fastest tested multigrid setting ({fastest[2:] if fastest!='mg_td' else '113'} nodes along the first axis) takes "
      f"${pooled.loc[fastest,'complete_design_seconds']:.2f}$\\,s and gives "
      f"${value(-1,fastest):.2f}\\%$/${value(1,fastest):.2f}\\%$ reduction. "
      f"Retaining the outer mesh in one process per complete adaptive design reduces wall time to "
      f"${pm.loc['twos_sparse','complete_design_seconds']:.2f}$\\,s for sparse \\method{{}}, "
      f"${pm.loc['mg57','complete_design_seconds']:.2f}$\\,s for coarse multigrid, and "
      f"${pm.loc['mg_td','complete_design_seconds']:.2f}$\\,s for primary multigrid. "
      "All 70 repeated designs have exactly the same six sequential cavity centers as their original runs, "
      "permitting reuse of the independent objective evaluations. "
      "These results demonstrate an advantage over hottest-point placement; the full budget sweep "
      "in \\cref{fig:regional,tab:regional} qualifies comparisons with volumetric topological derivatives.\n")
    (b.OUT/'regional_summary.tex').write_text(paragraph)
    lines=[r'\begin{table*}[t]',r'\centering\small',
      r'\caption{Regional design comparison. Reductions are means $\pm$ one standard error across five designs per sensor. Both wall-time columns include candidate preparation and the complete selection driver, averaged across sensor cases. Retained-mesh timings use one fresh server per design; dashes denote configurations not rerun with that backend. The extra multigrid resolutions are a supplementary sweep.}',
      r'\label{tab:regional}',r'\begin{tabular}{lrrrr}',r'\toprule',
      r'Method & Left reduction (\%) & Right reduction (\%) & Reload mesh (s) & Retain mesh (s) \\',r'\midrule']
    for method in ['random','wos_hot','mg_hot','twos_sparse','twos_dense','mg57','mg85','mg_td','mg169']:
        left=table.loc[(-1,method)];right=table.loc[(1,method)]
        lines.append(f"{LABEL[method]} & ${left.reduction_percent:.2f}\\pm{left.design_se_pp:.2f}$ & "
         f"${right.reduction_percent:.2f}\\pm{right.design_se_pp:.2f}$ & {pooled.loc[method,'complete_design_seconds']:.2f} & "
         +(f"{pm.loc[method,'complete_design_seconds']:.2f}" if method in pm.index else '--')+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (b.OUT/'regional_table.tex').write_text('\n'.join(lines)+'\n')
    (b.OUT/'regional_checks.tex').write_text(
        f"Across first-seed designs, the fourfold shell refinement changes measured reduction by at most "
        f"${report['max_shell_sensitivity_pp']:.3f}$ percentage points. Refining the evaluation grid "
        f"from $169\\times109\\times85$ to $225\\times145\\times113$ changes it by at most "
        f"${report['max_grid_sensitivity_pp']:.3f}$ points. Across all primary and supplementary designs, "
        f"WoS and the $169\\times109\\times85$ reference differ by at most "
        f"${report['max_reference_disagreement_pp']:.3f}$ points. These sensitivity checks "
        "do not bound the remaining discretization or sampling bias.\n")
    print(json.dumps(report,indent=2),flush=True)
    # Geometry-derived plane maps. These do not feed selection or evaluation.
    xx,yy=np.meshgrid(np.linspace(-1.2,1.2,161),np.linspace(-.78,.78,109))
    plane=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,-.03)))
    mask=b.housing.housing_sdf(plane)<-.005
    fld,_=b.solver('figure/initial_fields_full_outline',plane[mask],[],1,'mg',grid=[169,109,85])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(10.2,6.6),layout='constrained')
    gs=fig.add_gridspec(2,3,height_ratios=[1,1.2])
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]
    fields=[fld.u,fld.p,fld.score];titles=['a  Initial temperature $u$','b  Sensor influence $p$','c  Insertion score $up$']
    for ax,values,title in zip(axes[:3],fields,titles):
        vals=np.full(len(plane),np.nan);vals[mask]=values
        if ax is axes[2]:
            vals[np.linalg.norm(plane-b.center(1),axis=1)<.13+b.R+.02]=np.nan
        im=ax.pcolormesh(xx,yy,vals.reshape(xx.shape),cmap='magma',shading='auto',rasterized=True)
        ax.contour(xx,yy,b.housing.housing_sdf(plane).reshape(xx.shape),levels=[0],colors=['#555555'],linewidths=.65)
        if ax is axes[2]:
            ax.add_patch(Circle(b.center(1)[:2],.13+b.R+.02,facecolor='#efefef',edgecolor='#ababab',linewidth=.7,zorder=3))
            ax.text(*b.center(1)[:2],'No cavity\ncenters',ha='center',va='center',fontsize=6,color='#4d4d4d',zorder=5)
        ax.add_patch(Circle(b.center(1)[:2],.13,fill=False,edgecolor='#25bed0',linewidth=1.6,zorder=4))
        ax.set(aspect='equal',title=title,xlabel='$x$',ylabel='$y$')
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.04,fraction=.045)
        cb.locator=MaxNLocator(4);cb.formatter=ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2,2));cb.update_ticks();cb.ax.tick_params(labelsize=8)
    seed=b.CONFIG['seeds'][0];ax=axes[3]
    ax.contour(xx,yy,b.housing.housing_sdf(plane).reshape(xx.shape),levels=[0],colors=['#6e6e6e'],linewidths=1)
    for method,color,style in [('mg_hot','#d18a45','--'),('twos_sparse','#9653b4','-')]:
        holes=np.array(json.loads((b.OUT/f'side1/seed{seed}/{method}/design.json').read_text())['holes'])
        for i,c in enumerate(holes):
            ax.add_patch(Circle(c[:2],b.R,fill=False,edgecolor=color,linestyle=style,linewidth=1.5,
                              label=LABEL[method] if i==0 else None))
    ax.add_patch(Circle(b.center(1)[:2],.13,facecolor='#25bed030',edgecolor='#25bed0',label='Protected sensor'))
    ax.set(aspect='equal',title='d  Projected cavity locations',xlabel='$x$',ylabel='$y$',xlim=(-1.2,1.2),ylim=(-.78,.78))
    ax.legend(loc='lower right',bbox_to_anchor=(.96,.18),fontsize=7,ncol=1,frameon=False)
    ax=axes[4];methods=['random','wos_hot','mg_hot','twos_sparse','twos_dense','mg57','mg85','mg_td','mg169']
    short=['Random','WoS hot','MG hot','TWoS S','TWoS L','MG 57','MG 85','MG 113','MG 169']
    for k,side in enumerate([-1,1]):
        sub=summary[summary.side==side].set_index('method').loc[methods]
        ax.errorbar(np.arange(len(methods))+(k-.5)*.18,sub.reduction_percent,yerr=sub.design_se_pp,
            fmt='o' if k==0 else 's',color='#333333' if k==0 else '#929292',markersize=4,capsize=2,label='Left sensor' if k==0 else 'Right sensor')
    ax.set(title='e  Objective-aware placement helps',ylabel='Regional temperature reduction (%)',xticks=np.arange(len(methods)),xticklabels=short,ylim=(0,65))
    ax.tick_params(axis='x',rotation=65,labelsize=7);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=7,frameon=False)
    ax=axes[5];mean=pm
    mg=['mg57','mg_td'];mc=['twos_sparse','twos_dense']
    for names,color in [(mg,'#42849b'),(mc,'#9653b4')]:
        ax.plot(mean.loc[names].complete_design_seconds,100*mean.loc[names].reduction,'o-',color=color,markersize=5)
        for name in names:
            text={'mg57':'57','mg85':'85','mg_td':'113','mg169':'169','twos_sparse':'Sparse','twos_dense':'Larger'}[name]
            ax.annotate(text,(mean.loc[name].complete_design_seconds,100*mean.loc[name].reduction),xytext=(0,-14 if name=='twos_sparse' else 8),textcoords='offset points',ha='center',fontsize=7,color=color)
    for name in ['wos_hot','mg_hot']:
        ax.scatter(mean.loc[name].complete_design_seconds,100*mean.loc[name].reduction,color=COLOR[name],s=25)
        ax.annotate(LABEL[name],(mean.loc[name].complete_design_seconds,100*mean.loc[name].reduction),xytext=(0,-12),textcoords='offset points',ha='center',fontsize=7,color=COLOR[name])
    ax.set(title='f  Retained-mesh design cost',xlabel='Complete design wall time (s)',ylabel='Regional temperature reduction (%)',xscale='log',ylim=(25,65),xlim=(.9,9))
    ax.set_xticks([1,2,4,8]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.15);ax.text(.04,.95,'Purple: TWoS budgets\nBlue: multigrid resolutions\nOne fresh process per full design',transform=ax.transAxes,va='top',fontsize=7)
    fig.savefig(b.OUT/'distinct-adjoint-benchmark.pdf',bbox_inches='tight')
    fig.savefig(b.OUT/'distinct-adjoint-benchmark.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
