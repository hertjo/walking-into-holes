"""Measured comparisons and geometry overlays; no generated scientific imagery."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import igl
import run_accessible as a

LABEL={'twos_sparse':'TWoS proxy','wos_hot':'WoS hottest','mg57':'MG proxy (57)',
       'mg113':'MG proxy (113)','random':'Random'}
COLOR={'twos_sparse':'#9653b4','wos_hot':'#c17d35','mg57':'#42849b','mg113':'#205568','random':'#969696'}

def main():
    data=pd.read_csv(a.OUT/'results.csv');assert len(data)==50
    audit=json.loads((a.OUT/'audit.json').read_text())
    rows=[]
    for (side,method),g in data.groupby(['side','method']):
        rows.append({'side':int(side),'method':method,'reduction_percent':100*g.reduction.mean(),
          'se_pp':100*g.reduction.std(ddof=1)/np.sqrt(len(g)),'seconds':g.complete_design_seconds.mean(),
          'removed_percent':100*g.removed_material_fraction.mean(),
          'removed_source_percent':100*g.removed_source_fraction.mean()})
    table=pd.DataFrame(rows);table.to_csv(a.OUT/'summary.csv',index=False)
    means=data.groupby('method').mean(numeric_only=True)
    lines=[r'\begin{table*}[t]',r'\centering\small',
      r'\caption{Accessible-channel comparison. Regional temperature reductions are means $\pm$ one standard error across five designs per sensor. Time and removed material are averaged across both sensors. All methods use six channels with the same radius and extent; volume varies with local housing thickness. Point scores are placement proxies, not derivatives of an entire channel.}',
      r'\label{tab:accessible}',r'\begin{tabular}{lrrrr}',r'\toprule',
      r'Method & Left reduction (\%) & Right reduction (\%) & Design time (s) & Removed material (\%) \\',r'\midrule']
    for m in ['random','wos_hot','twos_sparse','mg57','mg113']:
        l=table[(table.side==-1)&(table.method==m)].iloc[0];r=table[(table.side==1)&(table.method==m)].iloc[0]
        lines.append(f'{LABEL[m]} & ${l.reduction_percent:.2f}\\pm{l.se_pp:.2f}$ & ${r.reduction_percent:.2f}\\pm{r.se_pp:.2f}$ & {means.loc[m,"complete_design_seconds"]:.2f} & {100*means.loc[m,"removed_material_fraction"]:.2f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (a.OUT/'channel_table.tex').write_text('\n'.join(lines)+'\n')
    (a.OUT/'channel_checks.tex').write_text(
      f'The saved-data audit verifies all 50 channel geometries and {audit["selected_argmax_checks"]} nonrandom selections. '
      f'All walks terminate before the step cap and all recorded multigrid solves meet the $10^{{-8}}$ residual tolerance. '
      f'The largest first-seed shell-refinement change is ${audit["max_shell_sensitivity_pp"]:.3f}$ percentage points; '
      f'the grid-refinement change is ${audit["max_grid_sensitivity_pp"]:.3f}$ points. '
      f'Across all designs, the two primary numerical references differ by at most ${audit["max_reference_disagreement_pp"]:.3f}$ points. '
      'These are sensitivity checks, not bounds on the remaining numerical error.\n')
    l=table[(table.side==-1)&(table.method=='twos_sparse')].iloc[0]
    r=table[(table.side==1)&(table.method=='twos_sparse')].iloc[0]
    hl=table[(table.side==-1)&(table.method=='wos_hot')].iloc[0]
    hr=table[(table.side==1)&(table.method=='wos_hot')].iloc[0]
    (a.OUT/'channel_summary.tex').write_text(
      f'On the left and right sensors, the TWoS proxy gives reductions of ${l.reduction_percent:.2f}\\%$ and ${r.reduction_percent:.2f}\\%$, '
      f'versus ${hl.reduction_percent:.2f}\\%$ and ${hr.reduction_percent:.2f}\\%$ '
      f'for hottest-point placement. Mean complete design times are ${means.loc["twos_sparse","complete_design_seconds"]:.2f}$\\,s '
      f'for TWoS and ${means.loc["mg57","complete_design_seconds"]:.2f}$\\,s for coarse multigrid. '
      f'Their mean reductions across both sensors are ${100*means.loc["twos_sparse","reduction"]:.2f}\\%$ '
      f'and ${100*means.loc["mg57","reduction"]:.2f}\\%$, respectively. '
      'This preserves a useful quality--cost tradeoff under access constraints, without establishing superiority over deterministic topological scoring.\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(12.0,8.3));gs=fig.add_gridspec(2,2,hspace=.35,wspace=.29)
    ax=fig.add_subplot(gs[0,0],projection='3d')
    v,f=igl.read_triangle_mesh(str(a.b.MESH))
    poly=Poly3DCollection(v[f],facecolor='#8f9fa5',edgecolor='none',alpha=.10,rasterized=True)
    ax.add_collection3d(poly)
    seed=a.CONFIG['seeds'][0]
    def centers(m):return np.array(json.loads((a.OUT/f'side1/seed{seed}/{m}/design.json').read_text())['centers'])
    x=centers('twos_sparse')
    zz=np.linspace(a.Z[0]-a.R,a.Z[-1]+a.R,300);tt=np.linspace(0,2*np.pi,48)
    rr=np.sqrt(np.maximum(0,a.R*a.R-np.min((zz[:,None]-a.Z[None,:])**2,axis=1)))
    for c in x:
        ax.plot_surface(c[0]+rr[:,None]*np.cos(tt),c[1]+rr[:,None]*np.sin(tt),
           np.broadcast_to(zz[:,None],(len(zz),len(tt))),color=COLOR['twos_sparse'],linewidth=0,shade=True,rasterized=True)
    ax.set(xlim=(-1.2,1.2),ylim=(-.8,.8),zlim=(-.55,.95));ax.set_box_aspect((2.4,1.6,1.5),zoom=1.2);ax.view_init(32,-38);ax.set_axis_off()
    ax.set_title('a  Open passages through the housing',loc='left',pad=0)
    ax.text2D(.03,.03,'Transparent boundary; purple shows removed channels',transform=ax.transAxes,fontsize=9)
    bx=fig.add_subplot(gs[0,1]);bx.set_title('b  Placement and protected regions',loc='left')
    gx,gy=np.meshgrid(np.linspace(-1.17,1.17,300),np.linspace(-.78,.78,200))
    sdf=a.b.housing.housing_sdf(np.column_stack((gx.ravel(),gy.ravel(),np.full(gx.size,-.03)))).reshape(gx.shape)
    bx.contour(gx,gy,sdf,levels=[0],colors=['#8b969c'],linewidths=1.5)
    for i,c in enumerate(centers('wos_hot')):bx.add_patch(Circle(c[:2],a.R,fill=False,edgecolor=COLOR['wos_hot'],ls='--',lw=1.7,label='Hottest' if i==0 else None))
    for i,c in enumerate(x):bx.add_patch(Circle(c[:2],a.R,facecolor=COLOR['twos_sparse']+'35',edgecolor=COLOR['twos_sparse'],lw=1.7,label='TWoS proxy' if i==0 else None))
    for i,c in enumerate(a.DEVICES):bx.add_patch(Circle(c[:2],.07,facecolor='#555b61',alpha=.8,label='Protected components' if i==0 else None))
    bx.add_patch(Circle(a.b.center(1)[:2],.13,facecolor='#21bdd044',edgecolor='#21a9bf',lw=1.6,label='Protected sensor'))
    bx.set(xlim=(-1.18,1.18),ylim=(-.8,.8),xlabel='$x$',ylabel='$y$',aspect='equal')
    bx.legend(loc='lower left',ncol=2,frameon=True,facecolor='white',edgecolor='white',framealpha=.97,fontsize=8)
    cx=fig.add_subplot(gs[1,0]);dx=fig.add_subplot(gs[1,1])
    cx.set_title('c  Complete design cost',loc='left');dx.set_title('d  Cooling and removed material',loc='left')
    for m in ['random','wos_hot','twos_sparse','mg57','mg113']:
        group=data[data.method==m]
        cx.scatter(means.loc[m,'complete_design_seconds'],100*means.loc[m,'reduction'],color=COLOR[m],s=55)
        offset={'random':(6,0),'wos_hot':(6,-15),'twos_sparse':(-65,-15),'mg57':(6,9),'mg113':(-83,-17)}[m]
        cx.annotate(LABEL[m],(means.loc[m,'complete_design_seconds'],100*means.loc[m,'reduction']),xytext=offset,textcoords='offset points',color=COLOR[m],fontsize=9)
        dx.scatter(100*group.removed_material_fraction,100*group.reduction,color=COLOR[m],s=16,alpha=.42)
        dx.scatter(100*means.loc[m,'removed_material_fraction'],100*means.loc[m,'reduction'],color=COLOR[m],s=70,edgecolor='white',linewidth=.8,label=LABEL[m])
    cx.set(xlabel='Complete design wall time (s)',ylabel='Regional temperature reduction (%)',xscale='log',ylim=(0,70))
    cx.set_xticks([.25,.5,1,2,4,8],labels=['0.25','0.5','1','2','4','8'])
    dx.set(xlabel='Removed material (%)',ylabel='Regional temperature reduction (%)',ylim=(0,70))
    for axis in [cx,dx]:axis.grid(alpha=.14);axis.set_axisbelow(True)
    dx.legend(frameon=False,fontsize=8,loc='lower right',ncol=2)
    fig.subplots_adjust(left=.06,right=.98,top=.96,bottom=.07)
    fig.savefig(a.OUT/'accessible-channels-diagnostic.pdf',dpi=300)
    fig.savefig(a.OUT/'accessible-channels-diagnostic.png',dpi=500)
    plt.close(fig)
    a.b.save(a.OUT/'summary.json',{'per_case':rows,'pooled':means.reset_index().to_dict(orient='records'),'audit':audit})

if __name__=='__main__':main()
