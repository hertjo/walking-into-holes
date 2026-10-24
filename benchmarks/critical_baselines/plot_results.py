"""Publication-size vector plots and 450-dpi previews from retained raw data."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from sampling import OUT
from connectivity import cavity

COLORS={'mc':'#535d65','rqmc':'#c27a32','array':'#506ebb','neighbors':'#27947d'}
NAMES={'mc':'MC','rqmc':'Sobol RQMC','array':'Array-RQMC','neighbors':'Neighbor reuse'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,
      'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
      'axes.labelsize':8.5,'legend.fontsize':7.5,'legend.frameon':False,
      'grid.alpha':.25,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=450,bbox_inches='tight')
    plt.close(fig)

def interval(values):
    rng=np.random.default_rng(34191)
    boots=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
    return np.quantile(boots,[.025,.975])

def sampling_figure():
    f=pd.read_csv(OUT/'sampling_runs.csv');s=pd.read_csv(OUT/'source_sampling_runs.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,6.25),layout='constrained')
    statistics=[]
    for row,(name,data) in enumerate([('Poisson compliance',s),('Harmonic square diagnostic',f[f.domain=='square'])]):
      for method,g in data.groupby('method',sort=False):
        records=[]
        for n,h in g.groupby('n'):
            lo,hi=interval(h.mse.to_numpy());records.append([h.paths.iloc[0],h.seconds.median(),
                            h['queries' if row==0 else 'distance_queries'].mean(),h.mse.mean(),lo,hi])
            statistics.append(dict(problem=name,method=method,n=int(n),mean_mse=float(h.mse.mean()),
                    mse_lo=float(lo),mse_hi=float(hi),median_seconds=float(h.seconds.median())))
        a=np.array(records)
        for col in range(3):
            ax=axes[row,col]
            ax.plot(a[:,col],a[:,3],'-o',ms=4,lw=1.5,color=COLORS[method],label=NAMES[method])
            ax.fill_between(a[:,col],a[:,4],a[:,5],color=COLORS[method],alpha=.12)
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='major')
      for col in range(3):
        ax=axes[row,col];ax.set_xlabel(['Total launched paths','Wall time (s)','Distance-query points'][col])
        ax.set_ylabel('Topological-derivative MSE' if row==0 else 'Harmonic-product MSE')
        ax.set_title(f'{chr(97+3*row+col)}   {name}',loc='left')
      axes[row,0].legend(loc='lower left')
    fig.get_layout_engine().set(hspace=.11,wspace=.07)
    save(fig,'accelerated-sampling')
    pd.DataFrame(statistics).to_csv(OUT/'plot_statistics.csv',index=False)

def diagnostic_figure():
    f=pd.read_csv(OUT/'shape_optimization.csv');audit=pd.read_csv(OUT/'shape_gradient_audit.csv')
    eig=pd.read_csv(OUT/'connectivity_spectrum.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,7.1),layout='constrained')
    a=axes[0,0];t=np.linspace(0,2*np.pi,300)
    a.plot(np.cos(t),.65*np.sin(t),color='#535d65',lw=2,label='Target')
    a.plot(.85*np.cos(t),.85*np.sin(t),'--',color='#c27a32',lw=1.5,label='Initial')
    for _,g in f[(f.method=='probabilistic')&(f.step==30)].iterrows():
        a.plot(g.a*np.cos(t),g.b*np.sin(t),color='#27947d',alpha=.55,lw=1)
    a.plot([],[],color='#27947d',label='MC final (5 seeds)')
    a.set_aspect('equal');a.set_title('a   Fixed-topology tracking',loc='left');a.legend(loc='upper right',fontsize=7)
    a.set_xlabel('x');a.set_ylabel('y')
    a=axes[0,1]
    for (method,seed),g in f.groupby(['method','seed']):
        a.semilogy(g.step,g.objective,color='#535d65' if method=='deterministic' else '#27947d',
                   lw=1.5,alpha=1 if method=='deterministic' else .5,
                   label=('Deterministic' if method=='deterministic' else 'Probabilistic') if seed==0 else None)
    a.set_title('b   Same state oracle and line search',loc='left');a.legend()
    a.set_xlabel('Accepted or attempted iteration');a.set_ylabel('Tracking objective');a.grid()
    a=axes[0,2]
    grouped=audit.groupby('n').apply(lambda g:np.sqrt(np.mean(g.error**2)),include_groups=False)
    a.loglog(grouped.index,grouped.values,'o-',color='#506ebb',label='Gradient RMSE')
    a.loglog(grouped.index,grouped.iloc[0]*np.sqrt(grouped.index[0]/grouped.index),'--',color='#9a9a9a',label=r'$n^{-1/2}$ reference')
    a.set_title('c   Exact-gradient audit',loc='left');a.set_xlabel('Exit paths');a.set_ylabel('Gradient RMSE');a.grid();a.legend()
    names=['sealed_channel','open_channel','two_open_components','open_plus_island','thin_open_channel']
    labels=['Sealed channel','Open channel','Two open components','Open + island','Thin open channel']
    colors=['#c36157','#27947d','#506ebb','#94649f','#c27a32']
    a=axes[1,0];x=np.linspace(-1,1,200);y=np.linspace(-1,1,80);xx,yy=np.meshgrid(x,y)
    for j,(kind,label) in enumerate(zip(names,labels)):
        mask=cavity(xx,yy,kind)
        a.contourf(xx,.16*yy-j*.5,mask,levels=[.5,1.5],colors=[colors[j]])
        a.plot([-1,1,1,-1,-1],[.16-j*.5,.16-j*.5,-.16-j*.5,-.16-j*.5,.16-j*.5],color='#aaa',lw=.6)
        a.text(1.08,-j*.5,label,va='center',fontsize=7.2)
    a.set_xlim(-1.05,2.7);a.set_ylim(-2.3,.4);a.axis('off');a.set_title('d   Connectivity test geometries',loc='left')
    a=axes[1,1]
    for kind,label,color in zip(names,labels,colors):
        g=eig[(eig.n==129)&(eig.kind==kind)].sort_values('eta')
        a.loglog(g.eta,g.lambda1,'o-',color=color,label=label,ms=4)
    a.set_title('e   Exterior reachability spectrum',loc='left');a.set_xlabel(r'Material contrast $k_0=m_0$')
    a.set_ylabel(r'First Dirichlet eigenvalue $\lambda_1$');a.grid();a.legend(fontsize=6.8,loc='lower right')
    a=axes[1,2];data=json.loads((OUT/'channel_overlap_audit.json').read_text())
    a.bar([d['stage'].capitalize() for d in data],[d['max_gap_over_diameter'] for d in data],color=['#27947d','#9a9a9a','#506ebb'])
    a.axhline(1,color='#c36157',ls='--',lw=1);a.set_ylim(0,1.15)
    a.text(.5,1.03,'Strict overlap requires ratio < 1',transform=a.get_yaxis_transform(),ha='center',fontsize=7)
    a.set_title('f   Saved 3D channel audit',loc='left');a.set_ylabel('Largest consecutive gap / diameter')
    save(fig,'baseline-diagnostics')

if __name__=='__main__':style();sampling_figure();diagnostic_figure()
