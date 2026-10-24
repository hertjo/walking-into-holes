"""Vector typography, paired statistics, matched data-grounded 3D views."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,ConnectionPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter, LogLocator
from PIL import Image
from run_study import HERE,OUT,SPECS,SEEDS
ASSETS=HERE/'render_assets';FIG=HERE.parents[1]/'figures';FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(FIG))
from layout_checks import check_text
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':7,'pdf.fonttype':42,'axes.titlesize':8})
C={'nearest':'#677c47','twos':'#8e5aab','mg':'#54889a','mg49':'#7fb5c4','hot':'#bd8e52','random':'#9b9b9b','coupled-fd':'#9a9fa4','hit-correction':'#438779'}
N={'nearest':'Nearest sensor','twos':'TWoS','mg':'Multigrid TD (65)','mg49':'Multigrid TD (49)','hot':'Hottest point','random':'Random'}
MARK={'twos':'o','mg':'s','mg49':'s','nearest':'D','hot':'^','random':'x'}
LINE={'twos':'-','mg':'-','mg49':'--','nearest':'-.','hot':'-','random':':'}

def image(ax,p):
 ax.imshow(Image.open(p));ax.set_axis_off()

def save(fig,name):
 check_text(fig,FIG/f'{name}-layout.json')
 fig.savefig(FIG/f'{name}.pdf',dpi=600,bbox_inches=None,facecolor='white');fig.savefig(FIG/f'{name}.png',dpi=850,facecolor='white');plt.close(fig)

def plate(study):
 data=pd.read_csv(OUT/'design-results.csv');group=pd.read_csv(OUT/'group-results.csv');vol=pd.read_csv(OUT/'volume-audit.csv')
 fig=plt.figure(figsize=(7.1,6.53));fig.patch.set_facecolor('white')
 title='From cavity placement to shape refinement' if study=='pipeline' else 'Accessible cooling at matched material removal'
 sub='Six cavities per design; Differential WoS: 8 rounds, 16 for hot/random.' if study=='pipeline' else 'Six open ball-union channels; a common removal target; all heat-source supports preserved.'
 fig.text(.035,.97,title,size=14,weight='bold',va='top');fig.text(.035,.937,sub,size=8,color='#55515b',va='top')
 fig.legend(handles=[Line2D([0],[0],color=C[m],marker=MARK[m],ls=LINE[m],ms=3,lw=1,label=N[m],markerfacecolor='white' if m=='mg49' else C[m]) for m in ['random','hot','nearest','twos','mg','mg49']],loc='upper center',bbox_to_anchor=(.5,.912),ncol=6,frameon=False,fontsize=6.5,columnspacing=1.4,handlelength=1.5)
 cols=[.025,.255,.485];width=.225
 for row,name in enumerate(SPECS):
  top=.873-row*.271
  fig.text(.035,top,SPECS[name]['label'],size=9,weight='bold',va='top')
  for col,method in enumerate(['nearest','twos','mg']):
   x=cols[col];state='refine16' if study=='pipeline' and method=='hot' else ('refine8' if study=='pipeline' else 'initial')
   g=group.query('study==@study and geometry==@name and method==@method and state==@state').iloc[0]
   label=N[method]+(' + DWoS' if study=='pipeline' else '')
   fig.text(x+.008,top-.03,label,size=8,weight='bold',color=C[method],va='top')
   ax=fig.add_axes([x,top-.225,width,.185]);image(ax,ASSETS/f'{name}-{study}-{method}.png')
   # Insets use separately rendered, common-camera closeups of the actual sensor.
   ia=fig.add_axes([x+.152,top-.137,.073,.079]);im=ia.imshow(Image.open(ASSETS/f'{name}-{study}-{method}-detail.png'));ia.set_axis_off();circle=Circle((.5,.5),.49,transform=ia.transAxes,facecolor='none',edgecolor=C[method],lw=.8);ia.add_patch(circle);im.set_clip_path(circle)
   meta=json.loads((ASSETS/f'{name}-{study}-{method}.json').read_text());px,py=meta['sensor_pixel']
   connector=ConnectionPatch(xyA=(.12,.18),coordsA=ia.transAxes,xyB=(px,1-py),coordsB=ax.transAxes,color=C[method],lw=.7,alpha=.6,zorder=1);fig.add_artist(connector)
   fig.text(x+.008,top-.231,f'{g.reduction_pct:.1f}% mean sensor cooling',size=7.5,weight='bold',va='top')
   fig.text(x+.008,top-.25,f'{g.seconds:.2f} s  |  {g.backend_peak_rss_mb:.1f} MiB backend' if study=='pipeline' else f'{g.seconds:.2f} s  |  mean radius {g.radius:.3f}' if 'radius' in g.index else f'{g.seconds:.2f} s  |  matched volume',size=6.5,color='#66616c',va='top')
  # Quality-versus-cost is a measured comparison, with one dot per paired seed.
  ax=fig.add_axes([.765,top-(.190 if study=='channels' else .213),.205,.137 if study=='channels' else .16]);a=data.query('study==@study and geometry==@name')
  for method in ['random','hot','nearest','twos','mg','mg49']:
   aa=a[a.method==method]
   states=['initial','refine4','refine8']+(['refine16'] if method in ['hot','random'] else []) if study=='pipeline' else ['initial']
   xx=[];yy=[];low=[];high=[]
   for state in states:
    b=aa[aa.state==state];gg=group.query('study==@study and geometry==@name and method==@method and state==@state').iloc[0];low.append(gg.cooling_ci95_low);high.append(gg.cooling_ci95_high);xx.append(b.seconds.mean());yy.append(b.reduction_pct.mean());ax.scatter(b.seconds,b.reduction_pct,s=5,alpha=.3,color=C[method],edgecolors='none')
   if len(xx)>1:ax.fill_between(xx,low,high,color=C[method],alpha=.12,linewidth=0)
   else:ax.vlines(xx,low,high,color=C[method],lw=1.3,alpha=.8)
   for state,x,y in zip(states,xx,yy):
    gg=group.query('study==@study and geometry==@name and method==@method and state==@state').iloc[0]
    ax.hlines(y,gg.seconds_ci95_low,gg.seconds_ci95_high,color=C[method],lw=.7,alpha=.6)
   ax.plot(xx,yy,marker=MARK[method],ls=LINE[method],lw=1.0,ms=3,color=C[method],label=N[method],markerfacecolor='white' if method=='mg49' else C[method])
  row_stats=group.query('study==@study and geometry==@name');ax.set_xlim(0,1.08*a.seconds.max());ax.set_ylim(min(0,1.1*float(row_stats.cooling_ci95_low.min())),max(.1,1.1*float(row_stats.cooling_ci95_high.max())));ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(4));ax.set_xlabel('Complete design time (s)');ax.set_ylabel('Sensor cooling (%)');ax.grid(axis='y',color='#dedce1',lw=.5);ax.set_title('Quality and total cost',loc='left',size=7.5)
  if study=='channels':
   v=vol[vol.geometry==name];fig.text(.766,top-.248,f'Volume target: 0.0168\nAudit: {v.volume.min():.5f}–{v.volume.max():.5f}',size=6.5,color='#66616c',va='top')
 # One color bar and one explicit summary of the different sampling populations.
 cax=fig.add_axes([.055,.054,.30,.010]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap='viridis'),cax=cax,orientation='horizontal');cb.set_label('Local cooling (%)',size=6.5);cb.ax.tick_params(labelsize=6.5)
 fig.text(.43,.045,'Cyan: protected sensor   •   Orange: complete heat-source support',size=7,va='center')
 fig.text(.43,.025,'First-seed views; mean curves; dots: runs; bands/bars: pointwise 95% CIs.',size=6.5,color='#66616c',va='center')
 save(fig,'submission-'+study)

def estimator():
 f=pd.read_csv(OUT/'estimator-results.csv');fig=plt.figure(figsize=(7.1,5.65));fig.text(.045,.97,'A direct insertion score avoids a vanishing trial target',size=13,weight='bold',va='top');fig.text(.045,.93,'Eight sites per geometry; 16 repeats. Variance: 2,048 paths/site. Error–cost: 256 and 2,048 paths/site.',size=7.5,color='#66616c',va='top')
 fig.legend(handles=[Line2D([0],[0],color=C[m],marker=mark,lw=1,label=label,ms=3) for m,mark,label in [('coupled-fd','^','CRN subtraction'),('hit-correction','D','Harmonic correction'),('twos','o','Direct TWoS'),('mg','s','Multigrid')]],loc='upper center',bbox_to_anchor=(.5,.906),ncol=4,frameon=False,fontsize=7,columnspacing=1.8,handlelength=1.5)
 for row,name in enumerate(SPECS):
  top=.815-row*.255;a=f[f.geometry==name]
  fig.text(.04,top+.035,SPECS[name]['label'],size=9,weight='bold')
  ax=fig.add_axes([.025,top-.20,.24,.21]);image(ax,ASSETS/f'{name}-opaque.png')
  ax=fig.add_axes([.32,top-.18,.28,.195]);b=a[(a.method=='coupled-fd')&(a.budget==2048)].sort_values('radius');ax.fill_between(b.radius,b.variance_ci95_low,b.variance_ci95_high,color=C['coupled-fd'],alpha=.15,lw=0);ax.loglog(b.radius,b.normalized_variance,'^--',color=C['coupled-fd'],label='CRN subtraction');b=a[(a.method=='hit-correction')&(a.budget==2048)&(a.variance_status!='unresolved_all_zero')].sort_values('radius');ax.fill_between(b.radius,b.variance_ci95_low,b.variance_ci95_high,color=C['hit-correction'],alpha=.15,lw=0);ax.loglog(b.radius,b.normalized_variance,'D-',color=C['hit-correction'],label='Harmonic correction');t=a[(a.method=='twos')&(a.budget==2048)].iloc[0];ax.axhspan(t.variance_ci95_low,t.variance_ci95_high,color=C['twos'],alpha=.14,lw=0);ax.axhline(t.normalized_variance,color=C['twos'],lw=2,label='Direct TWoS');ax.set_xlabel('Trial-cavity radius');ax.set_ylabel('Relative variance');ax.grid(which='major',color='#e6e3e8',lw=.5)
  ax.set_xticks([.01,.02,.04,.08],['0.01','0.02','0.04','0.08']);ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_major_locator(LogLocator(numticks=4))
  if name=='ring':ax.text(.04,.42,'r = 0.01, 0.02: all corrections zero\nVariance unresolved',transform=ax.transAxes,size=6.5,color=C['hit-correction'],bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
  ax=fig.add_axes([.69,top-.18,.265,.195])
  for method in ['coupled-fd','hit-correction','twos','mg']:
   b=a[a.method==method].sort_values('seconds');resolved=b[b.variance_status!='unresolved_all_zero'];unresolved=b[b.variance_status=='unresolved_all_zero'];marker={'mg':'s','twos':'o','coupled-fd':'^','hit-correction':'D'}[method];ax.scatter(resolved.seconds,resolved.relative_rmse,c=C[method],s=16,marker=marker);ax.scatter(unresolved.seconds,unresolved.relative_rmse,facecolors='none',edgecolors=C[method],s=22,marker=marker)
   for _,r in b.iterrows():
    ax.vlines(r.seconds,r.rmse_ci95_low,r.rmse_ci95_high,color=C[method],lw=.7,alpha=.5)
    if np.isfinite(r.seconds_ci95_low):ax.hlines(r.relative_rmse,r.seconds_ci95_low,r.seconds_ci95_high,color=C[method],lw=.7,alpha=.5)
   if method=='mg':
    for _,r in b.iterrows():ax.annotate(str(int(r.budget)),(r.seconds,r.relative_rmse),xytext=(5,-12),textcoords='offset points',size=6.5,color=C[method],arrowprops={'arrowstyle':'-','lw':.4,'color':C[method]})
  ax.set_xscale('log');ax.set_yscale('log');ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter());ax.xaxis.set_major_locator(LogLocator(numticks=4));ax.yaxis.set_major_locator(LogLocator(numticks=4));ax.margins(x=.15,y=.3);ax.set_xlabel('Time for eight query sites (s)');ax.set_ylabel('Relative RMS error');ax.grid(which='major',color='#e6e3e8',lw=.5)
 fig.text(.045,.025,'Finite-hole errors include radius bias; grid errors include discretization. Open markers: all-zero corrections.\nBands/bars: pointwise 95% bootstrap CIs; numerical bias and unobserved rare events are not covered.',size=6.5,color='#66616c')
 save(fig,'submission-estimators')

def geometry_plate():
 fig=plt.figure(figsize=(7.1,3.6));fig.text(.035,.97,'Three geometries, three different transport challenges',size=13,weight='bold',va='top')
 details={'housing':'Parallel fins, narrow gaps,\nand mounting bosses','branch':'Branching transport, thin ribs,\nand an elevated return loop','ring':'Curved transport, a central bridge,\nand closely spaced ribs'}
 for col,name in enumerate(SPECS):
  x=.02+col*.33;fig.text(x+.02,.88,SPECS[name]['label'],size=9,weight='bold');ax=fig.add_axes([x,.17,.32,.65]);image(ax,ASSETS/f'{name}-opaque.png')
  audit=json.loads((HERE/'assets/geometry-audit.json').read_text())[name];fig.text(x+.02,.16,details[name],size=7,va='top');fig.text(x+.02,.075,f'{audit["triangles"]:,} triangles; watertight input surface',size=7,color='#66616c')
 fig.text(.04,.025,'The same physical domain is used by every method. Presentation cuts and ghost shells are excluded from all PDE solves.',size=7,color='#66616c')
 save(fig,'submission-geometries')

if __name__=='__main__':
 plate('pipeline');plate('channels');estimator();geometry_plate()
