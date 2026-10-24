"""Reference-inspired scientific plate; vector labels, measured comparisons."""
import json
import numpy as np
import pandas as pd
import igl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
import run_accessible as a
from render_professional import ASSETS,centers,SEED,METHODS,project_points
from layout_checks import check_text

COL={'wos_hot':'#b77b38','twos_sparse':'#8e53a8','mg57':'#4e8c9f'}
NAMES={'wos_hot':'Hottest-point placement','twos_sparse':'TWoS placement','mg57':'Multigrid placement'}

def main():
    meta=json.loads((ASSETS/'manifest.json').read_text());assert not meta['preview']
    data=pd.read_csv(a.OUT/'results.csv');vmax=meta['temperature_scale_max']
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':17,
      'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
      'text.color':'#2c2930','axes.labelcolor':'#2c2930','axes.linewidth':.8})
    fig=plt.figure(figsize=(16,8.6),facecolor='white')
    # Same crop, same world-space camera, same world-space detail center.
    for i,m in enumerate(METHODS):
        x=.008+i*.331
        ax=fig.add_axes([x,.455,.325,.475]);ax.set(xlim=(0,1),ylim=(0,.79));ax.axis('off')
        im=Image.open(ASSETS/f'{m}-with-ghost.png');w,h=im.size
        im=im.crop((0,round(.14*h),w,round(.93*h)))
        ax.imshow(im,extent=[0,1,0,.79],zorder=1)
        sensor=np.load(ASSETS/f'{m}-projection.npz')['sensor_pixel']
        focus=(sensor[0],.93-sensor[1]);inset=(.82,.605);radius=.17
        ax.plot([focus[0],inset[0]-.10],[focus[1],inset[1]-.12],color=COL[m],lw=1,zorder=3)
        ax.add_patch(Circle(focus,.006,facecolor='white',edgecolor=COL[m],lw=1,zorder=4))
        clip=Circle(inset,radius,facecolor='white',edgecolor='none',zorder=5);ax.add_patch(clip)
        detail=ax.imshow(Image.open(ASSETS/f'{m}-detail.png'),extent=[inset[0]-radius,inset[0]+radius,inset[1]-radius,inset[1]+radius],zorder=6)
        detail.set_clip_path(clip)
        theta=np.linspace(0,2*np.pi,240);target=a.b.center(1)
        ring=target+np.column_stack((.13*np.cos(theta),.13*np.sin(theta),np.zeros_like(theta)))
        pixels=project_points(ring,target,.31,1)
        line,=ax.plot(inset[0]-radius+2*radius*pixels[:,0],inset[1]+radius-2*radius*pixels[:,1],
                     color='#19c1d1',lw=1.15,zorder=6.5)
        line.set_clip_path(clip)
        ax.add_patch(Circle(inset,radius,fill=False,edgecolor=COL[m],lw=1.4,zorder=7))
        fig.text(x+.017,.964,NAMES[m],fontsize=22,fontweight='bold',color=COL[m])
        subtitle={'wos_hot':'Ranks temperature $u$','twos_sparse':'Ranks regional benefit $up$','mg57':'Ranks $up$ on a volumetric grid'}[m]
        fig.text(x+.017,.934,subtitle,fontsize=17)
        row=data[(data.side==1)&(data.seed==SEED)&(data.method==m)].iloc[0]
        fig.text(x+.017,.426,f'{100*row.reduction:.1f}% lower regional temperature',fontsize=17,fontweight='bold',color=COL[m])
    fig.text(.025,.388,'First right-sensor seed. Cyan: sensor; faint shells: cut-away bars.',fontsize=13.5,color='#55515a')
    cax=fig.add_axes([.745,.375,.16,.010]);fig.colorbar(ScalarMappable(norm=Normalize(0,vmax),cmap='inferno'),cax=cax,orientation='horizontal',ticks=[0,vmax/2,vmax])
    cax.tick_params(labelsize=11,length=2,pad=2);fig.text(.745,.398,'Temperature $u$ (shared scale)',fontsize=13)

    fig.text(.025,.310,'Why the placements differ',fontsize=21,fontweight='bold')
    fig.text(.025,.279,'Temperature $u$ × target influence $p$ gives placement priority.',fontsize=14)
    initial=np.load(ASSETS/'initial-field.npz');xx,yy=np.meshgrid(initial['x'],initial['y'])
    mask=initial['mask'];sensor=a.b.center(1)
    forbidden=np.hypot(xx-sensor[0],yy-sensor[1])<a.R+.13+.02
    for c in a.DEVICES:forbidden |= np.hypot(xx-c[0],yy-c[1])<a.R+.07+.02
    forbidden |= (np.abs(xx)>.84)|(np.abs(yy)>.41)
    v,f=igl.read_triangle_mesh(str(a.b.MESH))
    points=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,-.03)))
    distance,_,_=igl.point_mesh_squared_distance(points,v,f)
    forbidden |= np.sqrt(distance).reshape(xx.shape)<=a.R+.023
    for i,(key,title,cmap) in enumerate([('u','Original temperature $u$','inferno'),('score','Admissible priority $up$','viridis')]):
        ax=fig.add_axes([.025+i*.222,.065,.202,.170]);ax.axis('off')
        val=initial[key].copy();val[~mask]=np.nan
        if key=='score':val[forbidden]=np.nan;val/=np.nanmax(val)
        ax.imshow(val,origin='lower',extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap=cmap,vmin=0,vmax=vmax if key=='u' else 1)
        ax.contour(xx,yy,mask.astype(float),levels=[.5],colors=['#b0adb2'],linewidths=.6)
        ax.add_patch(Circle(sensor[:2],.13,fill=False,ec='#19c1d1',lw=1.5))
        ax.set_title(title,fontsize=16,pad=4)
        if key=='score':
            for c in centers('twos_sparse'):ax.plot(c[0],c[1],'o',mfc='none',mec=COL['twos_sparse'],ms=6,mew=1.2)
            ax.text(.5,-.04,'Blank: excluded axes; circles: final TWoS axes',transform=ax.transAxes,ha='center',fontsize=11,color='#55515a')
    priority_bar=fig.add_axes([.458,.089,.006,.115])
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='viridis'),cax=priority_bar,ticks=[0,1])
    priority_bar.tick_params(labelsize=10,length=2);priority_bar.set_title('rel.',fontsize=10,pad=3)
    fig.text(.025,.017,'Ideal cold walls; point-score heuristic. Display cuts are excluded from the PDE.',fontsize=11,color='#55515a')

    fig.text(.535,.310,'The measured advantage',fontsize=21,fontweight='bold')
    fig.text(.535,.279,'Both sensor positions; five paired seeds; six channels per design.',fontsize=13)
    cool=fig.add_axes([.64,.115,.19,.120]);cost=fig.add_axes([.872,.115,.105,.120])
    order=METHODS;means=[];se=[];times=[]
    for m in order:
        g=data[data.method==m];paired=g.groupby('seed').reduction.mean()*100
        means.append(paired.mean());se.append(paired.std(ddof=1)/np.sqrt(len(paired)));times.append(g.complete_design_seconds.mean())
    y=np.arange(3)
    for j,m in enumerate(order):
        cool.barh(j,means[j],height=.46,color=COL[m],alpha=.92)
        cool.errorbar(means[j],j,xerr=se[j],fmt='none',ecolor='#28232b',capsize=3,lw=1)
        cool.text(means[j]+2,j,f'{means[j]:.1f}%',va='center',fontsize=14,fontweight='bold')
        cost.barh(j,times[j],height=.46,color=COL[m],alpha=.92)
        cost.text(times[j]+.04,j,f'{times[j]:.2f}',va='center',fontsize=14,fontweight='bold')
    cool.set(yticks=y,yticklabels=['Hottest','TWoS','Multigrid'],xlim=(0,70),xticks=[0,20,40,60],ylim=(-.6,2.6));cool.invert_yaxis()
    cost.set(yticks=[],xlim=(0,1.65),xticks=[0,.5,1,1.5],ylim=(-.6,2.6));cost.invert_yaxis()
    cool.set_title('Regional cooling (%)',fontsize=16,pad=7);cost.set_title('Design time (s)',fontsize=16,pad=7)
    for ax in [cool,cost]:ax.tick_params(labelsize=13);ax.grid(axis='x',color='#e5e1e7',lw=.6);ax.set_axisbelow(True)
    gain=means[1]-means[0]
    fig.text(.535,.047,f'+{gain:.1f} points vs. hottest-point placement at similar cost.',fontsize=15,fontweight='bold',color=COL['twos_sparse'])
    fig.text(.535,.017,'Multigrid cools more; TWoS is faster here. Full study: 50 designs.',fontsize=11,color='#55515a')
    check_text(fig,ASSETS/'text-layout-audit.json')
    fig.savefig(a.OUT/'accessible-channels.pdf',dpi=350)
    fig.savefig(a.OUT/'accessible-channels.png',dpi=500)
    plt.close(fig)
    (ASSETS/'composition-manifest.json').write_text(json.dumps({'means':dict(zip(order,means)),
      'se_across_five_paired_seed_averages':dict(zip(order,se)),'mean_seconds':dict(zip(order,times)),
      'single_design_seed':SEED,'single_design_side':1,'crop_relative':[0,.14,1,.93],
      'data_source':'accessible_channels/output/results.csv','png_width':8000},indent=2)+'\n')

if __name__=='__main__':main()
