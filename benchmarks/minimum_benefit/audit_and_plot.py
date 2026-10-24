"""Audit retained validation data; produce publication-quality vector figures."""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal,localcontext
import hashlib,json,math,sys
import igl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_application_validation as protocol

HERE=Path(__file__).resolve().parent
OUT=protocol.OUT

def main():
    results={k:json.loads((OUT/k/'result.json').read_text()) for k in ['bunny','housing']}
    exp_sum=sum((Q(17,2)**k/Q(math.factorial(k)) for k in range(26)),Q(0))
    assert exp_sum>4800
    rows=[];total_steps=0;max_excess=-math.inf;comparisons=0
    rng=np.random.default_rng(2026091699)
    for kind,res in results.items():
        V,F=igl.read_triangle_mesh(str(protocol.ROOT/res['mesh']))
        tree=igl.AABB();tree.init(V,F)
        accepted=[]
        for record in res['records']:
            sf=OUT/kind/f"attempt-{record['step']:02d}"
            samples=pd.read_csv(sf/'validation.csv',float_precision='round_trip')
            C=protocol.CONFIG['value_caps'][kind]
            assert len(samples)==65536 and np.array_equal(samples['sample'].values,np.arange(65536))
            assert np.all((samples.value>=0)&(samples.value<=C))
            assert samples.steps.max()<=128
            total_steps+=int(samples.steps.sum())
            assert record['containment_failures']==0
            z=np.asarray(record['center']);r=record['radius']
            inside,_=protocol.exact_inside(z,V,F);assert inside
            actual_distance=float(np.sqrt(tree.squared_distance(V,F,z[None])[0][0]))
            if accepted:actual_distance=min(actual_distance,min(np.linalg.norm(z-c)-r for c in accepted))
            d=record['stages'][0]['center_clearance_lower']
            max_excess=max(max_excess,d-actual_distance)
            assert d<=actual_distance+1e-13
            for st in record['stages']:
                dn=lambda x:float(np.nextafter(x,-np.inf))
                up=lambda x:float(np.nextafter(x,np.inf))
                exact_sum=sum((Q(float(x)) for x in samples.value.values[:st['n']]),Q(0))
                margin=up(C*up(math.sqrt(up(8.5/(2*st['n'])))))
                lower=max(0.,dn(dn(float(exact_sum/st['n']))-margin))
                assert lower==st['state_lower'],(kind,record['step'],lower,st['state_lower'])
                floor,geo=protocol.benefit(lower,z,r,d,kind)
                avg,_=protocol.average_benefit(lower,z,r,d,kind)
                assert st['benefit_lower']==max(floor,avg)
                assert st['passes']==(st['benefit_lower']>=record['target_absolute'])
            assert record['benefit_lower']==max(s['benefit_lower'] for s in record['stages'])
            assert record['accepted']==any(s['passes'] for s in record['stages'])
            if record['accepted']:accepted.append(z)
            # Independent high-precision source check at random points in the
            # selected ball. This supplements the all-points enclosure proof.
            for _ in range(20):
                direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
                p=z+r*rng.random()**(1/3)*direction
                flo,fhi=protocol.source_bounds(z,r,kind)
                with localcontext() as ctx:
                    ctx.prec=70
                    dd=lambda x:Decimal.from_float(float(x))
                    if kind=='bunny':truth=Decimal(1)+dd(.65)*(dd(p[1])+dd(.5))
                    else:
                        truth=dd(.08)
                        for amp,center,width in zip(protocol.AMPL,protocol.CENTERS,protocol.WIDTHS):
                            t=sum(((dd(x)-dd(c))/dd(w))**2 for x,c,w in zip(p,center,width))
                            truth+=dd(amp)*(-t).exp()
                    assert dd(flo)<=truth<=dd(fhi)
                comparisons+=1
            measured=record['measured_trial_benefit']
            rows.append(dict(geometry=kind,attempt=record['step']+1,accepted=record['accepted'],
                target=record['target_absolute'],bound=record['benefit_lower'],
                average=record['average_bound'],floor=record['floor_bound'],source_only=record['source_only_lower'],
                measured=measured,reference_se=record['measured_trial_benefit_se'],
                bound_over_reference=record['benefit_lower']/measured,
                average_over_floor=record['average_bound']/record['floor_bound'],
                first_passing_n=record['first_passing_n'],validation_seconds=record['validation_seconds']))
        assert len(accepted)==res['accepted_count']
        exact_total=sum(Q(float(x['benefit_lower'])) for x in res['records'] if x['accepted'])
        # Publish a downward-rounded cumulative benefit rather than a sum
        # rounded in an unspecified direction.
        res['sum_accepted_benefit_lower']=float(np.nextafter(float(exact_total),-np.inf))
        res['first_passing_prefix_seconds']=sum(next(s['elapsed_seconds'] for s in x['stages'] if s['passes'])
            if x['accepted'] else x['stages'][-1]['elapsed_seconds'] for x in res['records'])
        initial=np.asarray(res['initial_reference']['blocks']);final=np.asarray(res['final_reference']['blocks'])
        res['measured_total_benefit']=float((initial-final).mean())
        res['measured_total_benefit_se']=float((initial-final).std(ddof=1)/2)
        res['measured_reduction_percent']=float(100*(1-final.mean()/initial.mean()))
        protocol.save(OUT/kind/'result.json',res)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'validation-results.csv',index=False)
    audit=dict(sample_count=12*65536,guarded_steps=total_steps,containment_failures=0,
        stage_checks=48,high_precision_source_comparisons=comparisons,
        exact_initial_parity_checks=12,maximum_lower_distance_minus_libigl_distance=max_excess,
        exact_exp_series_sum_exceeds_4800=True,exact_exp_series_sum=float(exp_sum),
        total_ideal_sampling_error_budget='48/4800 = 0.01',
        arithmetic_audit='Pass: retained sample ranges, exact dyadic means, outward scalar bounds, exact initial parity, source enclosures, conditional acceptance logic.',
        remaining_scope='Ideal random sampling laws and their propagation through finite-precision coordinates are not proved by these numerical audits. Input is assumed to be an embedded closed surface.')
    protocol.save(OUT/'audit.json',audit)
    protocol.save(OUT/'summary.json',dict(configuration=protocol.CONFIG,results=results,audit=audit))
    attach_grid(results,frame)
    plot(results,frame)
    print(json.dumps({k:{key:v[key] for key in ['accepted_count','sum_accepted_benefit_lower','measured_total_benefit','measured_reduction_percent','validation_seconds','selection_seconds','first_passing_prefix_seconds']} for k,v in results.items()},indent=2))

def attach_grid(results,frame):
    for kind,res in results.items():
        folder=OUT/kind
        baseline=json.loads((folder/'initial/multigrid-reference.json').read_text())
        before=baseline['mean']
        for rec in res['records']:
            mg=json.loads((folder/f"attempt-{rec['step']:02d}/trial/multigrid-reference.json").read_text())
            gain=before-mg['mean']
            rec['multigrid_trial_benefit']=gain
            mask=(frame.geometry==kind)&(frame.attempt==rec['step']+1)
            frame.loc[mask,'multigrid_reference']=gain
            # Empirical comparison with a second approximation, not a proof.
            assert gain>rec['benefit_lower']
            if rec['accepted']:before=mg['mean']
            protocol.save(folder/f"attempt-{rec['step']:02d}/decision.json",rec)
        res['multigrid_total_benefit']=baseline['mean']-before
        res['reference_gain_relative_difference']=abs(res['multigrid_total_benefit']-res['measured_total_benefit'])/res['measured_total_benefit']
        protocol.save(folder/'result.json',res)
    frame.to_csv(OUT/'validation-results.csv',index=False)
    report=json.loads((OUT/'summary.json').read_text());report['results']=results
    protocol.save(OUT/'summary.json',report)

def plot(results,frame):
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.axisbelow':True})
    purple='#8754ac';blue='#5794aa';gray='#56616a';orange='#c48131'
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.6),layout='constrained')
    for ax,kind,title in zip(axes[0],['bunny','housing'],['a  Bunny: finite insertion benefit','b  Housing: finite insertion benefit']):
        r=frame[frame.geometry==kind];scale=protocol.CONFIG['fixed_scales'][kind]
        x=r.attempt.values
        ax.plot(x,r.multigrid_reference/scale*100,'D--',color='#9a9fa4',mfc='white',lw=1,ms=4,label='Multigrid reference')
        ax.errorbar(x,r.measured/scale*100,yerr=r.reference_se/scale*100,fmt='o-',color=gray,lw=1.3,ms=4,capsize=3,label='Independent WoS reference')
        ax.plot(x,r.bound/scale*100,'o-',color=purple,lw=1.7,ms=4,label='Average-trace certificate')
        ax.plot(x,r.floor/scale*100,'s--',color=blue,lw=1.2,ms=3.5,label='Temperature-floor certificate')
        ax.plot(x,r.source_only/scale*100,':',color=orange,lw=1.6,label='Source-only bound')
        ax.axhline(.1,color='#aaaaaa',lw=.8,ls='--')
        ax.set(xlabel='Attempted insertion',ylabel='Benefit / fixed scale (%)',title=title,xticks=x)
        ax.grid(axis='y',alpha=.18)
        if kind=='bunny':ax.legend(frameon=False,fontsize=7.5,loc='upper right')
    ax=axes[1,0]
    for kind,color in [('bunny',purple),('housing',blue)]:
        res=results[kind];st=res['records'][0]['stages'];scale=protocol.CONFIG['fixed_scales'][kind]
        ax.plot([s['n'] for s in st],[100*s['benefit_lower']/scale for s in st],'o-',color=color,label=kind.title())
    ax.set(xscale='log',xlabel='Independent walks at the first candidate',ylabel='Benefit bound / fixed scale (%)',title='c  Tightening the bound with more samples')
    ax.set_xticks(protocol.STAGES,[f'{n:,}' for n in protocol.STAGES]);ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
    ax=axes[1,1];x=np.arange(3);w=.34
    for off,kind,color in [(-w/2,'bunny',purple),(w/2,'housing',blue)]:
        r=results[kind];vals=[r['selection_seconds'],r['first_passing_prefix_seconds'],r['validation_seconds']]
        ax.bar(x+off,vals,w,color=color,label=kind.title())
        for xx,v in zip(x+off,vals):ax.text(xx,v*1.10,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    ax.set(yscale='log',ylabel='Seconds for six attempts',title='d  Added cost of conservative validation',xticks=x,
        xticklabels=['Scout + rescore','First passing prefixes','Full validation sweep'])
    ax.tick_params(axis='x',labelsize=8);ax.set_ylim(.06,max(r['validation_seconds'] for r in results.values())*3)
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8,loc='upper left')
    fig.savefig(OUT/'application-minimum-benefit.pdf',bbox_inches='tight')
    fig.savefig(OUT/'application-minimum-benefit.png',dpi=600,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
