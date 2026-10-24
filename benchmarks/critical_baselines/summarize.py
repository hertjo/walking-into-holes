"""Validate retained experiment coverage and generate manuscript numbers."""
import json
import numpy as np
import pandas as pd
from sampling import OUT


def scientific(x):
    mantissa, exponent = f'{x:.2e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


if __name__ == '__main__':
    f = pd.read_csv(OUT/'sampling_runs.csv')
    s = pd.read_csv(OUT/'source_sampling_runs.csv')
    shape = pd.read_csv(OUT/'shape_optimization.csv')
    eig = pd.read_csv(OUT/'connectivity_spectrum.csv')
    assert len(f) == 768 and len(s) == 288 and len(eig) == 45
    assert (f.groupby(['domain','method','n']).rep.nunique() == 32).all()
    assert (s.groupby(['method','n']).rep.nunique() == 32).all()
    assert not f.duplicated(['domain','method','n','rep']).any()
    assert not s.duplicated(['method','n','rep']).any()
    for frame in (f,s):
        assert np.isfinite(frame[['mse','seconds']]).all().all()
        assert (frame[['mse','seconds']] > 0).all().all()
    g=s[s.n==1024].groupby('method')
    mse=g.mse.mean(); seconds=g.seconds.median()
    neighbor=f[(f.n==1024)&(f.domain=='square')].groupby('method').mse.mean()
    final=shape[(shape.method=='probabilistic')&(shape.step==30)].objective
    assert len(final)==5
    rng=np.random.default_rng(60213)
    paired=s[s.n==1024].pivot(index='rep',columns='method',values='mse')
    ix=rng.integers(0,32,size=(10000,32))
    ratios=paired.mc.to_numpy()[ix].mean(axis=1)/paired['array'].to_numpy()[ix].mean(axis=1)
    ci=np.quantile(ratios,[.025,.975])
    metrics={
      'SourceArrayMSE':scientific(mse['array']),
      'SourceMCMSE':scientific(mse.mc),
      'SourceRatio':f'{mse.mc/mse["array"]:.1f}',
      'SourceArrayTime':f'{seconds["array"]:.3f}',
      'SourceMCTime':f'{seconds.mc:.3f}',
      'NeighborRatio':f'{neighbor.mc/neighbor.neighbors:.2f}',
      'ShapeInitial':scientific(shape[shape.step==0].objective.iloc[0]),
      'ShapeMinimum':scientific(final.min()),
      'ShapeMaximum':scientific(final.max()),
      'SealedEigen':scientific(eig[(eig.n==129)&(eig.eta==1e-6)&(eig.kind=='sealed_channel')].lambda1.iloc[0])}
    (OUT/'metrics.tex').write_text('% Generated from retained raw experiments; do not edit.\n'+
        '\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in metrics.items())+'\n')
    summary={'retained_harmonic_runs':len(f),'retained_poisson_runs':len(s),
      'source_equal_path_mse_ratio':float(mse.mc/mse['array']),
      'source_ratio_bootstrap95':ci.tolist(),
      'source_median_runtime_ratio':float(seconds['array']/seconds.mc),
      'square_neighbor_equal_path_mse_ratio':float(neighbor.mc/neighbor.neighbors),
      'latex_metrics':metrics}
    (OUT/'critical_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
