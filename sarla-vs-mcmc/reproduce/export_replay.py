"""Export a compact, faithful replay from the completed benchmark.

Usage: python export_replay.py /path/to/sarla-carbon-benchmark /path/to/replay.json
Requires the benchmark dependencies. No upstream source is edited.
"""
import sys, json, base64
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]); dest=Path(sys.argv[2]); sys.path.insert(0,str(root))
import benchmark as b
R=json.loads((root/'results/results.json').read_text())
S=json.loads((root/'deliverables/summary.json').read_text())

def packed(x):
    x=np.asarray(x)
    assert np.isfinite(x).all() and np.abs(x).max()<=1.00001
    return base64.b64encode(np.rint((np.clip(x,-1,1)+1)*32767.5).astype('<u2').tobytes()).decode()

def charts(atlas):
    return [dict(center=c.center.tolist(),vectors=c.eigvecs.tolist(),var=c.var.tolist(),rank=int(c.rank),id=c.id) for c in atlas.charts]

D=dict(cases={},source_commit=R['source_commit'],frames=96,chains=32)
for n in [4,2]:
    c=R['cases'][str(n)]; ref=np.load(root/f'results/reference_{n}.npz'); x=ref['x']
    rng=np.random.default_rng(501+n)
    ids=rng.choice(len(x)**2,1600,p=ref['marginal'].ravel()/ref['marginal'].sum())
    cloud=[]
    for index in ids:
        i,j=np.unravel_index(index,(len(x),len(x)))
        u=np.column_stack([np.full(len(x),x[i]),np.full(len(x),x[j]),x])
        r=(b.pred_np(u,b.TIMES[:n])-b.DATA[:n])/2
        w=np.exp(-.5*np.sum(r*r,axis=1)); k=rng.choice(len(x),p=w/w.sum()); cloud.append(u[k])
    C=dict(reference=packed(cloud),starts=[z['u'] for z in c['gn_solutions']],runs={},stats=S[str(n)]['stats'],fit_s=c['fit_s'],compile_s=c['compile_s'],gaussian=c['gaussian'],best=c['best_physical'],rank=c['gn_rank'])
    target,*_=b.make_target(n)
    centres=np.array(C['starts'])
    for tag,label,thresh in [('sarla','SARLA2 default + IMH',5),('sarla_tight','SARLA2 tighter audit + IMH',2),('emcee','DE ensemble MCMC',None),('rwm','Local random-walk MH',None)]:
        saved=np.load(root/f'results/{tag}_{n}_3.npz'); U=saved['samples']
        rec=next(v for v in c['runs'] if v['method']==label and v['seed']==3)
        frame_indices=np.rint(np.linspace(0,len(U)-1,D['frames'])).astype(int)
        flat=U.reshape(-1,3); inds=np.linspace(0,len(flat)-1,1800).astype(int)
        curve=b.pred_np(flat[::20],np.arange(61))
        run=dict(label=label,trace=packed(U[frame_indices,::2,:]),posterior=packed(flat[inds]),frame_indices=frame_indices.tolist(),steps=len(U),burn=rec.get('burn',(rec['draws']//64)//3),tv=rec['tv'],accept=rec['accept'],mode_mass=rec['mode1_mass'],switching=rec['fraction_chains_switching_mode'],history=rec.get('history',[]),predictive=np.quantile(curve,[.05,.5,.95],axis=0).round(3).tolist())
        if thresh is not None:
            snapshots=[]; original=b.S2.audit
            def record_audit(atlas,rng):
                snapshots.append(charts(atlas))
                return original(atlas,rng)
            b.S2.audit=record_audit
            try:
                atlas=b.S2.sarla2(target,centres,b.S2.SurgeryConfig(rounds=6,n_audit=4096,flag_thresh=thresh),seed=3,verbose=False)
            finally:
                b.S2.audit=original
            snapshots.append(charts(atlas))
            assert np.allclose(np.array([v.center for v in atlas.charts]),saved['chart_centres'])
            assert len(atlas.history)==len(rec['history'])
            for new,old in zip(atlas.history,rec['history']):
                assert new['K_after']==old['K_after'] and new['n_flags']==old['n_flags'] and np.isclose(new['ess'],old['ess'])
            run['charts']=snapshots
        C['runs'][tag]=run
    D['cases'][str(n)]=C
dest.write_text(json.dumps(D,separators=(',',':'),allow_nan=False))
print('Exported verified replay:',dest.stat().st_size,'bytes')
