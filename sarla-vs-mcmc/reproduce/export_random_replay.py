"""Build the updated replay, including actual SARLA optimization iterations.

Usage: python export_random_replay.py /path/to/original/benchmark
Run random_start_mcmc.py first. Reads the archived optimized-start replay.
"""
import sys, json, base64, re
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

HERE=Path(__file__).resolve().parent
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
import benchmark as b
R=json.loads((root/'random-start/results.json').read_text())
D=json.loads((HERE/'replay-optimized.json').read_text())

def packed(x):
    x=np.asarray(x)
    assert np.isfinite(x).all() and np.abs(x).max()<=1.00001
    return base64.b64encode(np.rint((np.clip(x,-1,1)+1)*32767.5).astype('<u2').tobytes()).decode()

for n in [4,2]:
    C=D['cases'][str(n)];c=R['cases'][str(n)]
    target,res,jac,counts,_=b.make_target(n)
    raw=np.random.default_rng(1729).uniform(-.9,.9,(16,3))
    paths=[];endpoints=[]
    for start in raw:
        path=[start.copy()]
        def callback(x): path.append(x.copy())
        fit=least_squares(res,start,jac=jac,bounds=(-np.ones(3),np.ones(3)),
            ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=400,callback=callback)
        if not np.array_equal(path[-1],fit.x):path.append(fit.x.copy())
        endpoints.append((float(fit.cost),fit.x))
        paths.append(np.asarray(path).tolist())
    endpoints.sort(key=lambda x:x[0])
    assert np.allclose([v[1] for v in endpoints],C['starts'],atol=1e-10)
    C['optimization']=dict(paths=paths,seed=1729,algorithm='Bounded trust-region Gauss-Newton with JAX residual Jacobian',
        counts=counts.copy(),frames=max(map(len,paths)))
    c['optimization_counts']=counts.copy()
    C['stats']=c['stats']; C['initialization']='random'
    for tag,label in [('emcee','DE ensemble MCMC'),('rwm','Local random-walk MH')]:
        saved=np.load(root/f'random-start/{tag}_{n}_3.npz');U=saved['samples'];chain=saved['chain']
        rec=next(v for v in c['runs'] if v['method']==label and v['seed']==3)
        old=C['runs'][tag]; flat=U.reshape(-1,3)
        frame_indices=np.rint(np.linspace(0,len(U)-1,D['frames'])).astype(int)
        warm=np.concatenate([saved['initial'][None],chain[:rec['burn']]],axis=0)
        warm_indices=np.rint(np.linspace(0,len(warm)-1,64)).astype(int)
        curves=b.pred_np(flat[::20],np.arange(61))
        old.update(trace=packed(U[frame_indices,::2,:]),posterior=packed(flat[np.linspace(0,len(flat)-1,1800).astype(int)]),
            frame_indices=frame_indices.tolist(),steps=len(U),burn=rec['burn'],tv=rec['tv'],accept=rec['accept'],
            mode_mass=rec['mode1_mass'],switching=rec['fraction_chains_switching_mode'],
            predictive=np.quantile(curves,[.05,.5,.95],axis=0).round(3).tolist(),
            initial=saved['initial'].tolist(),warmup=packed(warm[warm_indices,::2,:]),warmup_frames=64,
            initialization='Random uniform prior draws',target_backend='NumPy')
    # Check NumPy and JAX targets independently on prior and out-of-prior points.
    from random_start_mcmc import make_target
    np_target,_=make_target(n)
    check=np.random.default_rng(n).uniform(-1.2,1.2,(97,3))
    assert np.allclose(target['logpost_batch'](check),np_target(check),rtol=1e-12,atol=1e-10)
    print(n,'observations: verified optimization paths',len(paths),'max iterations',C['optimization']['frames']-1,flush=True)
(root/'random-start/replay.json').write_text(json.dumps(D,separators=(',',':'),allow_nan=False))
(root/'random-start/results.json').write_text(json.dumps(R,indent=2,allow_nan=False))
if '--update-page' in sys.argv:
    page=HERE.parent/'index.html'
    data=json.dumps(D,separators=(',',':'),allow_nan=False)
    page.write_text(re.sub(r'(<script id="data" type="application/json">).*?(</script>)',
        lambda m:m.group(1)+data+m.group(2),page.read_text(),flags=re.S))
    (HERE/'random-start-results.json').write_text(json.dumps(R,indent=2,allow_nan=False))
print('Exported random-start replay',flush=True)
