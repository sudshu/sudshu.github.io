"""JAX/NumPyro NUTS for the exact bounded three-parameter target.

python nuts_benchmark.py /path/to/original/benchmark
32 random-prior chains, 512 warm-up + 1536 retained steps, three seeds.
No optimized initial points. This is NOT an equal-evaluation-budget test.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,time,json,platform,base64,re
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC,NUTS
from numpyro.diagnostics import summary
from random_start_mcmc import metrics,prediction,make_target
jax.config.update('jax_enable_x64',True)
HERE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]);OUT=ROOT/'nuts';OUT.mkdir(exist_ok=True)
OLD=json.loads((HERE/'results.json').read_text())
WARM=512;STEPS=1536;CHAINS=32
EXTRA=('num_steps','accept_prob','diverging')

def potential(n):
 tt=jnp.array([2.5,6.,12.,24.][:n]);yy=jnp.array([84.21,73.09,61.08,47.23][:n])
 def fn(z):
  u=2*jax.nn.sigmoid(z)-1
  tau=10**(1+1.3*u[:2]);f=.5+.48*u[2]
  residual=(100*(f*jnp.exp(-tt/tau[0])+(1-f)*jnp.exp(-tt/tau[1]))-yy)/2
  logjac=jnp.sum(jnp.log(2.)+jax.nn.log_sigmoid(z)+jax.nn.log_sigmoid(-z))
  return .5*jnp.sum(residual**2)-logjac
 return fn

def pack(x):
 assert np.isfinite(x).all() and np.abs(x).max()<=1
 return base64.b64encode(np.rint((x+1)*32767.5).astype('<u2').tobytes()).decode()

R=dict(versions=dict(jax=jax.__version__,numpyro=numpyro.__version__,numpy=np.__version__),
 platform=platform.platform(),protocol=dict(chains=CHAINS,warmup=WARM,samples_per_chain=STEPS,
 seeds=[3,17,41],dense_mass=True,target_accept_prob=.9,max_tree_depth=10,
 coordinates='u=2*sigmoid(z)-1 with exact log-Jacobian; same uniform bounded prior',
 initialization='First 32 independent uniform prior draws from RNG seed+200; no optimizer',
 timing='Clear JAX caches before each run. Warm-up and production timings include their compilation; device synchronization included. Imports and diagnostics excluded.',
 budget='Not matched to the approximately 131k target evaluations of the earlier methods. num_steps counts leapfrog steps, not all function calls.'),cases={})
replays={}
for n in [4,2]:
 fn=potential(n);rng=np.random.default_rng(712)
 u=rng.uniform(-.99,.99,(40,3));z=np.log((u+1)/(1-u))
 lp,_=make_target(n)
 logjac=np.sum(np.log((1-u*u)/2),axis=1)
 assert np.allclose(np.asarray(jax.vmap(fn)(z)),-lp(u)-logjac,atol=1e-9)
 ref=np.load(ROOT/f'results/reference_{n}.npz')['joint'];runs=[]
 for seed in [3,17,41]:
  jax.clear_caches()
  t0=time.perf_counter()
  initial=np.random.default_rng(seed+200).uniform(-1,1,(CHAINS,3))
  initial_z=jnp.array(np.log((initial+1)/(1-initial)))
  kernel=NUTS(potential_fn=fn,dense_mass=True,target_accept_prob=.9,max_tree_depth=10)
  sampler=MCMC(kernel,num_warmup=WARM,num_samples=STEPS,num_chains=CHAINS,chain_method='vectorized',progress_bar=False)
  sampler.warmup(jax.random.key(seed+700),init_params=initial_z,collect_warmup=True,extra_fields=EXTRA)
  warm=np.asarray(sampler.get_samples(group_by_chain=True));we={k:np.asarray(v) for k,v in sampler.get_extra_fields(group_by_chain=True).items()}
  warm_s=time.perf_counter()-t0;t1=time.perf_counter()
  sampler.run(jax.random.key(seed+1700),extra_fields=EXTRA)
  Z=np.asarray(sampler.get_samples(group_by_chain=True));extra={k:np.asarray(v) for k,v in sampler.get_extra_fields(group_by_chain=True).items()}
  sample_s=time.perf_counter()-t1
  U=np.asarray(2*jax.nn.sigmoid(Z)-1);WU=np.asarray(2*jax.nn.sigmoid(warm)-1)
  samples=U.transpose(1,0,2);diag=summary({'u':U},group_by_chain=True)['u']
  rec=dict(method='JAX NUTS',seed=seed,**metrics(samples,ref,OLD['cases'][str(n)]['reference']),
   total_wall_s=warm_s+sample_s,wall_s=warm_s+sample_s,warmup_compile_s=warm_s,sampling_compile_s=sample_s,
   accept=float(extra['accept_prob'].mean()),divergences=int(extra['diverging'].sum()),
   warmup_divergences=int(we['diverging'].sum()),max_rhat=float(np.max(diag['r_hat'])),
   min_ess=float(np.min(diag['n_eff'])),leapfrog_steps=int(extra['num_steps'].sum()),
   warmup_leapfrog_steps=int(we['num_steps'].sum()),max_depth_hits=int((extra['num_steps']>=1023).sum()),
   initial=initial.tolist())
  runs.append(rec)
  np.savez_compressed(OUT/f'nuts_{n}_{seed}.npz',samples=samples.astype(np.float32),warmup=WU.transpose(1,0,2).astype(np.float32),initial=initial,**extra)
  print(n,seed,'TV',round(100*rec['tv'],2),'seconds',round(rec['total_wall_s'],2),'Rhat',round(rec['max_rhat'],3),'divergences',rec['divergences'],flush=True)
  if seed==3:
   frames=np.rint(np.linspace(0,STEPS-1,96)).astype(int);flat=samples.reshape(-1,3)
   warmstates=np.concatenate([initial[None],WU.transpose(1,0,2)],axis=0)
   wf=np.rint(np.linspace(0,len(warmstates)-1,64)).astype(int)
   replays[str(n)]=dict(label='JAX NUTS',trace=pack(samples[frames]),posterior=pack(flat[np.linspace(0,len(flat)-1,1800).astype(int)]),
    warmup=pack(warmstates[wf]),warmup_frames=64,initial=initial.tolist(),steps=STEPS,burn=WARM,
    tv=rec['tv'],accept=rec['accept'],mode_mass=rec['mode1_mass'],switching=rec['fraction_chains_switching_mode'],history=[],
    predictive=np.quantile(prediction(flat[::10],np.arange(61)),[.05,.5,.95],axis=0).round(3).tolist(),
    max_rhat=rec['max_rhat'],divergences=rec['divergences'],leapfrog_steps=rec['leapfrog_steps'])
 stats={k:dict(median=float(np.median([v[k] for v in runs])),min=float(min(v[k] for v in runs)),max=float(max(v[k] for v in runs))) for k in ['tv','total_wall_s','wall_s','max_rhat','min_ess','divergences','leapfrog_steps','warmup_leapfrog_steps']}
 R['cases'][str(n)]=dict(runs=runs,stats=stats)
 (HERE/'nuts-results.json').write_text(json.dumps(R,indent=2,allow_nan=False))
 (OUT/'replay.json').write_text(json.dumps(replays,separators=(',',':'),allow_nan=False))
print('Saved NUTS results and replay',flush=True)
