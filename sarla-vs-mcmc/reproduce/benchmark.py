"""Original optimized-start experiment: JAX Gauss-Newton, SARLA2, and MCMC.

The current page replaces these MCMC runs using random_start_mcmc.py.
Run this script first to retain the original SARLA and reference results.

The upstream sampler is used without changes. Run with JAX, scipy, numpy,
emcee installed. All coordinates use the webpage's prior and likelihood.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time, hashlib, platform, argparse
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.special import logsumexp, rel_entr
import jax
import jax.numpy as jnp
import emcee
jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream/code'))
import sarla2 as S2

TIMES = np.array([2.5, 6., 12., 24.])
DATA = np.array([84.21, 73.09, 61.08, 47.23])
COMMIT = '842e93785f2fc9220869e371a9d937a202922eee'
NBIN = 18

def physical(u):
    u = np.asarray(u)
    return np.stack([10**(1+1.3*u[...,0]), 10**(1+1.3*u[...,1]), .5+.48*u[...,2]], axis=-1)

def pred_np(u, times=TIMES):
    p = physical(u)
    return 100*(p[...,2,None]*np.exp(-np.asarray(times)/p[...,0,None]) +
                (1-p[...,2,None])*np.exp(-np.asarray(times)/p[...,1,None]))

def make_target(n):
    tt, yy = jnp.array(TIMES[:n]), jnp.array(DATA[:n])
    def residual(u):
        # Clipping is only a numerical guard outside the prior support.
        v = jnp.clip(u, -3., 3.)
        tau = 10**(1+1.3*v[:2]); f = .5+.48*v[2]
        return (100*(f*jnp.exp(-tt/tau[0])+(1-f)*jnp.exp(-tt/tau[1]))-yy)/2.
    def objective(u):
        r = residual(u)
        return .5*jnp.dot(r,r)
    def logpi(u):
        return jnp.where(jnp.all(jnp.abs(u)<=1), -objective(u), -jnp.inf)
    rj, jj = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    gb, hb = jax.jit(jax.grad(objective)), jax.jit(jax.hessian(objective))
    lb = jax.jit(jax.vmap(logpi))
    count = dict(logp=0, grad=0, hess=0, residual=0, jacobian=0)
    def batch(U):
        U = np.atleast_2d(U); count['logp'] += len(U)
        return np.asarray(lb(jnp.asarray(U)))
    def grad(u):
        count['grad'] += 1
        return np.asarray(gb(jnp.asarray(u)))
    def hess(u):
        count['hess'] += 1
        return np.asarray(hb(jnp.asarray(u)))
    def res(u):
        count['residual'] += 1
        return np.asarray(rj(jnp.asarray(u)))
    def jac(u):
        count['jacobian'] += 1
        return np.asarray(jj(jnp.asarray(u)))
    t0 = time.perf_counter()
    u = jnp.zeros(3)
    for fn in [rj,jj,gb,hb]: fn(u).block_until_ready()
    # Common batch shapes used by the algorithms are compiled before timing.
    for k in [1,5,16,32,64,4096]: lb(jnp.zeros((k,3))).block_until_ready()
    compile_s = time.perf_counter()-t0
    return dict(logpost_batch=batch,grad=grad,hess=hess,scale=np.ones(3)), res, jac, count, compile_s

def fit_modes(res, jac, nstart=16):
    rng = np.random.default_rng(1729)
    starts = rng.uniform(-.9,.9,(nstart,3))
    sols=[]; t0=time.perf_counter()
    for s in starts:
        fit=least_squares(res,s,jac=jac,bounds=(-np.ones(3),np.ones(3)),
                          ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=400)
        sols.append((float(fit.cost),fit.x,fit.nfev,fit.status))
    sols.sort(key=lambda v:v[0])
    return sols, time.perf_counter()-t0

def reference(n, N):
    assert N%NBIN==0
    x=-1+(np.arange(N)+.5)*2/N
    # Build in slabs to bound memory use.
    Y,Z=np.meshgrid(x,x,indexing='ij'); yz=np.stack([Y.ravel(),Z.ravel()],1)
    w=np.empty((N,N,N)); predhist=np.zeros(400)
    for i,a in enumerate(x):
        u=np.column_stack([np.full(N*N,a),yz])
        r=(pred_np(u,TIMES[:n])-DATA[:n])/2
        w[i]=np.exp(-.5*np.sum(r*r,axis=1)).reshape(N,N)
    norm=w.sum(); p=w/norm
    marg=[p.sum(axis=tuple(j for j in range(3) if j!=i)) for i in range(3)]
    mean=np.array([m@x for m in marg]); sd=np.sqrt([m@((x-mean[i])**2) for i,m in enumerate(marg)])
    frac=N//NBIN
    coarse=p.reshape(NBIN,frac,NBIN,frac,NBIN,frac).sum(axis=(1,3,5))
    pm=np.zeros(3); ps=np.zeros(3)
    for i,a in enumerate(x):
        u=np.column_stack([np.full(N*N,a),yz]); phy=physical(u); ww=p[i].ravel()
        pm+=ww@phy; ps+=ww@(phy*phy)
        predhist+=np.histogram(pred_np(u,[60]).ravel(),bins=400,range=(0,100),weights=ww)[0]
    return dict(N=N,x=x,p=p,coarse=coarse,mean=mean,sd=sd,physical_mean=pm,
                physical_sd=np.sqrt(ps-pm*pm),marginal=marg,predictive=predhist,
                logZ=float(np.log(norm)+(3*np.log(2/N))))

def js_tv(p,q):
    p=np.ravel(p).astype(float); q=np.ravel(q).astype(float)
    # Subnormal masses can underflow when averaged. Their total is negligible.
    p=np.where(p<1e-300,0,p);q=np.where(q<1e-300,0,q)
    p=p/p.sum();q=q/q.sum();m=(p+q)/2
    return dict(js_bits=float((rel_entr(p,m).sum()+rel_entr(q,m).sum())/(2*np.log(2))),
                tv=float(.5*np.abs(p-q).sum()))

def hist_samples(U):
    return np.histogramdd(U,bins=[np.linspace(-1,1,NBIN+1)]*3)[0]

def sample_metrics(U,ref):
    U=np.asarray(U); flat=U.reshape(-1,3); p=physical(flat); pred=pred_np(flat,[60]).ravel()
    out=js_tv(ref['coarse'],hist_samples(flat))
    out.update(draws=len(flat),mode1_mass=float(np.mean(flat[:,0]<flat[:,1])),
               max_mean_error_sd=float(np.max(np.abs(flat.mean(0)-ref['mean'])/ref['sd'])),
               max_sd_relative_error=float(np.max(np.abs(flat.std(0)/ref['sd']-1))),
               physical_mean=p.mean(0).tolist(),physical_q=np.quantile(p,[.025,.5,.975],axis=0).tolist(),
               predictive_mean=float(pred.mean()),predictive_q=np.quantile(pred,[.025,.5,.975]).tolist())
    if U.ndim==3:
        label=U[:,:,0]<U[:,:,1]
        out['fraction_chains_switching_mode']=float(np.mean(np.any(label[1:]!=label[:-1],axis=0)))
        out['mode_transitions']=int(np.sum(label[1:]!=label[:-1]))
    return out

def density_grid(logq,N=144):
    x=-1+(np.arange(N)+.5)*2/N
    Y,Z=np.meshgrid(x,x,indexing='ij'); yz=np.stack([Y.ravel(),Z.ravel()],1)
    lq=np.empty((N,N,N))
    for i,a in enumerate(x):lq[i]=logq(np.column_stack([np.full(N*N,a),yz])).reshape(N,N)
    logZ=logsumexp(lq)+3*np.log(2/N)
    q=np.exp(lq-logsumexp(lq));frac=N//NBIN
    return q.reshape(NBIN,frac,NBIN,frac,NBIN,frac).sum(axis=(1,3,5)),float(logZ),q.sum(axis=2)

def gaussian_logq(centres,covariances):
    inv=np.linalg.inv(covariances); logdet=np.linalg.slogdet(covariances)[1]
    def logq(U):
        terms=[]
        for c,ic,ld in zip(centres,inv,logdet):
            d=U-c;terms.append(-.5*(np.einsum('ij,jk,ik->i',d,ic,d)+ld+3*np.log(2*np.pi)))
        return logsumexp(np.stack(terms),axis=0)-np.log(len(terms))
    return logq

def run_rwm(target,centres,nsteps,seed):
    rng=np.random.default_rng(seed);nc=64
    X=centres[np.arange(nc)%len(centres)].copy()
    lp=target['logpost_batch'](X).copy();out=np.empty((nsteps,nc,3));accepted=0
    # Isotropic random walk: scale adjusted during burn-in only.
    burn=nsteps//4; logstep=np.log(.05);block=0
    t0=time.perf_counter()
    for t in range(nsteps):
        Y=X+np.exp(logstep)*rng.normal(size=X.shape);lpy=target['logpost_batch'](Y)
        take=np.log(rng.random(nc))<lpy-lp
        X[take]=Y[take];lp[take]=lpy[take];out[t]=X
        accepted+=take.sum();block+=take.sum()
        if t<burn and (t+1)%50==0:
            logstep+=.5*(block/(50*nc)-.3);block=0
    return out[burn:],dict(wall_s=time.perf_counter()-t0,accept=float(accepted/(nsteps*nc)),step=float(np.exp(logstep)),burn=burn)

def run_emcee(target,centres,nsteps,seed):
    rng=np.random.default_rng(seed);nc=64
    X=centres[np.arange(nc)%len(centres)]+rng.normal(0,.02,(nc,3));X=np.clip(X,-.999999,.999999)
    sampler=emcee.EnsembleSampler(nc,3,target['logpost_batch'],vectorize=True,
        moves=[(emcee.moves.DEMove(),.8),(emcee.moves.DESnookerMove(),.2)])
    sampler.random_state=np.random.RandomState(seed).get_state()
    t0=time.perf_counter();sampler.run_mcmc(X,nsteps,progress=False,skip_initial_state_check=True)
    elapsed=time.perf_counter()-t0;burn=nsteps//4
    return sampler.get_chain(discard=burn),dict(wall_s=elapsed,accept=float(sampler.acceptance_fraction.mean()),burn=burn)

def serial_ref(ref):
    return {k:np.asarray(ref[k]).tolist() for k in ['N','mean','sd','physical_mean','physical_sd','logZ']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    seeds=[3] if args.pilot else [3,17,41]
    outdir=ROOT/('pilot' if args.pilot else 'results');outdir.mkdir(exist_ok=True)
    results=dict(source_commit=COMMIT,source_sha256=hashlib.sha256((ROOT/'upstream/code/sarla2.py').read_bytes()).hexdigest(),
        versions=dict(jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__,emcee=emcee.__version__),
        platform=platform.platform(),devices=[str(d) for d in jax.devices()],
        protocol=dict(sigma=2.,times=TIMES.tolist(),data=DATA.tolist(),coarse_bins=NBIN,seeds=seeds,
                      common_optimization_starts=16,eval_budget=131072),cases={})
    for n in [4,2]:
        print('\nCASE',n,'harvests',flush=True)
        t0=time.perf_counter();ref=reference(n,144);ref2=reference(n,216) if not args.pilot else ref
        referr=js_tv(ref['coarse'],ref2['coarse']);print('Reference',time.perf_counter()-t0,'s',referr,flush=True)
        target,res,jac,count,compile_s=make_target(n)
        sols,fit_s=fit_modes(res,jac);best=sols[0][1]
        centres=np.array([s[1] for s in sols])
        # Preserve independently optimized seeds; no oracle reflection or true-posterior starts.
        J=jac(best);H=J.T@J;lam=np.linalg.eigvalsh(H)
        case=dict(reference=serial_ref(ref2),reference_convergence=referr,compile_s=compile_s,fit_s=fit_s,
                  best_sse=2*sols[0][0]*4,best_physical=physical(best).tolist(),gn_eigenvalues=lam.tolist(),
                  gn_rank=int(np.linalg.matrix_rank(H,tol=lam.max()*1e-9)),gn_solutions=[dict(cost=s[0],u=s[1].tolist(),status=int(s[3])) for s in sols],
                  runs=[],gaussian={})
        np.savez_compressed(outdir/f'reference_{n}.npz',joint=ref2['coarse'],marginal=ref2['p'].sum(axis=2),x=ref2['x'],
                            predictive=ref2['predictive'])
        print('Gauss-Newton',case['best_physical'],'SSE',case['best_sse'],'rank',case['gn_rank'],'fit s',fit_s,flush=True)
        if case['gn_rank']==3:
            other=next((s[1] for s in sols if (s[1][0]<s[1][1])!=(best[0]<best[1])),None)
            for name,cs in [('single',best[None]),('two_mode',np.array([best,other]) if other is not None else best[None])]:
                cvs=np.array([np.linalg.inv(jac(c).T@jac(c)) for c in cs]);lq=gaussian_logq(cs,cvs)
                q,lz,marg=density_grid(lq)
                case['gaussian'][name]=dict(**js_tv(ref2['coarse'],q),inside_prior_mass=float(np.exp(lz)),
                                            centres=cs.tolist(),covariances=cvs.tolist())
                np.savez_compressed(outdir/f'gaussian_{n}_{name}.npz',joint=q,marginal=marg)
                print('Gaussian',name,case['gaussian'][name]['tv'],flush=True)
        for seed in seeds:
            for label,thresh,tag in [('SARLA2 default + IMH',5.,'sarla'),('SARLA2 tighter audit + IMH',2.,'sarla_tight')]:
                count0=count.copy();cfg=S2.SurgeryConfig(rounds=6,n_audit=4096,flag_thresh=thresh)
                t0=time.perf_counter();atlas=S2.sarla2(target,centres,cfg,seed=seed,verbose=True)
                atlas_s=time.perf_counter()-t0;atlas_count={k:count[k]-count0[k] for k in count}
                remaining=max(4096,131072-atlas_count['logp']);steps=remaining//64
                t0=time.perf_counter();prod=S2.production_imh(atlas,target,n_steps=steps,n_chains=64,seed=seed+100)
                prod_s=time.perf_counter()-t0;U=prod['draws_z'][steps//4:]
                metrics=sample_metrics(U,ref2)
                q,lz,marg=density_grid(atlas.logq,N=72 if args.pilot else 144)
                uncorrected=js_tv(ref2['coarse'],q)
                rec=dict(method=label,seed=seed,**metrics,wall_s=atlas_s+prod_s+fit_s,atlas_s=atlas_s,
                         production_s=prod_s,accept=prod['accept'],charts=len(atlas.charts),atlas_metrics=uncorrected,
                         proposal_inside_mass=float(np.exp(lz)),atlas_evals=atlas_count,counts={k:count[k]-count0[k] for k in count},
                         cfg=asdict(cfg),history=atlas.history,ops=dict(Counter(op[1] for op in atlas.ops_log)))
                case['runs'].append(rec)
                np.savez_compressed(outdir/f'{tag}_{n}_{seed}.npz',samples=U.astype(np.float32),atlas_joint=q,atlas_marginal=marg,
                                    chart_centres=np.array([c.center for c in atlas.charts]))
                print(label,seed,'TV',metrics['tv'],'mode',metrics['mode1_mass'],'time',rec['wall_s'],'atlas TV',uncorrected['tv'],flush=True)
            for name,runner in [('Local random-walk MH',run_rwm),('DE ensemble MCMC',run_emcee)]:
                count0=count.copy();U,meta=runner(target,centres,2048,seed+200)
                metrics=sample_metrics(U,ref2)
                rec=dict(method=name,seed=seed,**metrics,**meta,counts={k:count[k]-count0[k] for k in count})
                rec['wall_s']+=fit_s;case['runs'].append(rec)
                np.savez_compressed(outdir/f'{"rwm" if name.startswith("Local") else "emcee"}_{n}_{seed}.npz',samples=U.astype(np.float32))
                print(name,seed,'TV',metrics['tv'],'mode',metrics['mode1_mass'],'time',rec['wall_s'],flush=True)
            # Independent posterior draws set the finite-histogram error floor.
            rng=np.random.default_rng(seed+900);flat=ref2['p'].ravel();idx=rng.choice(len(flat),size=len(U.reshape(-1,3)),p=flat)
            inds=np.array(np.unravel_index(idx,ref2['p'].shape)).T;iid=ref2['x'][inds]
            case.setdefault('iid_floor',[]).append(sample_metrics(iid,ref2))
            results['cases'][str(n)]=case
            def check_finite(obj,path='results'):
                if isinstance(obj,dict):
                    for k,v in obj.items():check_finite(v,path+'.'+str(k))
                elif isinstance(obj,list):
                    for k,v in enumerate(obj):check_finite(v,path+'.'+str(k))
                elif isinstance(obj,(float,np.floating)) and not np.isfinite(obj):
                    raise ValueError('Nonfinite result: '+path+' = '+str(obj))
            check_finite(results)
            (outdir/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        del ref,ref2
    print('Saved',outdir/'results.json',flush=True)

if __name__=='__main__':main()
