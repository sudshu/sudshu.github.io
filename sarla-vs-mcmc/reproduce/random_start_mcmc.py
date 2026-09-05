"""Rerun direct MCMC without JAX or optimized initial states.

Usage: python random_start_mcmc.py /path/to/completed/original/benchmark
Writes random-start samples beside the original samples without changing them.
Only NumPy evaluates the target. No optimizer or derivative library is imported.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, csv, platform
from pathlib import Path
import numpy as np
import emcee
from scipy.special import rel_entr

TIMES = np.array([2.5, 6., 12., 24.])
DATA = np.array([84.21, 73.09, 61.08, 47.23])
LABELS = {'rwm': 'Local random-walk MH', 'emcee': 'DE ensemble MCMC'}

def prediction(u, times=TIMES):
    u = np.asarray(u)
    tau = 10 ** (1 + 1.3 * u[..., :2])
    f = .5 + .48 * u[..., 2, None]
    return 100 * (f * np.exp(-times / tau[..., 0, None]) +
                  (1 - f) * np.exp(-times / tau[..., 1, None]))

def make_target(n):
    counts = dict(logp=0, grad=0, hess=0, residual=0, jacobian=0)
    def logpost(u):
        u = np.atleast_2d(u)
        counts['logp'] += len(u)
        inside = np.all(np.abs(u) <= 1, axis=1)
        lp = np.full(len(u), -np.inf)
        residual = (prediction(u[inside], TIMES[:n]) - DATA[:n]) / 2
        lp[inside] = -.5 * np.sum(residual * residual, axis=1)
        return lp
    return logpost, counts

def run(n, tag, seed, nsteps=2048):
    t0 = time.perf_counter()
    # Same independent prior draws for both MCMC variants at a given seed.
    rng = np.random.default_rng(seed + 200)
    initial = rng.uniform(-1, 1, (64, 3))
    logpost, counts = make_target(n)
    burn = nsteps // 4
    if tag == 'rwm':
        X = initial.copy(); lp = logpost(X)
        chain = np.empty((nsteps, 64, 3)); accepted = 0; block = 0
        logstep = np.log(.05)
        for t in range(nsteps):
            Y = X + np.exp(logstep) * rng.normal(size=X.shape)
            lpy = logpost(Y)
            take = np.log(rng.random(64)) < lpy - lp
            X[take] = Y[take]; lp[take] = lpy[take]; chain[t] = X
            accepted += take.sum(); block += take.sum()
            if t < burn and (t + 1) % 50 == 0:
                logstep += .5 * (block / (50 * 64) - .3); block = 0
        meta = dict(accept=float(accepted / (nsteps * 64)), step=float(np.exp(logstep)))
    else:
        sampler = emcee.EnsembleSampler(64, 3, logpost, vectorize=True,
            moves=[(emcee.moves.DEMove(), .8), (emcee.moves.DESnookerMove(), .2)])
        sampler.random_state = np.random.RandomState(seed + 200).get_state()
        sampler.run_mcmc(initial, nsteps, progress=False)
        chain = sampler.get_chain()
        meta = dict(accept=float(sampler.acceptance_fraction.mean()))
    elapsed = time.perf_counter() - t0
    return chain, initial, dict(**meta, wall_s=elapsed, total_wall_s=elapsed,
        setup_s=0., fit_s=0., counts=counts, burn=burn, seed=seed,
        initial_source='64 independent uniform prior draws',
        target_backend='NumPy', optimizer=None, derivative_calls=0)

def metrics(U, ref, info):
    flat = U.reshape(-1, 3)
    p = ref.ravel().astype(float); p[p < 1e-300] = 0; p /= p.sum()
    h = np.histogramdd(flat, bins=[np.linspace(-1, 1, 19)] * 3)[0]
    q = h.ravel() / h.sum(); m = (p + q) / 2
    label = U[:, :, 0] < U[:, :, 1]
    return dict(draws=len(flat), tv=float(.5 * np.abs(p - q).sum()),
        js_bits=float((rel_entr(p, m).sum() + rel_entr(q, m).sum()) / (2 * np.log(2))),
        mode1_mass=float(label.mean()),
        fraction_chains_switching_mode=float(np.any(label[1:] != label[:-1], axis=0).mean()),
        mode_transitions=int(np.sum(label[1:] != label[:-1])),
        max_mean_error_sd=float(np.max(np.abs(flat.mean(0) - info['mean']) / info['sd'])),
        max_sd_relative_error=float(np.max(np.abs(flat.std(0) / info['sd'] - 1))))

def main():
    root = Path(sys.argv[1]); out = root / 'random-start'; out.mkdir(exist_ok=True)
    old = json.loads((root / 'results/results.json').read_text())
    result = dict(protocol=dict(seeds=[3, 17, 41], observations=[4, 2], sigma=2,
        parameterization='u in [-1,1]^3; log10(tau)=1+1.3*u; f=.5+.48*u3',
        mcmc='64 random prior starts; NumPy target; no optimization or derivatives',
        sarla='Retained original runs: 16 random interior-prior seeds, JAX Gauss-Newton, Hessian atlas, MH correction',
        mcmc_steps=2048, mcmc_warmup=512, target_evaluations=131136,
        timing='MCMC rerun with NumPy. SARLA retained original timing; fit included, original JAX setup added for total. Timings are descriptive, not a controlled speedup test.'),
        platform=platform.platform(), versions=dict(numpy=np.__version__, emcee=emcee.__version__), cases={})
    rows=[]
    for n in [4, 2]:
        c=old['cases'][str(n)]; ref=np.load(root / f'results/reference_{n}.npz')['joint']
        kept=[]
        for v in c['runs']:
            if v['method'].startswith('SARLA'):
                v=dict(v, initial_source='16 random seeds optimized with JAX Gauss-Newton',
                    target_backend='JAX', fit_s=c['fit_s'], setup_s=c['compile_s'],
                    total_wall_s=v['wall_s']+c['compile_s'])
                kept.append(v)
        for seed in [3, 17, 41]:
            for tag, label in LABELS.items():
                chain, initial, meta=run(n, tag, seed)
                U=chain[meta['burn']:]
                rec=dict(method=label, **meta, **metrics(U, ref, c['reference']), initial=initial.tolist())
                kept.append(rec)
                np.savez_compressed(out / f'{tag}_{n}_{seed}.npz', samples=U.astype(np.float32),
                    chain=chain.astype(np.float32), initial=initial)
                print(n, label, seed, 'TV', round(rec['tv']*100, 2), 'time', round(rec['wall_s'], 3), flush=True)
        stats={}
        for label in dict.fromkeys(v['method'] for v in kept):
            runs=[v for v in kept if v['method']==label]
            stats[label]={k:dict(median=float(np.median([v[k] for v in runs])),
                min=float(min(v[k] for v in runs)), max=float(max(v[k] for v in runs)))
                for k in ['tv','js_bits','wall_s','total_wall_s','mode1_mass','fraction_chains_switching_mode']}
        result['cases'][str(n)]=dict(runs=kept,stats=stats,fit_s=c['fit_s'],compile_s=c['compile_s'],
            reference_convergence=c['reference_convergence'],gn_solutions=c['gn_solutions'])
        for v in kept:
            rows.append(dict(observations=n,method=v['method'],seed=v['seed'],
                initialization=v['initial_source'],target_backend=v['target_backend'],tv_percent=100*v['tv'],
                js_bits=v['js_bits'],cpu_with_setup_seconds=v['total_wall_s'],cpu_after_setup_seconds=v['wall_s'],
                jax_setup_seconds=v['setup_s'],fit_seconds=v['fit_s'],target_evaluations=v['counts']['logp'],
                gradient_calls=v['counts']['grad'],hessian_calls=v['counts']['hess'],
                mode1_mass=v['mode1_mass'],fraction_chains_switching=v['fraction_chains_switching_mode']))
    assert 'jax' not in sys.modules, 'MCMC driver must not import JAX'
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'benchmark-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved',out,flush=True)

if __name__=='__main__': main()
