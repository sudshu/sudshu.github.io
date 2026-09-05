"""Build an offline HTML report and a reproducible source/data bundle."""
from pathlib import Path
import json,base64,html,csv,zipfile,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'deliverables'
R=json.loads((ROOT/'results/results.json').read_text())
S=json.loads((OUT/'summary.json').read_text())
PIN=R['source_commit'];CODE=f'https://github.com/sudshu/sarla-autoresearch/blob/{PIN}/code/sarla2.py'
def image(name):
 return 'data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()
def fmt(v,n=1):return f'{v:.{n}f}'
def rows(n):
 c=R['cases'][str(n)];s=S[str(n)];out=[]
 for key,label in [('single','One Gauss–Newton Gaussian'),('two_mode','Gaussian at each mode')]:
  if key in c['gaussian']:
   g=c['gaussian'][key]
   out.append(f'<tr><td>{label}</td><td>{100*g["tv"]:.1f}%</td><td>Deterministic approximation</td><td>{c["fit_s"]:.2f} s*</td><td>Gaussian shape only</td></tr>')
 for name,label in [('SARLA2 default + IMH','SARLA · default audit + MH'),('SARLA2 tighter audit + IMH','SARLA · tighter audit + MH'),('Local random-walk MH','Local random-walk MH'),('DE ensemble MCMC','Differential-evolution MCMC')]:
  v=s['stats'][name];e=v['tv'];t=v['wall_s'];m=v['mode1_mass']
  cls=' class="focus-row"' if 'tighter' in name else ''
  note=f'{100*m["median"]:.1f}% in τ₁ &lt; τ₂'
  if n==4 and name.startswith('Local'):note='No chains crossed between modes'
  out.append(f'<tr{cls}><td>{label}</td><td><strong>{100*e["median"]:.1f}%</strong></td><td>{100*e["min"]:.1f}–{100*e["max"]:.1f}%</td><td>{t["median"]:.2f} s</td><td>{note}</td></tr>')
 return ''.join(out)
def audit_rows(n):
 out=[]
 for name in ['SARLA2 default + IMH','SARLA2 tighter audit + IMH']:
  rr=[v for v in R['cases'][str(n)]['runs'] if v['method']==name]
  out.append('<tr><td>'+('Default audit' if 'default' in name else 'Tighter audit')+'</td><td>'+str([v['charts'] for v in rr])+'</td><td>'+f'{100*np.median([v["atlas_metrics"]["tv"] for v in rr]):.1f}%</td><td>'+f'{100*np.median([v["tv"] for v in rr]):.1f}%</td></tr>')
 return ''.join(out)
def panel(n):
 c=R['cases'][str(n)]
 if n==4:
  take='<strong>The tighter SARLA audit beats the tested ensemble baseline at this budget.</strong> Its mismatch is 3.2–5.9% across seeds, compared with 7.9–9.0% for differential-evolution MCMC. Default SARLA gives 17.9–19.2%. The local random walk looks reasonably accurate in the joint histogram but never crosses between modes.'
  fit='<strong>Gauss–Newton solves the best-fit problem:</strong> τ<sub>fast</sub> = 3.284 months, τ<sub>slow</sub> = 46.903 months, and initial fast fraction = 0.2139. The squared residual sum is 0.4475 percentage-points². Sixteen starts take about 0.07 s after compilation.'
  fail='One Gaussian misses the other mode and the curved tails. Adding a Gaussian at the second mode improves the approximation, but the joint mismatch remains 39.0%. Capturing both peaks alone is insufficient to describe the posterior.'
 else:
  take='<strong>SARLA has no universal advantage in the weaker-data setting.</strong> With two observations, the tighter audit gives 12.2–14.5% mismatch, while the local random walk gives 9.0–9.5%. The ensemble baseline gives 16.9–17.1%. A broad connection between label orderings makes local exploration easier here.'
  fit='<strong>A perfect fit does not identify all three parameters.</strong> The Gauss–Newton matrix has rank 2 out of 3 at the optimum. Many combinations match the two observations; the matrix has no inverse, so a full local Gaussian covariance cannot be obtained without adding another assumption.'
  fail='The bounded prior still defines a proper posterior that numerical integration and sampling can explore. The two Gaussian rows are therefore omitted here: a pseudoinverse would not supply the missing uncertainty along the flat direction.'
 return f'''<section class="case" id="case-{n}" {'hidden' if n==2 else ''}>
 <div class="finding">{take}</div><div class="two-col"><div><h2>How far does Gauss–Newton get?</h2><p>{fit}</p><p>{fail}</p></div><div><h2>How to read the error</h2><p><strong>Joint posterior mismatch</strong> is total variation (TV) on a fixed 18 × 18 × 18 grid in the three prior coordinates. It is the fraction of probability mass that would need to be reassigned among those cells to match the numerical reference.</p><p>Zero means agreement on that grid. This measures the <em>distribution of possible parameters</em>, rather than the residual error of the best fit.</p></div></div>
 <h2>Measured comparison · {n} observations</h2><div class="table-wrap"><table><thead><tr><th>Method</th><th>Mismatch ↓</th><th>Range over 3 seeds</th><th>Median CPU time</th><th>Mode check</th></tr></thead><tbody>{rows(n)}</tbody></table></div>
 <p class="caption">Sampling methods receive about 131,000 target evaluations, including SARLA's atlas audits. All start from the same 16 independently optimized solutions. Timings include this common optimization and exclude the shared JAX compilation ({c['compile_s']:.2f} s for this setting). *The Gaussian timing is optimization only; diagnostic grid evaluation is excluded for every method. This is a small CPU benchmark.</p>
 <figure><img src="{image(f'posterior-{n}-observations.png')}" alt="Numerical posterior and approximations with {n} observations"><figcaption>All panels integrate over the initial fraction f. White contours mark 50% and 90% probability regions of the reference <em>two-parameter marginal</em>. The common color scale is relative to the reference peak; areas below 0.003 of that peak are blank. Sample panels use seed 3 and light smoothing for display; all reported metrics use unsmoothed three-parameter histograms and all three seeds.</figcaption></figure>
 <details><summary>What does the SARLA atlas achieve before MCMC correction?</summary><p>The atlas remains an approximate proposal distribution. The final frozen-proposal Metropolis–Hastings step corrects for differences between that proposal and the target posterior. A better exploration proposal need not minimize this particular approximation-error metric.</p><div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Charts, seeds 3 / 17 / 41</th><th>Atlas mismatch, median</th><th>After MH correction</th></tr></thead><tbody>{audit_rows(n)}</tbody></table></div><p>The atlas and Gaussian approximations are normalized within the same prior box for these shape comparisons. Proposals outside the box are rejected during sampling and their computation is included.</p></details>
 </section>'''

report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gauss–Newton, SARLA and MCMC · A measured posterior benchmark</title><style>
:root{{--ink:#172c36;--muted:#536570;--line:#dce4e6;--accent:#187f72;--paper:#fff;--bg:#f3f6f5}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}}main{{max-width:1210px;margin:auto;padding:36px 30px 60px}}header{{max-width:1040px}}.eyebrow{{font-size:12px;letter-spacing:.11em;color:var(--accent);font-weight:700;text-transform:uppercase}}h1{{font-size:38px;line-height:1.13;letter-spacing:-.035em;font-weight:650;margin:12px 0 16px}}.lead{{font-size:19px;color:var(--muted);margin:0 0 20px;max-width:1000px}}h2{{font-size:21px;line-height:1.25;margin:27px 0 12px}}p{{margin:10px 0 14px}}a{{color:#096e65}}.meta{{font-size:13px;color:var(--muted)}}.tabs{{display:flex;gap:10px;margin:26px 0 18px}}button{{font:inherit;padding:10px 17px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);cursor:pointer}}button[aria-selected=true]{{background:var(--accent);color:white;border-color:var(--accent)}}button:focus-visible,summary:focus-visible{{outline:3px solid #8cc8bb;outline-offset:3px}}.finding{{padding:18px 22px;border-left:4px solid var(--accent);background:#e6f0ec;border-radius:3px;font-size:17px}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:34px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid var(--line);padding:12px 14px;vertical-align:top}}th{{background:#edf3f1;font-weight:650;white-space:nowrap}}.focus-row{{background:#e9f4ef}}.caption,figcaption{{color:var(--muted);font-size:13px;line-height:1.55}}figure{{margin:25px 0;background:white;padding:15px;border:1px solid var(--line);border-radius:8px}}img{{display:block;width:100%;height:auto}}figcaption{{padding:10px 7px 3px}}details{{margin:22px 0;padding:16px 20px;background:white;border:1px solid var(--line);border-radius:7px}}summary{{font-weight:600;cursor:pointer}}.equation{{font:14px/1.8 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;background:white;padding:14px 18px;border-left:3px solid #9cc8bd}}.case[hidden]{{display:none}}.sources{{font-size:14px}}footer{{border-top:1px solid var(--line);margin-top:30px;padding-top:20px;color:var(--muted);font-size:13px}}@media(max-width:700px){{main{{padding:24px 17px}}h1{{font-size:30px}}.lead{{font-size:17px}}.two-col{{grid-template-columns:1fr;gap:0}}td,th{{padding:10px}}figure{{padding:5px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:10px}}.tabs{{display:none}}.case[hidden]{{display:block}}details{{break-inside:avoid}}}}
</style></head><body><main><header><div class="eyebrow">Nonlinear inversion / Measured computational experiment</div><h1>A good fit is easier than a good posterior</h1><p class="lead">JAX-based Gauss–Newton finds the best-fitting parameters quickly. Recovering the full curved, multimodal uncertainty requires more: in this example, SARLA becomes competitive with MCMC when its audit is tightened and its samples receive Metropolis–Hastings correction.</p><p class="meta">Run on 5 September 2026 · Three parameters · Two data settings · Three random seeds · Unmodified <a href="{CODE}">SARLA2 engine</a> at revision {PIN[:10]}</p></header>
<div class="tabs" role="tablist" aria-label="Number of observations"><button id="tab-4" role="tab" aria-controls="case-4" aria-selected="true" data-case="4">4 observations · current page</button><button id="tab-2" role="tab" aria-controls="case-2" aria-selected="false" data-case="2">2 observations · weaker constraints</button></div>
{panel(4)}{panel(2)}
<h2>How the advantage changes with the data</h2><figure><img src="{image('posterior-method-comparison.png')}" alt="Posterior mismatch for four samplers, with four or two observations"><figcaption>Whiskers show the observed minimum and maximum across three seeds, not confidence intervals. The dotted line gives the approximate finite-histogram error from 98,304 independent draws from the numerical posterior: about 1.4% with four observations and 3.3% with two. Monte Carlo draws are correlated, so their effective information is lower.</figcaption></figure>
<h2>The concrete SARLA finding</h2><p>In the four-observation pilot, the default audit found no weights more than 5 log units above its median, then froze after two flag-free rounds. Its importance-sampling ESS fractions were only 0.119 and 0.074. Setting the existing <code>flag_thresh</code> to <strong>2 instead of 5</strong> triggered actual splits, refinement, rank changes and merges, producing 13 charts in that run.</p><p>This is a change to an existing configuration knob. The tighter setting was chosen after inspecting the pilot and checked with two additional seeds; it needs validation on other problems. More charts and the absence of flagged points are both insufficient by themselves to establish posterior accuracy.</p>
<details><summary>Exact problem, comparison protocol, and limitations</summary><h2>Same target as the interactive page</h2><div class="equation">M(t) = 100[f exp(−t/τ₁) + (1−f) exp(−t/τ₂)]<br>t = [2.5, 6, 12, 24] months<br>observed M = [84.21, 73.09, 61.08, 47.23]%<br>independent Gaussian residuals; σ = 2 percentage points<br>uniform prior in (log₁₀τ₁, log₁₀τ₂, f)<br>log₁₀τ ∈ [−0.3, 2.3]; f ∈ [0.02, 0.98]</div><p>The two-observation setting uses the first two entries. Time zero is an initial normalization and is excluded from the likelihood. The uncertainty scale and prior are the page's analysis assumptions.</p><p>The model is differentiated in JAX with float64 arithmetic. SciPy's bounded least-squares solver uses the JAX Jacobian to find 16 local solutions from fixed uniform starts. The Gaussian covariance is (JᵀJ)⁻¹ where it exists. SARLA2 uses the exact JAX Hessian for its chart geometry. No mode reflection or numerical-reference draws are supplied as sampler starting points.</p><p>SARLA2 is called directly with its default <code>SurgeryConfig</code>, six maximum rounds and 4,096 audit proposals per round. The tighter experiment changes only <code>flag_thresh</code>. Its frozen atlas feeds the repository's own <code>production_imh</code> with 64 independent chains. The local baseline uses 64 Gaussian random-walk chains, tuning the step size only during burn-in. The stronger ensemble baseline uses emcee's differential-evolution and snooker moves (80% / 20%) with 64 walkers. The first quarter of each production trajectory is discarded.</p><p>Every sampler has approximately 131,000 target evaluations including initialization; SARLA's atlas evaluations use that same budget. Gradient and Hessian calls are also counted and included in wall time, with raw counts in the CSV. The 16-start optimization is shared and included in each runtime. Shared JAX compilation and post-hoc diagnostic calculations are reported separately. These CPU timings for a tiny model do not establish a GPU speedup or a CARDAMOM runtime improvement.</p><p>The numerical reference uses midpoint integration on a 216³ grid and is checked against 144³. Their 18³-bin TV differences are 0.17 percentage points (four observations) and 0.10 percentage points (two). The main conclusions persist with 12³ and 24³ scoring bins: tighter-audit SARLA has the lowest median mismatch with four observations, and the local random walk has the lowest with two. The relative ordering of the other baselines can change. TV and Jensen–Shannon divergence are saved; empirical KL is avoided because unsampled histogram bins would produce infinite values.</p><p>The modes in this example include exact pool-label symmetry. With four observations, the local random walk has no between-mode transitions in these runs and preserves its initial 43.75% allocation to τ₁ &lt; τ₂, instead of the symmetric target's 50%. Both modes were already present among the optimized starting points; this experiment tests posterior coverage and mixing, not blind discovery of an unknown mode. Three seeds and two data settings do not prove convergence or general superiority.</p><p>The Bayesian target is with respect to log turnover coordinates and f. Each approximate density is conditioned on the same prior box for the Gaussian/atlas shape comparisons. MCMC uses the original hard prior gate. The atlas is a proposal, and finite-length Metropolis–Hastings still has sampling error even though its stationary distribution is the target.</p><p>Software: JAX {R['versions']['jax']}, NumPy {R['versions']['numpy']}, SciPy {R['versions']['scipy']}, emcee {R['versions']['emcee']}; CPU; BLAS/OpenMP thread counts set to one. Seeds: 3, 17, 41. The downloadable reproduction bundle contains the actual source snapshot, scripts, raw samples, machine-readable results and source hashes.</p></details>
<div class="sources"><h2>Sources and context</h2><p><a href="https://sudshu.github.io/carbon-turnover/">Interactive nonlinear inverse-problem explorer</a> · <a href="{CODE}">Exact SARLA2 source used</a> · <a href="https://github.com/sudshu/sarla-autoresearch/blob/{PIN}/ATLAS_SURGERY.md">Atlas surgery documentation</a></p><p><a href="https://github.com/AgustinSarquis/aridec/blob/v1.0.2/data/Brandt2010/timeSeries.csv">Measured series: aridec v1.0.2, Brandt2010 column CBB</a> · <a href="https://link.springer.com/article/10.1007/s10021-010-9353-2">Original field experiment</a> · <a href="https://emcee.readthedocs.io/en/stable/user/sampler/">emcee sampler documentation</a> · <a href="https://docs.jax.dev/en/latest/installation.html">JAX</a></p></div>
<footer>This report contains precomputed experimental results and works offline. Its buttons switch between the two observation settings; they do not rerun the benchmark.</footer></main><script>document.querySelectorAll('[data-case]').forEach(button=>button.addEventListener('click',()=>{{const n=button.dataset.case;document.querySelectorAll('.case').forEach(panel=>panel.hidden=panel.id!=='case-'+n);document.querySelectorAll('[data-case]').forEach(tab=>tab.setAttribute('aria-selected',String(tab.dataset.case===n)));}}));</script></body></html>'''
(OUT/'sarla-gauss-newton-mcmc.html').write_text(report)

readme=f'''# Reproduce the carbon-turnover posterior comparison

Results are a small CPU benchmark, not a general performance claim.

1. Create a Python 3.12 environment and install requirements.txt.
2. Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.
3. Run `python benchmark.py` (the pilot can be run with --pilot).
4. Run `python analyze.py` and then `python make_report.py`.

The SARLA engine is copied without edits from:
{CODE}
SHA-256: {R['source_sha256']}

The target is the measured Brandt2010 CBB example shown at
https://sudshu.github.io/carbon-turnover/ with four or the first two observations,
sigma=2 percentage points and the identical bounded log-turnover/fraction prior.

The tighter flag_thresh=2 setting was selected after the seed-3 pilot; the
default is 5. Seeds 17 and 41 provide additional checks on this same problem.

results/*.npz contains post-burn-in samples and reference marginals.
results/results.json includes settings, counts, metrics and surgery history.
deliverables/benchmark-results.csv has one row per sampling run.

TV percentages are total variation on 18^3 equal prior-coordinate bins.
They are not best-fit residuals or guarantees about exact continuous density.
The HTML report describes the numerical-reference convergence and limitations.
'''
(ROOT/'REPRODUCE.md').write_text(readme)
(ROOT/'requirements.txt').write_text('\n'.join([f'jax[cpu]=={R["versions"]["jax"]}',f'numpy=={R["versions"]["numpy"]}',f'scipy=={R["versions"]["scipy"]}',f'emcee=={R["versions"]["emcee"]}','matplotlib>=3.9'])+'\n')
include=[ROOT/p for p in ['benchmark.py','analyze.py','make_report.py','REPRODUCE.md','requirements.txt','upstream/code/sarla2.py']]
include+=sorted((ROOT/'results').glob('*'))
include+=[OUT/'summary.json',OUT/'benchmark-results.csv']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in include}
(ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2));include.append(ROOT/'SHA256SUMS.json')
with zipfile.ZipFile(OUT/'sarla-carbon-benchmark-source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in include:z.write(p,'sarla-carbon-benchmark/'+str(p.relative_to(ROOT)))
print('Created',OUT/'sarla-gauss-newton-mcmc.html',len(report.encode()),'bytes')
print('Created reproduction bundle', (OUT/'sarla-carbon-benchmark-source.zip').stat().st_size,'bytes')
