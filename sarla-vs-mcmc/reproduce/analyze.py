"""Create diagnostic summaries and figures from the saved benchmark draws."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import benchmark as B

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
R=json.loads((ROOT/'results/results.json').read_text())
METHODS=['SARLA2 default + IMH','SARLA2 tighter audit + IMH','Local random-walk MH','DE ensemble MCMC']
TAGS=['sarla','sarla_tight','rwm','emcee']
LABELS=['SARLA · default audit','SARLA · tighter audit','Local random-walk MH','DE ensemble MCMC']
COLORS=['#ba6a35','#187f72','#737b88','#496eaf']
summary={}
rows=[]
for n in [4,2]:
 c=R['cases'][str(n)];rr=np.load(ROOT/f'results/reference_{n}.npz');ref=B.reference(n,216)
 stats={}
 for method,tag in zip(METHODS,TAGS):
  runs=[x for x in c['runs'] if x['method']==method]
  stats[method]={k:dict(median=float(np.median([v[k] for v in runs])),min=float(min(v[k] for v in runs)),max=float(max(v[k] for v in runs)))
                 for k in ['tv','js_bits','wall_s','mode1_mass','max_mean_error_sd','max_sd_relative_error','fraction_chains_switching_mode']}
  stats[method]['grid_sensitivity']={}
  for bins in [12,24]:
   f=216//bins;p=ref['p'].reshape(bins,f,bins,f,bins,f).sum(axis=(1,3,5));vals=[]
   for seed in R['protocol']['seeds']:
    a=np.load(ROOT/f'results/{tag}_{n}_{seed}.npz')['samples'].reshape(-1,3)
    h=np.histogramdd(a,bins=[np.linspace(-1,1,bins+1)]*3)[0];vals.append(B.js_tv(p,h)['tv'])
   stats[method]['grid_sensitivity'][str(bins)]=vals
  for run in runs:
   rows.append(dict(observations=n,method=method,seed=run['seed'],tv_percent=100*run['tv'],js_bits=run['js_bits'],
                    cpu_seconds=run['wall_s'],model_evaluations=run['counts']['logp'],gradient_calls=run['counts']['grad'],hessian_calls=run['counts']['hess'],
                    probability_tau1_less_tau2=run['mode1_mass'],fraction_chains_switching=run['fraction_chains_switching_mode']))
 summary[str(n)]=dict(stats=stats,fit_s=c['fit_s'],compile_s=c['compile_s'],gaussian=c['gaussian'],
                     iid_floor=[v['tv'] for v in c['iid_floor']],reference_convergence=c['reference_convergence'])
 print('\nOBSERVATIONS',n,flush=True)
 for name,s in stats.items():print(name,{k:round(s[k]['median'],5) for k in ['tv','wall_s','mode1_mass','max_mean_error_sd']},'bin sensitivity',s['grid_sensitivity'],flush=True)
 # Density panels use one documented seed; numeric summaries use all seeds.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 p0=rr['marginal'];N=len(p0);area=(2.6/N)**2;peak=(p0/area).max()
 def prob_density(a):return a/(2.6/len(a))**2/peak
 panels=[('Numerical reference',prob_density(p0))]
 if n==4:
  for tag,title in [('single','One Gauss–Newton Gaussian'),('two_mode','Gaussian at each mode')]:
   panels.append((title,prob_density(np.load(ROOT/f'results/gaussian_{n}_{tag}.npz')['marginal'])))
 else:
  panels.extend([('One Gauss–Newton Gaussian',None),('Gaussian covariance',None)])
 for tag,title in [('sarla','SARLA · default + MH'),('sarla_tight','SARLA · tighter audit + MH'),('emcee','DE ensemble MCMC')]:
  u=np.load(ROOT/f'results/{tag}_{n}_3.npz')['samples'].reshape(-1,3)
  h=np.histogram2d(1+1.3*u[:,0],1+1.3*u[:,1],bins=72,range=[[-.3,2.3],[-.3,2.3]])[0]
  h=gaussian_filter(h,.65);h/=h.sum();panels.append((title,prob_density(h)))
 sortedp=np.sort(p0.ravel())[::-1];cum=np.cumsum(sortedp)
 thresholds=sorted([sortedp[np.searchsorted(cum,v)]/area/peak for v in [.9,.5]])
 fig,axs=plt.subplots(2,3,figsize=(11.4,7.8),layout='constrained')
 for ax,(title,a) in zip(axs.flat,panels):
  ax.set_title(title,loc='left',fontweight='bold',fontsize=11)
  if a is None:
   ax.set_axis_off();ax.text(.04,.65,'Rank 2 / 3',transform=ax.transAxes,fontsize=21,color='#8b4930',weight='bold')
   ax.text(.04,.40,'A perfect fit still leaves\na locally unconstrained direction.\nThe Gauss–Newton matrix\nhas no inverse.',transform=ax.transAxes,fontsize=12,linespacing=1.5)
   continue
  img=ax.imshow(np.ma.masked_less(a.T,.003),origin='lower',extent=[-.3,2.3,-.3,2.3],norm=LogNorm(.003,4),cmap='viridis',aspect='equal')
  coords=-.3+(np.arange(N)+.5)*2.6/N
  ax.contour(coords,coords,prob_density(p0).T,levels=thresholds,colors='white',linewidths=.7,alpha=.85)
  ax.set_xticks([0,1,2],['1','10','100']);ax.set_yticks([0,1,2],['1','10','100'])
  ax.set_xlabel('τ₁ · months (log scale)');ax.set_ylabel('τ₂ · months (log scale)')
 fig.suptitle(f'Posterior shape with {n} observations',fontsize=18,fontweight='bold',x=.02,ha='left')
 cb=fig.colorbar(img,ax=axs.ravel().tolist(),shrink=.75,pad=.02,ticks=[.01,.1,1])
 cb.set_label('Marginal density / reference peak');cb.ax.set_yticklabels(['0.01','0.1','1'])
 fig.savefig(OUT/f'posterior-{n}-observations.png',dpi=165,facecolor='white');plt.close(fig)
 del ref

fig,axs=plt.subplots(1,2,figsize=(11.7,5.2),layout='constrained')
for ax,n in zip(axs,[4,2]):
 for i,(name,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
  s=summary[str(n)]['stats'][name]['tv'];med=100*s['median']
  ax.barh(i,med,color=color,height=.58)
  ax.errorbar(med,i,xerr=[[100*(s['median']-s['min'])],[100*(s['max']-s['median'])]],fmt='none',ecolor='#182632',capsize=4,lw=1.3)
  ax.text(34.2,i,f'{med:.1f}%',va='center',ha='right',fontsize=11)
 floor=100*np.median(summary[str(n)]['iid_floor']);ax.axvline(floor,color='#888',ls=':',lw=1)
 ax.set_yticks(range(4),LABELS);ax.invert_yaxis();ax.set_xlim(0,35);ax.set_xlabel('Joint posterior mismatch · TV (%) ↓')
 ax.set_title(f'{n} observations',loc='left',fontsize=14,fontweight='bold');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Tighter audits help SARLA; benefits depend on the observations',fontsize=15,fontweight='bold')
fig.supxlabel('Median of 3 seeds · Whiskers: observed range · Dotted line: independent-draw floor\n≈131,000 target evaluations per method; both SARLA rows include Metropolis–Hastings correction.',fontsize=9)
fig.savefig(OUT/'posterior-method-comparison.png',dpi=170,facecolor='white');plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
with (OUT/'benchmark-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Created figures and summaries in',OUT,flush=True)
