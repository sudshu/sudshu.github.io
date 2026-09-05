"""Visualize the saved pre-MCMC atlas; no optimization or MCMC is rerun.

Direct independent mixture draws are used only to render its density.
Run beside results.json and ../index.html after the other replay exporters.
"""
import json,re,base64
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,gammaln
HERE=Path(__file__).resolve().parent
page=HERE.parent/'index.html';html=page.read_text()
D=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',html,re.S).group(1))
R=json.loads((HERE/'results.json').read_text())
for n,C in D['cases'].items():
 for tag,label in [('sarla','SARLA2 default + IMH'),('sarla_tight','SARLA2 tighter audit + IMH')]:
  run=C['runs'][tag];charts=run['charts'][-1]
  saved=next(r for r in R['cases'][n]['runs'] if r['method']==label and r['seed']==3)
  cfg=saved['cfg'];eta=cfg['eta'];df=cfg['t_df']
  centers=np.array([c['center'] for c in charts]);variances=np.array([c['var'] for c in charts]);vectors=np.array([c['vectors'] for c in charts])
  lw=np.array([np.log(v[:c['rank']]).sum()/2 if cfg['weight_rule']=='volume' else 0 for c,v in zip(charts,variances)],dtype=float)
  lw-=logsumexp(lw);scale=max(2.,1.3*np.abs(centers).max())
  def logq(X):
   terms=[]
   for center,var,V,w in zip(centers,variances,vectors,lw):
    y=(X-center)@V/np.sqrt(var)
    terms.append(w-.5*(np.sum(y*y,axis=1)+np.log(var).sum()+3*np.log(2*np.pi)))
   t=gammaln((df+3)/2)-gammaln(df/2)-1.5*np.log(df*np.pi)-3*np.log(scale)-(df+3)/2*np.log1p(np.sum((X/scale)**2,axis=1)/df)
   return np.logaddexp(np.log1p(-eta)+logsumexp(terms,axis=0),np.log(eta)+t)
  rng=np.random.default_rng(914+int(n)+(tag=='sarla_tight'));inside=[];count=0;total=0
  while count<3200:
   k=rng.choice(len(charts),size=40000,p=np.exp(lw));X=np.empty((len(k),3))
   for j in range(len(charts)):
    mask=k==j;X[mask]=centers[j]+(rng.normal(size=(mask.sum(),3))*np.sqrt(variances[j]))@vectors[j].T
   tmask=rng.random(len(X))<eta;nt=tmask.sum()
   X[tmask]=scale*rng.normal(size=(nt,3))*np.sqrt(df/rng.chisquare(df,size=nt))[:,None]
   valid=X[np.all(np.abs(X)<=1,axis=1)];inside.append(valid);count+=len(valid);total+=len(X)
   assert total<4000000,'Unexpectedly low prior mass'
  X=np.concatenate(inside)[:3200]
  # Score colors at the quantized points actually rendered.
  encoded=np.rint((X+1)*32767.5).astype('<u2');X=encoded.astype(float)/32767.5-1
  density=logq(X);peak=max(density.max(),logq(centers).max())
  run['atlasCloud']=base64.b64encode(encoded.tobytes()).decode()
  run['atlasLevels']=np.rint(np.clip(1+(density-peak)/np.log(1000),0,1)*255).astype(int).tolist()
  run['atlasTV']=saved['atlas_metrics']['tv'];run['atlasInsideMass']=saved['proposal_inside_mass']
  run['atlasEta']=eta;run['atlasWeights']=np.exp(lw).tolist()
  assert abs(count/total-run['atlasInsideMass'])<.025
  print(n,tag,'charts',len(charts),'pre-MCMC TV',round(100*run['atlasTV'],2),'inside mass',round(count/total,3),flush=True)
data=json.dumps(D,separators=(',',':'),allow_nan=False)
page.write_text(re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m.group(1)+data+m.group(2),html,flags=re.S))
