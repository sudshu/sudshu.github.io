"""Embed NUTS results into the existing interactive page without changing other runs."""
from pathlib import Path
import sys,json,re,csv
HERE=Path(__file__).resolve().parent;root=Path(sys.argv[1])
R=json.loads((HERE/'nuts-results.json').read_text());replay=json.loads((root/'nuts/replay.json').read_text())
p=HERE.parent/'index.html';s=p.read_text()
D=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',s,re.S).group(1))
for n,c in D['cases'].items():
 c['runs']['nuts']=replay[n];c['stats']['JAX NUTS']=R['cases'][n]['stats'];c['nutsRecords']=R['cases'][n]['runs']
s=re.sub(r'(<script id="data" type="application/json">).*?(</script>)',lambda m:m.group(1)+json.dumps(D,separators=(',',':'),allow_nan=False)+m.group(2),s,flags=re.S)
p.write_text(s)
rows=[]
for n,c in R['cases'].items():
 for r in c['runs']:rows.append({k:r[k] for k in ['seed','tv','total_wall_s','warmup_compile_s','sampling_compile_s','divergences','max_rhat','min_ess','leapfrog_steps','warmup_leapfrog_steps','fraction_chains_switching_mode']}|{'observations':n})
with (HERE/'nuts-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('Embedded JAX NUTS replay and exported results table')
