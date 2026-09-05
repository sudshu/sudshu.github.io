# Reproduce the SARLA versus MCMC benchmark

The page shows the measured Brandt2010 carbon-turnover example with four or the first two observations. The fixed Bayesian target and full comparison protocol are documented in ../report.html.

1. Create a Python 3.12 environment and install `requirements.txt`.
2. Save the unmodified [pinned SARLA2 engine](https://raw.githubusercontent.com/sudshu/sarla-autoresearch/842e93785f2fc9220869e371a9d937a202922eee/code/sarla2.py) as `upstream/code/sarla2.py` beside `benchmark.py`.
3. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.
4. Run `python benchmark.py`. This writes reference arrays, raw chain states, and `results/results.json`.
5. Run `python analyze.py` and `python make_report.py` to generate the figures, summary, standalone report, and complete source/data ZIP.
6. Optionally run `python export_replay.py . replay.json` to rebuild the compact replay data.

`results.json` here records the original experiment. `benchmark-results.csv` is the compact per-run comparison. Seeds are 3, 17, 41. The tighter flag threshold (2 instead of 5) was chosen after the seed-3 pilot and checked with two further seeds. This small benchmark does not establish general speedup or convergence.

The interactive page embeds its replay data and can be downloaded to run offline. It shows seed 3, 32 of 64 chains, 96 times after warm-up. Display thinning is independent of the full-resolution scoring. Chart outlines are proposal components, not credible regions. Playback progress is not elapsed computation time.

`export_replay.py` is the page's replay exporter. It accepts a completed benchmark directory and a JSON output path; it also expects `deliverables/summary.json` from the accompanying benchmark analysis. Its audit observer records chart geometry without editing upstream source or consuming random numbers and validates the replay against original chart centers and audit histories.
