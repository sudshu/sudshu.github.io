---
layout: post
title: "I went to Yosemite. Then I made ML and 4D-Var fight over a waterfall."
date: 2026-06-08
tags: [machine-learning, fluid-dynamics, NWP, waterfall]
---

I went to Yosemite last spring. I came back with a 60-second cell-phone clip of
Bridalveil Falls and a question that wouldn't go away: **how well can a small
neural network predict the next frame of a chaotic waterfall, and how does that
compare to the physics-style methods that weather agencies actually use?**

A few weekends of GPU time later, I have an honest answer — and it's more
interesting than the usual *"ML beats physics, more news at 11"* take.

![Sample frame of Bridalveil Falls]({{ "/assets/figures/bbox_overlay_large.png" | relative_url }})
*The clip. We crop to the boxed region (216×88 pixels) to focus on the falling
water and its mist envelope, and discard the static cliff on the right.*

## The setup

The clip is 1,824 frames at 30 fps, 1080×1920 portrait, recorded handheld. I
downsampled to a manageable 192×320, cropped to 216×88 (cliff + falling column
+ splash), and ran a quick phase-correlation alignment to take out the 1–2 px
of camera shake from holding the phone. Then I split the timeline three ways:
first 60% for training, next 20% for validation, last 20% as a held-out test
set. The goal is **next-frame prediction**, then *autoregressive rollout* —
keep feeding each prediction back as input and ask: how long can you forecast
before the error exceeds "just guess yesterday"?

The two methods compared:

- **Machine-learning (ML) side:** a 1.7-M-parameter Swin-UNet that takes the
  last 5 frames and predicts the next frame's change. Curriculum-trained to be
  stable under autoregressive rollout (predict 1 step → 2 steps → 4 steps).
- **Physics side:** a 4D-Var-style variational assimilation, the same family of
  algorithms ECMWF and NOAA use for weather. At each test position, it solves
  for a single smooth velocity field that best explains the past 5 frames'
  transitions, then advects the last frame forward at constant velocity.

Both methods see the same data at inference; only the ML side has a training
set.

## What the rollouts actually look like

Here's a 60-frame (2-second) autoregressive forecast from the same starting
position, side by side: real waterfall, ML prediction, 4D-Var prediction,
and per-frame error bars below.

![ML vs 4D-Var rollout]({{ "/assets/figures/compare_with_bars_hires_slow3x.gif" | relative_url }})

You can see both methods losing detail as the rollout progresses, which is
exactly right — the underlying turbulence is chaotic and unpredictable past
about a second. ML stays a bit closer to the real frame at every step; the
red bars (ML error) sit consistently below the brown bars (4D-Var error).

So ML wins. Story over?

## The actual story: ML wins *only when given enough data*

If the comparison were always "ML wins, physics loses," we wouldn't need
physics. But here's what happens when I retrain the ML model on subsets of
the available training data — 50, 100, 200, 400, 700, or all 1,090 training
frames — and re-evaluate the same rollout on the same test set:

![Data sensitivity plot]({{ "/assets/figures/data_sensitivity_combined.png" | relative_url }})
*Red: ML's prediction error at 1-second lead, as a function of how many training
images it saw. Brown: 4D-Var's error at the same lead time, as a function of
how many past frames it uses for analysis at inference (the closest physics
analog to "data"). They use data differently — but both can have "more" of it.*

Three things to notice:

**1. The ML curve has a strong slope.** With only 50 training frames, ML is
*much* worse than 4D-Var (13.1 vs 7.4 in the L1 error). It needs about 300
training frames to draw level, and improves smoothly past that. With the
full 1,090 training frames, it's about 14% better than the best 4D-Var.

**2. The 4D-Var curve is essentially flat.** Going from a 5-frame analysis
window to a 30-frame window barely moves the error (7.37 → 6.95). Going
beyond 100 frames it actually starts getting *worse* — past observations get
old enough that they no longer reflect the current turbulent state. The
physics method has a built-in floor it cannot get below.

**3. There's a crossover.** Below ~300 training frames, *physics is better*.
Above ~300, *ML is better*. The boundary is dataset-dependent and
problem-dependent, but the shape of the picture is universal: ML's slope is
steep, physics has a floor, and the choice between them is really a question
about how much labelled data you have for the system you care about.

## Can NWP-style tricks help the physics side?

I tried two improvements weather agencies actually use:

- **Total-variation (TV) regularization on the velocity field** — preserves
  sharp discontinuities, ideal for systems with crisp edges between flowing
  and stationary regions.
- **Multi-incremental (coarse-to-fine) optimization** — solve at 1/4
  resolution first, then 1/2, then full. Avoids shallow local minima.

Multi-incremental gave a marginal ~1% improvement. TV did nothing (the
waterfall has no genuinely sharp velocity discontinuities). Combined, they
were no better than multi-incremental alone. The 4D-Var floor stayed at ~7.3
at the 1-second lead.

That's not a failure of NWP. It's a feature: this physics model (smooth
advection by a constant velocity field) is already near the best it can be.
Improving its analysis quality doesn't help, because the limit isn't the
analysis — it's the model class. To move the floor further, you'd need to
**change the model**: weak-constraint 4D-Var with a learned residual term,
which is exactly where operational NWP centres are headed (hybrid AI/physics
inside the variational system).

## Why this matters beyond a waterfall

The temptation, especially after seeing all the recent ML-beats-NWP papers, is
to take the "ML always wins" framing at face value. The actual physics is:

- **Physics methods have a floor set by their model assumptions.** No amount of
  optimization, no amount of additional observations, can get below it. If the
  real dynamics differs from your model class — as it does for spray, turbulent
  mixing, sub-grid-scale brightness changes — the gap is structural.
- **ML methods have a *slope* set by data.** With enough data, they can learn
  arbitrary deviations from the physics model class. With too little data, they
  fall below the physics floor and the comparison is meaningless.

For weather: NWP has decades of physical-model refinement and assimilates
millions of observations per cycle. ML weather models like Pangu-Weather,
GraphCast, FourCastNet, and Aurora work because they're trained on 40+ years
of ERA5 reanalysis — *that's the "more than 300 training frames" of their
domain*. If you tried to train them on a single 60-second clip of weather,
they'd lose to a hand-tuned 4D-Var.

For my waterfall: the dataset is single-clip-scale. The crossover happens at
N≈300 frames because that's what it takes for the network to see enough
samples of the local turbulent statistics to model them better than a smooth
velocity field can.

## Honest caveats

- This is *one clip of one waterfall*. The crossover point and the slope are
  specific to this system. I can't tell you a priori what they'd be for
  someone else's video.
- The physics model is intentionally simple — smooth advection. A real CFD
  simulation with Navier–Stokes, free-surface tracking, and white-water
  physics would do dramatically better, at dramatically more cost.
- The ML model is intentionally small. With a 100× larger model and pretraining
  on internet-scale video, the curve would shift but the shape would be similar.
- 4D-Var here uses a much weaker analysis system than operational centres
  (no flow-dependent B matrix, no hybrid 4D-EnVar, no bias correction).
  Real operational 4D-Var would push the floor down — but not eliminate it.

## Takeaway

I went up to Yosemite to look at a waterfall. Came back with a small
benchmark that lets you watch the ML-vs-physics scaling story play out at
60-second-clip scale: stable, reproducible, runnable on one consumer GPU.
The headline result isn't "ML wins" — it's that **the choice depends on
data**, and the comparison only makes sense once you tell people how much
data the ML side got.

When someone shows you ML beating a physics baseline, ask them how many
training samples were available, and what happens at half that.

---

**Code:** [github.com/sudshu/yosemite-waterfall-ml](https://github.com/sudshu/yosemite-waterfall-ml)

Reproducible: `pip install -r requirements.txt`, then `python train_rollout.py
--model swin --temporal-context --frames cache/frames_waterfall_stab_gray.npy`
will train the headline model in about an hour on a single A10G. The
data-scaling sweep is `python data_scaling_sweep.py`; the NWP-variants
benchmark is `python sweep_4dvar_variants_nwp.py`.

If you'd like the raw 40-MB MOV file, please email me — I'd rather not check
it into git.
