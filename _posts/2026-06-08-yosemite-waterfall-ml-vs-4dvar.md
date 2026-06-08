---
layout: post
title: "Yosemite, a Waterfall, and the Data Threshold Between ML and Physics"
date: 2026-06-08
tags: [machine-learning, fluid-dynamics, NWP, waterfall]
---

![Yosemite Valley with Bridalveil Falls]({{ "/assets/figures/hero_yosemite_valley.jpg" | relative_url }})
*Yosemite Valley from the Wawona Tunnel View. **The waterfall analysed in
this post is Bridalveil Falls** — the thin white streak halfway up the
right-hand cliff face.*

I went to Yosemite recently. It might be the most beautiful place I've ever
been — and standing in front of Bridalveil Falls, watching the white sheet
of water tear itself apart into spray about halfway down, I started
thinking about the recent stream of papers showing that AI weather models
can now outperform the operational physics-based forecasts —
[Pangu-Weather](https://www.nature.com/articles/s41586-023-06185-3),
[GraphCast](https://www.science.org/doi/10.1126/science.adi2336),
[FourCastNet](https://arxiv.org/abs/2202.11214),
[Aurora](https://arxiv.org/abs/2405.13063).

The waterfall is a small piece of pure turbulence: water dropping under
gravity, breaking, mixing with air, throwing off mist that wanders sideways
on its own little chaotic eddies. The pattern at any instant looks roughly
the same as a second earlier, but the details are wildly different and
genuinely impossible to predict frame-by-frame. A thought came to mind:

> *When will an AI model be able to predict the next moment of this
> waterfall better than a physics-based forecast can?*

I pulled out my phone and tried to shoot a clean, steady 60-second clip
that I could take home and run experiments on. A few weekends of GPU time
later, I have an honest answer. It's more interesting than the usual
*"AI beats physics, news at 11"* take — because the answer depends entirely
on **how much waterfall the neural network has been allowed to watch**.

![Bridalveil Falls — full frame and the cropped region we use]({{ "/assets/figures/figure1_waterfall_and_crop.png" | relative_url }})
*The clip. We focus on the boxed region — the cliff face, the falling column,
and the mist envelope — and leave the static rocks and sky out of the
comparison. The cropped region is 216 × 88 pixels.*

## A race: does the AI see the future better than physics does?

The video is 30 frames per second. The task is **autoregressive rollout**:
predict the next frame, then feed that prediction back as input and predict
the *next* one, and so on. Both contestants do this for 60 steps — two
seconds of water that hasn't happened yet.

```
real past frames                           predicted future
                                           (feeds back into itself)
┌──────────────────────────┐              ┌────────────────────┐
│  f(t-4) f(t-3) f(t-2)    │              │  f̂(t+1)  f̂(t+2) │
│  f(t-1) f(t)             │ ─► [model] ─►│   ↓ slide ↓        │
└──────────────────────────┘              │  predict f̂(t+3)   │
                                          │   ↓ ... up to t+60 │
                                          └────────────────────┘
```

Each step's prediction becomes the newest frame of the input window for the
next step. Errors made early in the rollout get fed back into the model and
can compound — which is part of why this task is harder than predicting
one frame at a time.

- **The neural network** — a small AI model that learns the waterfall's
  habits from past examples (an *optical-flow-aware Swin-UNet*; details in
  the [Nerdy note](#nerdy-note)). At inference, we hand it the last 5
  frames and ask it to predict the next one. Then we feed its prediction
  back as input and ask it again. And again. For two seconds.
- **The physics model** — the kind of method weather agencies have used
  for decades. At each starting position, it looks at the recent past,
  estimates an *optical flow field* (roughly: how each part of the image
  appears to be moving), and pushes the picture forward at that velocity.

A useful trick worth flagging: the neural network doesn't predict the raw
next-frame pixels directly. Instead, it predicts the **change** Δ from the
current frame:

> `f̂(t+1)  =  clamp( f(t) + Δ ,  0,  1 )`

Most of the next frame is identical to the current one (the cliff isn't
moving, only the water is), so the residual signal Δ is concentrated where
motion actually happens. This makes it a *much* easier target to learn than
trying to reproduce the whole frame from scratch every step.

Both contestants see the same recent past at inference. Only the neural
network gets *trained* on a separate chunk of the clip first — that's the
variable we'll sweep over.

We evaluate everything on a strictly held-out test slice: the last 20% of
the clip (about 12 seconds), which neither model has ever seen during
training or analysis.

Here's what a 2-second forecast looks like, side by side. Real waterfall on
the left, the two predictions in the middle, and per-frame L1 error
(average per-pixel difference, in 0–255 grayscale levels) growing in red
and brown bars at the bottom.

![ML vs 4D-Var rollout animation]({{ "/assets/figures/compare_with_bars_hires_slow3x.gif" | relative_url }})

Both methods lose detail as the rollout progresses — that's the underlying
turbulence being chaotic and unpredictable past about a second. The neural
network stays a little closer to the real waterfall at every step. The red
bars sit consistently below the brown bars.

So the AI wins. End of story?

## Not quite. It depends on practice.

Here's the actual fun part. I retrained the neural network on different
**fractions** of the available video — 50 frames, 100, 200, 400, 700, or
all 1,090 — and re-ran the same forecast against the same physics model.
This is what came out:

![How much waterfall does ML need before it beats physics]({{ "/assets/figures/headline_playful.png" | relative_url }})
*The neural network needs practice. With only 50 frames (less than 2 seconds
of footage) it is much worse than the physics model. Around 300 frames
(~10 seconds), it catches up. With the whole clip, it wins. The physics
model starts strong because it already knows a useful rule — **water
mostly moves downward** — but it quickly reaches a ceiling, because real
waterfall spray is more complicated than smooth downward motion.*

The picture tells the story:

- **The neural network has a slope.** Show it more waterfall, it gets
  better. With very little video, it's hopeless. With a lot, it wins.
- **The physics model has a ceiling.** Whether you give it 5 frames or 200
  to look back at, the answer barely changes. It already knows what it
  knows — its function class can't represent the residual unmodeled
  dynamics no matter how much it looks at.
- **There's a threshold.** Roughly 300 frames of training video — about 10
  seconds — is where the AI catches up. Before that, **physics is better.**
  After that, **AI is better.**

In bias-variance language: the physics model is **high-bias** (a
restrictive model class — smooth advection only) and the neural network is
**high-variance** (depends on how much data it has been allowed to see).
With enough data the variance shrinks below the bias floor, and the
neural network takes over.

Reasonable people sometimes describe this as "AI beating physics." It's
better described as **AI beating physics once it's been allowed to watch
the system enough**. The crossover point matters at least as much as the
end result.

## What's actually going on?

The physics model carries a strong built-in belief: water moves continuously
and mostly downward. That belief is free — it doesn't need any data to
acquire — and it gets you most of the way there.

What it can't capture is the rest: real spray, turbulent mixing, light
flickering off the water. These follow patterns that *exist* but aren't
easy to write down as smooth motion. Exactly the kind of thing a neural
network is good at noticing — given enough examples.

## Why this matters beyond a waterfall

The same shape of picture is showing up across science and engineering. ML
models like
[Pangu-Weather](https://www.nature.com/articles/s41586-023-06185-3),
[GraphCast](https://www.science.org/doi/10.1126/science.adi2336),
[FourCastNet](https://arxiv.org/abs/2202.11214), and
[Aurora](https://arxiv.org/abs/2405.13063) are beating operational weather
forecasts — but they were trained on **forty years of reanalysis data**
([ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)) to
do it. That's their "300 frames of waterfall." If you tried to train them
on a single 60-second clip of weather, they would lose to a hand-tuned
physics model in a heartbeat.

When someone shows you ML beating a physics baseline, the question to ask
isn't *"is the neural network better?"* — it's *"how much data did the
neural network get to look at?"* That sets the entire context.

---

### Nerdy note

The ML model is a 1.7-M-parameter **Swin-UNet** — a U-Net whose
convolutions are replaced with hierarchical shifted-window self-attention
blocks
([Liu et al. 2021](https://arxiv.org/abs/2103.14030)) — trained
autoregressively with a **rollout curriculum** (R = 1 → 2 → 4 steps), a
kind of scheduled-sampling fix for the exposure-bias problem in
time-series prediction. Full architecture, loss function, and training
pseudocode are in the [repo
README](https://github.com/sudshu/yosemite-waterfall-ml#the-ml-model).

The physics model is a **4D-Var-style variational data assimilation** — it
fits a single smooth velocity field `v(x, y)` to the past few frames by
minimising a data-fit cost (warped previous frame should match the next)
plus a smoothness prior on `v`, then advects the most recent frame forward
at that constant velocity. Full cost function and the TV /
multi-incremental NWP variants are
[here](https://github.com/sudshu/yosemite-waterfall-ml#the-physics-baseline-4d-var).
This is "physics" in the narrow sense of *smooth optical-flow advection* —
a full multi-phase Navier-Stokes solver with sub-grid turbulence closures
would do dramatically better, at ~10⁵× the compute.

The combined NWP-optimised physics baseline gets down to L1 ≈ 7.32 (in
0–255 grayscale levels) at the 1-second forecast lead. The ML model at
full data reaches **L1 ≈ 6.29** — about 14 % better. The ML data-scaling
curve is the median of three random subsamples per training size;
variance is tight at large N and noisier below ~200 frames, so the
crossover at "~300 frames" is best read as somewhere between 200 and 400.

Code, the source MOV, and the figure-generating scripts are at
[github.com/sudshu/yosemite-waterfall-ml](https://github.com/sudshu/yosemite-waterfall-ml).

### Honest caveats

- One clip of one waterfall. Single-clip data-scaling laws are fragile —
  the exact crossover and slope could easily differ by ~2× on a longer
  clip or a different turbulent system.
- The physics model is intentionally simple — smooth advection by a single
  velocity field. A full CFD simulation would do dramatically better, at
  dramatically more cost.
- The neural network is also intentionally small. A bigger model with
  internet-scale video pretraining would shift the curves but the shape of
  the story would be similar.
- The point isn't the waterfall. It's that the same shape of curve —
  slope versus ceiling, crossover at some data threshold — shows up almost
  everywhere the comparison gets made.

### The takeaway

The lesson isn't that ML magically beats physics. The lesson is that **ML
needs to watch the system long enough**. With too little waterfall, physics
wins. With enough waterfall, the neural network learns the parts of the
motion the simple physics model cannot represent.
