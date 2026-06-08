---
layout: post
title: "Yosemite, a Waterfall, a Neural Network, and the Data Threshold Where ML Beats Physics"
date: 2026-06-08
tags: [machine-learning, fluid-dynamics, NWP, waterfall]
---

I went to Yosemite last spring and stood for a while in front of Bridalveil
Falls with my phone out. I came back with a 60-second video clip and a
question that wouldn't leave me alone.

**How well can a small neural network predict the next frame of a chaotic
waterfall?** And how does it stack up against the kind of physics-style
methods that weather agencies use to forecast the atmosphere?

A few weekends of GPU time later, I have an honest answer. It's more
interesting than the usual *"AI beats physics, news at 11"* take — because
the answer depends entirely on **how much waterfall the neural network has
been allowed to watch**.

![Bridalveil Falls — full frame and the cropped region we use]({{ "/assets/figures/figure1_waterfall_and_crop.png" | relative_url }})
*The clip. We focus on the boxed region — the cliff face, the falling column,
and the mist envelope — and leave the static rocks and sky out of the
comparison. The cropped region is 216 × 88 pixels.*

## A race: does the AI see the future better than physics does?

Two contestants. Same task: see the next frame, then the next, then the next.
Sixty frames into the future. Two seconds of water that hasn't happened yet.

- **The neural network** — a small AI model that tries to learn the
  waterfall's habits from past examples. We give it the last 5 frames and
  ask it to predict the next one. Then we feed its prediction back as input
  and ask it again. And again. For two seconds.
- **The physics model** — the kind of method weather agencies have used
  for decades. At each starting position, it looks at the recent past,
  figures out roughly *how* the water appears to be moving, and pushes the
  image forward by that motion.

Both contestants see the same recent past when they make a prediction. Only
the neural network has been *trained* on a chunk of the clip first.

Here's what a 2-second forecast looks like, side by side. Real waterfall on
the left, the two predictions in the middle, and per-frame error growing in
red and brown bars at the bottom.

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
  knows.
- **There's a threshold.** Roughly 300 frames of training video — about 10
  seconds — is where the AI catches up. Before that, **physics is better.**
  After that, **AI is better.**

Reasonable people sometimes describe this as "AI beating physics." It's
better described as **AI beating physics once it's been allowed to watch
the system enough**. The crossover point matters at least as much as the
end result.

## What's actually going on?

The physics model in this race is doing something simple and elegant: it
estimates a smooth velocity field — basically a map of "this is roughly how
fast and in what direction each part of the picture is moving right now" —
and uses that map to push the current frame forward in time. That's a
strong inductive bias. The model already *believes* that water moves in a
mostly continuous way, mostly downward, mostly under gravity. It doesn't
need to learn that.

But real spray, turbulent mixing, light flickering off the water — these
don't follow a smooth velocity field. They follow patterns that *exist* but
aren't easy to write down. They are exactly the kind of thing a neural
network is good at noticing, given enough examples.

So with too few examples, the neural network can't see those patterns yet,
and it loses to the physics model's solid common-sense prior. With enough
examples, the neural network has noticed the patterns the physics model
can't represent, and it edges ahead.

## Why this matters beyond a waterfall

The same shape of picture is showing up across science and engineering. ML
models like Pangu-Weather, GraphCast, FourCastNet, and Aurora are beating
operational weather forecasts — but they were trained on **forty years of
reanalysis data** to do it. That's their "300 frames of waterfall." If you
tried to train them on a single 60-second clip of weather, they would lose
to a hand-tuned physics model in a heartbeat.

When someone shows you ML beating a physics baseline, the question to ask
isn't *"is the neural network better?"* — it's *"how much data did the
neural network get to look at?"* That sets the entire context.

---

### Nerd note

For people who want the technical details:

The ML model is a **1.7-M-parameter Swin-UNet**. It takes the last 5 frames
(with channel-wise pairwise differences appended) and predicts the next
frame as a residual over the most recent one. It's trained with a curriculum
of 1-step, 2-step, then 4-step autoregressive rollouts, so the gradient
signal stays meaningful when its own predictions are fed back as inputs.

The physics model is a **4D-Var-style variational data assimilation**. It
fits a single smooth velocity field `v(x, y)` to the past few frames by
minimizing `Σ ‖warp(f_i, v) − f_{i+1}‖_1 + λ‖∇v‖²`, then advects the most
recent frame forward at that constant velocity using a semi-Lagrangian
warp. I also tried TV regularization (no improvement on this scene — the
velocity field has no sharp discontinuities) and a multi-incremental
coarse-to-fine optimizer (~1% improvement). The combined NWP-style
optimization gets the physics floor down to about 7.32 in the L1 error;
the ML model at full data reaches 6.29 (about 14% better).

I aligned the camera shake out of the raw clip via phase correlation before
training. Both methods see the stabilized version. All evaluation is on a
held-out 20% test slice (last 12 seconds of the clip), and the ML
data-scaling sweep is the median of three random subsamples per training
size. Code, training scripts, sweeps, and figure-generating scripts are at
[github.com/sudshu/yosemite-waterfall-ml](https://github.com/sudshu/yosemite-waterfall-ml).

### Honest caveats

- One clip of one waterfall. The exact crossover frame count and the slope
  are specific to this system.
- The physics model is intentionally simple — smooth advection by a single
  velocity field. A full CFD simulation would do dramatically better, at
  dramatically more cost.
- The neural network is also intentionally small. A bigger model with
  internet-scale video pretraining would shift the curves but the shape of
  the story would be similar.
- This is a toy. The point isn't the waterfall. The point is that the same
  shape of curve — slope versus ceiling, crossover at some data threshold —
  shows up almost everywhere the comparison gets made.

### The takeaway

The lesson isn't that ML magically beats physics. The lesson is that **ML
needs to watch the system long enough**. With too little waterfall, physics
wins. With enough waterfall, the neural network learns the parts of the
motion the simple physics model cannot represent.

Next time you see a benchmark claiming AI overtakes physics, ask how much
data the AI got. The answer is the punchline.
