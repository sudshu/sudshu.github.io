---
layout: post
title: "The adjoint used to be a PhD. Now it's one line of JAX."
date: 2026-07-02
tags: [inverse-modeling, carbon-cycle, remote-sensing, data-assimilation, JAX, adjoint]
image: /assets/figures/jax_adjoint_og_card.png
excerpt: "In carbon-cycle science the adjoint is the engine behind every emission estimate — and it used to take years to build by hand. Then differentiable programming quietly deleted the hard part. A grad-student-level tour, with toy code, real satellites, and a laptop GPU."
---

<style>
  .iw-card{
    --ink:#1b1d1f; --muted:#68707a; --panel:#f6f8fa; --border:#e2e6ea;
    --green:#2f6b2c; --red:#c0392b; --steel:#37618e; --amber:#d97706;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink); line-height:1.55; -webkit-font-smoothing:antialiased;
  }
  .iw-card *{box-sizing:border-box;}

  .iw-card{
    border:1px solid var(--border); border-radius:12px; background:#fff;
    padding:20px 20px 18px; margin:22px 0;
    box-shadow:0 1px 2px rgba(20,25,35,.04), 0 8px 24px -18px rgba(20,25,35,.18);
  }
  .iw-eyebrow{
    text-transform:uppercase; letter-spacing:.12em; font-size:10.5px; font-weight:700;
    color:var(--amber); margin:0 0 4px;
  }
  .iw-title{font-size:18px; font-weight:650; margin:0 0 3px; letter-spacing:-.01em;}
  .iw-note{color:var(--muted); font-size:13.5px; margin:0 0 16px;}
  .iw-caption{color:var(--muted); font-size:12.5px; font-style:italic; margin:14px 2px 0; line-height:1.5;}

  .iw-panel{background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px;}
  .iw-canvases{display:flex; gap:12px;}
  .iw-canvases > div{flex:1 1 0; min-width:0;}
  .iw-cv-label{font-size:11px; color:var(--muted); font-weight:600; margin:0 0 6px 2px; letter-spacing:.02em;}
  canvas.iw{display:block; width:100%; border-radius:5px; touch-action:manipulation;}

  .iw-controls{display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin-top:14px;}
  .iw-sliderbox{display:flex; flex-direction:column; gap:3px; flex:1 1 220px; min-width:180px;}
  .iw-sliderbox label{font-size:11.5px; color:var(--muted); font-weight:600; display:flex; justify-content:space-between; gap:8px;}
  .iw-sliderbox label b{color:var(--ink); font-family:var(--mono); font-variant-numeric:tabular-nums; font-weight:600;}
  input[type=range].iw{-webkit-appearance:none; appearance:none; width:100%; height:5px; border-radius:3px; background:var(--border); outline:none; margin:6px 0;}
  input[type=range].iw::-webkit-slider-thumb{-webkit-appearance:none; appearance:none; width:17px; height:17px; border-radius:50%; background:var(--amber); border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,.28); cursor:pointer;}
  input[type=range].iw::-moz-range-thumb{width:15px; height:15px; border-radius:50%; background:var(--amber); border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,.28); cursor:pointer;}
  input[type=range].iw:focus-visible{box-shadow:0 0 0 3px rgba(217,119,6,.3);}

  button.iw{
    font:inherit; font-size:13px; font-weight:600; padding:8px 13px; border-radius:7px;
    border:1px solid var(--border); background:#fff; color:var(--ink); cursor:pointer;
    transition:background .12s, border-color .12s, transform .04s;
  }
  button.iw:hover{background:var(--panel); border-color:#cfd5db;}
  button.iw:active{transform:translateY(1px);}
  button.iw:focus-visible{outline:none; box-shadow:0 0 0 3px rgba(217,119,6,.3);}
  button.iw.primary{background:var(--ink); color:#fff; border-color:var(--ink);}
  button.iw.primary:hover{background:#000;}
  button.iw:disabled{opacity:.45; cursor:default; transform:none;}

  .iw-stat{font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:12.5px;}
  select.iw{font:inherit; font-size:12.5px; padding:6px 8px; border-radius:6px; border:1px solid var(--border); background:#fff; color:var(--ink);}

  @media (max-width:560px){
    .iw-canvases{flex-direction:column;}
  }
  @media (prefers-reduced-motion:reduce){
    button.iw{transition:none;}
  }
</style>

![Finite-difference cost versus the adjoint gradient, as the number of unknowns grows]({{ "/assets/figures/jax_adjoint_gradient_scaling.png" | relative_url }})

*The whole story in one plot. To estimate a gradient, the naive way (finite
differences) gets more expensive with every unknown you add. The adjoint
returns the entire gradient in one backward pass, for roughly the same price
regardless of how many unknowns there are. By 1,000 unknowns it is ~265×
cheaper — and the gap keeps widening.*

When I was a PhD student, there was a particular sentence that could silence a
room of atmospheric modellers: *"...and then we built the adjoint."*

It was said the way climbers talk about a summit. Building the **adjoint** of a
big atmospheric transport model — the thing you need to turn satellite
measurements into emission estimates — was a rite of passage that could eat a
year or two of someone's PhD. People wrote papers just about *how to write the
adjoint code* ([Giering & Kaminski, 1998](https://doi.org/10.1145/293686.293695)).
Whole model versions were named after it: the GEOS-Chem adjoint
([Henze et al., 2007](https://doi.org/10.5194/acp-7-2413-2007)), the TM5
four-dimensional variational system
([Meirink et al., 2008](https://doi.org/10.5194/acp-8-6341-2008)).

It ate the first six months of mine. I spent them learning to build adjoints by
hand, and being quietly floored that the recipe worked at all: apply a fixed set
of rules, mechanically, backwards through every line of your model, and out falls
the gradient of your cost with respect to every input. Not an approximation — the
*exact* gradient, to the last digit. A physical simulation could be made to hand
you its own exact sensitivities, if you were only disciplined enough about the
bookkeeping.

Then differentiable programming showed up and quietly deleted the hard part.

If your forward model is written in a framework like [JAX](https://github.com/jax-ml/jax),
the adjoint is **already there**. You do not derive it. You do not hand-code it.
You call `jax.grad`, and the framework walks backward through your model and
hands you the gradient. The thing that used to cost a PhD now costs one line.

Everything below is built from toy models small enough to run — and play with —
on a laptop.

## Why anyone wants an adjoint in the first place

Start with the picture that pays for a lot of Earth-observing satellites.

A spectrometer in orbit — [TROPOMI](https://doi.org/10.1016/j.rse.2011.09.027),
GOSAT, OCO-2 — measures how much methane or CO₂ sits in a column of air. What we
actually *want* is not the concentration. It is the **emissions**: which well,
which wetland, which city, leaking how much
([Jacob et al., 2016](https://doi.org/10.5194/acp-16-14371-2016)).

Between the two sits a forward model:

```text
emissions  ──►  [ atmospheric transport + chemistry ]  ──►  concentrations
   (what we want)                                            (what we see)
```

The **inverse problem** is to run that arrow backwards: find the emissions that,
pushed through the model, best reproduce what the satellite saw. In practice you
never invert the arrow directly. You define a mismatch — a scalar cost `J` that
measures how far the model is from the observations (plus a term keeping you near
a prior) — and you slide the emissions downhill until `J` is small. This is the
whole edifice of atmospheric inverse modelling, from global CO₂ budgets to
satellite methane. The textbook, if you want one, is
[Enting (2002)](https://doi.org/10.1017/CBO9780511535741).

To slide downhill you need the **gradient**: how does the mismatch change if I
nudge each emission parameter? And here is the catch that makes or breaks the
whole enterprise. A real flux inversion does not have four unknowns. It has
emissions on a grid — thousands to millions of them. You need the derivative of
one number (the mismatch) with respect to *all* of them.

There are two ways to get it.

- **Finite differences.** Nudge one parameter, re-run the whole model, see how
  the mismatch moved. Repeat for every parameter. For `P` unknowns that is about
  `2P` full model runs. For a million-parameter inversion, that is a
  non-starter — you would be re-running a chemistry-transport model a million
  times just to take *one* step downhill.
- **The adjoint.** Run the model forward once, then run a single backward pass
  that propagates the sensitivity of the mismatch back through every operation,
  accumulating the derivative with respect to *every* parameter along the way.
  One forward pass, one backward pass, the entire gradient — almost regardless of
  how many unknowns.

That backward pass **is** the adjoint. If you have ever trained a neural network,
you already know it under a different name: backpropagation. Reverse-mode
automatic differentiation, the adjoint method, and backprop are the same idea
wearing three different lab coats ([Errico, 1997](https://doi.org/10.1175/1520-0477(1997)078%3C2577:WIAAM%3E2.0.CO;2)
is still the friendliest one-page explanation for atmospheric scientists).

That is the trade-off in the plot at the top: at a hundred thousand parameters
it is the difference between "runs overnight" and "never finishes."

## The free lunch, in one line

Here is the part that still feels like cheating.

Take the simplest possible physical model — 1-D diffusion, heat (or a tracer)
spreading along a line:

```text
du/dt = kappa * d²u/dx²
```

with a single unknown, the diffusivity `kappa`. Write the time-stepper in JAX,
make some synthetic "observations" with a known `kappa_true`, and define the
mismatch:

```python
import jax, jax.numpy as jnp

def forward(kappa):
    u = u0
    for _ in range(n_steps):                 # explicit diffusion step
        u = u + kappa * dt/dx**2 * (jnp.roll(u,1) - 2*u + jnp.roll(u,-1))
    return u

def loss(kappa):
    return jnp.mean((forward(kappa) - u_obs)**2)

grad_loss = jax.grad(loss)      # <-- this is the discrete adjoint. that's it.
```

`jax.grad(loss)` is the adjoint of your time-stepping model. Not an
approximation of it — the exact derivative of the numerical code you wrote,
assembled by differentiating every operation in reverse, at least for the parts
of the model JAX can trace (more on where that breaks below). Conceptually:

```text
forward:    kappa ──► diffusion steps ──► final state ──► mismatch J
backward:   dJ ──► d(final state) ──► back through every time step ──► dJ/dkappa
```

No adjoint model was written. Feed that gradient to an optimizer and the inverse
problem just... solves:

```text
true kappa:            0.035000
initial guess:         0.006000
recovered kappa:       0.035513      (1.5% off, from a 6× wrong start)
```

Twenty years ago, that would have been a week of careful calculus and debugging.
Now it is a function call.

{% raw %}
<section class="iw-card" id="iw-diffuse">
  <p class="iw-eyebrow">Interactive · inverse problem</p>
  <p class="iw-title">Invert it yourself</p>
  <p class="iw-note">A hidden diffusivity <b>κ</b> blurred the sharp curve into the dots.
  Drag to guess κ by eye — then hit <em>Run the gradient</em> and watch calculus do it for you.</p>
  <div class="iw-canvases">
    <div>
      <p class="iw-cv-label">Model curve vs. observations</p>
      <div class="iw-panel"><canvas class="iw" id="d-profile"></canvas></div>
    </div>
    <div>
      <p class="iw-cv-label">Mismatch landscape J(κ)</p>
      <div class="iw-panel"><canvas class="iw" id="d-loss"></canvas></div>
    </div>
  </div>
  <div class="iw-controls">
    <div class="iw-sliderbox">
      <label>your guess for κ <b id="d-kval">0.0100</b></label>
      <input type="range" class="iw" id="d-slider" min="0" max="1" step="0.0001" value="0.35">
    </div>
    <button class="iw" id="d-step">Take one step</button>
    <button class="iw primary" id="d-run">Run the gradient ▸</button>
    <button class="iw" id="d-new">New hidden κ</button>
  </div>
  <p class="iw-stat" id="d-status" style="margin-top:12px; color:var(--muted);">
    mismatch J = <span id="d-loss-val" style="color:var(--ink)">—</span>
    &nbsp;·&nbsp; gradient steps: <span id="d-iter" style="color:var(--ink)">0</span></p>
  <p class="iw-caption">The dots are your data; the amber line is the model for your current κ.
  Slide it and the mismatch on the right moves with you. One button press does what your eye was
  doing — but automatically, and it works the same way when there are a million unknowns instead of one.</p>
</section>
<script>
(function(){
  var N=96, M=60, SCALE=9;
  function u0(){var u=new Float64Array(N);for(var i=0;i<N;i++){var x=i/N;
    u[i]=0.9*Math.exp(-(Math.pow((x-0.35)/0.08,2)))+0.6*Math.exp(-(Math.pow((x-0.68)/0.05,2)));}return u;}
  var U0=u0();
  function forward(kappa){var d=SCALE*kappa,u=U0.slice(),t,i,nu;
    for(t=0;t<M;t++){nu=new Float64Array(N);for(i=0;i<N;i++){var l=u[(i-1+N)%N],r=u[(i+1)%N];nu[i]=u[i]+d*(l-2*u[i]+r);}u=nu;}return u;}
  // slider maps 0..1 -> kappa 0.002..0.05 (log)
  var KMIN=0.002,KMAX=0.05;
  function s2k(s){return KMIN*Math.pow(KMAX/KMIN,s);}
  function k2s(k){return Math.log(k/KMIN)/Math.log(KMAX/KMIN);}
  var kappaTrue, uobs, lossGrid, lossMin, lossMax, kappaGuess, iter=0, converged=false, running=false, raf=null;
  function lossOf(k){var u=forward(k),s=0,i;for(i=0;i<N;i++){var e=u[i]-uobs[i];s+=e*e;}return s/N;}
  function buildGrid(){lossGrid=[];lossMin=1e9;lossMax=-1e9;for(var j=0;j<=90;j++){var s=j/90,k=s2k(s),L=Math.log10(lossOf(k)+1e-14);lossGrid.push([s,L]);if(L<lossMin)lossMin=L;if(L>lossMax)lossMax=L;}}
  function newTruth(){
    kappaTrue=0.024+Math.random()*0.020; // 0.024..0.044
    var base=forward(kappaTrue); uobs=new Float64Array(N);
    var mx=0,i; for(i=0;i<N;i++) if(base[i]>mx)mx=base[i];
    for(i=0;i<N;i++) uobs[i]=base[i]+(Math.random()-0.5)*0.004*mx;
    buildGrid();
    kappaGuess=s2k(0.30+Math.random()*0.12); iter=0; converged=false;
    sync(); draw();
  }
  var prof=document.getElementById('d-profile'), lossc=document.getElementById('d-loss');
  function fit(cv,h){var dpr=Math.min(window.devicePixelRatio||1,2);var w=cv.clientWidth||300;cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);cv.style.height=h+'px';var c=cv.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);return{c:c,w:w,h:h};}
  function drawProfile(){
    var o=fit(prof,150),c=o.c,w=o.w,h=o.h,padL=8,padR=8,padT=10,padB=10;
    c.clearRect(0,0,w,h);
    var yMax=1.0,gx=function(i){return padL+(i/(N-1))*(w-padL-padR);},gy=function(v){return padT+(1-v/yMax)*(h-padT-padB);};
    // guess model line (amber)
    var ug=forward(kappaGuess),i;
    c.strokeStyle='#d97706';c.lineWidth=2.4;c.beginPath();
    for(i=0;i<N;i++){var X=gx(i),Y=gy(ug[i]);i?c.lineTo(X,Y):c.moveTo(X,Y);}c.stroke();
    // obs dots (ink)
    c.fillStyle='#1b1d1f';for(i=0;i<N;i+=3){c.beginPath();c.arc(gx(i),gy(uobs[i]),1.9,0,6.2832);c.fill();}
    // truth line after converge
    if(converged){var ut=forward(kappaTrue);c.strokeStyle='rgba(47,107,44,.9)';c.lineWidth=1.4;c.setLineDash([4,3]);c.beginPath();for(i=0;i<N;i++){var X2=gx(i),Y2=gy(ut[i]);i?c.lineTo(X2,Y2):c.moveTo(X2,Y2);}c.stroke();c.setLineDash([]);}
  }
  function drawLoss(){
    var o=fit(lossc,150),c=o.c,w=o.w,h=o.h,padL=10,padR=10,padT=12,padB=14;
    c.clearRect(0,0,w,h);
    var span=Math.max(lossMax-lossMin,1e-6);
    var gx=function(s){return padL+s*(w-padL-padR);},gy=function(L){return padT+(1-(L-lossMin)/span)*(h-padT-padB);};
    // axis baseline
    c.strokeStyle='#e2e6ea';c.lineWidth=1;c.beginPath();c.moveTo(padL,h-padB);c.lineTo(w-padR,h-padB);c.stroke();
    // curve
    c.strokeStyle='#37618e';c.lineWidth=2;c.beginPath();
    for(var j=0;j<lossGrid.length;j++){var X=gx(lossGrid[j][0]),Y=gy(lossGrid[j][1]);j?c.lineTo(X,Y):c.moveTo(X,Y);}c.stroke();
    // ball at current guess
    var sg=k2s(kappaGuess),Lg=Math.log10(lossOf(kappaGuess)+1e-14);
    var bx=gx(Math.max(0,Math.min(1,sg))),by=gy(Math.max(lossMin,Math.min(lossMax,Lg)));
    c.fillStyle='#d97706';c.strokeStyle='#fff';c.lineWidth=2;c.beginPath();c.arc(bx,by,6,0,6.2832);c.fill();c.stroke();
    // labels
    c.fillStyle='#68707a';c.font='10px ui-monospace,Menlo,monospace';c.textAlign='left';
    c.fillText('more mismatch ↑',padL+2,padT+9);
    c.fillText('κ →',w-padR-24,h-4);
  }
  function draw(){drawProfile();drawLoss();}
  function sync(){
    document.getElementById('d-slider').value=k2s(kappaGuess).toFixed(4);
    document.getElementById('d-kval').textContent=kappaGuess.toFixed(4);
    document.getElementById('d-loss-val').textContent=lossOf(kappaGuess).toExponential(2);
    document.getElementById('d-iter').textContent=iter;
    var st=document.getElementById('d-status');
    if(converged){st.innerHTML='✓ recovered κ = <b style="color:var(--green)">'+kappaGuess.toFixed(4)+
      '</b> · true κ was <b style="color:var(--green)">'+kappaTrue.toFixed(4)+'</b> · '+iter+' gradient steps';}
    else{st.innerHTML='mismatch J = <span style="color:var(--ink)">'+lossOf(kappaGuess).toExponential(2)+
      '</span> &nbsp;·&nbsp; gradient steps: <span style="color:var(--ink)">'+iter+'</span>';}
  }
  // damped Newton step in theta=log(kappa)
  function gradStep(){
    var theta=Math.log(kappaGuess),h=1e-3;
    var Jm=lossOf(Math.exp(theta-h)),J0=lossOf(kappaGuess),Jp=lossOf(Math.exp(theta+h));
    var g=(Jp-Jm)/(2*h),H=(Jp-2*J0+Jm)/(h*h),step=-g/Math.max(Math.abs(H),1e-12);
    var lr=1.0,nt,nJ;
    for(var bt=0;bt<24;bt++){nt=theta+lr*step;nJ=lossOf(Math.exp(nt));if(nJ<=J0)break;lr*=0.5;}
    kappaGuess=Math.max(KMIN,Math.min(KMAX,Math.exp(nt))); iter++;
    if(Math.abs(g)<2e-6||nJ<1e-11) converged=true;
    return Math.abs(g);
  }
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion:reduce)').matches;
  function stopRun(){running=false;if(raf){cancelAnimationFrame(raf);raf=null;}document.getElementById('d-run').textContent='Run the gradient ▸';}
  function runLoop(){
    if(!running)return;
    var g=gradStep(); sync(); draw();
    if(converged||iter>60){stopRun();return;}
    setTimeout(function(){raf=requestAnimationFrame(runLoop);},260);
  }
  document.getElementById('d-slider').addEventListener('input',function(e){
    if(running)stopRun(); converged=false; kappaGuess=s2k(parseFloat(e.target.value)); sync(); draw();});
  document.getElementById('d-step').addEventListener('click',function(){if(running)stopRun();if(converged)return;gradStep();sync();draw();});
  document.getElementById('d-run').addEventListener('click',function(){
    if(running){stopRun();return;}
    if(converged){return;}
    if(reduce){var guard=0;while(!converged&&guard<80){gradStep();guard++;}sync();draw();return;}
    running=true;this.textContent='■ stop';runLoop();});
  document.getElementById('d-new').addEventListener('click',function(){if(running)stopRun();newTruth();});
  var rt;window.addEventListener('resize',function(){clearTimeout(rt);rt=setTimeout(draw,120);});
  newTruth();
  setTimeout(draw,30);
})();
</script>
{% endraw %}

## A cartoon of a real methane inversion

One unknown is a party trick. Let's make it look a little more like the day job.

I built a **two-box atmospheric chemistry model**: a "northern" box and a
"southern" box that exchange air, each carrying three tracers — **CH₄, CO, and
NOₓ**. The chemistry is deliberately nonlinear: CH₄ and CO are destroyed by the
hydroxyl radical OH, and OH itself depends on how much CH₄, CO, and NOₓ are
present. (That coupling — methane's lifetime depending on the very air it sits in — is a
real headache for explaining methane's ups and downs;
[Bousquet et al., 2006](https://doi.org/10.1038/nature05132).)

Then I hid the answer. I picked four "true" emission scale factors, generated
synthetic CH₄ and CO observations over five years, and asked the inversion to
recover the emissions it was never told:

```text
control        true      recovered
CH4 north      1.3000     1.2999
CO  north      0.7000     0.7000
CH4 south      0.7500     0.7501
CO  south      1.4000     1.4000
```

Four decimal places — in this clean, noise-free synthetic case — from one scalar
mismatch and its adjoint gradient, through a nonlinear coupled-chemistry model.
Same one line, `jax.value_and_grad(loss)`, doing the work. Recovering several
coupled tracers at once from column measurements is exactly the shape of a real
satellite inversion: the very first paper of my PhD was a joint CH₄/CO₂ column
inversion of just this kind
([Pandey et al., 2015](https://doi.org/10.5194/acp-15-8615-2015)). It is the same
machinery behind today's operational methane and CO₂ flux estimates
([Meirink et al., 2008](https://doi.org/10.5194/acp-8-6341-2008);
[Chevallier et al., 2005](https://doi.org/10.1029/2005JD006390)).

## Okay, you have the gradient. Now what do you do with it?

A gradient tells you which way is downhill. It does not tell you how big a step to
take, and this is where a lot of grad-student hours quietly disappear.

The plainest choice is a first-order optimizer — Adam, the workhorse from deep
learning — which only ever looks at the gradient. It works, but it can be slow,
wandering down long narrow valleys taking cautious steps.

The alternative is to use **curvature** — second-derivative information about how
the landscape bends. Two classic moves from the inverse-problems world:

- **Gauss-Newton / Levenberg-Marquardt** approximates the curvature from the
  Jacobian of the residuals (`JᵀJ`). It is the default for smooth
  least-squares problems for a reason.
- **Full Newton** uses the exact Hessian — which JAX will hand you for a small
  problem, `jax.hessian(loss)` (at real scale you never form it; you use
  Hessian-vector products instead).

Get a feel for the difference on the classic test valley below: gradient descent
only sees the slope and zig-zags for thousands of steps; curvature cuts across in
a dozen. Drag the start point and race them.

{% raw %}
<section class="iw-card" id="iw-race">
  <p class="iw-eyebrow">Interactive · optimizer race</p>
  <p class="iw-title">Slope vs. curvature</p>
  <p class="iw-note">Both optimizers start together and chase the same minimum down a curved valley.
  One follows the slope; the other uses curvature. Press <em>Race</em> — or click the map to move the start.</p>
  <div class="iw-panel"><canvas class="iw" id="r-canvas" style="cursor:crosshair;"></canvas></div>
  <div class="iw-controls">
    <button class="iw primary" id="r-race">Race ▸</button>
    <button class="iw" id="r-reset">Reset</button>
    <span class="iw-stat" style="margin-left:auto;">
      <span style="color:var(--steel); font-weight:700;">● gradient descent</span>
      <b id="r-gd" style="color:var(--steel);">0</b>
      &nbsp;&nbsp;<span style="color:var(--green); font-weight:700;">● Gauss–Newton</span>
      <b id="r-gn" style="color:var(--green);">0</b>
    </span>
  </div>
  <p class="iw-stat" id="r-status" style="margin-top:10px; color:var(--muted);">Ready. Both start at the amber dot.</p>
  <p class="iw-caption">Gradient descent (steel) only knows which way is downhill, so it zig-zags thousands of
  times along the valley floor. Gauss–Newton (green) also knows how the valley <em>bends</em>, so it cuts almost
  straight to the bottom in about a dozen steps. Same answer — the curvature is what pays for the shortcut.</p>
</section>
<script>
(function(){
  var A=1,B=40,MINX=1,MINY=1;
  var XLO=-1.7,XHI=1.7,YLO=-0.7,YHI=2.3;
  function f(x,y){return (A-x)*(A-x)+B*(y-x*x)*(y-x*x);}
  function grad(x,y){return [-2*(A-x)-4*B*x*(y-x*x), 2*B*(y-x*x)];}
  var cv=document.getElementById('r-canvas'),bg=null,bgW=0,bgH=0,dpr=Math.min(window.devicePixelRatio||1,2);
  var start=[-1.0,1.0];
  function fit(){var w=cv.clientWidth||520,h=Math.round(w*0.60);cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr);cv.style.height=h+'px';return[w,h];}
  var W,H;
  function px(x){return (x-XLO)/(XHI-XLO)*W;}
  function py(y){return (1-(y-YLO)/(YHI-YLO))*H;}
  function buildBG(){
    var d=cv.getContext('2d');d.setTransform(1,0,0,1,0,0);
    bg=d.createImageData(cv.width,cv.height);var data=bg.data;
    for(var j=0;j<cv.height;j++){for(var i=0;i<cv.width;i++){
      var x=XLO+(i/cv.width)*(XHI-XLO), y=YLO+(1-j/cv.height)*(YHI-YLO);
      var v=Math.log10(f(x,y)+0.02), t=Math.max(0,Math.min(1,(v+1.7)/3.4));
      // darker = lower cost (the valley reads as a basin you look down into); light = high ground
      var r=Math.round(44+(238-44)*t), g=Math.round(62+(241-62)*t), b=Math.round(88+(245-88)*t);
      var o=(j*cv.width+i)*4;data[o]=r;data[o+1]=g;data[o+2]=b;data[o+3]=255;
    }}
    bgW=cv.width;bgH=cv.height;
  }
  var gd,gdSteps,gdPath,gdDone,ln,lam,gnSteps,gnPath,gnDone,anim=null,racing=false,frames=0;
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion:reduce)').matches;
  function resetState(){
    gd=start.slice();gdSteps=0;gdPath=[start.slice()];gdDone=false;
    ln=start.slice();lam=1e-2;gnSteps=0;gnPath=[start.slice()];gnDone=false;
    document.getElementById('r-gd').textContent='0';
    document.getElementById('r-gn').textContent='0';
    document.getElementById('r-status').textContent='Ready. Both start at the amber dot.';
    draw();
  }
  function gdAdvance(n){var lr=0.003;for(var k=0;k<n&&!gdDone;k++){var g=grad(gd[0],gd[1]);gd[0]-=lr*g[0];gd[1]-=lr*g[1];gdSteps++;
    if(gdSteps%12===0)gdPath.push(gd.slice());
    if(f(gd[0],gd[1])<1e-3||gdSteps>=4000){gdDone=true;gdPath.push(gd.slice());}}}
  function gnAdvance(){ // one accepted LM step
    if(gnDone)return;
    var x=ln[0],y=ln[1],sb=Math.sqrt(B);
    var r0=(A-x),r1=sb*(y-x*x),j00=-1,j10=-2*sb*x,j11=sb;
    var Aa=j00*j00+j10*j10,Bb=j10*j11,Cc=j11*j11,gx=j00*r0+j10*r1,gy=j11*r1;
    for(var tries=0;tries<40;tries++){
      var A2=Aa+lam,C2=Cc+lam,det=A2*C2-Bb*Bb;
      var dx=(-gx*C2+gy*Bb)/det,dy=(gx*Bb-gy*A2)/det;
      if(f(x+dx,y+dy)<f(x,y)){ln[0]=x+dx;ln[1]=y+dy;lam=Math.max(lam*0.4,1e-9);gnSteps++;gnPath.push(ln.slice());break;}
      else lam*=3;
    }
    if(f(ln[0],ln[1])<1e-8)gnDone=true;
  }
  function path(d,pts,color,wdt){d.strokeStyle=color;d.lineWidth=wdt;d.beginPath();for(var i=0;i<pts.length;i++){var X=px(pts[i][0]),Y=py(pts[i][1]);i?d.lineTo(X,Y):d.moveTo(X,Y);}d.stroke();}
  function dot(d,p,color,rad){d.fillStyle=color;d.strokeStyle='#fff';d.lineWidth=2;d.beginPath();d.arc(px(p[0]),py(p[1]),rad,0,6.2832);d.fill();d.stroke();}
  function draw(){
    var d=cv.getContext('2d');d.setTransform(1,0,0,1,0,0);
    if(!bg||bgW!==cv.width||bgH!==cv.height)buildBG();
    d.putImageData(bg,0,0);
    d.setTransform(dpr,0,0,dpr,0,0);
    // white halos so both paths stay visible over the dark valley
    path(d,gdPath,'rgba(255,255,255,.85)',4.4);
    path(d,gnPath,'rgba(255,255,255,.9)',4.8);
    path(d,gdPath,'#5b83b0',2.4);
    path(d,gnPath,'#4e9a4a',2.6);
    dot(d,gd,'#5b83b0',4);
    dot(d,ln,'#4e9a4a',4);
    dot(d,start,'#f59e0b',4.5);
    // minimum marker (the basin floor) + label, drawn on top so it is always visible
    var mX=px(MINX),mY=py(MINY);
    d.fillStyle='#4e9a4a';d.strokeStyle='#fff';d.lineWidth=2;
    d.beginPath();d.arc(mX,mY,6,0,6.2832);d.fill();d.stroke();
    d.font='600 11px -apple-system,BlinkMacSystemFont,system-ui,sans-serif';d.textAlign='left';
    d.strokeStyle='rgba(15,17,21,.75)';d.lineWidth=3;d.strokeText('minimum',mX+10,mY+4);
    d.fillStyle='#fff';d.fillText('minimum',mX+10,mY+4);
    // legend
    d.fillStyle='rgba(15,17,21,.6)';d.fillRect(6,H-22,224,16);
    d.fillStyle='#eef1f5';d.font='10px ui-monospace,Menlo,monospace';d.textAlign='left';
    d.fillText("bird's-eye view · darker = lower cost",11,H-10);
  }
  function loop(){
    if(!racing)return;
    frames++;
    gdAdvance(22); if(!gnDone && frames%3===0) gnAdvance();
    document.getElementById('r-gd').textContent=gdSteps.toLocaleString('en-US');
    document.getElementById('r-gn').textContent=gnSteps;
    draw();
    if(gdDone&&gnDone){racing=false;finish();return;}
    anim=requestAnimationFrame(loop);
  }
  function finish(){
    document.getElementById('r-race').textContent='Race ▸';
    var gnf=gnSteps, gdf=gdSteps;
    document.getElementById('r-status').innerHTML=
      '<span style="color:var(--green)">Gauss–Newton reached the minimum in '+gnf+' steps.</span> '+
      '<span style="color:var(--steel)">Gradient descent needed '+gdf.toLocaleString('en-US')+
      (gdf>=4000?'+ and was still crawling':'')+'.</span>';
  }
  function race(){
    if(racing){racing=false;if(anim)cancelAnimationFrame(anim);document.getElementById('r-race').textContent='Race ▸';return;}
    if(gdDone&&gnDone)resetState();
    if(reduce){ // instant
      while(!gdDone)gdAdvance(200); var guard=0; while(!gnDone&&guard<60){gnAdvance();guard++;}
      document.getElementById('r-gd').textContent=gdSteps.toLocaleString('en-US');
      document.getElementById('r-gn').textContent=gnSteps; draw(); finish(); return;
    }
    racing=true;frames=0;document.getElementById('r-race').textContent='■ stop';
    document.getElementById('r-status').textContent='Racing…';
    loop();
  }
  document.getElementById('r-race').addEventListener('click',race);
  document.getElementById('r-reset').addEventListener('click',function(){racing=false;if(anim)cancelAnimationFrame(anim);document.getElementById('r-race').textContent='Race ▸';resetState();});
  cv.addEventListener('click',function(e){
    if(racing)return;
    var rect=cv.getBoundingClientRect();
    var x=XLO+((e.clientX-rect.left)/rect.width)*(XHI-XLO);
    var y=YLO+(1-(e.clientY-rect.top)/rect.height)*(YHI-YLO);
    x=Math.max(XLO+0.05,Math.min(XHI-0.05,x));y=Math.max(YLO+0.05,Math.min(YHI-0.05,y));
    start=[x,y];resetState();
  });
  var rt;window.addEventListener('resize',function(){clearTimeout(rt);rt=setTimeout(function(){var s=fit();W=s[0];H=s[1];bg=null;draw();},140);});
  var s=fit();W=s[0];H=s[1];resetState();
  setTimeout(function(){var s2=fit();W=s2[0];H=s2[1];bg=null;draw();},30);
})();
</script>
{% endraw %}

The same race on our actual two-box chemistry model, with an added nonlinear
penalty — all three reach the *same* answer, but look how long they take:

![Convergence of Adam, Gauss-Newton, and full Newton on the nonlinear chemistry penalty]({{ "/assets/figures/jax_adjoint_curvature_race.png" | relative_url }})

*Same destination, very different journeys. Adam (gradient only) needs ~180
iterations to settle. Gauss-Newton, using curvature from the residual Jacobian,
gets there in 7. Full Newton with the exact Hessian, in 9. Panel C shows why:
the curvature-aware methods (green, red) cut almost straight across the valley;
Adam (blue) zig-zags down it.*

180 iterations versus 7. And here is the part that should make a carbon-cycle
person sit up: **the curvature you compute to go fast is the same curvature you
need for error bars.** For Gaussian errors and a linear approximation near the
solution, the posterior covariance — your uncertainty on the recovered
emissions — is

```text
posterior covariance  ≈  ( Jᵀ R⁻¹ J  +  B⁻¹ )⁻¹
```

that same `JᵀJ`-shaped object, built from the prior (`B`) and observation (`R`)
error covariances. In our field a flux estimate without an uncertainty is not a
result — it is a rumour. So the method that converges fastest is *also* the one
that tells you how much to trust the answer. You rarely get to have both; here
you do.

**An aside for the inversion crowd.** That Hessian `A = JᵀR⁻¹J + B⁻¹` is doing
double duty: its inverse is *both* the posterior covariance *and* the one-step
solution, because minimising a quadratic cost is just `x* = x₀ − A⁻¹∇J`. So why
do operational CH₄/CO₂ inversions grind through hundreds of conjugate-gradient
iterations instead of solving in one shot? Because a single gradient is only
`A(x − x*)` — the Hessian times the error, not the error itself — and undoing
that needs `A⁻¹`. JAX will hand you `A` exactly (`jax.hessian`), but *forming* it
costs one adjoint pass per unknown and *inverting* it costs `N³` operations —
fine for a handful of controls, hopeless for a million-cell flux field. So at
scale you never build `A`; you use its action on vectors (matrix-free
Hessian–vector products) and let conjugate gradient rebuild `A⁻¹` one direction
at a time.

So which do you actually reach for, for smooth least-squares problems like these?

- **Default: Gauss-Newton / Levenberg-Marquardt** with autodiff Jacobians — fast,
  and it hands you the posterior covariance almost for free.
- **At scale: matrix-free Gauss-Newton** — never form the giant Jacobian; use its
  action on vectors via forward-mode (`Jv`) and reverse-mode (`Jᵀv`) products.
- **Fallback: adjoint Adam / L-BFGS** when Gauss-Newton is hard to stabilise.
- **Finite differences:** keep them only to check your fancy derivatives are right.

## The trap nobody warns you about: switches

Automatic differentiation feels like magic until it silently lies to you, and
the usual culprit is a **discontinuity**.

Atmospheric models are full of them: rain / no-rain switches in wet deposition,
threshold chemistry, hard `min`/`max` limiters, "if concentration > X then..."
branches. At the switch itself, the mathematical derivative doesn't exist, and
your framework will happily hand you the slope of whichever branch it happened to
take — a number that looks fine and points the wrong way. Higher-order
derivatives near a switch are even more treacherous.

The fixes are part of the craft, and worth internalising early:

- replace a hard `max(0, x)` with a smooth `softplus`;
- replace a hard threshold with a sigmoid ramp;
- interpolate lookup tables instead of snapping to bins;
- and for genuinely discrete decisions, step outside gradient-based methods
  entirely.

None of this is JAX-specific — a hand-written adjoint has exactly the same
disease. Differentiable programming doesn't rescue you from bad math; it just
makes the *good* math free.

## Why this is a bigger deal than a toy

The toys are toys. But the shape of the result is not.

For thirty years, the two great cultures of Earth-system modelling — the
physics-first modellers who wrote transport-and-chemistry codes, and the
statistics-first inverse modellers who wanted to fit them to data — were
separated by a moat, and the moat was the adjoint. Getting derivatives through a
physical model was so expensive that it defined what was and wasn't possible.

Differentiable programming drains the moat. When the gradient of any model you
can write is free, the boundary between "simulation" and "optimisation" dissolves.
It is the same current carrying the ML weather models — GraphCast
([Lam et al., 2023](https://doi.org/10.1126/science.adi2336)) and its cousins are
neural networks, differentiable by construction — and it flows straight back into classical inverse
problems, and into the hybrid physics-ML models now being built in between, which
need clean gradients through the physics to train at all.

The adjoint stopped being the summit. It became the base camp. And the interesting
work — the science — moved up a level, to what you point all that cheap gradient
*at*.

## Nerdy note

- **The models.** A 1-D periodic diffusion solver and a nonlinear two-box
  CH₄/CO/NOₓ chemistry-transport toy, both written in JAX and stepped with
  `jax.lax.scan`. Deliberately small and non-dimensionalised — they demonstrate
  algorithmic structure, not production chemistry.
- **The adjoint.** `jax.grad` / `jax.value_and_grad` give reverse-mode gradients
  (`JᵀJ` structure via `jax.vjp`); `jax.jvp` gives forward-mode tangent-linear
  products (`Jv`); `jax.hessian` gives the exact Hessian for small problems. The
  matrix-free Gauss-Newton route composes `jvp` then `vjp` to apply `JᵀJ` without
  ever forming it.
- **The scaling numbers.** Finite-difference vs adjoint gradient on the two-box
  model: 1.2× at 4 parameters, 33.7× at 128, and 264.8× at 1,024 — growing
  ∝ `P`, because central differences cost `2P` model runs while the adjoint costs
  one backward pass. Convergence on the nonlinear penalty: Adam 180 iterations,
  Gauss-Newton 7–8, full Newton 9, all to the identical objective.
- **The hardware.** Everything ran on a MacBook — no supercomputer — including on
  the laptop's own GPU via `jax-metal`, which peaked at ~1.96× over CPU near 2¹⁸
  grid cells (CPU won for both very small and very large grids).
- **Code, notebooks, and figures.** The full handoff — runnable scripts, two
  tutorial notebooks, benchmark CSVs, and every figure above — lives in the
  project folder accompanying this post.

![Decision matrix comparing inversion algorithms across cost, scaling, reliability, and uncertainty]({{ "/assets/figures/jax_adjoint_method_matrix.png" | relative_url }})

*The full decision matrix behind those three bullets — five algorithm families
scored on small-P speed, large-P scaling, convergence reliability, posterior
usefulness, and implementation effort.*

## Takeaway

The adjoint used to be the hardest, most artisanal part of quantitative
Earth science — a thing you built once, carefully, and then guarded. Differentiable
programming turned it into infrastructure: invisible, reliable, one line long.

That doesn't make the science easier. Deciding *what* to invert, which
observations to trust, how to regularise, where the model is lying to you — all of
that is exactly as hard as it ever was. But we no longer pay a tax of two years
and a thousand lines of hand-written derivative code just to get in the door. The
gradient is free now. Go spend it on something interesting.

---

*The JAX experiments and figures in this post were built in a scratch coding
session with an AI assistant; the full runnable handoff is linked above. More
about the author at [sudshu.github.io](https://sudshu.github.io/).*

*The views and opinions expressed in this article are those of the author alone
and do not reflect those of the JPL, NASA and CALTECH.*

## Further reading

**Adjoints and variational data assimilation**
- Errico (1997), *What is an adjoint model?* Bull. Amer. Meteor. Soc. — [doi](https://doi.org/10.1175/1520-0477(1997)078%3C2577:WIAAM%3E2.0.CO;2)
- Talagrand & Courtier (1987), *Variational assimilation with the adjoint vorticity equation.* QJRMS — [doi](https://doi.org/10.1002/qj.49711347812)
- Giering & Kaminski (1998), *Recipes for adjoint code construction.* ACM TOMS — [doi](https://doi.org/10.1145/293686.293695)

**Inverse modelling of the carbon cycle and methane**
- Enting (2002), *Inverse Problems in Atmospheric Constituent Transport.* Cambridge Univ. Press — [doi](https://doi.org/10.1017/CBO9780511535741)
- Gurney et al. (2002), *Robust regional estimates of CO₂ sources and sinks.* Nature — [doi](https://doi.org/10.1038/415626a)
- Rödenbeck et al. (2003), *CO₂ flux history 1982–2001 from a global inversion.* ACP — [doi](https://doi.org/10.5194/acp-3-1919-2003)
- Chevallier et al. (2005), *Inferring CO₂ sources and sinks from satellite observations.* JGR — [doi](https://doi.org/10.1029/2005JD006390)
- Henze et al. (2007), *Development of the adjoint of GEOS-Chem.* ACP — [doi](https://doi.org/10.5194/acp-7-2413-2007)
- Meirink et al. (2008), *4D-Var for inverse modelling of methane emissions.* ACP — [doi](https://doi.org/10.5194/acp-8-6341-2008)
- Bergamaschi et al. (2009), *Inverse modeling of CH₄ emissions using SCIAMACHY.* JGR — [doi](https://doi.org/10.1029/2009JD012287)
- Bousquet et al. (2006), *Sources of atmospheric methane variability.* Nature — [doi](https://doi.org/10.1038/nature05132)

**Remote sensing and satellite methane**
- Veefkind et al. (2012), *TROPOMI on Sentinel-5 Precursor.* Remote Sens. Environ. — [doi](https://doi.org/10.1016/j.rse.2011.09.027)
- Jacob et al. (2016), *Satellite observations of atmospheric methane.* ACP — [doi](https://doi.org/10.5194/acp-16-14371-2016)

**My own, in this lineage**
- Pandey et al. (2015), *On the use of satellite-derived CH₄:CO₂ columns in a joint inversion of CH₄ and CO₂ fluxes* — the first paper of my PhD. ACP — [doi](https://doi.org/10.5194/acp-15-8615-2015)
- Pandey et al. (2019), *Satellite detection of an extreme methane well blowout.* PNAS — [doi](https://doi.org/10.1073/pnas.1908712116)
- Pandey et al. (2022), *Order-of-magnitude speed-up of variational methane inversions.* GMD — [doi](https://doi.org/10.5194/gmd-15-4555-2022)

**Differentiable ML weather models**
- Lam et al. (2023), *Learning skillful medium-range global weather forecasting (GraphCast).* Science — [doi](https://doi.org/10.1126/science.adi2336)
