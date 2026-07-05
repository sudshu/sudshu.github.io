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

{% raw %}
<section class="iw-card" id="iw-fit">
  <p class="iw-eyebrow">Interactive · what this whole post is about</p>
  <p class="iw-title">Let the gradient fit the curve</p>
  <p class="iw-note">The dots are data, blurred by a hidden amount <b>κ</b>. The amber curve is a model that
  starts with the <em>wrong</em> κ — too sharp — and the shaded band is how badly it misses. Press
  <em>Solve</em> (or just watch) and a gradient walks κ until the curve lands on the dots and the mismatch vanishes.</p>
  <div class="iw-panel"><canvas class="iw" id="fit-canvas"></canvas></div>
  <div class="iw-controls">
    <div class="iw-sliderbox">
      <label>blur strength κ <b id="fit-kval">0.0035</b></label>
      <input type="range" class="iw" id="fit-slider" min="0" max="1" step="0.001" value="0.10">
    </div>
    <button class="iw primary" id="fit-solve">Solve ▸</button>
    <button class="iw" id="fit-new">New data</button>
  </div>
  <p class="iw-stat" id="fit-status" style="margin-top:12px; color:var(--muted);">mismatch J = — · step 0</p>
  <p class="iw-caption">That is the entire inverse problem in one picture: a wrong model, some data, and a
  gradient that knows which way to nudge the unknown. This post is about where that gradient comes from — and why
  the same one line of code (<code>jax.grad</code>) works whether there is one unknown here or a million in a
  real satellite flux inversion.</p>
</section>
<script>
(function(){
  var N=110, M=64, SCALE=9;
  function u0(){var u=new Float64Array(N),i;for(i=0;i<N;i++){var x=i/N;
    u[i]=0.95*Math.exp(-(Math.pow((x-0.34)/0.07,2)))+0.62*Math.exp(-(Math.pow((x-0.66)/0.05,2)));}return u;}
  var U0=u0();
  function forward(k){var d=SCALE*k,u=U0.slice(),t,i,nu;
    for(t=0;t<M;t++){nu=new Float64Array(N);for(i=0;i<N;i++){var l=u[(i-1+N)%N],r=u[(i+1)%N];nu[i]=u[i]+d*(l-2*u[i]+r);}u=nu;}return u;}
  var KMIN=0.002,KMAX=0.05;
  function s2k(s){return KMIN*Math.pow(KMAX/KMIN,s);}
  function k2s(k){return Math.log(k/KMIN)/Math.log(KMAX/KMIN);}
  var kTrue,uobs,kGuess,step=0,converged=false,anim=null,playing=false,userTouched=false;
  function lossOf(k){var u=forward(k),s=0,i;for(i=0;i<N;i++){var e=u[i]-uobs[i];s+=e*e;}return s/N;}
  var cv=document.getElementById('fit-canvas');
  var _dpr=Math.min(window.devicePixelRatio||1,2);
  function fit(h){var w=cv.clientWidth||600,W=Math.round(w*_dpr),Hh=Math.round(h*_dpr);
    if(cv.width!==W||cv.height!==Hh){cv.width=W;cv.height=Hh;cv.style.height=h+'px';}
    var c=cv.getContext('2d');c.setTransform(_dpr,0,0,_dpr,0,0);return{c:c,w:w,h:h};}
  function trace(c,arr,gx,gy){c.beginPath();for(var i=0;i<arr.length;i++){var X=gx(i),Y=gy(arr[i]);i?c.lineTo(X,Y):c.moveTo(X,Y);}}
  function draw(){
    if((cv.clientWidth||0)<20)return;
    var o=fit(224),c=o.c,w=o.w,h=o.h,padL=12,padR=12,padT=18,padB=14,i;
    c.clearRect(0,0,w,h);
    var yMax=1.05,gx=function(i){return padL+(i/(N-1))*(w-padL-padR);},gy=function(v){return padT+(1-v/yMax)*(h-padT-padB);};
    c.strokeStyle='#e2e6ea';c.lineWidth=1;c.beginPath();c.moveTo(padL,h-padB);c.lineTo(w-padR,h-padB);c.stroke();
    var ug=forward(kGuess),col=converged?'#2f6b2c':'#d97706';
    c.fillStyle=converged?'rgba(47,107,44,0.10)':'rgba(217,119,6,0.16)';
    c.beginPath();
    for(i=0;i<N;i++){var Xa=gx(i),Ya=gy(ug[i]);i?c.lineTo(Xa,Ya):c.moveTo(Xa,Ya);}
    for(i=N-1;i>=0;i--){c.lineTo(gx(i),gy(uobs[i]));}
    c.closePath();c.fill();
    c.fillStyle='#1b1d1f';for(i=0;i<N;i+=3){c.beginPath();c.arc(gx(i),gy(uobs[i]),2.0,0,6.2832);c.fill();}
    c.lineJoin='round';
    if(converged){c.strokeStyle='rgba(47,107,44,0.22)';c.lineWidth=8;trace(c,ug,gx,gy);c.stroke();}
    c.strokeStyle=col;c.lineWidth=2.8;trace(c,ug,gx,gy);c.stroke();
    c.font='11px ui-monospace,Menlo,monospace';c.textAlign='left';
    c.fillStyle='#1b1d1f';c.fillText('● data',padL+2,padT-4);
    c.fillStyle=col;c.fillText('— model  κ='+kGuess.toFixed(4),padL+56,padT-4);
    c.textAlign='right';
    if(converged){c.fillStyle='#2f6b2c';c.font='600 12px -apple-system,BlinkMacSystemFont,system-ui,sans-serif';c.fillText('✓ fitted',w-padR-2,padT-4);}
    else{c.fillStyle='#b0741f';c.fillText('shaded = mismatch',w-padR-2,padT-4);}
  }
  function setStatus(){
    var st=document.getElementById('fit-status'),J=lossOf(kGuess);
    document.getElementById('fit-kval').textContent=kGuess.toFixed(4);
    if(converged) st.innerHTML='✓ fitted — recovered κ = <b style="color:var(--green)">'+kGuess.toFixed(4)+
      '</b> (true κ = '+kTrue.toFixed(4)+') in '+step+' gradient steps · <b>New data</b> to try again';
    else st.innerHTML='mismatch J = <b style="color:var(--ink)">'+J.toExponential(2)+'</b> · step '+step+' · press <b>Solve</b> or drag κ';
  }
  function syncSlider(){document.getElementById('fit-slider').value=k2s(kGuess).toFixed(3);}
  function stop(){playing=false;if(anim){cancelAnimationFrame(anim);anim=null;}document.getElementById('fit-solve').textContent='Solve ▸';}
  function newData(){
    stop();
    kTrue=0.020+Math.random()*0.020;
    var base=forward(kTrue),mx=0,i;for(i=0;i<N;i++) if(base[i]>mx)mx=base[i];
    uobs=new Float64Array(N);for(i=0;i<N;i++) uobs[i]=base[i]+(Math.random()-0.5)*0.004*mx;
    kGuess=0.0035;step=0;converged=false;setStatus();syncSlider();draw();
  }
  function newtonPath(){
    var path=[kGuess],theta=Math.log(kGuess),hh=1e-3,it;
    for(it=0;it<12;it++){
      var Jm=lossOf(Math.exp(theta-hh)),J0=lossOf(Math.exp(theta)),Jp=lossOf(Math.exp(theta+hh));
      var g=(Jp-Jm)/(2*hh),Hs=(Jp-2*J0+Jm)/(hh*hh),stp=-g/Math.max(Math.abs(Hs),1e-12);
      var lr=1.0,nt,nJ;
      for(var bt=0;bt<24;bt++){nt=theta+lr*stp;nJ=lossOf(Math.exp(nt));if(nJ<=J0)break;lr*=0.5;}
      theta=nt;path.push(Math.exp(theta));
      if(Math.abs(g)<2e-6||nJ<1e-11)break;
    }
    return path;
  }
  var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion:reduce)').matches;
  function solve(){
    if(playing){stop();return;}
    if(converged){newData();}
    var path=newtonPath();
    if(reduce){kGuess=path[path.length-1];step=path.length-1;converged=true;setStatus();syncSlider();draw();return;}
    playing=true;document.getElementById('fit-solve').textContent='■ stop';
    var seg=0,TW=36,f=0;  // 36 frames/step ≈ 3× slower than before
    function ease(t){return t<0.5?2*t*t:1-Math.pow(-2*t+2,2)/2;}
    function frame(){
      if(!playing)return;
      if(seg>=path.length-1){kGuess=path[path.length-1];converged=true;step=path.length-1;stop();setStatus();syncSlider();draw();return;}
      var a=Math.log(path[seg]),b=Math.log(path[seg+1]),t=ease(f/TW);
      kGuess=Math.exp(a+(b-a)*t);step=seg;
      setStatus();syncSlider();draw();
      f++;if(f>TW){f=0;seg++;}
      anim=requestAnimationFrame(frame);
    }
    frame();
  }
  var _upd=false;
  function requestUpdate(){if(_upd)return;_upd=true;requestAnimationFrame(function(){_upd=false;setStatus();draw();});}
  document.getElementById('fit-slider').addEventListener('input',function(e){userTouched=true;stop();converged=false;kGuess=s2k(parseFloat(e.target.value));step=0;requestUpdate();});
  document.getElementById('fit-solve').addEventListener('click',function(){userTouched=true;solve();});
  document.getElementById('fit-new').addEventListener('click',function(){userTouched=true;newData();});
  var rt;window.addEventListener('resize',function(){clearTimeout(rt);rt=setTimeout(draw,120);});
  if(window.ResizeObserver){var ro=new ResizeObserver(function(){draw();});ro.observe(cv.parentNode);}
  var autoPlayed=false;
  function maybeAuto(){if(autoPlayed)return;autoPlayed=true;setTimeout(function(){if(!userTouched&&!playing&&!converged)solve();},900);}
  if(window.IntersectionObserver){var io=new IntersectionObserver(function(es){for(var k=0;k<es.length;k++){if(es[k].isIntersecting){maybeAuto();io.disconnect();break;}}},{threshold:0.35});io.observe(cv);}else maybeAuto();
  newData();
  setTimeout(draw,30);setTimeout(draw,250);setTimeout(draw,800);
})();
</script>
{% endraw %}


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

Here is that trade-off, measured on a toy chemistry model — finite differences
climb with every unknown you add, while the adjoint stays almost flat:

![Finite-difference cost versus the adjoint gradient, as the number of unknowns grows]({{ "/assets/figures/jax_adjoint_gradient_scaling.png" | relative_url }})

*Finite differences re-run the whole model twice per unknown; the adjoint gets the
entire gradient in one backward pass, for roughly the same price no matter how
many unknowns there are. By 1,000 it is ~265× cheaper — and the gap keeps widening.*

At a hundred thousand parameters, that gap is the difference between "runs
overnight" and "never finishes."

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
  <p class="iw-caption">Watch the steel path: gradient descent only knows which way is <em>locally</em> downhill,
  so it first dives to the nearest point on the valley floor — down-left, <em>away</em> from the minimum — and
  then has to crawl thousands of steps back along the floor. Gauss–Newton (green) also knows how the valley
  <em>bends</em>, so it ignores that plunge and cuts almost straight to the answer in about a dozen steps.
  Same destination — curvature is what pays for the shortcut.</p>
</section>
<script>
(function(){
  var A=1,B=40,MINX=1,MINY=1;
  var XLO=-1.7,XHI=1.7,YLO=-0.7,YHI=2.3;
  function f(x,y){return (A-x)*(A-x)+B*(y-x*x)*(y-x*x);}
  function grad(x,y){return [-2*(A-x)-4*B*x*(y-x*x), 2*B*(y-x*x)];}
  var cv=document.getElementById('r-canvas'),bg=null,bgW=0,bgH=0,dpr=Math.min(window.devicePixelRatio||1,2);
  var start=[-0.6,2.0];  // deliberately off the valley floor: steepest descent dives to the
                         // nearest floor point (down-left, AWAY from the minimum) before crawling back
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
    if(gdSteps<=50||gdSteps%12===0)gdPath.push(gd.slice());  // sample finely early to show the plunge
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
    gdAdvance(gdSteps<40?2:26); if(!gnDone && frames%3===0) gnAdvance();  // slow during the dive, fast on the crawl
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
- **Same code, any hardware.** The forward model and its adjoint are one JAX
  source; XLA JIT-compiles them to CPU, GPU, or TPU with no changes. Exact
  derivatives *and* accelerator speed from one codebase is the combination that
  makes large inversions practical.
- **Not a JAX-only story.** The same free-adjoint idea runs through Julia's SciML
  ecosystem — where adjoint sensitivity analysis differentiates through ODE/PDE
  solvers directly — as well as PyTorch and TensorFlow. JAX is one dialect of a
  broader shift to differentiable programming.

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
session with an AI assistant. More about the author at
[sudshu.github.io](https://sudshu.github.io/).*

*The views and opinions expressed in this article are those of the author alone
and do not reflect those of the JPL, NASA and CALTECH.*

## Further reading

**Adjoints and variational data assimilation**
- Errico (1997), *What is an adjoint model?* Bull. Amer. Meteor. Soc. — [doi](https://doi.org/10.1175/1520-0477(1997)078%3C2577:WIAAM%3E2.0.CO;2)
- Talagrand & Courtier (1987), *Variational assimilation of meteorological observations with the adjoint vorticity equation. I: Theory.* QJRMS — [doi](https://doi.org/10.1002/qj.49711347812)
- Giering & Kaminski (1998), *Recipes for adjoint code construction.* ACM TOMS — [doi](https://doi.org/10.1145/293686.293695)

**Inverse modelling of the carbon cycle and methane**
- Enting (2002), *Inverse Problems in Atmospheric Constituent Transport.* Cambridge Univ. Press — [doi](https://doi.org/10.1017/CBO9780511535741)
- Gurney et al. (2002), *Towards robust regional estimates of CO₂ sources and sinks using atmospheric transport models.* Nature — [doi](https://doi.org/10.1038/415626a)
- Rödenbeck et al. (2003), *CO₂ flux history 1982–2001 inferred from atmospheric data using a global inversion of atmospheric transport.* ACP — [doi](https://doi.org/10.5194/acp-3-1919-2003)
- Chevallier et al. (2005), *Inferring CO₂ sources and sinks from satellite observations: method and application to TOVS data.* JGR — [doi](https://doi.org/10.1029/2005JD006390)
- Henze et al. (2007), *Development of the adjoint of GEOS-Chem.* ACP — [doi](https://doi.org/10.5194/acp-7-2413-2007)
- Meirink et al. (2008), *Four-dimensional variational data assimilation for inverse modelling of atmospheric methane emissions: method and comparison with synthesis inversion.* ACP — [doi](https://doi.org/10.5194/acp-8-6341-2008)
- Bergamaschi et al. (2009), *Inverse modeling of global and regional CH₄ emissions using SCIAMACHY satellite retrievals.* JGR — [doi](https://doi.org/10.1029/2009JD012287)
- Bousquet et al. (2006), *Contribution of anthropogenic and natural sources to atmospheric methane variability.* Nature — [doi](https://doi.org/10.1038/nature05132)

**Remote sensing and satellite methane**
- Veefkind et al. (2012), *TROPOMI on the ESA Sentinel-5 Precursor: a GMES mission for global observations of the atmospheric composition for climate, air quality and ozone layer applications.* Remote Sens. Environ. — [doi](https://doi.org/10.1016/j.rse.2011.09.027)
- Jacob et al. (2016), *Satellite observations of atmospheric methane and their value for quantifying methane emissions.* ACP — [doi](https://doi.org/10.5194/acp-16-14371-2016)

**My own, in this lineage**
- Pandey et al. (2015), *On the use of satellite-derived CH₄:CO₂ columns in a joint inversion of CH₄ and CO₂ fluxes* — the first paper of my PhD. ACP — [doi](https://doi.org/10.5194/acp-15-8615-2015)
- Pandey et al. (2019), *Satellite observations reveal extreme methane leakage from a natural gas well blowout.* PNAS — [doi](https://doi.org/10.1073/pnas.1908712116)
- Pandey et al. (2022), *Order of magnitude wall time improvement of variational methane inversions by physical parallelization: a demonstration using TM5-4DVAR.* GMD — [doi](https://doi.org/10.5194/gmd-15-4555-2022)

**Differentiable ML weather models**
- Lam et al. (2023), *Learning skillful medium-range global weather forecasting (GraphCast).* Science — [doi](https://doi.org/10.1126/science.adi2336)
