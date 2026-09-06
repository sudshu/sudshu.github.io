---
layout: home
list_title: Recent posts
---

I'm a scientist at **NASA's Jet Propulsion Laboratory**. My research combines
satellite observations, atmospheric modeling, and inverse methods to understand
the carbon cycle and the processes shaping Earth's atmosphere and climate.

<section class="research-highlights" aria-labelledby="recent-preprints">
  <h2 id="recent-preprints">Recent preprints</h2>
  <article class="research-highlight" aria-labelledby="dynamics-preprint-title">
    <p class="research-highlight-meta">Preprint · 26 August 2026 · Sudhanshu Pandey</p>
    <h3 id="dynamics-preprint-title"><a href="https://doi.org/10.21203/rs.3.rs-10781568/v1">Recovering atmospheric dynamics from atmospheric composition snapshots using machine learning</a></h3>
    <p>Tests whether a single atmospheric-composition scene contains recoverable information about winds and boundary-layer height, using simulations and TEMPO satellite observations.</p>
    <p class="research-highlight-links"><a href="https://doi.org/10.21203/rs.3.rs-10781568/v1">Read preprint</a> <span aria-hidden="true">·</span> <a href="https://github.com/sudshu/compass-repro">Code and reproducibility package</a></p>
  </article>
  <article class="research-highlight" aria-labelledby="growth-preprint-title">
    <p class="research-highlight-meta">Preprint · 9 June 2026 · Sudhanshu Pandey et al.</p>
    <h3 id="growth-preprint-title"><a href="https://doi.org/10.21203/rs.3.rs-9854768/v1">Time-varying errors in the atmospheric CO₂ growth rate</a></h3>
    <p>Examines how changes in observing-network coverage affect annual CO₂ growth estimates and their interpretation in the global carbon budget.</p>
    <p class="research-highlight-links"><a href="https://doi.org/10.21203/rs.3.rs-9854768/v1">Read preprint</a></p>
  </article>
</section>

## About

Building on this foundation, I develop AI and machine learning approaches for
Earth-system inference and scientific discovery. I'm particularly interested in
how we can extract physical understanding from complex and incomplete
observations. My goal is to combine physics, learning from data, and uncertainty
quantification to uncover underlying processes, test scientific explanations,
and improve our understanding of a changing Earth.

I also explore how agentic AI can advance the research process, from formulating
hypotheses and designing experiments to evaluating evidence. Across these
efforts, I'm interested in developing methods that help us ask new scientific
questions and answer them rigorously.

My broader interests include the philosophy of science and artificial
intelligence: what constitutes scientific understanding, how prediction relates
to explanation, and how AI is reshaping science and society.

{% raw %}
<style>
.iw-card.iw-transport{margin:1.6rem 0;border:1px solid #e2e6ea;border-radius:10px;background:#f6f8fa;overflow:hidden}
.iw-transport .iw-transport-head{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;padding:.55rem .8rem .4rem;font-size:.78rem;color:#68707a;flex-wrap:wrap}
.iw-transport .iw-transport-title{font-weight:600;color:#3b4148;letter-spacing:.01em}
.iw-transport .iw-transport-hint{font-style:italic}
.iw-transport canvas{display:block;width:100%;cursor:crosshair;touch-action:none}
.iw-transport .iw-transport-cap{margin:0;padding:.5rem .8rem .65rem;font-size:.78rem;line-height:1.35;color:#68707a}
</style>
<section class="iw-card iw-transport" id="iw-transport-card" aria-label="Interactive atmospheric tracer transport">
<div class="iw-transport-head"><span class="iw-transport-title">Atmospheric tracer transport — a toy</span><span class="iw-transport-hint">move to steer the wind · click to place the source</span></div>
<canvas id="iw-transport-canvas"></canvas>
<p class="iw-transport-cap">A puff of tracer advecting and spreading by turbulent diffusion in a wind field — a cartoon of the atmospheric transport that turns emissions into what a satellite sees.</p>
</section>
<script>
(function(){
  var card = document.getElementById('iw-transport-card');
  var cv = document.getElementById('iw-transport-canvas');
  if (!card || !cv || !cv.getContext) return;
  var ctx = cv.getContext('2d');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var W = 0, H = 0, dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  var parts = [];
  var src = { x: 0, y: 0 };
  var wind = { u: 1.4, v: 0 };
  var mouse = { x: null, y: null, inside: false };
  var params = { turb: 0.6, maxAge: 120, emit: 5 };
  var running = false, raf = 0;
  var spriteColors = ['#fff3bf','#ffe066','#ffd43b','#ffa94d','#ff922b','#f76707','#e8590c','#d9480f'];
  var sprites = [];
  function buildSprites(){
    sprites = spriteColors.map(function(col){
      var r = 22, c = document.createElement('canvas');
      c.width = c.height = r * 2;
      var g = c.getContext('2d');
      var grd = g.createRadialGradient(r, r, 0, r, r, r);
      grd.addColorStop(0, col);
      grd.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = grd;
      g.beginPath(); g.arc(r, r, r, 0, Math.PI * 2); g.fill();
      return c;
    });
  }
  function randn(){
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function resize(){
    var w = Math.max(240, card.clientWidth);
    var h = Math.round(Math.min(260, Math.max(180, w * 0.42)));
    if (w === W && h === H) return;
    W = w; H = h;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (src.x === 0) { src.x = 0.18 * W; src.y = 0.52 * H; }
    if (reduce) staticFrame();
  }
  function setWindFromMouse(){
    if (mouse.inside && mouse.x !== null) {
      var dx = mouse.x - src.x, dy = mouse.y - src.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
      var sp = Math.max(0.6, Math.min(2.2, d / 90));
      wind.u = dx / d * sp; wind.v = dy / d * sp;
    } else { wind.u = 1.4; wind.v = 0; }
  }
  function step(){
    for (var i = 0; i < params.emit; i++) parts.push({ x: src.x + randn() * 3, y: src.y + randn() * 3, age: 0 });
    var next = [];
    for (var j = 0; j < parts.length; j++) {
      var p = parts[j];
      p.x += wind.u + randn() * params.turb;
      p.y += wind.v + randn() * params.turb * 0.9;
      p.age++;
      if (p.age <= params.maxAge && p.x > -24 && p.x < W + 24 && p.y > -24 && p.y < H + 24) next.push(p);
    }
    parts = next;
  }
  function drawWind(){
    ctx.strokeStyle = 'rgba(140,150,160,0.32)';
    ctx.fillStyle = 'rgba(140,150,160,0.45)';
    ctx.lineWidth = 1;
    var gx = 62, gy = 54, len = 9;
    var m = Math.sqrt(wind.u * wind.u + wind.v * wind.v) || 1;
    var ux = wind.u / m, uy = wind.v / m;
    for (var x = gx * 0.6; x < W; x += gx) {
      for (var y = gy * 0.6; y < H; y += gy) {
        ctx.beginPath(); ctx.moveTo(x - ux * len, y - uy * len); ctx.lineTo(x + ux * len, y + uy * len); ctx.stroke();
        ctx.beginPath(); ctx.arc(x + ux * len, y + uy * len, 1.3, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  function drawSource(){
    ctx.save(); ctx.translate(src.x, src.y);
    var s = 5;
    ctx.strokeStyle = 'rgba(20,22,25,0.85)'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(-s, 0); ctx.lineTo(s, 0); ctx.moveTo(0, -s); ctx.lineTo(0, s); ctx.stroke();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(-s, 0); ctx.lineTo(s, 0); ctx.moveTo(0, -s); ctx.lineTo(0, s); ctx.stroke();
    ctx.restore();
  }
  function render(){
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f6f8fa'; ctx.fillRect(0, 0, W, H);
    drawWind();
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      var f = p.age / params.maxAge;
      var idx = Math.min(sprites.length - 1, Math.floor(f * sprites.length));
      var rad = 8 + f * 16;
      ctx.globalAlpha = 0.12 * (1 - f * 0.5);
      ctx.drawImage(sprites[idx], p.x - rad, p.y - rad, rad * 2, rad * 2);
    }
    ctx.globalAlpha = 1;
    drawSource();
  }
  function frame(){
    if (!running) return;
    setWindFromMouse(); step(); render();
    raf = requestAnimationFrame(frame);
  }
  function start(){ if (running || reduce) return; running = true; raf = requestAnimationFrame(frame); }
  function stop(){ running = false; if (raf) cancelAnimationFrame(raf); }
  function staticFrame(){ parts = []; for (var t = 0; t < 170; t++) { setWindFromMouse(); step(); } render(); }
  cv.addEventListener('mousemove', function(e){ var r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; mouse.inside = true; });
  cv.addEventListener('mouseleave', function(){ mouse.inside = false; mouse.x = null; mouse.y = null; });
  cv.addEventListener('click', function(e){ var r = cv.getBoundingClientRect(); src.x = e.clientX - r.left; src.y = e.clientY - r.top; for (var i = 0; i < 80; i++) parts.push({ x: src.x + randn() * 4, y: src.y + randn() * 4, age: Math.floor(Math.random() * 10) }); if (reduce) staticFrame(); });
  cv.addEventListener('touchmove', function(e){ var r = cv.getBoundingClientRect(), t = e.touches[0]; mouse.x = t.clientX - r.left; mouse.y = t.clientY - r.top; mouse.inside = true; e.preventDefault(); }, { passive: false });
  cv.addEventListener('touchend', function(){ mouse.inside = false; });
  if (window.ResizeObserver) { new ResizeObserver(function(){ resize(); }).observe(card); }
  else { window.addEventListener('resize', resize); }
  buildSprites(); resize();
  if (reduce) { staticFrame(); }
  else if (window.IntersectionObserver) {
    new IntersectionObserver(function(es){ es.forEach(function(en){ if (en.isIntersecting) start(); else stop(); }); }, { threshold: 0.12 }).observe(card);
  } else { start(); }
})();
</script>
{% endraw %}

## Background

I did my **PhD in physics at Utrecht University** (2017) and an integrated
BS-MS in Earth Sciences at IISER Kolkata. Before JPL I spent nearly six years
at SRON, the Netherlands Institute for Space Research. My work is supported by
a NASA Early Career Investigator Program award (2023).

My earlier work includes satellite detection of major methane leaks
([PNAS, 2019](https://doi.org/10.1073/pnas.1908712116)) and methods to estimate
whole-atmosphere CO₂ growth and clarify the global carbon budget
([AGU Advances, 2024](https://doi.org/10.1029/2023AV001145);
[Nature Communications, 2025](https://doi.org/10.1038/s41467-025-61588-2);
[AGU Advances, 2025](https://doi.org/10.1029/2025AV002085)).

## Selected publications

Full list in my [CV](https://science.jpl.nasa.gov/documents/1475/CV_Sudhanshu_Pandey_20260415.pdf)
or [Google Scholar](https://scholar.google.com/citations?user=efFF_TEAAAAJ&hl=en).

- Worden, J., **Pandey, S.**, et al. (2026). Top-down benchmark of US methane
  inventories reveals regional discrepancies in activity-based estimates.
  *Atmospheric Chemistry and Physics* 26, 8855–8873.
  [doi:10.5194/acp-26-8855-2026](https://doi.org/10.5194/acp-26-8855-2026)
- Dasgupta, B., **Pandey, S.**, et al. (2026). Global methane emission estimates
  from a dual-isotope inversion: new constraints from δD-CH₄.
  *Atmospheric Chemistry and Physics* 26, 8601–8616.
  [doi:10.5194/acp-26-8601-2026](https://doi.org/10.5194/acp-26-8601-2026)
- Friedlingstein, P., et al. (incl. **Pandey, S.**) (2026). Global Carbon
  Budget 2025. *Earth System Science Data* 18, 3211–3288.
  [doi:10.5194/essd-18-3211-2026](https://doi.org/10.5194/essd-18-3211-2026)
- **Pandey, S.** (2025). Taking Earth's carbon pulse from space.
  *AGU Advances.*
  [doi:10.1029/2025AV002085](https://doi.org/10.1029/2025AV002085)
- **Pandey, S.**, et al. (2025). Reduction in Earth's carbon budget imbalance.
  *Nature Communications* 16, 6818.
  [doi:10.1038/s41467-025-61588-2](https://doi.org/10.1038/s41467-025-61588-2)
- **Pandey, S.**, et al. (2025). Relating multi-scale plume detection and area
  estimates of methane emissions: a theoretical and empirical analysis.
  *Environmental Science & Technology* 59, 7931–7947.
  [doi:10.1021/acs.est.4c07415](https://doi.org/10.1021/acs.est.4c07415)
- **Pandey, S.**, et al. (2024). Toward low-latency estimation of atmospheric
  CO₂ growth rates using satellite observations: evaluating sampling errors of
  satellite and in situ observing approaches. *AGU Advances* 5, e2023AV001145.
  [doi:10.1029/2023AV001145](https://doi.org/10.1029/2023AV001145)
- **Pandey, S.**, et al. (2023). Daily detection and quantification of methane
  leaks using Sentinel-3: a tiered satellite observation approach with
  Sentinel-2 and Sentinel-5P. *Remote Sensing of Environment* 296, 113716.
  [doi:10.1016/j.rse.2023.113716](https://doi.org/10.1016/j.rse.2023.113716)
- **Pandey, S.**, et al. (2022). Order-of-magnitude wall-time improvement of
  variational methane inversions by physical parallelization: a demonstration
  using TM5-4DVAR. *Geoscientific Model Development* 15, 4555–4567.
  [doi:10.5194/gmd-15-4555-2022](https://doi.org/10.5194/gmd-15-4555-2022)
- **Pandey, S.**, et al. (2019). Satellite observations reveal extreme methane
  leakage from a natural gas well blowout.
  *PNAS* 116(52), 26376–26381.
  [doi:10.1073/pnas.1908712116](https://doi.org/10.1073/pnas.1908712116)
- **Pandey, S.**, et al. (2017). Enhanced methane emissions from tropical
  wetlands during the 2011 La Niña.
  *Scientific Reports* 7, 45759.
  [doi:10.1038/srep45759](https://doi.org/10.1038/srep45759)

## Find me elsewhere

- 🌐 [JPL profile](https://science.jpl.nasa.gov/people/pandeysu/)
- 🎓 [Google Scholar](https://scholar.google.com/citations?user=efFF_TEAAAAJ&hl=en)
- 💻 [GitHub: @sudshu](https://github.com/sudshu)
- 📄 [CV (PDF)](https://science.jpl.nasa.gov/documents/1475/CV_Sudhanshu_Pandey_20260415.pdf)
- ✉️ sudhanshu.pandey [at] jpl.nasa.gov

*Views and weekend experiments here are my own and don't represent NASA or
JPL.*

---
