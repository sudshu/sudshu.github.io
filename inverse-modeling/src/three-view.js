import {
  WebGLRenderer, Scene, OrthographicCamera, BufferGeometry, Float32BufferAttribute,
  BufferAttribute, Points, ShaderMaterial, LineSegments, LineBasicMaterial,
  Color, Raycaster, Vector2, SRGBColorSpace
} from 'three';

// The posterior is computed by the teaching model; this module only displays it.
export function createPosterior3D({canvas, redraw, rendererFactory}) {
  const renderer = rendererFactory ? rendererFactory(canvas) : new WebGLRenderer({
    canvas, alpha: true, antialias: true, powerPreference: 'default'
  });
  renderer.setClearColor(0x050c15, 0);
  renderer.outputColorSpace = SRGBColorSpace;
  const scene = new Scene();
  const camera = new OrthographicCamera(-3, 3, 3, -3, .1, 30);
  camera.up.set(0, 0, 1);
  const geometry = new BufferGeometry();
  const material = new ShaderMaterial({
    transparent: true, depthWrite: false, depthTest: true,
    uniforms: {pixelRatio: {value: 1}, pointSize: {value: 4}},
    vertexShader: `
      attribute vec3 densityColor;
      attribute float support;
      varying vec3 vColor;
      varying float vSupport;
      uniform float pixelRatio;
      uniform float pointSize;
      void main() {
        vColor = densityColor;
        vSupport = support;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = pointSize * pixelRatio;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      varying float vSupport;
      void main() {
        float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
        if (r > 1.0) discard;
        float edge = 1.0 - smoothstep(0.72, 1.0, r);
        gl_FragColor = vec4(vColor, (0.12 + 0.72 * vSupport) * edge);
        #include <colorspace_fragment>
      }`
  });
  const points = new Points(geometry, material);
  scene.add(points);
  const wireVertices = [];
  const corners = [];
  for (const z of [-1, 1]) for (const y of [-1, 1]) for (const x of [-1, 1]) corners.push([x, y, z]);
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    if ([1, 2, 4].includes(i ^ j)) wireVertices.push(...corners[i], ...corners[j]);
  }
  const wireGeometry = new BufferGeometry();
  wireGeometry.setAttribute('position', new Float32BufferAttribute(wireVertices, 3));
  const wireMaterial = new LineBasicMaterial({color: 0x718ca9, transparent: true, opacity: .65, depthWrite: false});
  const wire = new LineSegments(wireGeometry, wireMaterial);
  wire.renderOrder = -1;
  scene.add(wire);
  const raycaster = new Raycaster(), pointer = new Vector2(), convertedColor = new Color();
  let source = null, ordered = null, cutoff = null, vertices = new Float32Array(),
      indices = new Uint32Array(), depths = new Float32Array(), width = 1, height = 1,
      scale = 1, yaw = NaN, pitch = NaN, lastDpr = 0, available = true;

  function rebuild(cloud, orderedValue, level, rgb) {
    const visible = cloud.filter(p => p.d <= level && (!orderedValue || p.u[0] <= p.u[1]));
    // A bounded display budget; the model still evaluates the complete posterior.
    const stride = Math.max(1, Math.ceil(visible.length / 24000));
    const count = Math.ceil(visible.length / stride);
    vertices = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3), support = new Float32Array(count);
    indices = new Uint32Array(count); depths = new Float32Array(count);
    for (let i = 0, j = 0; i < visible.length; i += stride, j++) {
      const p = visible[i], c = rgb(p.d);
      vertices.set(p.u, j * 3);
      convertedColor.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, SRGBColorSpace);
      colors.set([convertedColor.r, convertedColor.g, convertedColor.b], j * 3);
      support[j] = Math.exp(-p.d / 2); indices[j] = j;
    }
    // Release old GPU buffers when changing measurements or parameter ordering.
    geometry.dispose();
    geometry.setAttribute('position', new BufferAttribute(vertices, 3));
    geometry.setAttribute('densityColor', new BufferAttribute(colors, 3));
    geometry.setAttribute('support', new BufferAttribute(support, 1));
    geometry.setIndex(new BufferAttribute(indices, 1));
    geometry.computeBoundingSphere();
    source = cloud; ordered = orderedValue; cutoff = level; yaw = NaN;
  }

  function render({cloud, ordered: order, level, rgb, view, pose, pixelsPerUnit}) {
    if (!available) return false;
    if (source !== cloud || ordered !== order || cutoff !== level) rebuild(cloud, order, level, rgb);
    const dpr = Math.min(view.dpr || 1, view.w < 721 ? 1.5 : 2);
    if (width !== view.w || height !== view.h || lastDpr !== dpr) {
      width = Math.max(1, view.w); height = Math.max(1, view.h); lastDpr = dpr;
      renderer.setPixelRatio(dpr); renderer.setSize(width, height, false);
    }
    scale = pixelsPerUnit;
    camera.left = -width / (2 * scale); camera.right = width / (2 * scale);
    camera.top = height * .49 / scale; camera.bottom = -height * .51 / scale;
    camera.updateProjectionMatrix();
    const sy = Math.sin(pose.yaw), cy = Math.cos(pose.yaw), sp = Math.sin(pose.pitch), cp = Math.cos(pose.pitch);
    camera.position.set(sy * cp * 8, -cy * cp * 8, -sp * 8);
    camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    if (yaw !== pose.yaw || pitch !== pose.pitch) {
      // Transparent POINTS need per-point sorting; Three.js sorts objects only.
      for (let i = 0; i < indices.length; i++) depths[i] = -sy * cp * vertices[3*i] + cy * cp * vertices[3*i+1] + sp * vertices[3*i+2];
      indices.sort((a, b) => depths[b] - depths[a]);
      geometry.index.needsUpdate = true; yaw = pose.yaw; pitch = pose.pitch;
    }
    material.uniforms.pixelRatio.value = dpr;
    material.uniforms.pointSize.value = Math.max(2.6, Math.min(5.8, scale * .035));
    renderer.render(scene, camera);
    canvas.style.display = 'block';
    return true;
  }

  function pick(x, y, radius = 12) {
    if (!available || !vertices.length) return null;
    pointer.set(2 * x / width - 1, 1 - 2 * y / height);
    raycaster.params.Points.threshold = radius / scale;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(points, false);
    // Favor the closest dot on screen; raycaster resolves near/far ties.
    let best = null;
    for (const hit of hits) if (!best || hit.distanceToRay < best.distanceToRay - 1e-6) best = hit;
    return best ? Array.from(vertices.slice(best.index * 3, best.index * 3 + 3)) : null;
  }
  canvas.addEventListener('webglcontextlost', event => {event.preventDefault(); available = false; canvas.style.display = 'none'; redraw();});
  canvas.addEventListener('webglcontextrestored', () => {available = true; redraw();});
  return {render, pick, hide: () => {canvas.style.display = 'none';},
    isAvailable: () => available,
    // Exposed for projection and selection checks without a browser.
    inspect: () => ({camera, points, renderer, count: vertices.length / 3, width, height, indices, depths}),
    dispose: () => {available = false; geometry.dispose(); material.dispose(); wireGeometry.dispose(); wireMaterial.dispose(); renderer.dispose();}
  };
}

if (typeof window !== 'undefined') window.createPosterior3D = createPosterior3D;
