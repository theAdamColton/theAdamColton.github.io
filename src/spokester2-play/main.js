// spokester2 in the browser: the game simulated in JS, its depth and flow rasterized on the GPU, and every frame
// rendered by the neural renderer (engine.js). Frames are made two at a time (a framelet), at most two
// framelets in flight; the game's time step follows the GPU's throughput, at most 60 fps.
//
// The renderer is built for one token grid (resolution x aspect); changing either rebuilds it (the game goes on,
// the renderer starts a fresh rollout). With the "fit window" aspect a resize that would make rendering slower
// pauses the game instead, offering to resume at the new shape or at the native 16:9.
import { Engine, parseModel } from "./engine.js";
import { Raster, gameParams } from "./raster.js";
import { CFG, World } from "./world.js";

const $ = (id) => document.getElementById(id);
const SIZES = [224, 288, 352, 432]; // the shorter side, px
const ASPECTS = { native: 16 / 9, "4:3": 4 / 3, "1:1": 1, window: null };
const MAX_FPS = 60;
const MIN_FPS = 12;
const LAYERS = 12; // output frame ring: 6 framelets
const SLOWER = 1.15; // fit window: a new shape with this many times the tokens pauses instead of rebuilding

function status(msg, err = false) {
  $("status").textContent = msg;
  $("status").className = err ? "err" : "";
  $("status").style.display = msg ? "block" : "none";
}

function loadSettings() {
  let s = {};
  try {
    s = JSON.parse(localStorage.getItem("spokester2") ?? "{}");
  } catch {}
  const want = s.size || 224;
  const size = SIZES.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
  return { size, aspect: s.aspect in ASPECTS ? s.aspect : "native" };
}

// the token grid [nph, npw] for a shorter side SIZE and an aspect (window: the window's)
function grid(size, aspect) {
  const a = ASPECTS[aspect] ?? innerWidth / Math.max(innerHeight, 1);
  const w = a >= 1 ? size * a : size, h = a >= 1 ? size : size / a;
  return [Math.max(2, Math.round(h / 16)), Math.max(2, Math.round(w / 16))];
}

async function fetchModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status(`downloading the model… ${(got / 1e6).toFixed(1)}${total ? ` / ${(total / 1e6).toFixed(1)}` : ""} MB`);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  return buf.buffer;
}

async function gpu() {
  if (!navigator.gpu) throw new Error("This browser has no WebGPU. Try a recent Chrome, Edge or Safari.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  const L = adapter.limits;
  const f16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice({
    requiredFeatures: f16 ? ["shader-f16"] : [],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: Math.min(32768, L.maxComputeWorkgroupStorageSize),
      maxComputeInvocationsPerWorkgroup: Math.min(1024, L.maxComputeInvocationsPerWorkgroup),
      maxComputeWorkgroupSizeX: Math.min(1024, L.maxComputeWorkgroupSizeX),
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
    },
  });
  device.lost.then((i) => status(`GPU device lost: ${i.message}`, true));
  device.addEventListener("uncapturederror", (e) => console.error(e.error.message));
  return { device, f16 };
}

const DISPLAY = /* wgsl */ `
struct U { layer: u32, pad: u32, aspect: f32, canvasAspect: f32 }
@group(0) @binding(0) var tex: texture_2d_array<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: U;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) v: u32) -> VO {
  let p = vec2f(array<f32, 3>(-1.0, 3.0, -1.0)[v], array<f32, 3>(-1.0, -1.0, 3.0)[v]);
  var o: VO; o.pos = vec4f(p, 0.0, 1.0); o.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let a = u.aspect;
  var uv = i.uv;
  if (u.canvasAspect > a) { uv.x = (uv.x - 0.5) * u.canvasAspect / a + 0.5; } else { uv.y = (uv.y - 0.5) * a / u.canvasAspect + 0.5; }
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  return vec4f(textureSampleLevel(tex, samp, uv, u.layer, 0.0).rgb, 1.0);
}`;

async function main() {
  const { device, f16 } = await gpu();
  const model = parseModel(await fetchModel("/spokester2/model.bin"));
  const settings = loadSettings();

  // display
  const canvas = $("view"), ctx = canvas.getContext("webgpu"), format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });
  const dmod = device.createShaderModule({ code: DISPLAY });
  const dpipe = device.createRenderPipeline({ layout: "auto", vertex: { module: dmod, entryPoint: "vs" },
    fragment: { module: dmod, entryPoint: "fs", targets: [{ format }] }, primitive: { topology: "triangle-list" } }); // prettier-ignore
  const dUni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const samp = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  // the renderer for an nph x npw token grid: the engine, its rasterizer, the display's bind group
  const build = (nph, npw) => {
    const W = npw * 16, H = nph * 16;
    CFG.camera.width = W;
    CFG.camera.height = H;
    const eng = new Engine(device, model, { nph, npw, f16, frames: LAYERS });
    const raster = new Raster(device, { W, H, dimg: eng.B.dimg, fimg: eng.B.fimg, game: gameParams(model.game) });
    const dBG = device.createBindGroup({ layout: dpipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: eng.tex.createView({ dimension: "2d-array" }) }, { binding: 1, resource: samp },
      { binding: 2, resource: { buffer: dUni } }] }); // prettier-ignore
    return { eng, raster, dBG, W, H, nph, npw };
  };
  status("compiling shaders…");
  await new Promise((r) => setTimeout(r, 0));
  let R = build(...grid(settings.size, settings.aspect));

  // ------------------------------------------------------------------------------------------- the game
  let seed = 1, world = new World(seed);
  let taps = 0, restart = false, crashAt = null, gameT = 0, prevT = null, slot = 0, framelets = 0, snap = null;
  let dt = 1 / MAX_FPS, gpuMs = 0, lastDone = 0, inflight = 0, gen = 0, paused = false, busy = false, layer = -1;
  let nextShow = 0, hudT = 0;
  const ready = [];

  // two game frames -> rasterized conditioning -> the neural renderer, into ring layers 2 slot, 2 slot + 1
  function produce() {
    if (restart) {
      world = new World(++seed);
      R.eng.reset();
      restart = false;
      crashAt = null;
      gameT = 0;
      prevT = null;
      snap = null;
    }
    const times = [], fdt = dt;
    for (let f = 0; f < 2; f++) {
      const action = Math.sign(taps);
      if (taps) taps -= Math.sign(taps);
      const cur = snap ?? { cam: world.camera(), boxes: world.boxes() };
      world.step(action, fdt);
      snap = { cam: world.camera(), boxes: world.boxes() }; // the next frame's, and the frame after's current one
      R.raster.setFrame(f, cur.cam, snap.cam, cur.boxes, new Map(snap.boxes.map((b) => [b.id, b])));
      times.push(gameT);
      gameT += fdt;
      if (world.crashed && crashAt === null) crashAt = gameT;
    }
    // frame gaps (train.frame_gaps): within the framelet, and since the last one (the first: its own again)
    const gaps = [fdt, prevT === null ? fdt : times[0] - prevT];
    prevT = times[1];
    const enc = device.createCommandEncoder();
    const s = (framelets * 2654435761) >>> 0;
    R.raster.encode(enc, { seed: s });
    const l = 2 * slot, g = gen;
    R.eng.encode(enc, { t: (times[0] + times[1]) / 2, gaps, seed: s ^ 0x5bd1e995, frame: l });
    const submitted = performance.now();
    device.queue.submit([enc.finish()]);
    inflight++;
    framelets++;
    slot = (slot + 1) % (LAYERS / 2);
    const r = world.rider, hud = { v: r.v, z: r.z, crashed: world.crashed };
    device.queue.onSubmittedWorkDone().then(() => {
      inflight--;
      if (g !== gen) return; // made by a renderer since rebuilt
      const now = performance.now();
      const busyMs = now - Math.max(submitted, lastDone);
      lastDone = now;
      gpuMs = gpuMs ? 0.9 * gpuMs + 0.1 * busyMs : busyMs;
      // the step that the GPU keeps up with: two frames per framelet, with some margin
      dt = Math.min(Math.max((gpuMs * 1.1) / 2000, 1 / MAX_FPS), 1 / MIN_FPS);
      ready.push({ layer: l, dt: fdt, hud }, { layer: l + 1, dt: fdt, hud });
      pump();
    });
  }

  function pump() {
    while (!paused && !busy && inflight < 2 && ready.length + 2 * inflight < 4) produce();
  }

  // a new renderer for NPH x NPW once the GPU is idle; the game state stays, the rollout starts over
  async function rebuild(nph, npw) {
    busy = true;
    gen++;
    status(`building the renderer for ${npw * 16}×${nph * 16}…`);
    await device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 0));
    ready.length = 0;
    R.eng.destroy();
    R.raster.destroy();
    R = build(nph, npw);
    snap = null; // the camera's intrinsics changed
    prevT = null;
    slot = 0;
    layer = -1;
    gpuMs = 0;
    await device.queue.onSubmittedWorkDone();
    status("");
    busy = false;
  }

  // ------------------------------------------------------------------------------------------- the menu
  const menu = $("menu");
  const fillSizes = () => {
    const aspect = $("aspect").value;
    $("size").innerHTML = SIZES.map((s) => {
      const [nph, npw] = grid(s, aspect);
      return `<option value="${s}">${npw * 16}×${nph * 16}</option>`;
    }).join("");
    $("size").value = String(settings.size);
  };
  $("aspect").addEventListener("change", fillSizes);

  function openMenu(notice = "") {
    paused = true;
    $("aspect").value = settings.aspect;
    fillSizes();
    $("notice").textContent = notice;
    $("notice").style.display = notice ? "block" : "none";
    $("nativeRow").style.display = notice ? "flex" : "none";
    menu.style.display = "block";
  }

  // close the menu with its settings (or ASPECT), rebuilding the renderer if the grid changed
  async function resume({ again = false, aspect = null } = {}) {
    if (busy) return;
    settings.size = Number($("size").value);
    settings.aspect = aspect ?? $("aspect").value;
    localStorage.setItem("spokester2", JSON.stringify(settings));
    menu.style.display = "none";
    const [nph, npw] = grid(settings.size, settings.aspect);
    if (nph !== R.nph || npw !== R.npw) await rebuild(nph, npw);
    if (again) restart = true;
    paused = false;
    nextShow = performance.now();
    pump();
  }

  $("gear").addEventListener("click", () => (paused ? resume() : openMenu()));
  $("resume").addEventListener("click", () => resume());
  $("restart").addEventListener("click", () => resume({ again: true }));
  $("native").addEventListener("click", () => resume({ aspect: "native" }));
  document.addEventListener("visibilitychange", () => document.hidden && !paused && openMenu());

  // fit window: follow the window's shape, unless that's slower to render (then pause and ask)
  let resizeTimer = null;
  async function onShape() {
    if (settings.aspect !== "window" || busy) return;
    const [nph, npw] = grid(settings.size, "window");
    if (paused) return void (menu.style.display === "block" && fillSizes());
    if (nph === R.nph && npw === R.npw) return;
    if (nph * npw > SLOWER * R.nph * R.npw)
      return openMenu(`The window's new shape needs ${npw * 16}×${nph * 16}, which renders slower than ${R.W}×${R.H}.`);
    paused = true;
    await rebuild(nph, npw);
    paused = false;
    nextShow = performance.now();
    pump();
  }
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(onShape, 300);
  };
  window.addEventListener("resize", resize);
  resize();

  // ------------------------------------------------------------------------------------------- input
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLSelectElement) return;
    const k = e.key.toLowerCase();
    if (e.repeat) return void (["arrowleft", "arrowright"].includes(k) && e.preventDefault());
    if (k === "escape") paused ? resume() : openMenu();
    else if (k === "r") paused ? resume({ again: true }) : (restart = true);
    else if (paused) return;
    else if (k === "arrowleft" || k === "a") taps -= 1;
    else if (k === "arrowright" || k === "d") taps += 1;
    else return;
    e.preventDefault();
  });
  canvas.addEventListener("pointerdown", (e) => {
    if (paused) return;
    if (crashAt !== null) return void (restart = true);
    const r = canvas.getBoundingClientRect();
    taps += e.clientX - r.left < r.width / 2 ? -1 : 1;
  });

  // ------------------------------------------------------------------------------------------- display
  // show the ready frames at their game time step; the current frame is redrawn every animation frame
  const dUniData = new ArrayBuffer(16), dU32 = new Uint32Array(dUniData), dF32 = new Float32Array(dUniData);
  function frame(now) {
    requestAnimationFrame(frame);
    if (!paused && !busy && ready.length && now >= nextShow - 4) {
      const f = ready.shift();
      nextShow = Math.max(nextShow + f.dt * 1000, now - f.dt * 500);
      layer = f.layer;
      if (now - hudT > 100) {
        hudT = now;
        $("hud").textContent = `${(f.hud.v * 3.6).toFixed(0)} km/h · ${f.hud.z.toFixed(0)} m`;
        $("banner").textContent = "Crashed — tap or press R";
        $("banner").style.display = f.hud.crashed ? "block" : "none";
      }
    }
    if (layer >= 0 && !busy) {
      dU32[0] = layer;
      dF32[2] = R.W / R.H;
      dF32[3] = canvas.width / canvas.height;
      device.queue.writeBuffer(dUni, 0, dUniData);
      const enc = device.createCommandEncoder();
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] }); // prettier-ignore
      rp.setPipeline(dpipe);
      rp.setBindGroup(0, R.dBG);
      rp.draw(3);
      rp.end();
      device.queue.submit([enc.finish()]);
    }
    pump();
  }
  await device.queue.onSubmittedWorkDone();
  status("");
  pump();
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  status(String(e.message ?? e), true);
});
