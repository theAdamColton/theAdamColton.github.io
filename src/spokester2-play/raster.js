// The game's conditioning on the GPU: depth and exact optical flow rasterized from the world's boxes (the
// equivalent of spokester2/game/render.py), then degraded like train.game_clip: depth remapped toward the labels'
// distribution and log-encoded, flow magnitudes remapped, both blurred (a torch-exact antialiased bilinear
// downsample by 2.5 and a bilinear upsample), plus noise. Writes the engine's DIMG / FIMG for one framelet.
//
// Every box carries its pose in this frame and the next: a fragment's box-local point is moved with the next pose
// and projected through the next camera (the ground is static, the sky only rotates). Coordinates are relative to
// this frame's camera, so f32 stays exact far down the road.

// the run's TrainConfig.game_* (in model.bin's header)
export const gameParams = (g) => ({ depthScale: g.depth_scale, depthPower: g.depth_power, flowScale: g.flow_scale,
  flowPower: g.flow_power, skyM: g.sky_m, blur: g.blur, depthNoise: g.noise[0], flowNoise: g.noise[1] }); // prettier-ignore

const NEAR = 0.05, SKY = 1e30, SEG = 24; // cylinders: SEG-gons (circumscribed)
const INST = 36; // floats per instance: 9 vec4

const CAM = /* wgsl */ `
struct Cam { pos: vec4f, r0: vec4f, r1: vec4f, r2: vec4f, npos: vec4f, n0: vec4f, n1: vec4f, n2: vec4f, k: vec4f, size: vec4f }
@group(0) @binding(0) var<uniform> cam: Cam;
// world -> camera of this frame (rows r of the camera -> world rotation are its columns' transposes)
fn toCam(p: vec3f) -> vec3f { let d = p - cam.pos.xyz; return vec3f(dot(vec3f(cam.r0.x, cam.r1.x, cam.r2.x), d), dot(vec3f(cam.r0.y, cam.r1.y, cam.r2.y), d), dot(vec3f(cam.r0.z, cam.r1.z, cam.r2.z), d)); }
fn rotNext(d: vec3f) -> vec3f { return vec3f(dot(vec3f(cam.n0.x, cam.n1.x, cam.n2.x), d), dot(vec3f(cam.n0.y, cam.n1.y, cam.n2.y), d), dot(vec3f(cam.n0.z, cam.n1.z, cam.n2.z), d)); }
// a point seen from the next camera -> flow from pixel FRAG (0 where it's behind the camera)
fn flowTo(q: vec3f, frag: vec2f) -> vec2f {
  if (q.z <= 1e-3) { return vec2f(0.0); }
  return vec2f(cam.k.x * q.x / q.z + cam.k.z, cam.k.y * q.y / q.z + cam.k.w) - frag;
}
`;

const RASTER = /* wgsl */ `${CAM}
struct Inst { c: vec4f, h: vec4f, r0: vec4f, r1: vec4f, r2: vec4f, nc: vec4f, n0: vec4f, n1: vec4f, n2: vec4f }
@group(0) @binding(1) var<storage, read> inst: array<Inst>;
struct VO { @builtin(position) pos: vec4f, @location(0) local: vec3f, @location(1) z: f32, @location(2) @interpolate(flat) i: u32 }

const CUBE = array<vec3f, 36>(
  vec3f(-1,-1,-1), vec3f(1,-1,-1), vec3f(1,1,-1), vec3f(-1,-1,-1), vec3f(1,1,-1), vec3f(-1,1,-1),
  vec3f(-1,-1,1), vec3f(1,1,1), vec3f(1,-1,1), vec3f(-1,-1,1), vec3f(-1,1,1), vec3f(1,1,1),
  vec3f(-1,-1,-1), vec3f(-1,1,-1), vec3f(-1,1,1), vec3f(-1,-1,-1), vec3f(-1,1,1), vec3f(-1,-1,1),
  vec3f(1,-1,-1), vec3f(1,1,1), vec3f(1,1,-1), vec3f(1,-1,-1), vec3f(1,-1,1), vec3f(1,1,1),
  vec3f(-1,-1,-1), vec3f(1,-1,1), vec3f(1,-1,-1), vec3f(-1,-1,-1), vec3f(-1,-1,1), vec3f(1,-1,1),
  vec3f(-1,1,-1), vec3f(1,1,-1), vec3f(1,1,1), vec3f(-1,1,-1), vec3f(1,1,1), vec3f(-1,1,1));

fn cylinder(v: u32) -> vec3f {  // along x, unit radius (circumscribed polygon), SEG sides + 2 caps
  let seg = ${SEG}u; let rr = 1.0 / cos(3.14159265 / f32(seg));
  let ang = 6.28318531 / f32(seg);
  if (v < seg * 6u) {
    let k = v / 6u; let c = v % 6u;
    let kk = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u)[c] + k; let xs = array<f32, 6>(-1, -1, 1, -1, 1, 1)[c];
    return vec3f(xs, rr * cos(f32(kk) * ang), rr * sin(f32(kk) * ang));
  }
  let w = v - seg * 6u; let cap = w / (seg * 3u); let k = (w % (seg * 3u)) / 3u; let c = w % 3u;
  let xs = select(-1.0, 1.0, cap == 1u);
  if (c == 0u) { return vec3f(xs, 0.0, 0.0); }
  let kk = k + c - 1u;
  return vec3f(xs, rr * cos(f32(kk) * ang), rr * sin(f32(kk) * ang));
}

@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VO {
  let b = inst[i];
  let unit = select(CUBE[v % 36u], cylinder(v), b.c.w > 0.5);
  let local = unit * b.h.xyz;
  let world = b.c.xyz + vec3f(dot(b.r0.xyz, local), dot(b.r1.xyz, local), dot(b.r2.xyz, local));
  let q = toCam(world);
  let W = cam.size.x; let H = cam.size.y;
  var o: VO;
  o.pos = vec4f(2.0 * cam.k.x * q.x / W + (2.0 * cam.k.z / W - 1.0) * q.z,
                -(2.0 * cam.k.y * q.y / H) + (1.0 - 2.0 * cam.k.w / H) * q.z, ${NEAR}, q.z);
  o.local = local; o.z = q.z; o.i = i;
  return o;
}

@fragment fn fs(v: VO) -> @location(0) vec4f {
  let b = inst[v.i];
  var f = vec2f(0.0);
  if (b.h.w > 0.5) {
    let p = b.nc.xyz + vec3f(dot(b.n0.xyz, v.local), dot(b.n1.xyz, v.local), dot(b.n2.xyz, v.local));
    f = flowTo(rotNext(p - cam.npos.xyz), v.pos.xy);
  }
  return vec4f(v.z, f, 1.0);
}

struct GO { @builtin(position) pos: vec4f }
@vertex fn vsGround(@builtin(vertex_index) v: u32) -> GO {
  var o: GO;
  o.pos = vec4f(array<f32, 3>(-1.0, 3.0, -1.0)[v], array<f32, 3>(-1.0, -1.0, 3.0)[v], 0.0, 1.0);
  return o;
}
struct GF { @location(0) c: vec4f, @builtin(frag_depth) depth: f32 }
@fragment fn fsGround(g: GO) -> GF {
  let dc = vec3f((g.pos.x - cam.k.z) / cam.k.x, (g.pos.y - cam.k.w) / cam.k.y, 1.0);
  let d = vec3f(dot(cam.r0.xyz, dc), dot(cam.r1.xyz, dc), dot(cam.r2.xyz, dc));  // unit camera z: t = z depth
  var o: GF;
  if (d.y < -1e-6) {
    let t = -cam.pos.y / d.y;
    let p = cam.pos.xyz + d * t;
    o.c = vec4f(t, flowTo(rotNext(p - cam.npos.xyz), g.pos.xy), 1.0);
    o.depth = ${NEAR} / t;
  } else {
    o.c = vec4f(${SKY}, flowTo(rotNext(d), g.pos.xy), 1.0);
    o.depth = 0.0;
  }
  return o;
}
`;

// torch's antialiased bilinear weights (aten _compute_weights_aa, align_corners = False): output index i of
// in -> out (scale = in / out >= 1): taps xmin .. xmin + xsize, triangle filter of support scale
const AA = /* wgsl */ `
fn aaRange(i: u32, scale: f32, n: u32) -> vec2u {
  let center = scale * (f32(i) + 0.5);
  let xmin = u32(max(center - scale + 0.5, 0.0));
  let xmax = min(u32(center + scale + 0.5), n);
  return vec2u(xmin, xmax);
}
fn aaW(j: u32, i: u32, scale: f32) -> f32 {
  let center = scale * (f32(i) + 0.5);
  return max(0.0, 1.0 - abs((f32(j) - center + 0.5) / scale));
}
`;

function prepShaders(W, H, w, h, g) {
  const sx = W / w, sy = H / h;
  const near = 0.5, far = 1000.0;
  return {
    // raw depth / flow -> depth code and remapped flow [2][H][W] vec4
    remap: `
@group(0) @binding(0) var tex: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> PRE: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${W}u) { return; }
  let v = textureLoad(tex, vec2u(id.x, id.y), id.z, 0);
  var d = v.x;
  var code = 1.0;
  if (d < ${SKY / 10}) {
    d = ${near} * pow(d / ${near}, ${g.depthPower}) * ${g.depthScale};
    code = clamp(log(min(d, ${g.skyM}) / ${near}) / ${Math.log(far / near)}, 0.0, 1.0);
  } else { code = clamp(log(${g.skyM} / ${near}) / ${Math.log(far / near)}, 0.0, 1.0); }
  var f = clamp(v.yz, vec2f(-512.0), vec2f(512.0));
  if (any(f != f)) { f = vec2f(0.0); }
  f = f * pow(max(length(f), 1e-3), ${g.flowPower - 1});
  f = clamp(f * ${g.flowScale}, vec2f(-512.0), vec2f(512.0));
  PRE[(id.z * ${H}u + id.y) * ${W}u + id.x] = vec4f(code, f, 0.0);
}`,
    // antialiased downsample along x: [2][H][W] -> [2][H][w]
    blurX: `${AA}
@group(0) @binding(0) var<storage, read> PRE: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> BX: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${w}u) { return; }
  let r = aaRange(id.x, ${sx}, ${W}u);
  var acc = vec4f(0.0); var tot = 0.0;
  for (var j = r.x; j < r.y; j++) { let wt = aaW(j, id.x, ${sx}); acc += wt * PRE[(id.z * ${H}u + id.y) * ${W}u + j]; tot += wt; }
  BX[(id.z * ${H}u + id.y) * ${w}u + id.x] = acc / tot;
}`,
    // antialiased downsample along y: [2][H][w] -> [2][h][w]
    blurY: `${AA}
@group(0) @binding(0) var<storage, read> BX: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> BS: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${w}u) { return; }
  let r = aaRange(id.y, ${sy}, ${H}u);
  var acc = vec4f(0.0); var tot = 0.0;
  for (var j = r.x; j < r.y; j++) { let wt = aaW(j, id.y, ${sy}); acc += wt * BX[(id.z * ${H}u + j) * ${w}u + id.x]; tot += wt; }
  BS[(id.z * ${h}u + id.y) * ${w}u + id.x] = acc / tot;
}`,
    // bilinear upsample (align_corners = False) + noise -> the engine's DIMG [2][H][W], FIMG [2][H][W][2]
    up: `
struct U { seed: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> BS: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> DIMG: array<f32>;
@group(0) @binding(2) var<storage, read_write> FIMG: array<vec2f>;
@group(0) @binding(3) var<uniform> u: U;
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn unif(h: u32) -> f32 { return (f32(h >> 8u) + 0.5) / 16777216.0; }
fn src(i: u32, scale: f32, n: u32) -> vec3f {  // i0, i1, lambda
  let s = max(scale * (f32(i) + 0.5) - 0.5, 0.0);
  let i0 = min(u32(floor(s)), n - 1u);
  let l = clamp(s - f32(i0), 0.0, 1.0);
  return vec3f(f32(i0), f32(select(i0, i0 + 1u, i0 < n - 1u)), l);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${W}u) { return; }
  let ys = src(id.y, ${h / H}, ${h}u); let xs = src(id.x, ${w / W}, ${w}u);
  let b = id.z * ${h}u;
  let r0 = (b + u32(ys.x)) * ${w}u; let r1 = (b + u32(ys.y)) * ${w}u;
  let top = mix(BS[r0 + u32(xs.x)], BS[r0 + u32(xs.y)], xs.z);
  let bot = mix(BS[r1 + u32(xs.x)], BS[r1 + u32(xs.y)], xs.z);
  var v = mix(top, bot, ys.z);
  let i = (id.z * ${H}u + id.y) * ${W}u + id.x;
  {
    let h1 = pcg(u.seed ^ pcg(i)); let h2 = pcg(h1 ^ 0x9e3779b9u); let h3 = pcg(h2 ^ 0x85ebca6bu); let h4 = pcg(h3);
    let r1 = sqrt(-2.0 * log(unif(h1))); let r2 = sqrt(-2.0 * log(unif(h3)));
    let n = vec3f(r1 * cos(6.2831853 * unif(h2)), r1 * sin(6.2831853 * unif(h2)), r2 * cos(6.2831853 * unif(h4)));
    v = vec4f(clamp(v.x + ${g.depthNoise} * n.x, 0.0, 1.0), v.yz + ${g.flowNoise} * n.yz, 0.0);
  }
  DIMG[i] = v.x; FIMG[i] = v.yz;
}`,
  };
}

export class Raster {
  // dimg, fimg: the engine's conditioning buffers; W x H the model's resolution
  constructor(device, { W, H, dimg, fimg, game, maxBoxes = 2048 }) {
    this.dev = device;
    this.W = W;
    this.H = H;
    this.maxBoxes = maxBoxes;
    const dev = device;
    this.tex = dev.createTexture({ size: [W, H, 2], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC }); // prettier-ignore
    this.depth = dev.createTexture({ size: [W, H], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.cams = [0, 1].map(() => dev.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    this.insts = [0, 1].map(() => dev.createBuffer({ size: maxBoxes * INST * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
    const mod = dev.createShaderModule({ code: RASTER });
    const target = { format: "rgba32float" };
    const ds = (cmp) => ({ format: "depth32float", depthWriteEnabled: true, depthCompare: cmp });
    this.boxPipe = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: mod, entryPoint: "vs" },
      fragment: { module: mod, entryPoint: "fs", targets: [target] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: ds("greater"),
    });
    this.groundPipe = dev.createRenderPipeline({
      layout: "auto",
      vertex: { module: mod, entryPoint: "vsGround" },
      fragment: { module: mod, entryPoint: "fsGround", targets: [target] },
      primitive: { topology: "triangle-list" },
      depthStencil: ds("always"),
    });
    this.boxBG = [0, 1].map((f) => dev.createBindGroup({ layout: this.boxPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cams[f] } }, { binding: 1, resource: { buffer: this.insts[f] } }] })); // prettier-ignore
    this.groundBG = [0, 1].map((f) => dev.createBindGroup({ layout: this.groundPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cams[f] } }] })); // prettier-ignore
    this.views = [0, 1].map((f) => this.tex.createView({ dimension: "2d", baseArrayLayer: f, arrayLayerCount: 1 }));
    this.depthView = this.depth.createView();

    // the degradation
    const w = Math.round(W / game.blur), h = Math.round(H / game.blur);
    this.small = [w, h];
    const sh = prepShaders(W, H, w, h, game);
    const pipe = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } }); // prettier-ignore
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.pre = dev.createBuffer({ size: 2 * H * W * 16, usage: S });
    this.bx = dev.createBuffer({ size: 2 * H * w * 16, usage: S });
    this.bs = dev.createBuffer({ size: 2 * h * w * 16, usage: S });
    this.uni = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.P = { remap: pipe(sh.remap), blurX: pipe(sh.blurX), blurY: pipe(sh.blurY), up: pipe(sh.up) };
    const bg = (p, list) => dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map((r, i) => ({ binding: i, resource: r.buffer === undefined && !(r instanceof GPUTextureView) ? { buffer: r } : r })) }); // prettier-ignore
    this.G = {
      remap: bg(this.P.remap, [this.tex.createView({ dimension: "2d-array" }), this.pre]),
      blurX: bg(this.P.blurX, [this.pre, this.bx]),
      blurY: bg(this.P.blurY, [this.bx, this.bs]),
      up: bg(this.P.up, [this.bs, dimg, fimg, this.uni]),
    };
  }

  // free the GPU memory (the raster is unusable after)
  destroy() {
    for (const x of [this.tex, this.depth, ...this.cams, ...this.insts, this.pre, this.bx, this.bs, this.uni]) x.destroy();
  }

  // frame f's camera and boxes; cam / next: World.camera() of this frame and the next, boxes: World.boxes() of
  // this frame, nextById: Map id -> box of the next frame. Everything relative to this frame's camera.
  setFrame(f, cam, next, boxes, nextById) {
    const o = [cam.pos[0], 0, cam.pos[2]], data = new Float32Array(Math.min(boxes.length, this.maxBoxes) * INST);
    const order = boxes.filter((b) => !b.round).concat(boxes.filter((b) => b.round)).slice(0, this.maxBoxes);
    this.counts = this.counts ?? [];
    this.counts[f] = [order.filter((b) => !b.round).length, order.filter((b) => b.round).length];
    order.forEach((b, i) => {
      const n = nextById.get(b.id), base = i * INST, R = b.rot;
      data.set([b.center[0] - o[0], b.center[1] - o[1], b.center[2] - o[2], b.round ? 1 : 0,
        b.half[0], b.half[1], b.half[2], n ? 1 : 0, R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0], base); // prettier-ignore
      if (n) {
        const N = n.rot;
        data.set([n.center[0] - o[0], n.center[1] - o[1], n.center[2] - o[2], 0,
          N[0], N[1], N[2], 0, N[3], N[4], N[5], 0, N[6], N[7], N[8], 0], base + 20); // prettier-ignore
      }
    });
    const c = new Float32Array(40), R = cam.rot, N = next.rot;
    c.set([0, cam.pos[1], 0, 0, R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0]);
    c.set([next.pos[0] - o[0], next.pos[1] - o[1], next.pos[2] - o[2], 0, N[0], N[1], N[2], 0, N[3], N[4], N[5], 0, N[6], N[7], N[8], 0], 16); // prettier-ignore
    c.set([cam.fx, cam.fy, cam.cx, cam.cy, this.W, this.H, NEAR, 0], 32);
    this.dev.queue.writeBuffer(this.cams[f], 0, c);
    this.dev.queue.writeBuffer(this.insts[f], 0, data);
  }

  // rasterize both frames (set with setFrame)
  encode(enc, { seed = 0 } = {}) {
    for (let f = 0; f < 2; f++) {
      const rp = enc.beginRenderPass({
        colorAttachments: [{ view: this.views[f], loadOp: "clear", storeOp: "store", clearValue: [SKY, 0, 0, 1] }],
        depthStencilAttachment: { view: this.depthView, depthLoadOp: "clear", depthStoreOp: "discard", depthClearValue: 0 },
      });
      rp.setPipeline(this.groundPipe);
      rp.setBindGroup(0, this.groundBG[f]);
      rp.draw(3);
      rp.setPipeline(this.boxPipe);
      rp.setBindGroup(0, this.boxBG[f]);
      const [nb, nc] = this.counts[f];
      if (nb) rp.draw(36, nb, 0, 0);
      if (nc) rp.draw(SEG * 12, nc, 0, nb);
      rp.end();
    }
    this.dev.queue.writeBuffer(this.uni, 0, new Uint32Array([seed >>> 0, 0, 0, 0]));
    const cp = enc.beginComputePass();
    const [w, h] = this.small;
    const run = (k, x, y) => {
      cp.setPipeline(this.P[k]);
      cp.setBindGroup(0, this.G[k]);
      cp.dispatchWorkgroups(Math.ceil(x / 64), y, 2);
    };
    run("remap", this.W, this.H);
    run("blurX", w, this.H);
    run("blurY", w, h);
    run("up", this.W, this.H);
    cp.end();
  }
}
