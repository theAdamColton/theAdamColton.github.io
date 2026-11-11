// WebGPU inference for spokester2.model.net: the causal one-step video transformer, two-step, streamed framelet by
// framelet. Hardcoded to the exported architecture (tools/web_export.py): shared attention / adaLN, gated
// attention, RoPE (t, y, x), anchors, an optional pixel head and the patch decoder (its width and halo from the
// model header, see decArch).
//
// Per framelet (2 frames): the condition c and every adaLN (shared by all blocks) once, then two passes of the
// trunk, each with its own KV cache: step 1 from noise, step 2 from (1 - s) step 1 + s fresh noise (s: the run's
// two_step). Framelet 0
// also writes the anchors (2n rows). The KV cache holds 4 slots of n rows: anchors, then a ring of the 3 framelets
// a token sees (the 2 before it and its own), so keys are always rows [0, Lk) of the cache.

const D = 384, HEADS = 6, HD = 64, FF = 1536, CD = 256, PD = 1536, DEPTH = 12;
const CIN = 1538, CIN_PAD = 1552; // condition input (depth, flow, 2 gaps), padded to the GEMM's BK
const MODW = 3072; // shared adaLN (6 x 384) + output adaLN (2 x 384)
// the architecture the kernels are written for (tools/web_export.py checks the rest); the pixel head's code width
// and the patch decoder's width and halo vary (decArch)
const ARCH = { dim: D, depth: DEPTH, heads: HEADS, mlp_ratio: FF / D, cond_dim: CD, window: 3, patch_size: 16,
  patch_duration: 2, time_pairs: 8, pixel_kernel: 5 }; // prettier-ignore

// The output stage's sizes. headcat's columns: head (1536), pixel code (k, if any), the patch decoder's adaLNs
// (2 c1, 2 c2, 2 c2, 2 c1). The decoder sees a window of 16 + 2 hl px per patchlet: G1 x G1 (c1 channels) at 1/2
// res, G2 x G2 (c2) at 1/4, P1 / P2 positions per token (2 frames).
function decArch(cfg) {
  const k = cfg.pixel_head, c1 = 2 * cfg.pixel_decoder, c2 = 4 * cfg.pixel_decoder, hl = cfg.pixel_halo;
  const G1 = (16 + 2 * hl) / 2, G2 = G1 / 2;
  const codeCol = PD, enc = PD + k, mid0 = enc + 2 * c1, mid1 = mid0 + 2 * c2, dec = mid1 + 2 * c2;
  const headw = dec + 2 * c1;
  return { k, c1, c2, hl, G1, G2, P1: 2 * G1 * G1, P2: 2 * G2 * G2, headw, hpad: Math.ceil(headw / 64) * 64,
    codeCol, decMod: { enc, mid0, mid1, dec }, xpw: Math.ceil((k + 3) / 16) * 16 }; // prettier-ignore
}

function checkArch(cfg) {
  for (const [k, v] of Object.entries(ARCH))
    if (cfg[k] !== v) throw new Error(`model.bin has ${k} = ${cfg[k]}, the player is written for ${v}`);
  const bad = (msg) => { throw new Error(`model.bin: ${msg}, the player can't run it`); }; // prettier-ignore
  if (!cfg.pixel_decoder) bad("no patch decoder");
  if (cfg.pixel_decoder % 2 || 2 * cfg.pixel_decoder > 128) bad(`pixel_decoder = ${cfg.pixel_decoder}`);
  if ((16 + 2 * cfg.pixel_halo) % 4 || cfg.pixel_halo > 8) bad(`pixel_halo = ${cfg.pixel_halo}`);
  if (cfg.pixel_head && cfg.pixel_hidden !== 64) bad(`pixel_hidden = ${cfg.pixel_hidden}`);
}

// ---------------------------------------------------------------------------------------------- weights

export function parseModel(buf) {
  const u8 = new Uint8Array(buf);
  if (String.fromCharCode(...u8.slice(0, 4)) !== "SPKW") throw new Error("not a spokester2 model");
  const hlen = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(u8.slice(8, 8 + hlen)));
  const base = 8 + hlen + ((16 - ((8 + hlen) % 16)) % 16);
  checkArch(header.cfg);
  const tensors = {};
  for (const t of header.tensors) tensors[t.name] = { ...t, data: dequant(buf, base, t) };
  return { cfg: header.cfg, twoStep: header.two_step, game: header.game, step: header.step, tensors };
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

function dequant(buf, base, t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  if (t.fmt === "f16") {
    const h = new Uint16Array(buf, base + t.offset, n);
    for (let i = 0; i < n; i++) out[i] = halfToFloat(h[i]);
  } else if (t.fmt === "q8") {
    const [rows, cols] = t.shape;
    const q = new Int8Array(buf, base + t.offset, n), sc = new Uint16Array(buf, base + t.scales, rows);
    for (let r = 0; r < rows; r++) {
      const s = halfToFloat(sc[r]);
      for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * s;
    }
  } else if (t.fmt === "q5") {
    const groups = n / 64;
    const q = new Uint8Array(buf, base + t.offset, groups * 40), sc = new Uint16Array(buf, base + t.scales, groups);
    for (let g = 0; g < groups; g++) {
      const s = halfToFloat(sc[g]);
      for (let i = 0; i < 64; i++) {
        const bit = g * 320 + i * 5, byte = bit >> 3, sh = bit & 7;
        const v = ((q[byte] | (q[byte + 1] << 8)) >> sh) & 31; // a group is 40 bytes: byte + 1 stays inside it
        out[g * 64 + i] = (v - 15) * s;
      }
    }
  } else throw new Error(t.fmt);
  return out;
}

const f16buf = new Float32Array(1), f16bits = new Uint32Array(f16buf.buffer);
function toHalf(x) {
  f16buf[0] = x;
  const b = f16bits[0], s = (b >>> 16) & 0x8000, e = (b >>> 23) & 0xff, m = b & 0x7fffff;
  if (e === 255) return s | 0x7c00 | (m ? 0x200 : 0);
  const ee = e - 127 + 15;
  if (ee >= 31) return s | 0x7c00;
  if (ee <= 0) {
    if (ee < -10) return s;
    const mm = (m | 0x800000) >> (1 - ee);
    return s | ((mm + 0x1000 + ((mm >> 13) & 1) - 1) >> 13);
  }
  const r = (ee << 10) | (m >> 13), rest = m & 0x1fff;
  return s | (rest > 0x1000 || (rest === 0x1000 && r & 1) ? r + 1 : r);
}

// --------------------------------------------------------------------------------------------- shaders

const UNIFORMS = /* wgsl */ `
struct P {
  M: u32, rowOff: u32, kvOff: u32, Lk: u32,
  first: u32, seed: u32, gw: f32, gs: f32,
  tc: array<vec4f, 2>, ts: array<vec4f, 2>,
  frame: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<uniform> p: P;
`;

const GELU = /* wgsl */ `
fn gelu(x: f32) -> f32 { let u = clamp(0.7978845608 * (x + 0.044715 * x * x * x), -15.0, 15.0); return 0.5 * x * (1.0 + tanh(u)); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn sigm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
`;

function header(f16) {
  return (f16 ? "enable f16;\nalias T = f16;\n" : "alias T = f32;\n") + UNIFORMS + GELU;
}

// C[rows, N] = A[rows, K] . W[K, N] (W is the transposed Linear weight): tiles BM x BN x BK, TM x TN per thread
// (rows ty * TM .., cols tx * TN ..), A and W tiles in workgroup memory as f32 vec4s (A transposed: 4 rows per vec4).
function gemmShader(o) {
  const { f16, AT, K, N, NREAL, lda, rows, epi, extra = "", BM = 64, BN = 64, TM = 4, TN = 4, BK = 16 } = o;
  const hm = f16 && o.halfMath; // f16 tiles and products, summed in f16 over a BK chunk, then into the f32 sums
  const V = hm ? "vec4<f16>" : "vec4f";
  const aIdx = o.aIdx ?? "(r * LDA + k) / 4u"; // the vec4 of A holding (row r, k .. k + 3)
  const NT = (BM / TM) * (BN / TN), TM4 = TM / 4, TN4 = TN / 4;
  return `${header(f16)}
alias AT = ${AT === "f32" ? "f32" : "T"};
@group(0) @binding(1) var<storage, read> A: array<vec4<AT>>;
@group(0) @binding(2) var<storage, read> W: array<vec4<T>>;
${extra}
const K = ${K}u; const N = ${N}u; const NREAL = ${NREAL}u; const LDA = ${lda}u;
const BM = ${BM}u; const BN = ${BN}u; const BK = ${BK}u; const NT = ${NT}u;
var<workgroup> As: array<${V}, ${(BK * BM) / 4}>;
var<workgroup> Bs: array<${V}, ${(BK * BN) / 4}>;
@compute @workgroup_size(${NT})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  let m0 = wg.y * BM; let n0 = wg.x * BN;
  let rows = ${rows};
  let tx = li % ${BN / TN}u; let ty = li / ${BN / TN}u;
  var acc: array<array<vec4f, ${TN4}>, ${TM}>;
  for (var k0 = 0u; k0 < K; k0 += BK) {
    for (var i = li; i < ${(BM / 4) * (BK / 4)}u; i += NT) {  // a 4 x 4 block (4 rows, 4 k) per thread, transposed
      let r4 = i / ${BK / 4}u; let kq = i % ${BK / 4}u;
      var v: array<vec4f, 4>;
      for (var e = 0u; e < 4u; e++) {
        let r = m0 + r4 * 4u + e;
        let k = k0 + kq * 4u;
        if (r < rows) { v[e] = vec4f(A[${aIdx}]); } else { v[e] = vec4f(0.0); }
      }
      let a = (kq * 4u) * (BM / 4u) + r4;
      As[a] = ${V}(vec4f(v[0].x, v[1].x, v[2].x, v[3].x));
      As[a + BM / 4u] = ${V}(vec4f(v[0].y, v[1].y, v[2].y, v[3].y));
      As[a + BM / 2u] = ${V}(vec4f(v[0].z, v[1].z, v[2].z, v[3].z));
      As[a + 3u * BM / 4u] = ${V}(vec4f(v[0].w, v[1].w, v[2].w, v[3].w));
    }
    for (var i = li; i < ${(BK * BN) / 4}u; i += NT) {
      let k = i / (BN / 4u); let nq = i % (BN / 4u);
      Bs[i] = ${V}(W[((k0 + k) * N + n0) / 4u + nq]);
    }
    workgroupBarrier();
    var part = array<array<${V}, ${TN4}>, ${TM}>();  // explicitly zeroed every chunk
    for (var k = 0u; k < BK; k++) {
      var b: array<${V}, ${TN4}>;
      for (var j = 0u; j < ${TN4}u; j++) { b[j] = Bs[k * (BN / 4u) + tx * ${TN4}u + j]; }
      for (var i4 = 0u; i4 < ${TM4}u; i4++) {
        let a = As[k * (BM / 4u) + ty * ${TM4}u + i4];
        for (var j = 0u; j < ${TN4}u; j++) {
          part[i4 * 4u][j] += a.x * b[j]; part[i4 * 4u + 1u][j] += a.y * b[j];
          part[i4 * 4u + 2u][j] += a.z * b[j]; part[i4 * 4u + 3u][j] += a.w * b[j];
        }
      }
    }
    for (var i = 0u; i < ${TM}u; i++) { for (var j = 0u; j < ${TN4}u; j++) { acc[i][j] += vec4f(part[i][j]); } }
    workgroupBarrier();
  }
  for (var i = 0u; i < ${TM}u; i++) {
    let r = m0 + ty * ${TM}u + i;
    if (r >= rows) { break; }
    for (var j = 0u; j < ${TN}u; j++) {
      let c = n0 + tx * ${TN}u + j;
      if (c < NREAL) { let v = acc[i][j / 4u][j % 4u]; ${epi} }
    }
  }
}`;
}

function lnShader(f16, { shift, scale, xOff, ntok }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> MOD: array<f32>;
@group(0) @binding(3) var<storage, read_write> H: array<T>;
var<workgroup> red: array<f32, 128>;
fn reduce(li: u32, v: f32) -> f32 {
  red[li] = v; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let r = red[0]; workgroupBarrier(); return r;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  let r = wg.x; let xr = r + ${xOff};
  let a = X[xr * ${D}u + li]; let b = X[xr * ${D}u + li + 128u]; let c = X[xr * ${D}u + li + 256u];
  let mean = reduce(li, a + b + c) / ${D}.0;
  let da = a - mean; let db = b - mean; let dc = c - mean;
  let inv = inverseSqrt(reduce(li, da * da + db * db + dc * dc) / ${D}.0 + 1e-6);
  let m = (r % ${ntok}u) * ${MODW}u;
  H[r * ${D}u + li] = T(da * inv * (1.0 + MOD[m + ${scale}u + li]) + MOD[m + ${shift}u + li]);
  H[r * ${D}u + li + 128u] = T(db * inv * (1.0 + MOD[m + ${scale}u + li + 128u]) + MOD[m + ${shift}u + li + 128u]);
  H[r * ${D}u + li + 256u] = T(dc * inv * (1.0 + MOD[m + ${scale}u + li + 256u]) + MOD[m + ${shift}u + li + 256u]);
}`;
}

function qkRopeShader(f16, { ntok, mpad, kvn }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> QKV: array<T>;
@group(0) @binding(2) var<storage, read> NW: array<f32>;
@group(0) @binding(3) var<storage, read> ROPE: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> Q: array<T>;
@group(0) @binding(5) var<storage, read_write> KC: array<T>;
@group(0) @binding(6) var<storage, read_write> VC: array<T>;
var<workgroup> rq: array<f32, 64>; var<workgroup> rk: array<f32, 64>;
var<workgroup> sq: array<f32, 64>; var<workgroup> sk: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) d: u32) {
  _ = p.M;
  let r = wg.x; let h = wg.y;
  let base = r * ${3 * D + D}u + h * 64u + d;
  let qv = f32(QKV[base]); let kv = f32(QKV[base + ${D}u]); let vv = QKV[base + ${2 * D}u];
  sq[d] = qv * qv; sk[d] = kv * kv; workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (d < s) { sq[d] += sq[d + s]; sk[d] += sk[d + s]; } workgroupBarrier(); }
  rq[d] = qv * inverseSqrt(sq[0] / 64.0 + 1e-6) * NW[d];
  rk[d] = kv * inverseSqrt(sk[0] / 64.0 + 1e-6) * NW[64u + d];
  workgroupBarrier();
  let i = d / 2u; let tok = r % ${ntok}u;
  var c = 0.0; var s = 0.0;
  if (i < 8u) {
    if (!(p.first == 1u && r < ${ntok}u)) { c = p.tc[i / 4u][i % 4u]; s = p.ts[i / 4u][i % 4u]; }
  } else { let cs = ROPE[tok * 24u + i - 8u]; c = cs.x; s = cs.y; }
  let q1 = rq[2u * i]; let q2 = rq[2u * i + 1u]; let k1 = rk[2u * i]; let k2 = rk[2u * i + 1u];
  let even = (d % 2u) == 0u;
  let oq = select(q1 * s + q2 * c, q1 * c - q2 * s, even);
  let ok = select(k1 * s + k2 * c, k1 * c - k2 * s, even);
  Q[(h * ${mpad}u + r) * 64u + d] = T(oq);
  let kr = p.kvOff + r;
  KC[(h * ${kvn}u + kr) * 64u + d] = T(ok);
  VC[(h * ${kvn}u + kr) * 64u + d] = vv;
}`;
}

// Flash attention over a split of the keys (flash-decoding): a workgroup = 32 queries of one head x one split of
// the keys, 32 keys at a time with the online softmax; writes the split's unnormalized output and (max, sum).
// Scores 2 queries x 4 keys per thread, the output 4 queries x 4 dims.
function attnShader(f16, { mpad, kvn, ns, hm }) {
  const V4 = hm ? "vec4<f16>" : "vec4f", V2 = hm ? "vec2<f16>" : "vec2f", S1 = hm ? "f16" : "f32";
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> Q: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> KC: array<vec4<T>>;
@group(0) @binding(3) var<storage, read> VC: array<vec4<T>>;
@group(0) @binding(4) var<storage, read_write> PO: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> PML: array<vec2f>;
var<workgroup> Qs: array<${V2}, 1024>;  // [d][query pair]
var<workgroup> Ks: array<${V4}, 512>;   // [d][key quad]
var<workgroup> Vs: array<${V4}, 512>;   // [key][d quad]
var<workgroup> Ps: array<${S1}, 1024>;    // [key][query]
var<workgroup> rm: array<f32, 256>;     // [query][key quad]
var<workgroup> rs: array<f32, 256>;
var<workgroup> mrow: array<f32, 32>;
var<workgroup> lrow: array<f32, 32>;
var<workgroup> crow: array<f32, 32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let h = wg.y; let split = wg.z; let r0 = wg.x * 32u;
  let ks = ((p.Lk + ${ns - 1}u) / ${ns}u + 31u) / 32u * 32u;
  let kbeg = split * ks; let kend = min(kbeg + ks, p.Lk);
  for (var b = li; b < 256u; b += 128u) {  // Q, scaled, transposed: 2 queries x 4 dims per block
    let qp = b / 16u; let d4 = b % 16u;
    var a = vec4f(0.0); var c = vec4f(0.0);
    if (r0 + 2u * qp < p.M) { a = vec4f(Q[(h * ${mpad}u + r0 + 2u * qp) * 16u + d4]) * 0.125; }
    if (r0 + 2u * qp + 1u < p.M) { c = vec4f(Q[(h * ${mpad}u + r0 + 2u * qp + 1u) * 16u + d4]) * 0.125; }
    for (var e = 0u; e < 4u; e++) { Qs[(d4 * 4u + e) * 16u + qp] = ${V2}(vec2f(a[e], c[e])); }
  }
  if (li < 32u) { mrow[li] = -1e30; lrow[li] = 0.0; }
  let qp = li / 8u; let kq = li % 8u;   // scores: queries 2qp, 2qp + 1 x keys 4kq .. 4kq + 3
  let qg = li / 16u; let dg = li % 16u; // output: queries 4qg .. 4qg + 3 x dims 4dg .. 4dg + 3
  var acc: array<vec4f, 4>;
  for (var k0 = kbeg; k0 < kend; k0 += 32u) {
    {  // K transposed: 4 keys x 4 dims per thread; V as is
      let kk = li / 16u; let d4 = li % 16u;
      var v: array<vec4f, 4>;
      for (var e = 0u; e < 4u; e++) {
        let key = k0 + kk * 4u + e;
        if (key < kend) { v[e] = vec4f(KC[(h * ${kvn}u + key) * 16u + d4]); } else { v[e] = vec4f(0.0); }
      }
      for (var e = 0u; e < 4u; e++) { Ks[(d4 * 4u + e) * 8u + kk] = ${V4}(vec4f(v[0][e], v[1][e], v[2][e], v[3][e])); }
      for (var i = li; i < 512u; i += 128u) {
        let key = k0 + i / 16u;
        if (key < kend) { Vs[i] = ${V4}(VC[(h * ${kvn}u + key) * 16u + i % 16u]); } else { Vs[i] = ${V4}(); }
      }
    }
    workgroupBarrier();
    var s0 = vec4f(0.0); var s1 = vec4f(0.0);
    ${hm ? `for (var d0 = 0u; d0 < 64u; d0 += 16u) {  // f16 products, summed in f32 every 16 dims
      var h0 = vec4<f16>(); var h1 = vec4<f16>();
      for (var d = d0; d < d0 + 16u; d++) { let q = Qs[d * 16u + qp]; let k = Ks[d * 8u + kq]; h0 += q.x * k; h1 += q.y * k; }
      s0 += vec4f(h0); s1 += vec4f(h1);
    }` : `for (var d = 0u; d < 64u; d++) { let q = Qs[d * 16u + qp]; let k = Ks[d * 8u + kq]; s0 += q.x * k; s1 += q.y * k; }`}
    for (var j = 0u; j < 4u; j++) { if (k0 + kq * 4u + j >= kend) { s0[j] = -1e30; s1[j] = -1e30; } }
    rm[(2u * qp) * 8u + kq] = max(max(s0.x, s0.y), max(s0.z, s0.w));
    rm[(2u * qp + 1u) * 8u + kq] = max(max(s1.x, s1.y), max(s1.z, s1.w));
    workgroupBarrier();
    var mn = vec2f(-1e30);
    for (var j = 0u; j < 8u; j++) { mn = max(mn, vec2f(rm[(2u * qp) * 8u + j], rm[(2u * qp + 1u) * 8u + j])); }
    mn = max(mn, vec2f(mrow[2u * qp], mrow[2u * qp + 1u]));
    let e0 = exp(s0 - mn.x); let e1 = exp(s1 - mn.y);
    for (var j = 0u; j < 4u; j++) { Ps[(kq * 4u + j) * 32u + 2u * qp] = ${S1}(e0[j]); Ps[(kq * 4u + j) * 32u + 2u * qp + 1u] = ${S1}(e1[j]); }
    rs[(2u * qp) * 8u + kq] = e0.x + e0.y + e0.z + e0.w;
    rs[(2u * qp + 1u) * 8u + kq] = e1.x + e1.y + e1.z + e1.w;
    workgroupBarrier();
    if (kq == 0u) {
      for (var i = 0u; i < 2u; i++) {
        let q = 2u * qp + i;
        var sum = 0.0;
        for (var j = 0u; j < 8u; j++) { sum += rs[q * 8u + j]; }
        let corr = exp(mrow[q] - mn[i]);
        lrow[q] = lrow[q] * corr + sum; mrow[q] = mn[i]; crow[q] = corr;
      }
    }
    workgroupBarrier();
    for (var i = 0u; i < 4u; i++) { acc[i] *= crow[4u * qg + i]; }
    ${hm ? `var pv = array<vec4<f16>, 4>();  // f16 products over the 32 keys (probabilities <= 1), then into f32
    for (var key = 0u; key < 32u; key++) {
      let v = Vs[key * 16u + dg];
      let pb = key * 32u + 4u * qg;
      pv[0] += Ps[pb] * v; pv[1] += Ps[pb + 1u] * v; pv[2] += Ps[pb + 2u] * v; pv[3] += Ps[pb + 3u] * v;
    }
    for (var i = 0u; i < 4u; i++) { acc[i] += vec4f(pv[i]); }` : `for (var key = 0u; key < 32u; key++) {
      let v = Vs[key * 16u + dg];
      let pb = key * 32u + 4u * qg;
      acc[0] += Ps[pb] * v; acc[1] += Ps[pb + 1u] * v; acc[2] += Ps[pb + 2u] * v; acc[3] += Ps[pb + 3u] * v;
    }`}
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = r0 + 4u * qg + i;
    if (r < p.M) {
      let o = (split * ${HEADS}u + h) * ${mpad}u + r;
      PO[o * 16u + dg] = acc[i];
      if (dg == 0u) { PML[o] = vec2f(mrow[4u * qg + i], lrow[4u * qg + i]); }
    }
  }
}`;
}

// merge the splits' partial attention, then the gate: O = softmax(q k) v * sigmoid(gate)
function attnCombineShader(f16, { mpad, ns }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> PO: array<f32>;
@group(0) @binding(2) var<storage, read> PML: array<vec2f>;
@group(0) @binding(3) var<storage, read> QKV: array<T>;
@group(0) @binding(4) var<storage, read_write> O: array<T>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) d: u32) {
  _ = p.M;
  let r = wg.x; let h = wg.y;
  var m = -1e30;
  for (var s = 0u; s < ${ns}u; s++) { m = max(m, PML[(s * ${HEADS}u + h) * ${mpad}u + r].x); }
  var l = 0.0; var o = 0.0;
  for (var s = 0u; s < ${ns}u; s++) {
    let i = (s * ${HEADS}u + h) * ${mpad}u + r;
    let w = exp(PML[i].x - m);
    l += w * PML[i].y; o += w * PO[i * 64u + d];
  }
  let col = h * 64u + d;
  O[r * ${D}u + col] = T(o / l * sigm(f32(QKV[r * ${4 * D}u + ${3 * D}u + col])));
}`;
}

// shared by the kernels that read the conditioning images: DIMG [2][H][W] depth code, FIMG [2][H][W][2] flow px
const fl = (x) => (Number.isInteger(x) ? `${x}.0` : `${x}`);

function imgCommon(g) {
  return `
const NPW = ${g.npw}u; const W_IMG = ${g.npw * 16}u; const HW = ${g.nph * 16 * g.npw * 16}u;
fn fcode(f: f32) -> f32 { return sign(f) * log(1.0 + ${fl(g.knee)} * abs(f) / ${fl(g.side)}) / ${fl(Math.log1p(g.knee))}; }
// pixel q = (t, py, px) of token tok -> (depth * 2 - 1, flow codes)
fn pixc(tok: u32, q: u32) -> vec3f {
  let t = q / 256u; let y = (tok / NPW) * 16u + (q / 16u) % 16u; let x = (tok % NPW) * 16u + q % 16u;
  let i = t * HW + y * W_IMG + x;
  return vec3f(DIMG[i] * 2.0 - 1.0, fcode(FIMG[2u * i]), fcode(FIMG[2u * i + 1u]));
}`;
}

function condInShader(f16, g) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> DIMG: array<f32>;
@group(0) @binding(2) var<storage, read> FIMG: array<f32>;
@group(0) @binding(3) var<storage, read_write> CI: array<T>;
${imgCommon(g)}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  _ = p.M;
  let tok = gid.y; let j = gid.x;
  if (j >= ${CIN_PAD}u) { return; }
  var v = 0.0;
  if (j < 512u) { v = pixc(tok, j).x; }
  else if (j < 1536u) { let c = pixc(tok, (j - 512u) / 2u); v = select(c.z, c.y, j % 2u == 0u); }
  else if (j == 1536u) { v = p.gw; } else if (j == 1537u) { v = p.gs; }
  CI[tok * ${CIN_PAD}u + j] = T(v);
}`;
}

function noiseShader(f16, { n }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read_write> Z: array<f32>;
@group(0) @binding(2) var<storage, read_write> FR: array<f32>;
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn unif(h: u32) -> f32 { return (f32(h >> 8u) + 0.5) / 16777216.0; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  _ = p.M;
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= ${(n * PD) / 2}u) { return; }  // pairs: Box-Muller gives two normals
  for (var b = 0u; b < 2u; b++) {
    let h1 = pcg(pcg(p.seed * 2u + b) ^ (2u * i)); let h2 = pcg(h1 ^ (2u * i + 1u));
    let rad = sqrt(-2.0 * log(unif(h1))); let th = 6.283185307 * unif(h2);
    if (b == 0u) { Z[2u * i] = rad * cos(th); Z[2u * i + 1u] = rad * sin(th); }
    else { FR[2u * i] = rad * cos(th); FR[2u * i + 1u] = rad * sin(th); }
  }
}`;
}

function anchorShader(f16) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> AN: array<f32>;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  for (var d = li; d < ${D}u; d += 128u) { X[wg.x * ${D}u + d] = AN[d]; }
}`;
}


// ---- pixel head: per pixel, its code (bilinear over the 3 x 3 neighboring tokens' codes) + position code and its
// depth / flow -> 64 -> 3. XP [pixels][xpw] (k + 3 used), then a GEMM (gelu) to HP [pixels][64], then 64 -> 3 into Y.
function pixInShader(f16, g, a) {
  const { k, xpw, hpad, codeCol } = a;
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> NBR: array<u32>;
@group(0) @binding(3) var<storage, read> PP: array<f32>;
@group(0) @binding(4) var<storage, read> DIMG: array<f32>;
@group(0) @binding(5) var<storage, read> FIMG: array<f32>;
@group(0) @binding(6) var<storage, read_write> XP: array<T>;
${imgCommon(g)}
var<workgroup> codes: array<f32, ${9 * k}>;
fn bw(i: u32) -> vec3f { let u = (f32(i) - 7.5) / 16.0; return vec3f(max(-u, 0.0), 1.0 - abs(u), max(u, 0.0)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  let tok = wg.x / 2u; let q = (wg.x % 2u) * 256u + li;
  for (var i = li; i < ${9 * k}u; i += 256u) { codes[i] = Y[NBR[tok * 9u + i / ${k}u] * ${hpad}u + ${codeCol}u + i % ${k}u]; }
  workgroupBarrier();
  let wy = bw((q / 16u) % 16u); let wx = bw(q % 16u);
  let o = (tok * 512u + q) * ${xpw}u;
  for (var k = 0u; k < ${k}u; k++) {
    var z = PP[q * ${k}u + k];
    for (var a = 0u; a < 3u; a++) {
      for (var b = 0u; b < 3u; b++) { z += wy[a] * wx[b] * codes[(a * 3u + b) * ${k}u + k]; }
    }
    XP[o + k] = T(z);
  }
  let pc = pixc(tok, q);
  XP[o + ${k}u] = T(pc.x); XP[o + ${k + 1}u] = T(pc.y); XP[o + ${k + 2}u] = T(pc.z);
  for (var k = ${k + 3}u; k < ${xpw}u; k++) { XP[o + k] = T(0.0); }
}`;
}

function pixOutShader(f16, { n, hpad }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> HP: array<vec4<T>>;
@group(0) @binding(2) var<storage, read> W2: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> w2: array<vec4f, 48>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  if (li < 48u) { let c = li / 16u; let j = li % 16u; w2[li] = vec4f(W2[c * 64u + 4u * j], W2[c * 64u + 4u * j + 1u], W2[c * 64u + 4u * j + 2u], W2[c * 64u + 4u * j + 3u]); }
  workgroupBarrier();
  let row = gid.x;
  if (row >= ${n * 512}u) { return; }
  var o = vec3f(0.0);
  for (var j = 0u; j < 16u; j++) {
    let h = vec4f(HP[row * 16u + j]);
    o += vec3f(dot(h, w2[j]), dot(h, w2[16u + j]), dot(h, w2[32u + j]));
  }
  let yi = (row / 512u) * ${hpad}u + (row % 512u) * 3u;
  Y[yi] += o.x; Y[yi + 1u] += o.y; Y[yi + 2u] += o.z;
}`;
}

// ---- patch decoder stem input: the 2 x 2 px of every 1/2 res position of the haloed window, 9 channels each
// (rgb in, rgb so far, depth / flow) -> XS [n * P1][48] (36 used)
function stemInShader(f16, g, { G1, P1, hl, hpad }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> ZIN: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;
@group(0) @binding(3) var<storage, read> NBR: array<u32>;
@group(0) @binding(4) var<storage, read> DIMG: array<f32>;
@group(0) @binding(5) var<storage, read> FIMG: array<f32>;
@group(0) @binding(6) var<storage, read_write> XS: array<T>;
${imgCommon(g)}
@compute @workgroup_size(${G1 * G1})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  for (var t = 0u; t < 2u; t++) {
  let tok = wg.x; let pos = t * ${G1 * G1}u + li; let i = (pos / ${G1}u) % ${G1}u; let j = pos % ${G1}u;
  let row = (tok * ${P1}u + pos) * 48u;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let py = i32(2u * i + dy) - ${hl}; let px = i32(2u * j + dx) - ${hl};
      let a = select(select(2, 1, py < 16), 0, py < 0); let b = select(select(2, 1, px < 16), 0, px < 0);
      let nb = NBR[tok * 9u + u32(a * 3 + b)];
      let q = t * 256u + u32(py - (a - 1) * 16) * 16u + u32(px - (b - 1) * 16);
      let o = row + (dy * 2u + dx) * 9u;
      let pc = pixc(nb, q);
      for (var c = 0u; c < 3u; c++) { XS[o + c] = T(ZIN[nb * ${PD}u + q * 3u + c]); XS[o + 3u + c] = T(Y[nb * ${hpad}u + q * 3u + c]); }
      XS[o + 6u] = T(pc.x); XS[o + 7u] = T(pc.y); XS[o + 8u] = T(pc.z);
    }
  }
  for (var k = 36u; k < 48u; k++) { XS[row + k] = T(0.0); }
  }
}`;
}

// ---- patch decoder: per token a 2 x (16 + 2 hl)^2 px window (its patchlet + an hl px halo), 2 x 2 merged: 2 x G1 x G1
// positions of c1 channels, then 2 x G2 x G2 of c2 (e.g. hl = 4, c1 = 32: 2 x 12 x 12 x 32, then 2 x 6 x 6 x 64).

// depthwise (3, 5, 5) conv, factored: 5 x 5 per frame (zero padded), then 3 temporal taps over the 2 frames; then
// per position the LayerNorm over channels and the token's adaLN -> HN, the ConvNeXt MLP's input
const dwPositions = (C) => Math.floor(256 / C); // positions of a token per workgroup, C threads (channels) each
function dwShader(f16, { C, G, modCol, hpad }) {
  const P = 2 * G * G, NP = dwPositions(C), C2 = 2 ** Math.ceil(Math.log2(C)); // C2: C rounded up to a power of 2
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> DW: array<f32>;
@group(0) @binding(3) var<storage, read> DT: array<f32>;
@group(0) @binding(4) var<storage, read> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> HN: array<T>;
var<workgroup> red: array<vec2f, ${NP * C}>;
fn reduce(li: u32, c: u32, v: vec2f) -> vec2f {  // over the C channels of a position
  red[li] = v; workgroupBarrier();
  for (var s = ${C2 / 2}u; s > 0u; s >>= 1u) { if (c < s && c + s < ${C}u) { red[li] += red[li + s]; } workgroupBarrier(); }
  let r = red[li - c]; workgroupBarrier(); return r;
}
@compute @workgroup_size(${NP * C})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  let c = li % ${C}u; let pos = wg.x * ${NP}u + li / ${C}u;  // a 1/G grid position of the first frame
  let tok = wg.y; let i = min(pos, ${G * G - 1}u) / ${G}u; let j = min(pos, ${G * G - 1}u) % ${G}u;
  var h = vec2f(0.0);
  for (var u = 0u; u < 5u; u++) {
    let ii = i32(i + u) - 2;
    if (ii < 0 || ii >= ${G}) { continue; }
    for (var v = 0u; v < 5u; v++) {
      let jj = i32(j + v) - 2;
      if (jj < 0 || jj >= ${G}) { continue; }
      let w = DW[c * 25u + u * 5u + v];
      let a = (tok * ${P}u + u32(ii) * ${G}u + u32(jj)) * ${C}u + c;
      h += w * vec2f(X[a], X[a + ${G * G * C}u]);
    }
  }
  let o = vec2f(h.x * DT[${C}u + c] + h.y * DT[${2 * C}u + c], h.x * DT[c] + h.y * DT[${C}u + c]);
  let mean = reduce(li, c, o) / ${C}.0;
  let d = o - mean;
  let inv = inverseSqrt(reduce(li, c, d * d) / ${C}.0 + 1e-6);
  let m = tok * ${hpad}u + ${modCol}u;
  let v = d * inv * (1.0 + Y[m + ${C}u + c]) + Y[m + c];
  if (pos < ${G * G}u) {
    let r = (tok * ${P}u + i * ${G}u + j) * ${C}u + c;
    HN[r] = T(v.x); HN[r + ${G * G * C}u] = T(v.y);
  }
}`;
}

// LayerNorm -> c1 -> 12 = 2 x 2 px of rgb residual; the patchlet's own pixels are finished: clamped, then pass 1
// renoises them for pass 2, pass 2 writes them out (and to the frame texture)
function decOutShader(f16, g, last, S, { c1, G1, P1, hl, hpad }) {
  return `${header(f16)}
@group(0) @binding(1) var<storage, read> D0: array<f32>;
@group(0) @binding(2) var<storage, read> WO: array<f32>;
@group(0) @binding(3) var<storage, read> Y: array<f32>;
${last ? "" : "@group(0) @binding(4) var<storage, read> FR: array<f32>;"}
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
${last ? "@group(0) @binding(6) var TEX: texture_storage_2d_array<rgba8unorm, write>;" : ""}
@compute @workgroup_size(${G1 * G1})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  _ = p.M;
  for (var t = 0u; t < 2u; t++) {
  let tok = wg.x; let pos = t * ${G1 * G1}u + li; let i = (pos / ${G1}u) % ${G1}u; let j = pos % ${G1}u;
  var v: array<f32, ${c1}>;
  var mean = 0.0;
  for (var c = 0u; c < ${c1}u; c++) { v[c] = D0[(tok * ${P1}u + pos) * ${c1}u + c]; mean += v[c]; }
  mean /= ${c1}.0;
  var var_ = 0.0;
  for (var c = 0u; c < ${c1}u; c++) { let d = v[c] - mean; var_ += d * d; }
  let inv = inverseSqrt(var_ / ${c1}.0 + 1e-6);
  for (var c = 0u; c < ${c1}u; c++) { v[c] = (v[c] - mean) * inv; }
  for (var sub = 0u; sub < 4u; sub++) {
    let py = i32(2u * i + sub / 2u) - ${hl}; let px = i32(2u * j + sub % 2u) - ${hl};
    if (py < 0 || py >= 16 || px < 0 || px >= 16) { continue; }
    let q = t * 256u + u32(py) * 16u + u32(px);
    var rgb = vec3f(0.0);
    for (var c = 0u; c < 3u; c++) {
      var s = 0.0;
      for (var k = 0u; k < ${c1}u; k++) { s += WO[(sub * 3u + c) * ${c1}u + k] * v[k]; }
      let y = clamp(Y[tok * ${hpad}u + q * 3u + c] + s, -1.0, 1.0);
      ${last ? "OUT[tok * 1536u + q * 3u + c] = y;" : `OUT[tok * 1536u + q * 3u + c] = ${1 - S} * y + ${S} * FR[tok * 1536u + q * 3u + c];`}
      rgb[c] = y;
    }
    ${last ? `textureStore(TEX, vec2u((tok % ${g.npw}u) * 16u + u32(px), (tok / ${g.npw}u) * 16u + u32(py)), p.frame + t, vec4f(rgb * 0.5 + 0.5, 1.0));` : ""}
  }
  }
}`;
}

// ---------------------------------------------------------------------------------------------- engine

export class Engine {
  // geometry: nph x npw tokens (16 px each); frames: layers of the output texture ring (a multiple of 2)
  // f16: activations and weights stored as f16, and the GEMMs' products, summed in f32 every 32
  constructor(device, model, { nph = 11, npw = 20, f16 = false, frames = 8 } = {}) {
    this.dev = device;
    this.halfMath = f16;
    this.ns = 4;
    this.f16 = f16;
    this.nph = nph;
    this.npw = npw;
    this.n = nph * npw;
    this.frames = frames;
    const n = this.n;
    this.mpad = Math.ceil((2 * n) / 64) * 64;
    this.npad = Math.ceil(n / 64) * 64;
    this.kvn = 4 * n;
    this.g = { nph, npw, side: ((nph + npw) / 2) * 16, knee: model.cfg.flow_knee };
    this.cfg = model.cfg;
    this.a = decArch(model.cfg);
    this.twoStep = model.twoStep;
    this.framelet = 0;
    this.makeBuffers(model.tensors);
    this.makePipelines();
  }

  buf(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    return this.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  }

  upload(arr, usage) {
    const b = this.buf(arr.byteLength, usage);
    this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }

  // the T type: f16 halves of a Float32Array, or the floats
  typed(f32) {
    if (!this.f16) return f32;
    const h = new Uint16Array(f32.length + (f32.length & 1));
    for (let i = 0; i < f32.length; i++) h[i] = toHalf(f32[i]);
    return h;
  }

  // a Linear weight [N, K] -> W^T [Kpad, Npad] in T, zero padded
  gemmWeight(w, N, K, Kpad = K, Npad = N) {
    const t = new Float32Array(Kpad * Npad);
    for (let o = 0; o < N; o++) for (let k = 0; k < K; k++) t[k * Npad + o] = w[o * K + k];
    return this.upload(this.typed(t));
  }

  makeBuffers(t) {
    const n = this.n, mp = this.mpad, np = this.npad, Tb = this.f16 ? 2 : 4, H = 16 * this.nph, W = 16 * this.npw;
    const { k, c1, c2, P1, P2, headw, hpad, xpw } = this.a;
    const w = (name) => t[name].data;
    this.W = {
      patch_in: this.gemmWeight(w("patch_in"), D, PD),
      cond0: this.gemmWeight(w("cond0"), CD, CIN, CIN_PAD),
      cond2: this.gemmWeight(w("cond2"), CD, CD),
      qkv: this.gemmWeight(w("qkv"), 4 * D, D),
      proj: this.gemmWeight(w("proj"), D, D),
      modcat: this.gemmWeight(w("modcat"), MODW, CD),
      headcat: this.gemmWeight(w("headcat"), headw, D, D, hpad),
      mlp0: [...Array(DEPTH).keys()].map((i) => this.gemmWeight(w(`mlp0_${i}`), FF, D)),
      mlp2: [...Array(DEPTH).keys()].map((i) => this.gemmWeight(w(`mlp2_${i}`), D, FF)),
      anchor: this.upload(w("anchor")),
      qknorm: this.upload(new Float32Array([...w("q_norm"), ...w("k_norm")])),
    };
    const gw = (name, Kpad) => this.gemmWeight(w(name), t[name].shape[0], t[name].shape[1], Kpad);
    if (k) {
      this.W.pix_pos = this.upload(w("pix_pos"));
      this.W.pix_mlp0 = gw("pix_mlp0", xpw);
      this.W.pix_mlp2 = this.upload(w("pix_mlp2"));
    }
    this.W.dec_stem = gw("dec_stem", 48);
    this.W.dec_down = gw("dec_down");
    this.W.dec_up = gw("dec_up");
    this.W.dec_out = this.upload(w("dec_out"));
    for (const c of ["enc", "mid0", "mid1", "dec"]) {
      for (const k of ["dt", "dw"]) this.W[`dec_${c}_${k}`] = this.upload(w(`dec_${c}_${k}`));
      for (const k of ["mlp0", "mlp2"]) this.W[`dec_${c}_${k}`] = gw(`dec_${c}_${k}`);
    }

    // static tables: spatial RoPE (cos, sin) [n][24] and every token's 3 x 3 neighborhood (clamped) [n][9]
    const cfg = this.cfg, hd2 = HD / 2, ns = (hd2 - cfg.time_pairs) / 2;
    const [s0, s1] = cfg.space_freqs;
    const fs = [...Array(ns).keys()].map((i) => 10 ** (Math.log10(s0) + ((Math.log10(s1) - Math.log10(s0)) * i) / (ns - 1)));
    const rope = new Float32Array(n * 2 * ns * 2), nbr = new Uint32Array(n * 9);
    for (let r = 0; r < this.nph; r++)
      for (let c = 0; c < this.npw; c++) {
        const tok = r * this.npw + c, yx = [((r + 0.5) / this.nph) * 2 - 1, ((c + 0.5) / this.npw) * 2 - 1];
        for (let a = 0; a < 2; a++)
          for (let i = 0; i < ns; i++) {
            const ang = Math.PI * yx[a] * fs[i], o = (tok * 2 * ns + a * ns + i) * 2;
            rope[o] = Math.cos(ang);
            rope[o + 1] = Math.sin(ang);
          }
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++) {
            const rr = Math.min(Math.max(r + dr, 0), this.nph - 1), cc = Math.min(Math.max(c + dc, 0), this.npw - 1);
            nbr[tok * 9 + (dr + 1) * 3 + dc + 1] = rr * this.npw + cc;
          }
      }
    const tf = cfg.time_freqs, tp = cfg.time_pairs;
    this.timeFreqs = [...Array(tp).keys()].map((i) => 10 ** (Math.log10(tf[0]) + ((Math.log10(tf[1]) - Math.log10(tf[0])) * i) / (tp - 1)));
    this.rope = this.upload(rope);
    this.nbr = this.upload(nbr);

    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this.uni = this.buf(112, U);
    this.B = {
      x: this.buf(mp * D * 4),
      h: this.buf(mp * D * Tb),
      qkvg: this.buf(mp * 4 * D * Tb),
      q: this.buf(HEADS * mp * HD * Tb),
      o: this.buf(mp * D * Tb),
      u: this.buf(mp * FF * Tb),
      cin: this.buf(np * CIN_PAD * Tb),
      c1: this.buf(np * CD * Tb),
      c: this.buf(np * CD * Tb),
      mod: this.buf(np * MODW * 4),
      hout: this.buf(np * D * Tb),
      y: this.buf(np * hpad * 4),
      z1: this.buf(n * PD * 4),
      fresh: this.buf(n * PD * 4),
      z2: this.buf(n * PD * 4),
      out: this.buf(n * PD * 4),
      dimg: this.buf(2 * H * W * 4),
      fimg: this.buf(2 * H * W * 2 * 4),
      xp: k ? this.buf(n * 512 * xpw * Tb) : null,
      hp: k ? this.buf(n * 512 * 64 * Tb) : null,
      xs: this.buf(n * P1 * 48 * Tb),
      s0: this.buf(n * P1 * c1 * 4),
      d0: this.buf(n * P1 * c1 * 4),
      m0: this.buf(n * P2 * c2 * 4),
      hn: this.buf(n * Math.max(P1 * c1, P2 * c2) * Tb),
      un: this.buf(n * Math.max(P1 * c1, P2 * c2) * 2 * Tb),
      po: this.buf(this.ns * HEADS * mp * HD * 4),
      pml: this.buf(this.ns * HEADS * mp * 8),
      kv: [0, 1].map(() => [this.buf(DEPTH * HEADS * this.kvn * HD * Tb), this.buf(DEPTH * HEADS * this.kvn * HD * Tb)]),
    };
    this.tex = this.dev.createTexture({
      size: [W, H, this.frames],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.texView = this.tex.createView({ dimension: "2d-array" });
  }

  pipe(code) {
    const m = this.dev.createShaderModule({ code });
    return this.dev.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
  }

  bind(pipe, bufs) {
    const entries = bufs
      .map((b, i) =>
        b === null ? null : b instanceof GPUTextureView ? { binding: i, resource: b } : { binding: i, resource: b.buffer ? b : { buffer: b } },
      )
      .filter((e) => e);
    return this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  }

  makePipelines() {
    const f16 = this.f16, n = this.n, B = this.B, g = this.g, a = this.a;
    const { k, c1, c2, G1, G2, P1, P2, hpad, xpw } = a;
    const bn = (N, pref = 64) => [pref, 64, 48, 32, 16].find((b) => N % b === 0); // a column tile dividing N
    this.tiles = new Map();
    const gemm = (o) => {
      const cfg = { BM: 64, BN: 64, BK: o.K % 32 ? 16 : 32, halfMath: this.halfMath, ...o };
      const pipe = this.pipe(gemmShader({ f16, AT: "T", ...cfg }));
      this.tiles.set(pipe, cfg);
      return pipe;
    };
    const storeOut = (bind, ty = "T") => `@group(0) @binding(3) var<storage, read_write> C: array<${ty}>;` + (bind || "");
    const P = {};
    P.patch_in = gemm({ AT: "f32", K: PD, N: D, NREAL: D, lda: PD, rows: `${n}u`, BM: 32,
      extra: storeOut("", "f32"), epi: `C[(r + p.rowOff) * ${D}u + c] = v;` }); // prettier-ignore
    P.cond0 = gemm({ K: CIN_PAD, N: CD, NREAL: CD, lda: CIN_PAD, rows: `${n}u`, BM: 32, BN: 32,
      extra: storeOut(), epi: `C[r * ${CD}u + c] = T(silu(v));` }); // prettier-ignore
    P.cond2 = gemm({ K: CD, N: CD, NREAL: CD, lda: CD, rows: `${n}u`, BM: 32, BN: 32,
      extra: storeOut(), epi: `C[r * ${CD}u + c] = T(silu(v));` }); // prettier-ignore
    P.modcat = gemm({ K: CD, N: MODW, NREAL: MODW, lda: CD, rows: `${n}u`,
      extra: storeOut("", "f32"), epi: `C[r * ${MODW}u + c] = v;` }); // prettier-ignore
    P.qkv = gemm({ K: D, N: 4 * D, NREAL: 4 * D, lda: D, rows: "p.M", BM: 32, extra: storeOut(), epi: `C[r * ${4 * D}u + c] = T(v);` }); // prettier-ignore
    const resgate = (col) => ({
      extra: `@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> MOD: array<f32>;`,
      epi: `C[r * ${D}u + c] += MOD[(r % ${n}u) * ${MODW}u + ${col}u + c] * v;`,
    });
    P.proj = gemm({ K: D, N: D, NREAL: D, lda: D, rows: "p.M", BM: 32, BN: 32, ...resgate(2 * D) });
    P.mlp0 = gemm({ K: D, N: FF, NREAL: FF, lda: D, rows: "p.M", BM: 32, extra: storeOut(), epi: `C[r * ${FF}u + c] = T(gelu(v));` }); // prettier-ignore
    P.mlp2 = gemm({ K: FF, N: D, NREAL: D, lda: FF, rows: "p.M", BM: 32, BN: 32, ...resgate(5 * D) });
    P.headcat = gemm({ K: D, N: hpad, NREAL: a.headw, lda: D, rows: `${n}u`,
      extra: storeOut("", "f32"), epi: `C[r * ${hpad}u + c] = v;` }); // prettier-ignore
    P.ln1 = this.pipe(lnShader(f16, { shift: 0, scale: D, xOff: "0u", ntok: n }));
    P.ln2 = this.pipe(lnShader(f16, { shift: 3 * D, scale: 4 * D, xOff: "0u", ntok: n }));
    P.lnout = this.pipe(lnShader(f16, { shift: 6 * D, scale: 7 * D, xOff: "p.rowOff", ntok: n }));
    P.qkrope = this.pipe(qkRopeShader(f16, { ntok: n, mpad: this.mpad, kvn: this.kvn }));
    P.attn = this.pipe(attnShader(f16, { mpad: this.mpad, kvn: this.kvn, ns: this.ns, hm: this.halfMath }));
    P.attnc = this.pipe(attnCombineShader(f16, { mpad: this.mpad, ns: this.ns }));
    P.condin = this.pipe(condInShader(f16, g));
    P.noise = this.pipe(noiseShader(f16, { n }));
    P.anchor = this.pipe(anchorShader(f16));
    const half = n * P1, quarter = n * P2, f32out = storeOut("", "f32");
    if (k) {
      P.pixin = this.pipe(pixInShader(f16, g, a));
      P.pix1 = gemm({ K: xpw, N: 64, NREAL: 64, lda: xpw, rows: `${n * 512}u`, extra: storeOut(), epi: `C[r * 64u + c] = T(gelu(v));` }); // prettier-ignore
      P.pixout = this.pipe(pixOutShader(f16, { n, hpad }));
    }
    P.stemin = this.pipe(stemInShader(f16, g, a));
    P.stem = gemm({ K: 48, N: c1, NREAL: c1, lda: 48, rows: `${half}u`, BN: bn(c1, 32), extra: f32out, epi: `C[r * ${c1}u + c] = v;` }); // prettier-ignore
    for (const [key, C, G] of [["enc", c1, G1], ["mid0", c2, G2], ["mid1", c2, G2], ["dec", c1, G1]]) {
      const rows = `${n * 2 * G * G}u`;
      P[`dw_${key}`] = this.pipe(dwShader(f16, { C, G, modCol: a.decMod[key], hpad }));
      P[`cnx1_${key}`] = gemm({ K: C, N: 2 * C, NREAL: 2 * C, lda: C, rows, BN: bn(2 * C), extra: storeOut(), epi: `C[r * ${2 * C}u + c] = T(gelu(v));` }); // prettier-ignore
      P[`cnx2_${key}`] = gemm({ K: 2 * C, N: C, NREAL: C, lda: 2 * C, rows, BN: bn(C, 32), extra: f32out, epi: `C[r * ${C}u + c] += v;` }); // prettier-ignore
    }
    // the 2 x 2 merge of the skip (S0) as the A operand: row (token, 1/4 res position), k = (sub-pixel, channel)
    const merged = (tok, rest) => `((${tok} * ${P1}u + (${rest} / ${G2 * G2}u) * ${G1 * G1}u
      + (2u * ((${rest} / ${G2}u) % ${G2}u) + k / ${2 * c1}u) * ${G1}u + 2u * (${rest} % ${G2}u) + (k / ${c1}u) % 2u) * ${c1}u
      + k % ${c1}u) / 4u`;
    P.down = gemm({ AT: "f32", K: 4 * c1, N: c2, NREAL: c2, lda: 4 * c1, rows: `${quarter}u`, BN: bn(c2), extra: f32out,
      aIdx: merged(`(r / ${P2}u)`, `(r % ${P2}u)`), epi: `C[r * ${c2}u + c] = v;` }); // prettier-ignore
    P.up = gemm({ AT: "f32", K: c2, N: 4 * c1, NREAL: 4 * c1, lda: c2, rows: `${quarter}u`, BN: bn(4 * c1),
      extra: `@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> S0: array<f32>;`,
      epi: `let tok = r / ${P2}u; let pos = r % ${P2}u; let sub = c / ${c1}u;
      let a = (tok * ${P1}u + (pos / ${G2 * G2}u) * ${G1 * G1}u + (2u * ((pos / ${G2}u) % ${G2}u) + sub / 2u) * ${G1}u
        + 2u * (pos % ${G2}u) + sub % 2u) * ${c1}u + c % ${c1}u;
      C[a] = v + S0[a];` }); // prettier-ignore
    P.out1 = this.pipe(decOutShader(f16, g, false, this.twoStep, a));
    P.out2 = this.pipe(decOutShader(f16, g, true, this.twoStep, a));
    this.P = P;

    // bind groups, all static: per pass (its input z and KV cache), per layer (its weights, its KV slice)
    const u = this.uni, layerBytes = HEADS * this.kvn * HD * (f16 ? 2 : 4);
    const G = {};
    G.condin = this.bind(P.condin, [u, B.dimg, B.fimg, B.cin]);
    G.cond0 = this.bind(P.cond0, [u, B.cin, this.W.cond0, B.c1]);
    G.cond2 = this.bind(P.cond2, [u, B.c1, this.W.cond2, B.c]);
    G.modcat = this.bind(P.modcat, [u, B.c, this.W.modcat, B.mod]);
    G.noise = this.bind(P.noise, [u, B.z1, B.fresh]);
    G.anchor = this.bind(P.anchor, [u, this.W.anchor, B.x]);
    G.ln1 = this.bind(P.ln1, [u, B.x, B.mod, B.h]);
    G.ln2 = this.bind(P.ln2, [u, B.x, B.mod, B.h]);
    G.lnout = this.bind(P.lnout, [u, B.x, B.mod, B.hout]);
    G.qkv = this.bind(P.qkv, [u, B.h, this.W.qkv, B.qkvg]);
    G.proj = this.bind(P.proj, [u, B.o, this.W.proj, B.x, B.mod]);
    G.mlp0 = this.W.mlp0.map((wt) => this.bind(P.mlp0, [u, B.h, wt, B.u]));
    G.mlp2 = this.W.mlp2.map((wt) => this.bind(P.mlp2, [u, B.u, wt, B.x, B.mod]));
    G.headcat = this.bind(P.headcat, [u, B.hout, this.W.headcat, B.y]);
    G.attnc = this.bind(P.attnc, [u, B.po, B.pml, B.qkvg, B.o]);
    if (k) {
      G.pixin = this.bind(P.pixin, [u, B.y, this.nbr, this.W.pix_pos, B.dimg, B.fimg, B.xp]);
      G.pix1 = this.bind(P.pix1, [u, B.xp, this.W.pix_mlp0, B.hp]);
      G.pixout = this.bind(P.pixout, [u, B.hp, this.W.pix_mlp2, B.y]);
    }
    const kvSlice = (b, l) => ({ buffer: b, offset: l * layerBytes, size: layerBytes });
    G.pass = [0, 1].map((ps) => {
      const [kc, vc] = B.kv[ps], zin = ps === 0 ? B.z1 : B.z2;
      return {
        patch_in: this.bind(P.patch_in, [u, zin, this.W.patch_in, B.x]),
        qkrope: [...Array(DEPTH).keys()].map((l) =>
          this.bind(P.qkrope, [u, B.qkvg, this.W.qknorm, this.rope, B.q, kvSlice(kc, l), kvSlice(vc, l)]),
        ),
        attn: [...Array(DEPTH).keys()].map((l) => this.bind(P.attn, [u, B.q, kvSlice(kc, l), kvSlice(vc, l), B.po, B.pml])),
        stemin: this.bind(P.stemin, [u, zin, B.y, this.nbr, B.dimg, B.fimg, B.xs]),
        out: ps === 0
          ? this.bind(P.out1, [u, B.d0, this.W.dec_out, B.y, B.fresh, B.z2])
          : this.bind(P.out2, [u, B.d0, this.W.dec_out, B.y, null, B.out, this.texView]),
      };
    });
    G.stem = this.bind(P.stem, [u, B.xs, this.W.dec_stem, B.s0]);
    G.down = this.bind(P.down, [u, B.s0, this.W.dec_down, B.m0]);
    G.up = this.bind(P.up, [u, B.m0, this.W.dec_up, B.d0, B.s0]);
    for (const [c, x] of [["enc", B.s0], ["mid0", B.m0], ["mid1", B.m0], ["dec", B.d0]]) {
      G[`dw_${c}`] = this.bind(P[`dw_${c}`], [u, x, this.W[`dec_${c}_dw`], this.W[`dec_${c}_dt`], B.y, B.hn]);
      G[`cnx1_${c}`] = this.bind(P[`cnx1_${c}`], [u, B.hn, this.W[`dec_${c}_mlp0`], B.un]);
      G[`cnx2_${c}`] = this.bind(P[`cnx2_${c}`], [u, B.un, this.W[`dec_${c}_mlp2`], x]);
    }
    this.G = G;
  }

  reset() {
    this.framelet = 0;
  }

  // free the GPU memory (the engine is unusable after)
  destroy() {
    const kill = (x) => {
      if (x instanceof GPUBuffer || x instanceof GPUTexture) x.destroy();
      else if (Array.isArray(x)) x.forEach(kill);
      else if (x && typeof x === "object" && !(x instanceof GPUTextureView)) Object.values(x).forEach(kill);
    };
    kill([this.W, this.B, this.uni, this.rope, this.nbr, this.tex]);
  }

  // the per-framelet uniforms: t = the framelet's mean time (s), gaps = [within, since] (s)
  params(t, gaps, seed, frame) {
    const first = this.framelet === 0, n = this.n, i = this.framelet;
    const a = new ArrayBuffer(112), u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = first ? 2 * n : n; // M
    u[1] = first ? n : 0; // rowOff
    u[2] = first ? 0 : (1 + (i % 3)) * n; // kvOff
    u[3] = Math.min(i + 2, 4) * n; // Lk
    u[4] = first ? 1 : 0;
    u[5] = seed >>> 0;
    const g = (x) => Math.log(Math.max(x * 24, 0.05)) / 2;
    f[6] = g(gaps[0]);
    f[7] = g(gaps[1]);
    for (let k = 0; k < 8; k++) {
      const ang = (2 * Math.PI * t * this.timeFreqs[k]) % (2 * Math.PI);
      f[8 + k] = Math.cos(ang);
      f[16 + k] = Math.sin(ang);
    }
    u[24] = frame;
    return a;
  }

  // Encode one framelet: the conditioning images in B.dimg / B.fimg, its time t and gaps; noise generated from
  // SEED. Output frames go to texture layers
  // FRAME, FRAME + 1, and the rgb patchlets to B.out.
  encode(enc, { t, gaps, seed = 0, frame = 0 }) {
    const P = this.P, G = this.G, n = this.n, first = this.framelet === 0;
    const { k, c1, c2, G1, G2, P1, P2, hpad } = this.a;
    this.dev.queue.writeBuffer(this.uni, 0, this.params(t, gaps, seed, frame));
    const M = first ? 2 * n : n;
    const cp = enc.beginComputePass();
    const run = (pipe, bg, x, y = 1, z = 1) => {
      cp.setPipeline(pipe);
      cp.setBindGroup(0, bg);
      cp.dispatchWorkgroups(x, y, z);
    };
    const gm = (pipe, bg, rows, N) => {
      const { BM, BN } = this.tiles.get(pipe);
      run(pipe, bg, Math.ceil(N / BN), Math.ceil(rows / BM));
    };
    run(P.condin, G.condin, Math.ceil(CIN_PAD / 256), n);
    gm(P.cond0, G.cond0, n, CD);
    gm(P.cond2, G.cond2, n, CD);
    gm(P.modcat, G.modcat, n, MODW);
    run(P.noise, G.noise, Math.ceil((n * PD) / 2 / 256));
    for (let ps = 0; ps < 2; ps++) {
      const GP = G.pass[ps];
      gm(P.patch_in, GP.patch_in, n, D);
      if (first) run(P.anchor, G.anchor, n);
      for (let l = 0; l < DEPTH; l++) {
        run(P.ln1, G.ln1, M);
        gm(P.qkv, G.qkv, M, 4 * D);
        run(P.qkrope, GP.qkrope[l], M, HEADS);
        run(P.attn, GP.attn[l], Math.ceil(M / 32), HEADS, this.ns);
        run(P.attnc, G.attnc, M, HEADS);
        gm(P.proj, G.proj, M, D);
        run(P.ln2, G.ln2, M);
        gm(P.mlp0, G.mlp0[l], M, FF);
        gm(P.mlp2, G.mlp2[l], M, D);
      }
      run(P.lnout, G.lnout, n);
      gm(P.headcat, G.headcat, n, hpad);
      if (k) {
        run(P.pixin, G.pixin, 2 * n);
        gm(P.pix1, G.pix1, n * 512, 64);
        run(P.pixout, G.pixout, (n * 512) / 256);
      }
      run(P.stemin, GP.stemin, n);
      gm(P.stem, G.stem, n * P1, c1);
      const cnx = (c, C, GG) => {  // GG: positions per frame
        run(P[`dw_${c}`], G[`dw_${c}`], Math.ceil(GG / dwPositions(C)), n);
        gm(P[`cnx1_${c}`], G[`cnx1_${c}`], n * 2 * GG, 2 * C);
        gm(P[`cnx2_${c}`], G[`cnx2_${c}`], n * 2 * GG, C);
      };
      cnx("enc", c1, G1 * G1);
      gm(P.down, G.down, n * P2, c2);
      cnx("mid0", c2, G2 * G2);
      cnx("mid1", c2, G2 * G2);
      gm(P.up, G.up, n * P2, 4 * c1);
      cnx("dec", c1, G1 * G1);
      run(ps === 0 ? P.out1 : P.out2, GP.out, n);
    }
    cp.end();
    this.framelet++;
  }
}
