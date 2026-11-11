// Port of spokester2/game/world.py: the procedural street, cars, pedestrians, the fixed-gear
// rider (LQR on the bicycle) and the chase camera. Same rules and constants (but a slower rider); its own random
// numbers.
import { GAINS, STEP as GAIN_STEP, V0 as GAIN_V0 } from "./lqr.js";

export const CFG = {
  camera: {
    width: 320, height: 176, hfov_deg: 94.0, fps: 23.75, height_m: 2.0, pitch_deg: -7.5, dist_mean: 4.0,
    dist_std: 1.0, dist_tau: 2.5, lat_mean: 0.6, lat_std: 0.97, lat_tau: 2.5, follow_tau: 2.0, view_dist: 300.0,
    // rising over whatever would pass through the camera: the space kept clear around the eye (half x, half z),
    // the clearance over a top, how far ahead (s) movers are anticipated, the up / down time constants (s) of each of
    // the two smoothing stages (an S-curve, no jolt), the ceiling
    clear_hx: 0.5, clear_hz: 0.6, clearance: 0.35, anticipate: 1.6, rise_tau: 0.2, fall_tau: 0.8, max_height: 5.0,
    curb_margin: 0.5, // the eye stays this far inside the road's edge, clear of parked cars, lampposts and trees
  },
  world: {
    lane_dirs: [-1, -1, 1, 1], rider_lane: 2, lane_w: 3.3, sidewalk_w: 8.0, block_len: 80.0, street_w: 14.0,
    building_h: [6.0, 45.0], setback: [0.0, 10.0], alley_p: 0.4, cars_per_100m: 3.0, car_speed: 9.0, peds_per_100m: 12.0, jaywalk_rate: 0.01,
    level_every_m: 250.0, ramp: 0.25,
    curb_h: 0.15, // sidewalk slab height above the road (m); 0: flat ground
    parked_per_100m: 8.0, // parked cars per side, in a parking lane between the road and the curb
    detail: true, // per-car proportions with wheels and raked windows, swinging arms, lumpy tree crowns, building
    // cornices / upper tiers / awnings
    ped_hit: 0.7, // pedestrians' collision distances (x 0.55 m, z 1.1 m) scaled by this
  },
  rider: {
    start_speed: 27 / 3.6, speed_gain: 0.1, gear_ratio: 48 / 17, wheel_circ: 2.1, wheelbase: 1.0, com_height: 1.0,
    com_ahead: 0.4, max_steer_deg: 30.0, max_steer_rate: 6.0, aim_clip: 2.5, aim_speed: 4.0, wobble: 0.5,
    wobble_speed: 10.0, fall_lean_deg: 70.0, calm_speed: 8.0, aim_frac: 0.5,
  },
}; // prettier-ignore

const STATIC_ID = 2 ** 40;
const PARK_W = 2.4; // parking lane width (m)
const rad = (d) => (d * Math.PI) / 180;
const clip = (x, a, b) => Math.min(Math.max(x, a), b);

// ------------------------------------------------------------------------------------------------- rng

function splitmix(a) {
  a = (a + 0x9e3779b9) | 0;
  let t = a ^ (a >>> 16);
  t = Math.imul(t, 0x21f0aaad);
  t ^= t >>> 15;
  t = Math.imul(t, 0x735a2d97);
  return (t ^ (t >>> 15)) >>> 0;
}

class Rng {
  constructor(...seeds) {
    let h = 0x12345678;
    for (const s of seeds) h = splitmix(h ^ splitmix(s | 0) ^ splitmix(Math.floor(s / 2 ** 32)));
    this.s = [splitmix(h), splitmix(h + 1), splitmix(h + 2), splitmix(h + 3)];
    this.spare = null;
  }
  random() {  // sfc32
    let [a, b, c, d] = this.s;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    this.s = [a, b, c, d];
    return (t >>> 0) / 4294967296;
  }
  uniform(a, b) {
    return a + (b - a) * this.random();
  }
  normal() {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    const u = 1 - this.random(), v = this.random(), r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  integers(n) {
    return Math.floor(this.random() * n);
  }
  choice(xs) {
    return xs[this.integers(xs.length)];
  }
}

// ---------------------------------------------------------------------------------------- 3x3 matrices

// row-major [9]
const mul3 = (A, B) => {
  const C = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
};
const mulv = (A, v) => [
  A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
  A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
  A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
];
const EYE = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function pitchM(p) {
  const c = Math.cos(p), s = Math.sin(p);
  return [1, 0, 0, 0, c, -s, 0, s, c];
}

// local -> world: heading yaw (toward +x) after leaning `lean` about the forward axis (top toward +x)
function rotM(yaw, lean = 0) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cl = Math.cos(lean), sl = Math.sin(lean);
  return mul3([cy, 0, sy, 0, 1, 0, -sy, 0, cy], [cl, sl, 0, -sl, cl, 0, 0, 0, 1]);
}

// a part: off (entity frame), half extents, pitch about the entity's x axis, round (cylinder along x)
const part = (off, half, pitch = 0, round = false) => ({ off, half, pitch, round });

function seg(a, b, wx, wt) {
  const dy = b[1] - a[1], dz = b[2] - a[2];
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  return part(mid, [wx, Math.hypot(dy, dz) / 2 + wt, wt], Math.atan2(dz, dy));
}

// WorldConfig.detail proportions: wheel radius, body height, cabin half length, cabin offset, cabin half height
function carShape(rng, van) {
  if (van) return [rng.uniform(0.33, 0.4), rng.uniform(1.9, 2.4), 0, 0, 0];
  return [rng.uniform(0.3, 0.36), rng.uniform(0.5, 0.75), rng.uniform(0.85, 1.4), rng.uniform(-0.6, 0), rng.uniform(0.24, 0.33)];
}

// a car or van along +z; with a carShape: on wheels, with raked windows (vans: a short hood)
function carParts(van, hx, hz, shape) {
  if (!shape.length) {
    if (van) return [part([0, 1.55, 0], [hx, 1.25, hz])];
    return [part([0, 0.65, 0], [hx, 0.4, hz]), part([0, 1.3, -0.2], [hx - 0.1, 0.27, 1.3])];
  }
  const [wr, bh, cl, co, ch] = shape, b0 = wr * 0.6;
  const parts = [];
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) parts.push(part([sx * (hx - 0.13), wr, sz * (hz - 0.8)], [0.12, wr, wr], 0, true));
  if (van) {
    const hood = 0.75;
    return [...parts,
      part([0, b0 + bh / 2, -hood / 2], [hx, bh / 2, hz - hood / 2]),
      part([0, b0 + 0.45, hz - hood / 2], [hx - 0.05, 0.45, hood / 2]),
      part([0, b0 + bh * 0.72, hz - hood - 0.02], [hx - 0.12, bh * 0.16, 0.04])]; // prettier-ignore
  }
  const top = b0 + bh, rake = 0.75, lean = ch * Math.tan(rake);
  return [...parts,
    part([0, b0 + bh / 2, 0], [hx, bh / 2, hz]),
    part([0, top + ch, co], [hx - 0.12, ch, cl]),
    part([0, top + ch, co + cl + lean / 2], [hx - 0.14, ch / Math.cos(rake), 0.03], -rake),
    part([0, top + ch, co - cl - lean / 2], [hx - 0.14, ch / Math.cos(rake), 0.03], rake)]; // prettier-ignore
}

// a cornice, a smaller upper tier set back from the street, a shop awning
function buildingDetail(add, rng, side, x, dx, h, zc, wz) {
  let top = h;
  if (rng.random() < 0.4) {
    const dx2 = dx * rng.uniform(0.45, 0.85), wz2 = wz * rng.uniform(0.6, 1), h2 = rng.uniform(3, 12);
    add([x + (side * (dx - dx2)) / 2, h + h2 / 2, zc], [dx2 / 2, h2 / 2, wz2 / 2]);
    top = h + h2;
  }
  if (rng.random() < 0.6) add([x, top + 0.2, zc], [dx / 2 + 0.3, 0.2, wz / 2 + 0.3]);
  if (rng.random() < 0.35) {
    const front = x - (side * dx) / 2;
    add([front - side * 0.75, 3.1, zc], [0.8, 0.06, (wz / 2) * 0.8], rotM(0, -side * 0.3)); // dips out
  }
}

// ----------------------------------------------------------------------------------------------- world

export class World {
  constructor(seed = 0, cfg = CFG) {
    this.cfg = cfg;
    const w = cfg.world, c = cfg.camera;
    this.seed = seed;
    this.rng = new Rng(seed);
    this.roadHalf = (w.lane_dirs.length * w.lane_w) / 2;
    this.curbX = this.roadHalf + (w.parked_per_100m ? PARK_W : 0);
    this.detailRng = new Rng(seed, 7); // apart, so detail doesn't change the episode
    this.xLimit = this.roadHalf + w.sidewalk_w - 0.4;
    const x0 = this.laneX(w.rider_lane);
    this.rider = { x: x0, lane: w.rider_lane, z: 0, v: cfg.rider.start_speed, yaw: 0, lean: 0, leanRate: 0, aim: x0, steer: 0, crank: 0 };
    this.t = 0;
    this.crashed = false;
    this.nextId = 1;
    this.stuck = [null, 0, 0];
    this.agents = [];
    this.blocks = new Map();
    const xc = this.rider.x - c.lat_mean;
    this.cam = { d_raw: c.dist_mean, d1: c.dist_mean, d: c.dist_mean, lat_raw: c.lat_mean, x_raw: xc, x1: xc, x: xc };
    this.cam.z = this.rider.z - c.dist_mean;
    this.cam.y = this.cam.y1 = c.height_m;
    this.populate(true);
  }

  laneX(i) {
    return -this.roadHalf + (i + 0.5) * this.cfg.world.lane_w;
  }
  get level() {
    return 1 + Math.floor(Math.max(this.rider.z, 0) / this.cfg.world.level_every_m);
  }
  get scale() {
    return 1 + this.cfg.world.ramp * (this.level - 1);
  }
  window() {
    return [this.cam.z - 30, this.cam.z + this.cfg.camera.view_dist];
  }

  // buildings (with detail), lampposts, trees, the sidewalk slab and parked cars of block k:
  // {boxes: [{center, half, rot, id, round}], coll: [[x, z, hx, hz]]}
  block(k) {
    if (!this.blocks.has(k)) {
      const w = this.cfg.world, rng = new Rng(this.seed, k + 1_000_000);
      const drng = new Rng(this.seed, k + 2_000_000); // detail / parked cars: the rest stays the same
      const z0 = k * w.block_len + w.street_w, z1 = (k + 1) * w.block_len, parts = [], coll = [];
      const add = (center, half, rot = EYE, round = false) => parts.push({ center, half, rot, round });
      for (const side of [-1, 1]) {
        let z = z0;
        while (z < z1 - 2) {
          const wz = Math.min(rng.uniform(8, 25), z1 - z), dx = rng.uniform(8, 20), h = rng.uniform(...w.building_h);
          const back = w.setback[1] ? rng.uniform(w.setback[0], w.setback[1]) : 0;
          const x = side * (this.roadHalf + w.sidewalk_w + back + dx / 2);
          add([x, h / 2, z + wz / 2], [dx / 2, h / 2, wz / 2]);
          if (w.detail) buildingDetail(add, drng, side, x, dx, h, z + wz / 2, wz);
          z += wz + (w.alley_p && rng.random() < w.alley_p ? rng.uniform(4, 12) : 0);
        }
        for (let z = z0 + 4; z < z1 - 2; z += 12.5) {
          const xp = side * (this.curbX + 0.5);
          if (Math.round((z - z0) / 12.5) % 2 === 0) { // lamppost with an arm over the road
            add([xp, 3, z], [0.08, 3, 0.08]);
            add([xp - side * 0.8, 6, z], [0.8, 0.06, 0.08]);
            coll.push([xp, z, 0.08, 0.08]);
          } else if (rng.random() < 0.7) { // tree
            const xt = xp + side * 0.8;
            add([xt, 1.2, z], [0.13, 1.2, 0.13]);
            if (w.detail) { // a few turned, uneven lumps instead of one cube
              const n = 2 + drng.integers(3);
              for (let i = 0; i < n; i++) {
                const sx = drng.uniform(0.5, 1), sy = drng.uniform(0.5, 1) * 0.8, sz = drng.uniform(0.5, 1);
                const ox = drng.uniform(-0.7, 0.7), oy = drng.uniform(-0.7, 0.7) * 0.6, oz = drng.uniform(-0.7, 0.7);
                add([xt + ox, 3.4 + oy, z + oz], [sx, sy, sz], rotM(drng.uniform(0, Math.PI)));
              }
            } else add([xt, 3.4, z], [1.1, 1.0, 1.1]);
            coll.push([xt, z, 0.13, 0.13]);
          }
        }
        if (w.curb_h) { // sidewalk slab out to past the buildings; the cross street stays at road level
          const span = w.sidewalk_w + w.setback[1] + 22 + this.roadHalf - this.curbX, hz = (z1 - z0) / 2;
          add([side * (this.curbX + span / 2), w.curb_h / 2, z0 + hz], [span / 2, w.curb_h / 2, hz]);
        }
        if (w.parked_per_100m) {
          const slot = 5.6;
          for (let z = z0 + 1; z < z1 - slot; z += slot) {
            if (drng.random() >= Math.min(1, (w.parked_per_100m * slot) / 100)) continue;
            const van = drng.random() < 0.15, [hx, hz] = van ? [1.05, 2.9] : [0.9, 2.25];
            const xc = side * (this.roadHalf + PARK_W / 2), zc = z + slot / 2, R = rotM(drng.choice([0, Math.PI]));
            const shape = w.detail ? carShape(drng, van) : [];
            for (const pt of carParts(van, hx, hz, shape)) {
              const o = mulv(R, pt.off);
              add([xc + o[0], o[1], zc + o[2]], pt.half, mul3(R, pitchM(pt.pitch)), pt.round);
            }
            coll.push([xc, zc, hx, hz]);
          }
        }
      }
      const base = STATIC_ID + (k + 1_000_000) * 1024;
      this.blocks.set(k, { boxes: parts.map((pt, i) => ({ ...pt, id: base + i })), coll });
    }
    return this.blocks.get(k);
  }

  static() {
    const [z0, z1] = this.window(), bl = this.cfg.world.block_len;
    const k0 = Math.floor(z0 / bl), k1 = Math.floor(z1 / bl);
    for (const k of this.blocks.keys()) if (k < k0 || k > k1) this.blocks.delete(k);
    const out = [];
    for (let k = k0; k <= k1; k++) out.push(this.block(k));
    return out;
  }

  // ------------------------------------------------------------------------------------------- agents
  spawn(kind, initial) {
    const w = this.cfg.world, rng = this.rng;
    const [z0, z1] = this.window();
    let a;
    if (kind === "ped") {
      const side = rng.choice([-1, 1]);
      const x = side * (this.curbX + rng.uniform(0.3, this.roadHalf + w.sidewalk_w - this.curbX - 0.6));
      const vz = rng.choice([-1, 1]) * rng.uniform(1.0, 1.6);
      const z = initial ? rng.uniform(z0, z1) : z1 - rng.uniform(0, 20);
      a = { id: this.nextId, kind: "ped", x, z, vx: 0, vz, yaw: vz > 0 ? 0 : Math.PI, speed: Math.abs(vz), hx: 0.25, hz: 0.25,
        height: rng.uniform(1.55, 1.9), phase: 0, targetX: null }; // prettier-ignore
    } else {
      const lane = rng.integers(w.lane_dirs.length), sgn = w.lane_dirs[lane];
      const speed = w.car_speed * rng.uniform(0.6, 1.25) * (1 + 0.5 * w.ramp * (this.level - 1));
      const van = rng.random() < 0.2;
      const ahead = sgn < 0 || speed < this.rider.v;
      const z = initial ? rng.uniform(z0, z1) : ahead ? z1 - rng.uniform(0, 10) : z0 + rng.uniform(0, 5);
      const x = this.laneX(lane);
      if (this.agents.some((o) => Math.abs(o.x - x) < 1 && Math.abs(o.z - z) < 12) || (z - this.rider.z > -15 && z - this.rider.z < 80))
        return;
      a = { id: this.nextId, kind: van ? "van" : "car", x, z, vx: 0, vz: sgn * speed, yaw: sgn > 0 ? 0 : Math.PI, speed,
        hx: van ? 1.1 : 0.9, hz: van ? 3.3 : 2.25, height: 1.7, phase: 0, targetX: null }; // prettier-ignore
    }
    if (w.detail) a.shape = a.kind === "ped" ? [1] : carShape(this.detailRng, a.kind === "van");
    this.agents.push(a);
    this.nextId++;
  }

  populate(initial = false) {
    const w = this.cfg.world;
    const [z0, z1] = this.window();
    this.agents = this.agents.filter((a) => z0 - 10 < a.z && a.z < z1 + 20);
    const span = ((z1 - z0) / 100) * this.scale;
    for (const [kind, target] of [["car", w.cars_per_100m * span], ["ped", w.peds_per_100m * span]]) {
      const have = this.agents.filter((a) => (a.kind === "ped") === (kind === "ped")).length;
      for (let i = 0; i < Math.floor(target) - have; i++) this.spawn(kind, initial);
    }
  }

  stepCar(a, dt) {
    const sgn = a.vz > 0 || a.yaw === 0 ? 1 : -1;
    let gap = Infinity;
    for (const o of this.agents)
      if (o !== a && Math.abs(o.x - a.x) < a.hx + o.hx && (o.z - a.z) * sgn > 0) gap = Math.min(gap, (o.z - a.z) * sgn - a.hz - o.hz);
    const r = this.rider;
    if (this.crashed && Math.abs(r.x - a.x) < a.hx + 1.0 && (r.z - a.z) * sgn > 0) gap = Math.min(gap, (r.z - a.z) * sgn - a.hz - 1.0);
    else if (sgn > 0 && Math.abs(r.x - a.x) < a.hx + 1.0 && r.z > a.z) gap = Math.min(gap, this.cam.z - 1.5 - a.z - a.hz);
    let v = Math.abs(a.vz);
    const target = a.speed * clip((gap - 2.5) / (1.2 * a.speed + 1e-6), 0, 1);
    v = target > v ? Math.min(v + 3 * dt, target) : Math.max(v - 8 * dt, target);
    a.vz = sgn * v;
  }

  stepPed(a, dt) {
    const w = this.cfg.world;
    if (a.targetX === null) {
      const zb = ((a.z % w.block_len) + w.block_len) % w.block_len;
      const atCrosswalk = (0.5 < zb && zb < 3.5) || (w.street_w - 3.5 < zb && zb < w.street_w - 0.5);
      const rate = w.jaywalk_rate * this.scale * (atCrosswalk ? 20 : 1);
      if (Math.abs(a.x) > this.roadHalf && Math.abs(a.z - this.rider.z) > 6 && this.rng.random() < rate * dt) {
        const side = a.x > 0 ? 1 : -1;
        a.targetX = -side * (this.curbX + this.rng.uniform(0.3, 1.5));
        a.vx = -side * this.rng.uniform(1.2, 1.7);
        a.vz = 0;
      }
    } else if ((a.targetX - a.x) * a.vx <= 0) {
      a.targetX = null;
      a.vx = 0;
      a.vz = this.rng.choice([-1, 1]) * a.speed;
    }
    a.yaw = Math.atan2(a.vx, a.vz);
    a.phase += Math.hypot(a.vx, a.vz) * dt * 4;
  }

  // --------------------------------------------------------------------------------------- simulation
  riderGain(v) {
    const f = clip((v - GAIN_V0) / GAIN_STEP, 0, GAINS.length - 1), i = Math.min(Math.floor(f), GAINS.length - 2), u = f - i;
    return GAINS[i].map((g, j) => g * (1 - u) + GAINS[i + 1][j] * u);
  }

  stepRider(dt, substeps = 10) {
    const c = this.cfg.rider, r = this.rider, g = 9.81;
    const b = c.wheelbase, h = c.com_height, a = c.com_ahead;
    r.v += c.speed_gain * dt;
    const v = Math.max(r.v, 2.0), K = this.riderGain(v);
    dt /= substeps;
    const maxSteer = rad(c.max_steer_deg);
    // below calm_speed the wobble shrinks and the aim slides sideways at most aim_frac v (the slow web rider:
    // steering's leverage on the lean goes as v^2, so the full wobble and a 4 m/s aim tip it over at 3 m/s)
    const wobble = c.wobble * Math.min(1.0, c.wobble_speed / v) * Math.min(1.0, v / c.calm_speed);
    const aimSpeed = Math.min(c.aim_speed, c.aim_frac * v);
    for (let s = 0; s < substeps; s++) {
      r.aim += clip(this.laneX(r.lane) - r.aim, -aimSpeed * dt, aimSpeed * dt);
      const err = clip(r.x - r.aim, -c.aim_clip, c.aim_clip);
      let u = -(K[0] * err + K[1] * r.yaw + K[2] * r.steer + K[3] * r.lean + K[4] * r.leanRate);
      u = clip(u, -c.max_steer_rate, c.max_steer_rate);
      u = (clip(r.steer + u * dt, -maxSteer, maxSteer) - r.steer) / dt;
      r.steer += u * dt;
      const acc = (g * Math.sin(r.lean) - (Math.cos(r.lean) * (v ** 2 * r.steer + a * v * u)) / b) / h;
      r.leanRate += acc * dt + wobble * Math.sqrt(dt) * this.rng.normal();
      r.lean += r.leanRate * dt;
      r.yaw += ((v * Math.tan(r.steer)) / b) * dt;
      r.x += r.v * Math.sin(r.yaw) * dt;
      r.z += r.v * Math.cos(r.yaw) * dt;
    }
    if (Math.abs(r.x) > this.xLimit) {
      r.x = Math.sign(r.x) * this.xLimit;
      r.yaw = 0;
    }
    r.crank += ((r.v / c.wheel_circ / c.gear_ratio) * 2 * Math.PI * dt * substeps);
  }

  // advance DT seconds; action -1 = tap left, +1 = tap right
  step(action, dt) {
    const r = this.rider;
    if (!this.crashed) {
      r.lane = clip(r.lane + action, 0, this.cfg.world.lane_dirs.length - 1);
      this.stepRider(dt);
    }
    for (const a of this.agents) {
      if (a.kind === "ped") this.stepPed(a, dt);
      else this.stepCar(a, dt);
      a.x += a.vx * dt;
      a.z += a.vz * dt;
    }
    if (this.crashed) {
      const [a, ox, oz] = this.stuck;
      if (a) {
        r.x = a.x + ox;
        r.z = a.z + oz;
      }
      r.leanRate += (9.81 / this.cfg.rider.com_height) * Math.sin(r.lean) * dt;
      r.lean = clip(r.lean + r.leanRate * dt, -1.45, 1.45);
    }
    this.stepCamera(dt);
    this.t += dt;
    this.populate();
    if (this.crashed) return;
    const obs = this.agents.map((a) => [a, a.x, a.z, a.hx, a.hz]);
    for (const b of this.static()) for (const [x, z, hx, hz] of b.coll) obs.push([null, x, z, hx, hz]);
    const hit = obs.find((o) => {
      const k = o[0]?.kind === "ped" ? this.cfg.world.ped_hit : 1;
      return Math.abs(o[1] - r.x) < k * (o[3] + 0.3) && Math.abs(o[2] - r.z) < k * (o[4] + 0.85);
    });
    this.crashed = hit !== undefined || Math.abs(r.lean) > rad(this.cfg.rider.fall_lean_deg);
    if (hit) {
      const [a, x, z] = hit;
      this.stuck = [a, 0.8 * (r.x - x), 0.8 * (r.z - z)];
      r.x = x + this.stuck[1];
      r.z = z + this.stuck[2];
      if (a) {
        a.speed = 0;
        if (a.kind === "ped") a.vx = a.vz = 0;
      }
    }
  }

  stepCamera(dt) {
    const c = this.cfg.camera, s = this.cam, rng = this.rng;
    const b = c.follow_tau / 2, k = 1 - Math.exp(-dt / b);
    const ou = (x, mu, std, tau) => {
      std *= (tau + b) / Math.sqrt(tau * (tau + b / 2));
      return x + ((mu - x) * dt) / tau + std * Math.sqrt((2 * dt) / tau) * rng.normal();
    };
    s.d_raw = ou(s.d_raw, c.dist_mean, c.dist_std, c.dist_tau);
    s.lat_raw = ou(s.lat_raw, c.lat_mean, c.lat_std, c.lat_tau);
    s.x_raw = this.rider.x - s.lat_raw;
    for (const v of ["d", "x"]) {
      s[v + "1"] += (s[v + "_raw"] - s[v + "1"]) * k;
      s[v] += (s[v + "1"] - s[v]) * k;
    }
    s.z = this.crashed ? s.z + this.rider.v * dt : this.rider.z - Math.max(1.5, s.d);
    const want = this.clearHeight(), tau = want > s.y1 ? c.rise_tau : c.fall_tau, ky = 1 - Math.exp(-dt / tau);
    s.y1 += (want - s.y1) * ky;
    s.y += (s.y1 - s.y) * ky;
  }

  // the height that clears every agent (and the crashed rider) inside the space around the eye, now or within
  // `anticipate` s at its velocity relative to the camera's; tops above max_height are left alone
  clearHeight() {
    const c = this.cfg.camera, s = this.cam, r = this.rider, ex = this.eyeX(), vc = r.v; // the camera follows the rider (crashed: keeps going at v)
    let h = c.height_m;
    const consider = (x, z, hx, hz, vx, vz, top) => {
      const dx = vx * c.anticipate, dz = (vz - vc) * c.anticipate;
      const x0 = Math.min(x, x + dx) - hx, x1 = Math.max(x, x + dx) + hx;
      const z0 = Math.min(z, z + dz) - hz, z1 = Math.max(z, z + dz) + hz;
      if (x1 < ex - c.clear_hx || x0 > ex + c.clear_hx || z1 < s.z - c.clear_hz || z0 > s.z + c.clear_hz) return;
      if (top + c.clearance <= c.max_height) h = Math.max(h, top + c.clearance);
    };
    for (const a of this.agents) {
      const top = this.groundY(a.x, a.kind) + Math.max(...this.parts(a).map((p) => p.off[1] + Math.hypot(p.half[1], p.half[2] * Math.sin(p.pitch))));
      consider(a.x, a.z, a.hx, a.hz, a.vx, a.vz, top);
    }
    consider(r.x, r.z, 0.3, 0.85, 0, this.crashed ? 0 : r.v, 1.75);
    return h;
  }

  eyeX() {
    return clip(this.cam.x, -(this.roadHalf - this.cfg.camera.curb_margin), this.roadHalf - this.cfg.camera.curb_margin);
  }

  // the chase camera: eye, rotation camera -> world (row-major, columns right, down, forward), intrinsics
  camera() {
    const c = this.cfg.camera, s = this.cam;
    const eye = [this.eyeX(), s.y, s.z];
    // raised, it pitches down by as much as the rider (~1 m up) moved down in the view
    const d = Math.max(this.rider.z - s.z, 1.5), drop = Math.atan2(s.y - 1, d) - Math.atan2(c.height_m - 1, d);
    const f = [0, Math.tan(rad(c.pitch_deg) - drop), 1], fn = Math.hypot(...f);
    const fwd = f.map((x) => x / fn);
    const right = [fwd[2], 0, -fwd[0]], rn = Math.hypot(...right); // cross((0, 1, 0), fwd)
    const rt = right.map((x) => x / rn);
    const down = [rt[1] * fwd[2] - rt[2] * fwd[1], rt[2] * fwd[0] - rt[0] * fwd[2], rt[0] * fwd[1] - rt[1] * fwd[0]];
    const rot = [rt[0], down[0], fwd[0], rt[1], down[1], fwd[1], rt[2], down[2], fwd[2]];
    // the diagonal field of view of hfov_deg at 16:9 (the training camera), whatever the aspect: wider or taller
    // screens trade some width for height instead of stretching one way
    const tanD = Math.tan(rad(c.hfov_deg) / 2) * Math.hypot(1, 9 / 16);
    const fx = Math.hypot(c.width, c.height) / 2 / tanD;
    return { pos: eye, rot, fx, fy: fx, cx: c.width / 2, cy: c.height / 2 };
  }

  // ------------------------------------------------------------------------------- renderer geometry
  parts(a) {
    if (a.kind !== "ped") return carParts(a.kind === "van", a.hx, a.hz, a.shape ?? []);
    const k = a.height / 1.75, sw = 0.4 * Math.sin(a.phase), hip = 0.84 * k, leg = 0.77 * k;
    const legs = [-1, 1].map((sx) => seg([sx * 0.1, hip, 0], [sx * 0.1, hip - leg * Math.cos(sw), sx * leg * Math.sin(sw)], 0.07, 0.07));
    if (a.shape?.length) { // arms swing against the legs
      const arm = 0.6 * k;
      for (const sx of [-1, 1]) legs.push(seg([sx * 0.25, 1.36 * k, 0], [sx * 0.27, 1.36 * k - arm * Math.cos(sw), -sx * arm * Math.sin(sw)], 0.05, 0.05));
    }
    return [...legs, part([0, 1.12 * k, 0], [0.2, 0.3 * k, 0.12]), part([0, a.height - 0.12, 0], [0.11, 0.12, 0.11])];
  }

  // a spoked wheel at C (radius 0.34): RIM chords of a tire, SPOKES diameters through a hub, turning with the
  // fixed gear's cranks (top forward)
  wheel(C, { rad = 0.34, rim = 18, spokes = 6 } = {}) {
    const turn = this.rider.crank * this.cfg.rider.gear_ratio;
    const at = (phi, r) => [C[0], C[1] + r * Math.cos(phi), C[2] + r * Math.sin(phi)];
    const out = [part(C, [0.04, 0.04, 0.04], 0, true)];
    for (let i = 0; i < rim; i++) {
      const a = turn + (2 * Math.PI * i) / rim, b = turn + (2 * Math.PI * (i + 1)) / rim;
      out.push(seg(at(a, rad - 0.015), at(b, rad - 0.015), 0.018, 0.018));
    }
    for (let i = 0; i < spokes; i++) {
      const a = turn + (Math.PI * (i + 0.5)) / spokes;
      out.push(seg(at(a, rad - 0.03), at(a + Math.PI, rad - 0.03), 0.008, 0.008));
    }
    return out;
  }

  riderParts() {
    const rear = [0, 0.34, -0.5], front = [0, 0.34, 0.5], bb = [0, 0.3, -0.05], saddle = [0, 0.92, -0.25], head = [0, 0.86, 0.38];
    const parts = [
      ...this.wheel(rear),
      ...this.wheel(front),
      seg(rear, bb, 0.02, 0.02),
      seg(bb, saddle, 0.02, 0.02),
      seg(bb, head, 0.025, 0.025),
      seg(saddle, head, 0.02, 0.02),
      seg(front, head, 0.02, 0.02),
      part([0, 0.95, 0.42], [0.22, 0.02, 0.02]),
      seg([0, 0.98, -0.22], [0, 1.42, 0.12], 0.17, 0.1),
      part([0, 1.6, 0.22], [0.11, 0.12, 0.13]),
    ];
    for (const [sx, ph] of [[-1, 0], [1, Math.PI]]) {
      const hip = [sx * 0.1, 0.98, -0.22], th = this.rider.crank + ph;
      const ped = [sx * 0.12, bb[1] + 0.17 * Math.cos(th), bb[2] + 0.17 * Math.sin(th)];
      const dy = ped[1] - hip[1], dz = ped[2] - hip[2], d = Math.hypot(dy, dz);
      const out = Math.sqrt(Math.max(0.47 ** 2 - d ** 2 / 4, 0));
      const knee = [sx * 0.11, hip[1] + dy / 2 + (out * dz) / d, hip[2] + dz / 2 - (out * dy) / d];
      parts.push(seg(hip, knee, 0.075, 0.075), seg(knee, ped, 0.06, 0.06), seg([sx * 0.17, 1.38, 0.08], [sx * 0.2, 0.95, 0.42], 0.045, 0.045));
    }
    return parts;
  }

  // riders and pedestrians step up on the sidewalk
  groundY(x, kind) {
    return kind !== "car" && kind !== "van" && Math.abs(x) > this.curbX ? this.cfg.world.curb_h : 0;
  }

  // every box near the camera: {center, half, rot (row-major local -> world), id, round}
  boxes() {
    const r = this.rider, out = [];
    const dyn = [[r.x, r.z, rotM(r.yaw, r.lean), 0, this.riderParts(), this.groundY(r.x, "rider")]];
    for (const a of this.agents) dyn.push([a.x, a.z, rotM(a.yaw), a.id, this.parts(a), this.groundY(a.x, a.kind)]);
    for (const [x, z, R, eid, parts, y0] of dyn)
      parts.forEach((p, j) => {
        const o = mulv(R, p.off);
        out.push({ center: [x + o[0], y0 + o[1], z + o[2]], half: p.half, rot: mul3(R, pitchM(p.pitch)), id: eid * 128 + j, round: p.round });
      });
    for (const b of this.static()) out.push(...b.boxes);
    const cz = this.cam.z, vd = this.cfg.camera.view_dist;
    return out.filter((b) => b.center[2] + b.half[2] > cz - 5 && b.center[2] - b.half[2] < cz + vd);
  }
}
