// The race: replays of the recorded final-test races, drawn top-down from the game's own road polygons.
// Data: data/tracks/<track>.json (scripts/export_site_data.py), every path checked against its recorded result.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const PLAYFIELD = 2000 / 6;            // the game's map edge (leaving it ends the race, -100)
const GRASS_DIM = PLAYFIELD / 20;      // the game's grass squares
const POINTS_PER_SECOND = 12.5;        // one path point every 2 decisions (25 decisions per second)
const CAR = { A: "#e8e5de", B: "#8b7dff" };   // A: warm white, B: violet (distinct on any color scheme, dark outline)

export function initRace(site) {
  const canvas = $("[data-race-canvas]"), ctx = canvas.getContext("2d");
  const select = $("[data-track]");
  const state = { track: null, data: null, colors: "unseen", seed: "42", t: 0, playing: false, cache: new Map() };
  site.tracks.forEach((t, i) => {
    const o = document.createElement("option");
    o.value = t; o.textContent = `Hidden test track ${i + 1} (seed ${t})`;
    select.appendChild(o);
  });

  const radio = (attr) => $$(`[${attr}]`).forEach((b) => b.addEventListener("click", () => {
    $$(`[${attr}]`).forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    state[attr === "data-colors" ? "colors" : "seed"] = b.getAttribute(attr);
    start();
  }));
  radio("data-colors"); radio("data-seed");
  select.addEventListener("change", () => load(Number(select.value)).then(start));
  $("[data-go]").addEventListener("click", () => start());
  $("[data-random]").addEventListener("click", () => {
    select.value = site.tracks[Math.floor(Math.random() * site.tracks.length)];
    load(Number(select.value)).then(start);
  });

  async function load(track) {
    if (!state.cache.has(track)) state.cache.set(track, fetch(`data/tracks/${track}.json`).then((r) => r.json()));
    state.data = await state.cache.get(track);
    state.track = track;
  }

  function size() {
    const d = Math.min(devicePixelRatio || 1, 2);
    canvas.width = canvas.clientWidth * d; canvas.height = canvas.clientHeight * d;
  }
  addEventListener("resize", () => { size(); draw(); });

  function view() {
    // fit the road (plus a margin) into the canvas; world y points up, screen y down
    const xs = [], ys = [];
    state.data.road.forEach((p) => p.slice(0, 4).forEach(([x, y]) => { xs.push(x); ys.push(y); }));
    const pad = 40, minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad, minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
    const s = Math.min(canvas.width / (maxX - minX), canvas.height / (maxY - minY));
    const ox = (canvas.width - s * (maxX - minX)) / 2, oy = (canvas.height - s * (maxY - minY)) / 2;
    return { s, X: (x) => ox + (x - minX) * s, Y: (y) => canvas.height - (oy + (y - minY) * s) };
  }

  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

  function drawTrack(v) {
    const col = state.data.colors[state.colors];
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = rgb(col.background);
    ctx.fillRect(v.X(-PLAYFIELD), v.Y(PLAYFIELD), PLAYFIELD * 2 * v.s, PLAYFIELD * 2 * v.s);
    ctx.fillStyle = rgb(col.grass);
    for (let x = -20; x < 20; x += 2) for (let y = -20; y < 20; y += 2)
      ctx.fillRect(v.X(GRASS_DIM * x), v.Y(GRASS_DIM * (y + 1)), GRASS_DIM * v.s + 0.5, GRASS_DIM * v.s + 0.5);
    for (const p of state.data.road) {
      const kind = p[4];
      ctx.fillStyle = kind === 0 ? rgb(col.road) : kind === 1 ? "#ff0000" : "#ffffff";
      ctx.beginPath();
      p.slice(0, 4).forEach(([x, y], i) => (i ? ctx.lineTo(v.X(x), v.Y(y)) : ctx.moveTo(v.X(x), v.Y(y))));
      ctx.closePath(); ctx.fill();
    }
  }

  function at(path, t) {
    const f = Math.min(t * POINTS_PER_SECOND, path.length - 1), i = Math.floor(f), k = f - i, a = path[i], b = path[Math.min(i + 1, path.length - 1)];
    let da = b[2] - a[2];
    da = Math.atan2(Math.sin(da), Math.cos(da));
    return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + da * k, f];
  }

  function drawCar(v, race, agent, t) {
    const path = race.path, [x, y, ang, f] = at(path, t);
    ctx.lineWidth = Math.max(2, v.s * 1.1); ctx.strokeStyle = CAR[agent]; ctx.globalAlpha = 0.85; ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i <= Math.floor(f); i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, v.X(path[i][0]), v.Y(path[i][1]));
    ctx.lineTo(v.X(x), v.Y(y)); ctx.stroke(); ctx.globalAlpha = 1;
    // the car: forward is the body's +y axis
    const L = Math.max(9, v.s * 5), W = L * 0.55, fx = -Math.sin(ang), fy = Math.cos(ang);
    const cx = v.X(x), cy = v.Y(y), sx = fx, sy = -fy, px = -sy, py = sx;
    ctx.fillStyle = CAR[agent]; ctx.strokeStyle = "#0b0b0c"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx + sx * L, cy + sy * L);
    ctx.lineTo(cx - sx * L * 0.7 + px * W, cy - sy * L * 0.7 + py * W);
    ctx.lineTo(cx - sx * L * 0.7 - px * W, cy - sy * L * 0.7 - py * W);
    ctx.closePath(); ctx.fill(); ctx.stroke();
    // a small name tag above the car
    const r = Math.max(9, L * 0.9), tx = cx, ty = cy - L * 1.6 - r;
    ctx.fillStyle = "rgba(6,6,9,0.78)"; ctx.beginPath(); ctx.arc(tx, ty, r, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = CAR[agent]; ctx.font = `600 ${Math.round(r * 1.2)}px Geist, system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(agent, tx, ty + 1);
  }

  function hud(t) {
    $("[data-hud-time]").textContent = `${Math.min(t, 20).toFixed(1)} s`;
    for (const agent of ["A", "B"]) {
      const r = state.data.races[state.colors][agent][state.seed];
      const done = t >= r.frames / 50;
      const how = r.off_track ? "left the map" : r.lap ? "lap complete" : `${r.covered.toFixed(0)}% of the road`;
      $(`[data-hud="${agent}"]`).textContent = done ? `${r.score.toFixed(0)} pts · ${how}` : "racing…";
    }
  }

  function draw() {
    if (!state.data) return;
    const v = view();
    drawTrack(v);
    for (const agent of ["A", "B"]) drawCar(v, state.data.races[state.colors][agent][state.seed], agent, state.t);
    hud(state.t);
  }

  let last = 0;
  function frame(now) {
    if (!state.playing) return;
    state.t += Math.max(0, Math.min(0.1, (now - last) / 1000)); last = now;
    const end = Math.max(...["A", "B"].map((a) => state.data.races[state.colors][a][state.seed].frames / 50)) + 0.5;
    if (state.t >= end) { state.t = end; state.playing = false; }
    draw();
    if (state.playing) requestAnimationFrame(frame);
  }

  function start() {
    if (!state.data) return;
    state.t = 0; state.playing = true; last = performance.now();
    requestAnimationFrame(frame);
  }

  size();
  // first track: the one in the film above (shown by the page's data), else the first test track
  const first = Number(site.film_track || site.tracks[0]);
  select.value = first;
  load(first).then(() => {
    draw();
    new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { start(); o.disconnect(); } }, { threshold: 0.4 }).observe(canvas);
  });
}
