// Same Road, New Colors: page behaviour. Numbers come from data/site.json (exported from the committed results).
import { initRace } from "./race.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const motion = document.documentElement.classList.contains("motion");
const fmt = (x, d = 0) => (x === undefined || x === null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));

// ---- Nerd mode ----
const nerd = $("[data-nerd]");
const setNerd = (on) => {
  document.documentElement.classList.toggle("nerd-mode", on);
  nerd.setAttribute("aria-checked", String(on));
  try { localStorage.setItem("nerd", on ? "1" : "0"); } catch (e) {}
};
setNerd(document.documentElement.classList.contains("nerd-mode"));
nerd.addEventListener("click", () => setNerd(!document.documentElement.classList.contains("nerd-mode")));

// ---- Navigation: blur when scrolled, dark text over light sections ----
const nav = $("[data-nav]");
const lights = $$('[data-theme="light"], .half-b');
const onScroll = () => {
  nav.classList.toggle("scrolled", scrollY > 30);
  const y = nav.getBoundingClientRect().bottom;
  nav.classList.toggle("light", lights.some((el) => { const r = el.getBoundingClientRect(); return r.top < y && r.bottom > y && r.left < innerWidth / 2; }));
};
addEventListener("scroll", onScroll, { passive: true });
onScroll();

// ---- Reveal on scroll ----
const io = new IntersectionObserver((entries) => entries.forEach((e) => {
  if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}), { rootMargin: "0px 0px -8% 0px" });
$$(".reveal").forEach((el) => io.observe(el));

// ---- Videos: play on click ----
function playable(frameSel, btnSel, videoSel) {
  const frame = $(frameSel), btn = $(btnSel), video = $(videoSel);
  if (!btn || !video) return;
  btn.addEventListener("click", () => {
    video.controls = true;
    video.play();
    frame.classList.add("playing");
  });
}
playable("[data-film]", "[data-video-play]", "[data-video]");
playable(".sees-frame", "[data-sees-play]", "[data-sees-video]");

// ---- B's color swatches: a new random scheme every couple of seconds (Gymnasium's formula) ----
function randomScheme() {
  const u = () => Math.round(Math.random() * 210);
  const road = [u(), u(), u()], bg = [u(), u(), u()], grass = [...bg];
  grass[Math.floor(Math.random() * 3)] += 20;
  return { road: `rgb(${road})`, grass: `rgb(${grass})` };
}
const sw = $("[data-swatches]");
if (sw) {
  for (let i = 0; i < 6; i++) sw.appendChild(document.createElement("i"));
  const paint = () => $$("i", sw).forEach((el) => { const s = randomScheme(); el.style.setProperty("--c", s.road); el.style.setProperty("--g", s.grass); });
  paint();
  if (motion) setInterval(paint, 1800);
}

// ---- Hero: a winding road whose colors keep changing ----
function heroRoad() {
  const c = $("[data-hero-road]");
  if (!c) return;
  const ctx = c.getContext("2d");
  let w, h, t = 0, scheme = randomScheme(), next = randomScheme(), mix = 0;
  const resize = () => { const d = Math.min(devicePixelRatio || 1, 2); w = c.clientWidth; h = c.clientHeight; c.width = w * d; c.height = h * d; ctx.setTransform(d, 0, 0, d, 0, 0); };
  resize(); addEventListener("resize", resize);
  const draw = () => {
    ctx.clearRect(0, 0, w, h);
    const pts = [];
    for (let i = 0; i <= 160; i++) {
      const s = i / 160, x = w * (0.08 + 0.84 * s);
      const y = h * (0.66 + 0.11 * Math.sin(s * 6.3 + t * 0.6) + 0.05 * Math.sin(s * 13 - t * 0.9));
      pts.push([x, y]);
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = scheme.grass; ctx.globalAlpha = 0.18; ctx.lineWidth = Math.max(60, h * 0.11); stroke(pts);
    ctx.strokeStyle = scheme.road; ctx.globalAlpha = 0.65; ctx.lineWidth = Math.max(24, h * 0.045); stroke(pts);
    ctx.setLineDash([18, 22]); ctx.lineDashOffset = -t * 60; ctx.strokeStyle = "rgba(243,242,238,0.5)"; ctx.lineWidth = 2; ctx.globalAlpha = 1; stroke(pts); ctx.setLineDash([]);
    const k = (t * 0.07) % 1, p = pts[Math.floor(k * 160)];
    ctx.fillStyle = "#e0402b"; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, Math.PI * 2); ctx.fill();
  };
  const stroke = (pts) => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke(); };
  let last = performance.now();
  const loop = (now) => {
    t += Math.min(0.05, (now - last) / 1000); last = now;
    mix += 0.004; if (mix >= 1) { scheme = next; next = randomScheme(); mix = 0; }
    draw();
    if (motion) requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}
heroRoad();

// ---- Results from site.json ----
fetch("data/site.json").then((r) => r.json()).then((site) => {
  window.__site = site;
  const mean = (agent, colors) => site.table.find((r) => r.agent === agent && r.colors === colors && r.seed === "mean of seeds");
  const top = 900;
  for (const key of ["A_unseen", "B_unseen", "A_normal", "B_normal"]) {
    const [agent, colors] = key.split("_");
    const row = mean(agent, colors);
    const n = $(`[data-n="${key}"]`), f = $(`[data-fill="${key}"]`);
    if (!row) continue;
    n.textContent = fmt(row.mean_score);
    const set = () => (f.style.width = `${Math.max(0.6, (Math.max(0, row.mean_score) / top) * 100)}%`);
    motion ? new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { set(); o.disconnect(); } }).observe(f) : set();
  }
  const main = site.tests[1], price = site.tests[2], aDrop = site.tests[0], bDrop = site.tests[3];
  const sig = (t) => t.p_holm < 0.05;
  if (sig(main) && main.mean_diff > 0) {
    $("[data-result-title]").innerHTML = 'New colors broke A.<br><span class="soft">B kept driving.</span>';
  } else if (sig(main) && main.mean_diff < 0) {
    $("[data-result-title]").innerHTML = 'New colors hurt B more.<br><span class="soft">A did better.</span>';
  } else {
    $("[data-result-title]").innerHTML = 'New colors, no clear winner.<br><span class="soft">The difference wasn\'t proven.</span>';
  }
  const facts = [
    [`${main.mean_diff > 0 ? "+" : ""}${fmt(main.mean_diff)} points`, `B over A on unseen colors, better on ${main.first_better} of 100 tracks. ${sig(main) ? "Proven by a paired test" : "Not proven"} (Holm p = ${main.p_holm.toPrecision(2)}).`],
    [`${fmt(price.mean_diff)} points`, `B vs A on normal colors: the price of practising in random colors. ${sig(price) ? "Significant" : "Not significant"}.`],
    [`${fmt(aDrop.mean_diff)} points`, `What new colors cost A, against ${fmt(bDrop.mean_diff)} for B.`],
  ];
  $("[data-facts]").innerHTML = facts.map(([b, t]) => `<li><b>${b}</b>${t}</li>`).join("");
  $("[data-tests]").innerHTML = "<ul>" + site.tests.map((t) =>
    `<li><b>${t.test}</b>: ${t.mean_diff > 0 ? "+" : ""}${fmt(t.mean_diff, 1)} (95% CI ${fmt(t.ci95[0], 1)} to ${fmt(t.ci95[1], 1)}), better on ${t.first_better}, worse on ${t.second_better}; Wilcoxon p = ${t.p.toPrecision(2)}, Holm p = ${t.p_holm.toPrecision(2)}</li>`).join("") +
    `</ul><p style="margin-top:10px">Unit = a track; a track's score = mean over the 3 seeds. Untrained car (random actions): ${fmt(site.untrained.mean_score, 1)}. Plan committed before the test: results/final_test/TEST_PLAN.md.</p>`;

  // What it sees
  const att = site.attention, v = Object.fromEntries(att.verdicts.map((x) => [x.test.slice(0, 1), x.verdict]));
  const blur = (name) => att.tests.find((t) => t.test.startsWith(name) && t.method === "blur");
  const e2 = blur("2."), o4 = blur("4.");
  $("[data-sees-facts]").innerHTML = [
    [`${fmt(e2.second_mean, 2)} → ${fmt(e2.first_mean, 2)}`, `Road-edge focus on unseen colors, A → B (1 = the edge gets its size's worth of attention). ${v["2"] === "supported" ? "Both methods agree: B watches the road edges more." : "The two methods don't agree, so this isn't claimed."}`],
    [`${fmt(o4.first_mean, 2)} vs ${fmt(o4.second_mean, 2)}`, `Off-road focus on unseen colors, A vs B. ${v["4"] === "supported" ? "A looks at the grass and background more." : v["4"] === "opposite" ? "The opposite of the guess: B looks off the road more." : "Not shown by both methods."}`],
    [`${fmt(site.focus.A_normal.dashboard, 1)}× / ${fmt(site.focus.B_normal.dashboard, 1)}×`, `Dashboard focus of A / B on normal colors: the black bar at the bottom (speed and steering gauges) gets that many times its size's worth of attention.`],
  ].map(([b, t]) => `<li><b>${b}</b>${t}</li>`).join("");
  $("[data-sees-nerd]").innerHTML = "<ul>" + att.tests.map((t) =>
    `<li><b>${t.test} [${t.method}]</b>: ${fmt(t.first_mean, 3)} vs ${fmt(t.second_mean, 3)}, Holm p = ${t.p_holm.toPrecision(2)}</li>`).join("") +
    "</ul><p style=\"margin-top:10px\">Blur test (Greydanus et al. 2018) and gradients; a finding counts only if both agree. Analysis tracks, 3 seeds per agent, every 5th decision.</p>";
  initRace(site);
}).catch((err) => console.error("site data", err));
