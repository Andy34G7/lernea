/* Lernea benchmark page. Reads window.LERNEA_DATA (site/data.js, written by `python -m tokenisers.dashboard`). */
(() => {
  'use strict';
  const D = window.LERNEA_DATA;
  const $ = (id) => document.getElementById(id);
  if (!D) {
    const e = $('load-error');
    e.hidden = false;
    e.textContent = 'data.js did not load, so there is nothing to show. Build it with: python -m tokenisers.dashboard';
    return;
  }

  // ---------- helpers ----------
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const missing = (v) => v === null || v === undefined || Number.isNaN(v);
  const DASH = '<span class="muted">–</span>';
  const fmt = (v, d = 3) => (isNum(v) ? v.toFixed(d) : DASH);
  const txt = (v, d = 3) => (isNum(v) ? v.toFixed(d) : '–');
  const pct = (v) => (isNum(v) ? (v * 100).toFixed(1) + '%' : DASH);
  const signed = (v, d = 3) => (isNum(v) ? `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}` : '–');
  const int = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);
  const mean = (a) => (Array.isArray(a) && a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const spread = (a) => (Array.isArray(a) && a.length > 1 ? Math.max(...a) / Math.min(...a) : null);
  const range = (a, d = 2, unit = '') => (a.length ? `${Math.min(...a).toFixed(d)}${unit} to ${Math.max(...a).toFixed(d)}${unit}` : '–');
  const mb = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;
  const median = (vals) => {
    const s = [...vals].sort((a, b) => a - b);
    const n = s.length;
    return n === 0 ? null : n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  };
  // Inline Markdown of the report lines: `code` and **bold** only
  const md = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  const code = (k) => `<code>${esc(k)}</code>`;
  const setHtml = (id, html) => { const el = $(id); if (el) el.innerHTML = html; };
  const R = (key) => D.results?.[key] ?? { available: false, path: key, data: null, provenance: [], notes: [] };
  const VOCABS = [16384, 32768, 65536];
  const vLabel = (v) => Number(v).toLocaleString('en-US');

  // ---------- families: fixed colour slot per family, never per rank ----------
  const FAMILIES = [
    ['Frequency BPE', 'Frequency'],
    ['Information-Theoretic', 'Information-theoretic'],
    ['Information-Theoretic (Improved)', 'Information-theoretic, calibrated'],
    ['Structural / Multi-Stage', 'Two-stage'],
    ['Morphology / Guided', 'Morpheme prior'],
    ['EM / Probabilistic', 'Unigram LM'],
    ['Regularized BPE', 'BPE-dropout'],
  ];
  const EXTERNAL = { gpt2: 'GPT-2 (tiktoken)', tm_englishcode_50256: 'TokenMonster englishcode 50,256', tm_englishcode_32000: 'TokenMonster englishcode 32,000' };
  const baseOf = (k) => String(k).replace(/@\d+$/, '').replace(/_(sp300|sp|g2)$/, '');
  const metaOf = (k) => D.metadata?.[baseOf(k)];
  const famSlot = (k) => { const i = FAMILIES.findIndex(([f]) => f === metaOf(k)?.family); return i + 1; };
  const famVar = (k) => `var(--fam-${famSlot(k)})`;
  const famLabel = (k) => FAMILIES[famSlot(k) - 1]?.[1] ?? 'External';
  const dispName = (k) => metaOf(k)?.display_name ?? EXTERNAL[k] ?? k;
  const nameHtml = (k, shown = k) =>
    `<span class="name" data-tip="${esc(`<b>${esc(dispName(k))}</b><br>${esc(famLabel(k))}`)}"><span class="dot" style="--c:${famVar(k)}"></span><span class="key">${esc(shown)}</span></span>`;
  const KEYS = Object.keys(D.metadata || {});
  // Position of a variant inside its family: picks the dash pattern of its line
  const dashOf = (k) => KEYS.filter((x) => famSlot(x) === famSlot(k)).indexOf(baseOf(k)) % 3;

  // ---------- provenance ----------
  function provLine(m) {
    if (!m) return 'no provenance recorded';
    const b = m.backend || {};
    const hw = Object.entries(m.runs_by_hardware || {});
    const parts = [];
    if (m.synthetic_data) parts.push('SYNTHETIC data');
    const where = hw.length > 1 ? hw.map(([k, v]) => `${Array.isArray(v) ? v.length : '?'} × ${k.replace(/`/g, '')}`).join(', ') : b.gpu ? b.gpu.split(',')[0] : b.kind === 'colab' ? `Colab CPU (${b.cpu ?? '?'})` : b.cpu;
    if (where) parts.push(where);
    if (isNum(m.byte_budget)) parts.push(`${mb(m.byte_budget)} LM budget`);
    else if (isNum(m.train_bytes)) parts.push(`${mb(m.train_bytes)} train`);
    else if (isNum(m.prepared?.train_bytes)) parts.push(`${mb(m.prepared.train_bytes)} train`);
    if (m.recipe?.args?.num_iterations) parts.push(`${m.recipe.args.num_iterations} steps`);
    if (Array.isArray(m.seeds)) parts.push(`${m.seeds.length} seed${m.seeds.length === 1 ? '' : 's'}`);
    if (isNum(m.passes)) parts.push(`${m.passes} passes`);
    if (m.eval_split) parts.push(`${m.eval_split} split`);
    if (m.git_commit) parts.push(`commit ${m.git_commit.slice(0, 7)}${m.git_dirty ? ' (dirty tree)' : ''}`);
    return parts.join(' · ');
  }
  const notRun = (e) => `<p class="note">No results committed for this part: ${code(e.path)} is missing${e.error ? ` (${esc(e.error)})` : ''}.</p>`;
  function provHtml(e, label = 'Provenance') {
    if (!e.available) return notRun(e);
    const lines = (e.provenance || []).map((l) => `<li>${md(l)}</li>`).join('');
    return `<details class="prov"><summary>${esc(label)}: ${esc(provLine(e.data?.meta))}</summary><ul>${lines}<li>file ${code(e.path)}</li></ul></details>`;
  }

  // ---------- tooltip ----------
  const tip = $('tip');
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let L = x + 14;
    let T = y + 16;
    if (L + r.width > innerWidth - 8) L = x - r.width - 14;
    if (L < 8) L = 8;
    if (T + r.height > innerHeight - 8) T = y - r.height - 14;
    tip.style.left = `${L}px`;
    tip.style.top = `${Math.max(8, T)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  let tipFrom = null;
  const tipTarget = (e) => (e.target instanceof Element ? e.target.closest('[data-tip]') : null);
  document.addEventListener('pointerover', (e) => {
    const t = tipTarget(e);
    if (t) { tipFrom = t; showTip(t.dataset.tip, e.clientX, e.clientY); } else if (tipFrom) { tipFrom = null; hideTip(); }
  });
  document.addEventListener('pointermove', (e) => { if (tipFrom && tipTarget(e) === tipFrom) showTip(tipFrom.dataset.tip, e.clientX, e.clientY); });
  document.addEventListener('focusin', (e) => {
    const t = tipTarget(e);
    if (!t) return;
    const r = t.getBoundingClientRect();
    tipFrom = t;
    showTip(t.dataset.tip, r.left, r.bottom - 8);
  });
  document.addEventListener('focusout', () => { tipFrom = null; hideTip(); });
  addEventListener('scroll', () => { tipFrom = null; hideTip(); }, { passive: true });

  // ---------- scales ----------
  function ticks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const raw = span / n;
    const p = 10 ** Math.floor(Math.log10(raw));
    const m = raw / p;
    const step = (m >= 5 ? 10 : m >= 2 ? 5 : m >= 1 ? 2 : 1) * p;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return { values: out, dec: Math.max(0, -Math.floor(Math.log10(step) + 1e-9)) };
  }

  // ---------- row chart (HTML): one row per tokeniser, marks on a shared horizontal scale ----------
  // o.rows: [{key, shown?, marks: [{v, lo?, hi?, s?}], val, tip?}]; o.series: [{label, c, shape}] for more than one mark
  // per row; o.mode 'bar' (zero baseline) or 'dot'; o.ref {v, label}; o.max clips the scale (marks beyond show as arrows)
  function rowChart(id, o) {
    const el = $(id);
    if (!el) return;
    const rows = o.rows.filter((r) => r.marks.some((m) => isNum(m.v)));
    if (!rows.length) { el.innerHTML = '<p class="note">No values to plot.</p>'; return; }
    const all = rows.flatMap((r) => r.marks.flatMap((m) => [m.v, m.lo, m.hi])).filter(isNum).concat(o.ref && isNum(o.ref.v) ? [o.ref.v] : []);
    const inScale = isNum(o.max) ? all.filter((v) => v <= o.max) : all;
    let lo;
    let hi;
    if (o.mode === 'bar' || o.zero) { lo = 0; hi = Math.max(...inScale) * 1.04 || 1; } else {
      const a = Math.min(...inScale);
      const b = Math.max(...inScale);
      const pad = (b - a) * 0.08 || Math.abs(a) * 0.05 || 1;
      lo = a - pad;
      hi = b + pad;
    }
    if (o.domain) [lo, hi] = o.domain;
    const tk = o.ticks ? { values: o.ticks, dec: 0 } : ticks(lo, hi, 5);
    const pos = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
    const tf = o.tickFmt ?? ((v) => v.toFixed(tk.dec));
    const grid = tk.values.map((v) => `<i class="rc-tick${v === 0 ? ' zero' : ''}" style="left:${pos(v)}%"></i>`).join('') +
      (o.ref && isNum(o.ref.v) ? `<i class="rc-ref" style="left:${pos(o.ref.v)}%"></i>` : '');
    const series = o.series || [];
    const body = rows.map((r) => {
      const ms = r.marks.filter((m) => isNum(m.v));
      let marks = '';
      if (o.link && ms.length > 1) {
        const ps = ms.map((m) => pos(m.v));
        marks += `<i class="rc-link" style="left:${Math.min(...ps)}%;width:${Math.max(...ps) - Math.min(...ps)}%"></i>`;
      }
      ms.forEach((m) => {
        const s = series[m.s ?? 0] || {};
        const c = s.c || famVar(r.key);
        if (isNum(m.lo) && isNum(m.hi) && m.v <= hi) marks += `<i class="rc-whisk" style="--c:${c};left:${pos(m.lo)}%;width:${pos(m.hi) - pos(m.lo)}%"></i>`;
        if (o.mode === 'bar') marks += `<i class="rc-bar" style="--c:${c};width:${pos(m.v)}%"></i>`;
        else marks += `<i class="rc-pt ${m.v > hi ? 'off' : s.shape || ''}" style="--c:${c};left:${pos(m.v)}%"></i>`;
      });
      const t = r.tip ? ` data-tip="${esc(r.tip)}"` : '';
      return `<div class="rc-row"${t}><div class="rc-name">${nameHtml(r.key, r.shown ?? r.key)}</div><div class="rc-track">${grid}${marks}</div><div class="rc-val">${r.val ?? ''}</div></div>`;
    }).join('');
    const legend = series.length > 1 || o.ref?.label
      ? `<div class="legend rc-legend">${series.map((s) => `<span><i class="dot ${s.shape || ''}" style="--c:${s.c}"></i>${esc(s.label)}</span>`).join('')}${o.ref?.label ? `<span><i class="ref-key"></i>${esc(o.ref.label)}</span>` : ''}</div>`
      : '';
    const axis = `<div class="rc-axis"><div></div><div class="rc-track">${tk.values.map((v) => `<span style="left:${pos(v)}%">${tf(v)}</span>`).join('')}</div><div></div></div>`;
    el.innerHTML = `${legend}<div class="rc" role="img" aria-label="${esc(o.label || o.title || '')}">${body}${axis}${o.title ? `<div class="rc-axis"><div class="rc-title">${esc(o.title)}</div></div>` : ''}</div>`;
  }

  // ---------- SVG charts: drawn at the container's pixel width, redrawn when it changes ----------
  const plots = new Map();
  const ro = new ResizeObserver((entries) => entries.forEach((en) => plots.get(en.target)?.()));
  const NS = 'http://www.w3.org/2000/svg';
  const DASH_ARR = ['', '7 4', '2 4'];
  let clipSeq = 0;

  function frame(W, H, m, xs, ys, o) {
    const x = (v) => m.l + ((v - xs[0]) / (xs[1] - xs[0] || 1)) * (W - m.l - m.r);
    const y = (v) => H - m.b - ((v - ys[0]) / (ys[1] - ys[0] || 1)) * (H - m.t - m.b);
    const yt = ticks(ys[0], ys[1], H < 280 ? 4 : 5);
    const xt = o.xTicks ? { values: o.xTicks.map((t) => t.v), dec: 0 } : ticks(xs[0], xs[1], W < 480 ? 4 : 7);
    const xl = (v, i) => (o.xTicks ? o.xTicks[i].label : (o.xFmt ?? ((n) => n.toFixed(xt.dec)))(v));
    let g = '<g class="grid">';
    yt.values.forEach((v) => { g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(yt.dec)}</text>`; });
    g += '</g>';
    xt.values.forEach((v, i) => { g += `<text x="${x(v)}" y="${H - m.b + 18}" text-anchor="middle">${esc(xl(v, i))}</text>`; });
    g += `<line class="axis-line" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    if (o.xTitle) g += `<text class="title" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xTitle)}</text>`;
    if (o.yTitle) g += `<text class="title" x="${m.l - 38}" y="${m.t - 12}">${esc(o.yTitle)}</text>`;
    return { x, y, g };
  }

  // o.series: [{key, pts: [[x, y]], c?, dash?}]; o.dots marks every point; o.tipFn(series, point) -> html
  function lineChart(id, o) {
    const el = $(id);
    if (!el) return;
    plots.forEach((_, k) => { if (el.contains(k)) { ro.unobserve(k); plots.delete(k); } });
    const series = o.series.filter((s) => s.pts.length);
    if (!series.length) { el.innerHTML = '<p class="note">No values to plot.</p>'; return; }
    el.innerHTML = `<div class="plot"><svg role="img" aria-label="${esc(o.label || '')}"></svg></div><div class="series" aria-label="Series"></div>`;
    const plot = el.firstChild;
    const svg = plot.firstChild;
    const leg = el.lastChild;
    const col = (s) => s.c || famVar(s.key);
    const dash = (s) => s.dash ?? dashOf(s.key);
    leg.innerHTML = series.map((s, i) => `<button type="button" data-i="${i}" style="--c:${col(s)}" aria-pressed="false"><i class="d${dash(s)}"></i>${esc(s.key)}</button>`).join('');
    const xsAll = series.flatMap((s) => s.pts.map((p) => p[0]));
    const ysAll = series.flatMap((s) => s.pts.map((p) => p[1]));
    const xs = o.xDomain || [Math.min(...xsAll), Math.max(...xsAll)];
    let ys = o.yDomain;
    if (!ys) { const a = Math.min(...ysAll); const b = Math.max(...ysAll); const pad = (b - a) * 0.06 || 1; ys = [a - pad, b + pad]; }
    let pinned = null;
    let geo = null;
    const setHl = (i) => {
      plot.classList.toggle('has-hl', i !== null);
      svg.querySelectorAll('[data-s]').forEach((n) => n.classList.toggle('hl', +n.dataset.s === i));
      leg.querySelectorAll('button').forEach((b) => { b.classList.toggle('hl', +b.dataset.i === i); b.setAttribute('aria-pressed', String(+b.dataset.i === pinned)); });
    };
    const draw = () => {
      const W = plot.clientWidth;
      if (!W) return;
      const H = Math.round(Math.max(230, Math.min(o.height || 380, W * 0.56)));
      const m = { l: 46, r: 16, t: 26, b: 42 };
      const f = frame(W, H, m, xs, ys, o);
      const cid = `clip${++clipSeq}`;
      let h = `<defs><clipPath id="${cid}"><rect x="${m.l}" y="${m.t - 4}" width="${W - m.l - m.r}" height="${H - m.t - m.b + 4}"/></clipPath></defs>${f.g}<g clip-path="url(#${cid})">`;
      series.forEach((s, i) => {
        h += `<path class="s" data-s="${i}" style="--c:${col(s)}" stroke-dasharray="${DASH_ARR[dash(s)]}" d="${s.pts.map((p, j) => `${j ? 'L' : 'M'}${f.x(p[0]).toFixed(1)},${f.y(p[1]).toFixed(1)}`).join('')}"/>`;
      });
      if (o.dots) series.forEach((s, i) => s.pts.forEach((p) => { h += `<circle class="p" data-s="${i}" style="--c:${col(s)}" cx="${f.x(p[0])}" cy="${f.y(p[1])}" r="4"/>`; }));
      h += `</g><circle class="p cur" r="5" visibility="hidden"/>`;
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.setAttribute('height', H);
      svg.innerHTML = h;
      geo = { f, m, W, H };
      setHl(pinned);
    };
    const nearest = (e) => {
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      let best = null;
      series.forEach((s, i) => s.pts.forEach((p) => {
        if (p[1] < ys[0] || p[1] > ys[1]) return;
        const d = (geo.f.x(p[0]) - px) ** 2 + (geo.f.y(p[1]) - py) ** 2;
        if (!best || d < best.d) best = { d, i, p };
      }));
      return best;
    };
    svg.addEventListener('pointermove', (e) => {
      if (!geo) return;
      const b = nearest(e);
      if (!b) return;
      const s = series[b.i];
      setHl(b.i);
      const cur = svg.querySelector('.cur');
      cur.setAttribute('cx', geo.f.x(b.p[0]));
      cur.setAttribute('cy', geo.f.y(b.p[1]));
      cur.style.setProperty('--c', col(s));
      cur.setAttribute('visibility', 'visible');
      showTip(o.tipFn ? o.tipFn(s, b.p) : `<b class="mono">${esc(s.key)}</b><br>${b.p[1].toFixed(3)}`, e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', () => { hideTip(); svg.querySelector('.cur')?.setAttribute('visibility', 'hidden'); setHl(pinned); });
    leg.addEventListener('pointerover', (e) => { const b = e.target.closest('button'); if (b) setHl(+b.dataset.i); });
    leg.addEventListener('pointerleave', () => setHl(pinned));
    leg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; pinned = pinned === +b.dataset.i ? null : +b.dataset.i; setHl(pinned); });
    plots.set(plot, draw);
    ro.observe(plot);
    draw();
  }

  // o.pts: [{key, x, y}]: one labelled point per tokeniser
  function scatter(id, o) {
    const el = $(id);
    if (!el) return;
    const pts = o.pts.filter((p) => isNum(p.x) && isNum(p.y));
    el.innerHTML = `<div class="plot"><svg role="img" aria-label="${esc(o.label || '')}"></svg></div>`;
    const plot = el.firstChild;
    const svg = plot.firstChild;
    const ext = (a) => { const lo = Math.min(...a); const hi = Math.max(...a); const pad = (hi - lo) * 0.08 || 1; return [lo - pad, hi + pad]; };
    const xs = ext(pts.map((p) => p.x));
    const ys = ext(pts.map((p) => p.y));
    const draw = () => {
      const W = plot.clientWidth;
      if (!W) return;
      const H = Math.round(Math.max(280, Math.min(400, W * 0.8)));
      const m = { l: 46, r: 16, t: 26, b: 42 };
      const f = frame(W, H, m, xs, ys, o);
      const placed = pts.map((p) => ({ x0: f.x(p.x) - 7, x1: f.x(p.x) + 7, y0: f.y(p.y) - 7, y1: f.y(p.y) + 7 }));
      const hit = (b) => placed.some((q) => b.x0 < q.x1 && b.x1 > q.x0 && b.y0 < q.y1 && b.y1 > q.y0) || b.x0 < m.l || b.x1 > W - 2 || b.y0 < 2 || b.y1 > H - m.b;
      let h = f.g;
      let labels = '';
      pts.forEach((p) => {
        const cx = f.x(p.x);
        const cy = f.y(p.y);
        const w = p.key.length * 6.7 + 4;
        // Try right, left, above, below; a label with no free spot is left to the tooltip
        const tries = [[cx + 9, cy + 4, 'start', cx + 9, cx + 9 + w], [cx - 9, cy + 4, 'end', cx - 9 - w, cx - 9], [cx, cy - 10, 'middle', cx - w / 2, cx + w / 2], [cx, cy + 19, 'middle', cx - w / 2, cx + w / 2]];
        for (const [tx, ty, anchor, x0, x1] of tries) {
          const b = { x0, x1, y0: ty - 10, y1: ty + 3 };
          if (hit(b)) continue;
          placed.push(b);
          labels += `<text class="lbl" x="${tx}" y="${ty}" text-anchor="${anchor}">${esc(p.key)}</text>`;
          break;
        }
        h += `<circle class="p" style="--c:${famVar(p.key)}" cx="${cx}" cy="${cy}" r="5.5" tabindex="0" data-tip="${esc(`<b class="mono">${esc(p.key)}</b><br>${esc(o.xName)}: ${p.x.toFixed(3)}<br>${esc(o.yName)}: ${p.y.toFixed(3)}`)}"/>`;
      });
      svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      svg.setAttribute('height', H);
      svg.innerHTML = h + labels;
    };
    plots.set(plot, draw);
    ro.observe(plot);
    draw();
  }

  // ---------- tables ----------
  const tables = {};
  // cols: {label, get(row), fmt?(value, row) -> html, dir: 1 higher is better / -1 lower / 0 none, text, tip, sub}
  function table(id, cols, rows, sortCol, shade = false) {
    const prev = tables[id];
    tables[id] = { cols, rows, shade, sort: prev && prev.cols.length === cols.length ? prev.sort : { i: sortCol, best: true } };
    drawTable(id);
  }
  // Rank-based tint within one column, direction-aware: the best value is darkest
  function rankShade(vals, v, dir, cap = 52) {
    if (!isNum(v) || vals.length < 2) return '';
    let s = (vals.indexOf(v) + vals.lastIndexOf(v)) / 2 / (vals.length - 1);
    if (dir === -1) s = 1 - s;
    return `background-color:color-mix(in srgb,var(--seq) ${Math.round(4 + s * cap)}%,var(--surface))`;
  }
  function drawTable(id) {
    const el = $(id);
    const t = tables[id];
    if (!el || !t) return;
    const { cols, rows, sort } = t;
    const c = cols[sort.i];
    const sign = (c.dir === 1 ? -1 : 1) * (sort.best ? 1 : -1);
    const sorted = [...rows].sort((a, b) => {
      const va = c.get(a);
      const vb = c.get(b);
      if (missing(va) || missing(vb)) return missing(va) - missing(vb);
      return sign * (typeof va === 'string' ? va.localeCompare(vb) : va - vb);
    });
    const colVals = cols.map((col) => (col.dir ? rows.map((r) => col.get(r)).filter(isNum).sort((a, b) => a - b) : []));
    const best = colVals.map((v, i) => (v.length > 1 ? (cols[i].dir === 1 ? v[v.length - 1] : v[0]) : null));
    const arrow = (d) => (d === 1 ? ' <span class="dir" title="higher is better">↑</span>' : d === -1 ? ' <span class="dir" title="lower is better">↓</span>' : '');
    const head = cols.map((col, i) => {
      const s = i === sort.i ? ` aria-sort="${sign < 0 ? 'descending' : 'ascending'}"` : '';
      const sub = col.sub === undefined ? '' : ` <span class="sd">${esc(col.sub)}</span>`;
      return `<th class="${col.text ? '' : 'r'}"${s}><button type="button" class="th-btn" data-t="${id}" data-i="${i}"${col.tip ? ` data-tip="${esc(esc(col.tip))}"` : ''}>${esc(col.label)}${sub}${arrow(col.dir)}</button></th>`;
    }).join('');
    const body = sorted.map((r, k) => {
      const cells = cols.map((col, i) => {
        const v = col.get(r);
        const cls = `${col.text ? '' : 'r num'}${best[i] !== null && v === best[i] ? ' best' : ''}`;
        const st = t.shade && col.dir ? ` style="${rankShade(colVals[i], v, col.dir, 34)}"` : '';
        return `<td class="${cls}"${st}>${col.fmt ? col.fmt(v, r) : missing(v) ? DASH : esc(v)}</td>`;
      }).join('');
      return `<tr><td class="rank num">${k + 1}</td>${cells}</tr>`;
    }).join('');
    el.innerHTML = `<thead><tr><th class="rank">#</th>${head}</tr></thead><tbody>${body}</tbody>`;
  }
  document.addEventListener('click', (e) => {
    const b = e.target instanceof Element ? e.target.closest('.th-btn[data-t]') : null;
    if (!b) return;
    const t = tables[b.dataset.t];
    const i = +b.dataset.i;
    t.sort = t.sort.i === i ? { i, best: !t.sort.best } : { i, best: true };
    drawTable(b.dataset.t);
  });
  const NAME_COL = { label: 'Tokeniser', text: true, get: (r) => r.key, fmt: (k) => nameHtml(k) };

  // ---------- controls ----------
  const state = { vocab: 16384, tokkaMetric: 'bytes_per_token', intrinsicMetric: 'renyi_e_2_5', tpMetric: 'encode_mb_per_sec', tier: 'proxy_small', vocabMetric: 'bytes_per_token', swapMode: 'val', flores: 'intrinsic@16384', pgSample: null, pgVocab: null };
  function seg(id, key, after) {
    const g = $(id);
    if (!g) return;
    g.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      state[key] = b.dataset.v;
      g.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      after();
    });
  }
  function sel(id, key, after, cast = String) {
    const s = $(id);
    if (!s) return;
    s.value = state[key];
    s.addEventListener('change', () => { state[key] = cast(s.value); after(); });
  }
  // Each section must render even when another one meets a result file it cannot read
  const safe = (fns) => fns.forEach((f) => { try { f(); } catch (err) { console.error(`${f.name} failed:`, err); } });

  // ---------- 01 overview ----------
  function renderAbout() {
    const tk = R('tokka_16384').data?.meta || {};
    const sets = Object.values(D.results || {});
    const n = Object.keys(R('intrinsic_16384').data?.models || {}).length;
    const items = [
      ['Variants benchmarked', n || '–', ''],
      ['Tokeniser training text', isNum(tk.train_bytes) ? `${(tk.train_bytes / 1e6).toFixed(1)} MB` : '–', isNum(tk.train_docs) ? `${int(tk.train_docs)} documents` : ''],
      ['Held-out test text', isNum(tk.eval_bytes) ? mb(tk.eval_bytes) : '–', `${(tk.macro_languages || []).length} languages`],
      ['Vocabulary sizes', VOCABS.length, '16,384 to 65,536'],
      ['Result sets', sets.filter((e) => e.available).length, `of ${sets.length} committed`],
    ];
    setHtml('scope', items.map(([k, v, s]) => `<div><dt>${esc(k)}</dt><dd>${v} ${s ? `<small>${s}</small>` : ''}</dd></div>`).join(''));
    if (D.timestamp) setHtml('built', `Data built ${esc(new Date(D.timestamp).toISOString().slice(0, 10))}`);
  }

  // ---------- 02 variants ----------
  const SHORT = {
    bpe_frequency: 'Merge the most frequent pair. The baseline every difference is measured against.',
    wordpiece_pmi: 'Pair count over the product of the smoothed part counts.',
    wp_bpe_base: 'The raw PMI ratio, with no frequency cutoff.',
    wp_bpe_ft: 'The PMI ratio, for pairs seen at least twice.',
    wp_bpe_pig: 'PMI weighted by the pair’s count.',
    wp_bpe_scp: 'Geometric-mean PMI with a Kneser-Ney style discount.',
    wp_bpe_npmi: 'Normalised PMI weighted by sub-linear frequency.',
    wp_bpe_ppig: 'Positive information gain divided by a length penalty.',
    mdl_bits: 'Corpus bits saved by the merge, minus the cost of the new entry.',
    length_penalized: 'Frequency divided by a power of the merged length.',
    super_bpe: 'Word-internal BPE first, then merges across words.',
    morpheme_bpe: 'Frequency scaled by a morpheme-boundary prior.',
    unigram_lm: 'Top-down: prune a large seed vocabulary with EM.',
    'bpe_dropout_p0.1': 'The bpe_frequency vocabulary, with merges skipped at random during LM training.',
  };
  const MORE = {
    wp_bpe_base: 'The ratio is at most 1 / max(freq(u), freq(v)) and reaches its maximum, 1, for a pair seen once whose parts occur nowhere else. So the rarest pairs are merged first. Each merge creates a rare token whose pairs are rare too, and the vocabulary fills with long strings that hardly recur in new text, which then falls back to near byte-level tokens.',
    wp_bpe_ft: 'The threshold removes singleton pairs. Pairs seen twice still outrank frequent ones, and the result stays close to wordpiece_pmi.',
    super_bpe: 'Up to the switch point it is ordinary BPE with pre-tokenisation. After it, the whitespace boundary is lifted and merges may join words into tokens such as “ of the”. The benchmark suite switches after 90% of the merges; the playground at 0.75 V.',
    morpheme_bpe: 'M is the share of the pair’s occurrences, in lexicon words, whose merged token stays inside one morpheme, minus the share that would span a morpheme boundary. Pairs with a scaled count below 1 are never merged. The formula shows the playground’s λ = 2.5; the benchmark suite trains with λ = 2 and a lexicon from the SIGMORPHON 2022 train split with every gold test word removed.',
    unigram_lm: 'Start from about 3V frequent substrings found with a suffix array, each with a probability. The E-step runs forward-backward over each word’s segmentation lattice; the M-step renormalises the expected counts; 10 EM iterations per round. Each round keeps the 75% of tokens whose removal would cost the most likelihood, until V remain. Encoding is the Viterbi segmentation.',
    'bpe_dropout_p0.1': 'In the proxy LM, each applicable merge is skipped with probability 0.1 while encoding training text, so the model sees several segmentations of the same word. Validation ids are deterministic.',
  };
  function renderVariants() {
    const used = FAMILIES.map(([f, label], i) => ({ label, i: i + 1, n: KEYS.filter((k) => D.metadata[k].family === f).length })).filter((f) => f.n);
    setHtml('family-legend', used.map((f) => `<span><i class="dot" style="--c:var(--fam-${f.i})"></i>${esc(f.label)} (${f.n})</span>`).join(''));
    const tk = R('tokka_16384').data?.models || {};
    const intr = R('intrinsic_16384').data?.models || {};
    setHtml('variant-list', KEYS.map((k) => {
      const m = D.metadata[k];
      const measured = isNum(intr[k]?.bytes_per_token) && isNum(tk[k]?.unique_tokens)
        ? `<p class="note">Measured at V = 16,384: ${intr[k].bytes_per_token.toFixed(2)} bytes per token; ${int(tk[k].unique_tokens)} of ${int(tk[k].vocab_size)} tokens occur in the held-out text.</p>`
        : '';
      return `<details><summary><span class="dot" style="--c:${famVar(k)}"></span><span class="v-name">${esc(m.display_name)}<small>${esc(k)}</small></span><span class="v-desc">${esc(SHORT[k] ?? m.description)}</span></summary>
        <div class="v-body"><div class="formula">$$${m.formula}$$</div><p>${esc(MORE[k] ?? m.description)}</p>${measured}</div></details>`;
    }).join(''));
  }

  // ---------- shared: ranking and seed margins ----------
  // Best-first [key, value] pairs
  const ranked = (models, get, dir) => Object.entries(models || {}).map(([k, m]) => [k, get(m)]).filter(([, v]) => isNum(v)).sort((a, b) => (b[1] - a[1]) * dir);
  // Margin between the two leaders of a seeded run, in standard errors of the difference
  function seedZ(models, r) {
    if (r.length < 2) return null;
    const se = (m) => (m.macro_val_bpb_sd ?? 0) ** 2 / Object.keys(m.macro_val_bpb_by_seed || {}).length;
    const [a, b] = [models[r[0][0]], models[r[1][0]]];
    const z = (b.macro_val_bpb - a.macro_val_bpb) / Math.sqrt(se(a) + se(b));
    return Number.isFinite(z) ? z : null;
  }
  const resolved = (z) => `${z.toFixed(1)} standard errors over the seeds, ${z < 2 ? 'within noise' : 'resolved'}`;

  // ---------- 04 overall ----------
  function renderOverall() {
    const I = R('intrinsic_16384').data?.models || {};
    const S = R('proxy_small').data?.models || {};
    const B = R('proxy_big').data?.models || {};
    const T = R('throughput_local').data?.models?.['16384'] || {};
    const F = R('flores').data?.models || {};
    const rows = KEYS.map((key) => ({
      key, small: S[key]?.macro_val_bpb, big: B[key]?.macro_val_bpb, bpt: I[key]?.bytes_per_token, renyi: I[key]?.renyi_e_2_5,
      morph: I[key]?.morph_f1, ident: I[key]?.identifier_f1, enc: mean(T[key]?.encode_mb_per_sec), prem: F[`${key}@16384`]?.aggregates?.in_mix?.mean_premium,
    }));
    table('overall-table', [
      NAME_COL,
      { label: 'Small LM BPB', dir: -1, get: (r) => r.small, fmt: (v) => fmt(v), tip: 'Macro validation bits per byte, small proxy LM, mean of 3 seeds' },
      { label: 'Big LM BPB', dir: -1, get: (r) => r.big, fmt: (v) => fmt(v), tip: 'Macro validation bits per byte, big proxy LM, mean of 3 seeds' },
      { label: 'Bytes/token', dir: 1, get: (r) => r.bpt, fmt: (v) => fmt(v, 2) },
      { label: 'Rényi E₂.₅', dir: 1, get: (r) => r.renyi, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 2.5' },
      { label: 'Morph F1', dir: 1, get: (r) => r.morph, fmt: (v) => fmt(v), tip: 'Morpheme-boundary F1, mean over 5 languages' },
      { label: 'Identifier F1', dir: 1, get: (r) => r.ident, fmt: (v) => fmt(v) },
      { label: 'Encode MB/s', dir: 1, get: (r) => r.enc, fmt: (v) => fmt(v, 1), tip: 'One pinned laptop core, mean of 2 runs' },
      { label: 'In-mix premium', dir: -1, get: (r) => r.prem, fmt: (v) => fmt(v, 2), tip: 'FLORES+ mean tokenisation premium over the 10 in-mix languages' },
    ], rows, 1, true);

    const pts = rows.map((r) => ({ key: r.key, x: r.bpt, y: r.small })).filter((p) => isNum(p.x) && isNum(p.y));
    const sc = { xTitle: 'Bytes per token ↑', yTitle: 'Small-LM validation BPB ↓', xName: 'Bytes per token', yName: 'Small-LM BPB' };
    scatter('scatter', { ...sc, pts, label: 'Scatter plot of bytes per token against small-LM validation BPB, one point per tokeniser' });
    // Most variants sit in one tight group that the full range cannot separate: a second panel on that group alone
    const mx = median(pts.map((p) => p.x));
    const y0 = Math.min(...pts.map((p) => p.y));
    const group = pts.filter((p) => Math.abs(p.x - mx) < 0.5 && p.y < y0 + 0.3);
    if (group.length > 3 && group.length < pts.length) {
      scatter('scatter-zoom', { ...sc, pts: group, label: 'The same plot restricted to the main group of tokenisers' });
      setHtml('scatter-zoom-sub', `<b>The main group, enlarged.</b> The ${group.length} variants within 0.5 bytes per token of the median. Axes do not start at zero.`);
    }
    setHtml('scatter-sub', 'One point per variant at V = 16,384. Towards the bottom right is better on both axes. Colour is the algorithm family.');

    const rs = ranked(S, (m) => m.macro_val_bpb, -1);
    const rc = ranked(I, (m) => m.bytes_per_token, 1);
    const low = rc.filter(([, v]) => v < 1.5).map(([k]) => k);
    const parts = [];
    if (rs.length) parts.push(`With the small proxy LM, ${code(rs[0][0])} has the lowest validation BPB (${rs[0][1].toFixed(3)}); ${code('bpe_frequency')} is at ${txt(S.bpe_frequency?.macro_val_bpb)}.`);
    if (rc.length) parts.push(`${code(rc[0][0])} covers the most text per token (${rc[0][1].toFixed(2)} bytes).`);
    if (low.length) parts.push(`${low.length} variant${low.length === 1 ? ' stays' : 's stay'} below 1.5 bytes per token, close to raw bytes: ${low.map(code).join(', ')}.`);
    setHtml('overall-finding', parts.join(' '));
    renderBest();
  }

  // GPT-2 swap runs whose final BPB compares with the original's (same recipe and validation documents)
  const SWAP_RUNS = { swap: 'original run', swap_space_prefix: 'space-prefix rerun', swap_ablation: 'ablation' };
  function comparableSwap() {
    const out = {};
    Object.entries(SWAP_RUNS).forEach(([key, run]) => {
      const e = R(key);
      if (!e.available || (key !== 'swap' && !(Array.isArray(e.not_comparable) && !e.not_comparable.length))) return;
      Object.entries(e.data.models || {}).forEach(([k, m]) => {
        if (!isNum(m.final_bpb)) return;
        if (!(k in out) || m.final_bpb < out[k].final_bpb) out[k] = { ...m, run };
      });
    });
    return out;
  }
  function renderBest() {
    const tp = R('throughput_local');
    const spreads = tp.available ? Object.values(tp.data.models?.['16384'] || {}).map((m) => spread(m.encode_mb_per_sec)).filter(isNum) : [];
    const seedNote = (r, M) => { const z = seedZ(M, r); return z === null ? '' : `Lead: ${resolved(z)}.`; };
    const specs = [
      { set: 'proxy_small', what: 'Small proxy LM, macro validation BPB ↓', d: 3, dir: -1, href: '#proxy', get: (m) => m.macro_val_bpb, note: seedNote },
      { set: 'proxy_big', what: 'Big proxy LM, macro validation BPB ↓', d: 3, dir: -1, href: '#proxy', get: (m) => m.macro_val_bpb, note: seedNote },
      { set: 'swap', what: 'GPT-2 small after 600 steps, validation BPB ↓', d: 4, dir: -1, href: '#swap', models: comparableSwap, get: (m) => m.final_bpb, note: (r, M) => `One run each, so no seed spread. ${code(r[0][0])} is from the ${M[r[0][0]].run}${r[1] ? `, ${code(r[1][0])} from the ${M[r[1][0]].run}` : ''}.` },
      { set: 'intrinsic_16384', what: 'Bytes per token ↑', d: 2, dir: 1, href: '#compression', get: (m) => m.bytes_per_token, note: () => 'SuperBPE tokens can span words.' },
      { set: 'intrinsic_16384', what: 'Rényi efficiency, α = 2.5 ↑', d: 3, dir: 1, href: '#intrinsic', get: (m) => m.renyi_e_2_5, note: () => 'Measures how evenly tokens are used.' },
      { set: 'intrinsic_16384', what: 'Morph F1 ↑', d: 3, dir: 1, href: '#intrinsic', get: (m) => m.morph_f1, note: () => 'morpheme_bpe is trained with a morpheme lexicon.' },
      { set: 'intrinsic_16384', what: 'Identifier F1 ↑', d: 3, dir: 1, href: '#intrinsic', get: (m) => m.identifier_f1, note: (r) => `${r.filter(([, v]) => r[0][1] - v < 0.01).length - 1} others are within 0.01.` },
      { set: 'throughput_local', what: 'Encode MB/s, pinned core ↑', d: 1, dir: 1, href: '#throughput', models: () => tp.data?.models?.['16384'], get: (m) => mean(m.encode_mb_per_sec), note: () => (spreads.length ? `Run-to-run spread: median ${median(spreads).toFixed(2)}×.` : '') },
    ];
    const body = specs.map((s) => {
      const e = R(s.set);
      if (!e.available) return `<tr><td><a href="${s.href}">${esc(s.what)}</a></td><td colspan="4" class="muted">not run</td></tr>`;
      const M = (s.models ? s.models() : e.data.models) || {};
      const r = ranked(M, s.get, s.dir);
      if (!r.length) return '';
      const run = r[1] ? `${nameHtml(r[1][0])} <span class="num">${r[1][1].toFixed(s.d)}</span> <span class="sd">(${signed(r[1][1] - r[0][1], s.d)})</span>` : DASH;
      return `<tr><td><a href="${s.href}">${esc(s.what)}</a></td><td>${nameHtml(r[0][0])}</td><td class="r num"><b>${r[0][1].toFixed(s.d)}</b></td><td>${run}</td><td class="muted">${s.note(r, M)}</td></tr>`;
    }).join('');
    setHtml('best-table', `<thead><tr><th>Metric</th><th>Best</th><th class="r">Value</th><th>Runner-up</th><th>What limits the comparison</th></tr></thead><tbody>${body}</tbody>`);
  }

  // ---------- 05 compression ----------
  const TOKKA = {
    bytes_per_token: ['Bytes per token', 1, 2], subword_fertility: ['Fertility (tokens per word)', -1, 2], word_split_pct: ['Word split %', -1, 1],
    continued_word_rate: ['Continued %', -1, 1], unique_tokens: ['Unique tokens used', 0, 0],
  };
  const LANG = { code: 'Code', python: 'Python', javascript: 'JS', java: 'Java', cpp: 'C++', go: 'Go', rust: 'Rust' };
  const langLabel = (l) => LANG[l] ?? l.toUpperCase();
  function renderTokka() {
    const e = R(`tokka_${state.vocab}`);
    setHtml('tokka-prov', provHtml(e));
    if (!e.available) return;
    const M = e.data.models ?? e.data;
    const metric = state.tokkaMetric;
    const [label, dir, dec] = TOKKA[metric];
    const avg = (k) => (metric === 'unique_tokens' ? M[k]?.unique_tokens : M[k]?.macro_avg?.[metric]);
    const order = Object.keys(M).sort((a, b) => ((avg(b) ?? -Infinity) - (avg(a) ?? -Infinity)) * (dir || 1));
    const langs = D.tokka_languages || [];
    const show = (v) => (isNum(v) ? (dec === 0 ? Math.round(v).toLocaleString('en-US') : v.toFixed(dec)) : '–');

    setHtml('tokka-chart-title', `${esc(label)}, ${metric === 'unique_tokens' ? 'over all languages' : 'macro average'}.`);
    setHtml('tokka-chart-sub', `V = ${vLabel(state.vocab)}. ${dir === 1 ? 'Higher is better.' : dir === -1 ? 'Lower is better.' : 'Of the vocabulary’s tokens, how many occur in the held-out text.'}`);
    rowChart('tokka-chart', { mode: 'bar', label, title: label, rows: order.map((k) => ({ key: k, marks: [{ v: avg(k) }], val: show(avg(k)) })) });

    const cols = [...langs, 'macro_avg'];
    const sorted = Object.fromEntries(cols.map((c) => [c, order.map((k) => M[k]?.[c]?.[metric]).filter(isNum).sort((a, b) => a - b)]));
    setHtml('tokka-lo', dir === 0 ? 'fewer' : 'worse');
    setHtml('tokka-hi', dir === 0 ? 'more' : 'better');
    setHtml('tokka-heat', `<thead><tr><th>Tokeniser</th>${langs.map((l) => `<th>${esc(langLabel(l))}</th>`).join('')}<th class="avg">Macro</th></tr></thead><tbody>${order.map((k) =>
      `<tr><td>${nameHtml(k)}</td>${cols.map((c) => { const v = M[k]?.[c]?.[metric]; return `<td class="cell${c === 'macro_avg' ? ' avg' : ''}" style="${rankShade(sorted[c], v, dir || 1)}">${show(v)}</td>`; }).join('')}</tr>`).join('')}</tbody>`);

    const macro = (m) => (r) => r.macro_avg?.[m];
    table('tokka-table', [
      NAME_COL,
      { label: '|V|', dir: 0, get: (r) => r.vocab_size, fmt: int },
      { label: 'Macro B/tok', dir: 1, get: macro('bytes_per_token'), fmt: (v) => fmt(v), tip: 'Unweighted mean over the languages' },
      { label: 'Micro B/tok', dir: 1, get: (r) => r.micro_avg?.bytes_per_token, fmt: (v) => fmt(v), tip: 'Totals over all languages (token-weighted)' },
      { label: 'Fertility', dir: -1, get: macro('subword_fertility'), fmt: (v) => fmt(v) },
      { label: 'Word split %', dir: -1, get: macro('word_split_pct'), fmt: (v) => fmt(v, 1) },
      { label: 'Continued %', dir: -1, get: macro('continued_word_rate'), fmt: (v) => fmt(v, 1) },
      { label: 'Unique tokens used', dir: 0, get: (r) => r.unique_tokens, fmt: int, tip: 'Distinct tokens over all languages' },
    ], Object.entries(M).map(([key, m]) => ({ key, ...m })), 2);

    const r = ranked(M, (m) => m.macro_avg?.bytes_per_token, 1);
    if (r.length > 1) {
      const worst = r[r.length - 1];
      const above = r.filter(([, v]) => v > (M.bpe_frequency?.macro_avg?.bytes_per_token ?? Infinity)).length;
      setHtml('tokka-finding', `At V = ${vLabel(state.vocab)}, macro bytes per token runs from ${worst[1].toFixed(2)} for ${code(worst[0])} to ${r[0][1].toFixed(2)} for ${code(r[0][0])}. ${code('bpe_frequency')} is at ${txt(M.bpe_frequency?.macro_avg?.bytes_per_token, 2)}; ${above} of the ${r.length} variants cover${above === 1 ? 's' : ''} more text per token than it does.`);
    }
  }

  // ---------- 06 intrinsic ----------
  const INTRINSIC = [
    { key: 'bytes_per_token', label: 'Bytes/token', long: 'Bytes per token', dir: 1, fmt: (v) => fmt(v, 2), tip: 'Bytes of raw text one token covers' },
    { key: 'fertility', label: 'Fertility', long: 'Fertility (tokens per word)', dir: -1, fmt: (v) => fmt(v, 2), tip: 'Average tokens per word' },
    { key: 'renyi_e_1_0', label: 'E₁', long: 'Rényi efficiency, α = 1 (Shannon)', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 1 (Shannon)' },
    { key: 'renyi_e_2_0', label: 'E₂', long: 'Rényi efficiency, α = 2', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 2' },
    { key: 'renyi_e_2_5', label: 'E₂.₅', long: 'Rényi efficiency, α = 2.5', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 2.5 (Zouhar et al. 2023)' },
    { key: 'gini_coefficient', label: 'Gini', long: 'Gini coefficient of token frequencies', dir: -1, fmt: (v) => fmt(v), tip: 'Inequality of token frequencies' },
    { key: 'active_fraction', label: 'Active', long: 'Active fraction of the vocabulary', dir: 1, fmt: pct, tip: 'Share of the vocabulary used on the held-out text' },
    { key: 'morph_f1', label: 'Morph F1', long: 'Morph F1, mean over 5 languages', dir: 1, fmt: (v) => fmt(v), tip: 'Morpheme-boundary F1, mean over languages' },
    { key: 'morph_precision', label: 'Morph P', long: 'Morph precision', dir: 1, fmt: (v) => fmt(v), tip: 'Of the cuts made, the share on true morpheme boundaries' },
    { key: 'morph_recall', label: 'Morph R', long: 'Morph recall', dir: 1, fmt: (v) => fmt(v), tip: 'Of the true morpheme boundaries, the share cut' },
    { key: 'identifier_f1', label: 'Ident F1', long: 'Identifier F1', dir: 1, fmt: (v) => fmt(v), tip: 'Boundary F1 on code identifiers' },
    { key: 'encode_mb_per_sec', label: 'Enc MB/s', dir: 1, fmt: (v) => fmt(v, 1), tip: 'Encode speed, unpinned laptop run, one pass', tableOnly: true },
    { key: 'decode_mb_per_sec', label: 'Dec MB/s', dir: 1, fmt: (v) => fmt(v, 1), tip: 'Decode speed, unpinned laptop run, one pass', tableOnly: true },
  ];
  const CHARTABLE = INTRINSIC.filter((c) => !c.tableOnly);
  function renderIntrinsic() {
    const e = R(`intrinsic_${state.vocab}`);
    setHtml('intrinsic-prov', provHtml(e));
    if (!e.available) return;
    const M = e.data.models ?? e.data;
    const c = INTRINSIC.find((x) => x.key === state.intrinsicMetric);
    const r = ranked(M, (m) => m[c.key], c.dir);
    setHtml('intrinsic-chart-title', `${esc(c.long)}.`);
    setHtml('intrinsic-chart-sub', `V = ${vLabel(state.vocab)}. ${c.dir === 1 ? 'Higher is better.' : 'Lower is better.'}`);
    rowChart('intrinsic-chart', { mode: 'bar', label: c.long, title: c.long, rows: r.map(([k, v]) => ({ key: k, marks: [{ v }], val: c.fmt(v) })) });

    const words = Object.values(M)[0]?.morph_words_by_lang || {};
    const langs = Object.keys(words);
    const order = ranked(M, (m) => m.morph_f1, 1).map(([k]) => k);
    const cols = [...langs.map((l) => ({ h: `${esc(l)}<small>${int(words[l])}</small>`, get: (m) => m.morph_f1_by_lang?.[l] })), { h: 'Mean', avg: true, get: (m) => m.morph_f1 }, { h: 'Word-weighted', get: (m) => m.morph_f1_word_weighted }];
    const sorted = cols.map((col) => order.map((k) => col.get(M[k])).filter(isNum).sort((a, b) => a - b));
    setHtml('morph-heat', `<thead><tr><th>Tokeniser</th>${cols.map((col) => `<th class="${col.avg ? 'avg' : ''}">${col.h}</th>`).join('')}</tr></thead><tbody>${order.map((k) =>
      `<tr><td>${nameHtml(k)}</td>${cols.map((col, i) => `<td class="cell${col.avg ? ' avg' : ''}" style="${rankShade(sorted[i], col.get(M[k]), 1)}">${txt(col.get(M[k]))}</td>`).join('')}</tr>`).join('')}</tbody>`);

    table('intrinsic-table', [NAME_COL, ...INTRINSIC.map((x) => ({ label: x.label, dir: x.dir, get: (row) => row[x.key], fmt: x.fmt, tip: x.tip }))], Object.entries(M).map(([key, m]) => ({ key, ...m })), 5, true);

    const mf = ranked(M, (m) => m.morph_f1, 1);
    const idf = ranked(M, (m) => m.identifier_f1, 1);
    const re = ranked(M, (m) => m.renyi_e_2_5, 1);
    const parts = [];
    if (re.length) parts.push(`At V = ${vLabel(state.vocab)}, ${code(re[0][0])} uses its tokens most evenly (Rényi E₂.₅ ${re[0][1].toFixed(3)}, at ${txt(M[re[0][0]].bytes_per_token, 2)} bytes per token).`);
    if (mf.length > 1) {
      const unsup = mf.find(([k]) => k !== 'morpheme_bpe');
      parts.push(`${code(mf[0][0])} has the highest Morph F1 (${mf[0][1].toFixed(3)})${mf[0][0] === 'morpheme_bpe' && unsup ? `; the highest without a morpheme lexicon is ${code(unsup[0])} (${unsup[1].toFixed(3)})` : ''}.`);
    }
    if (idf.length) parts.push(`On code identifiers, ${idf.filter(([, v]) => idf[0][1] - v < 0.01).length} variants are within 0.01 of the best F1 (${idf[0][1].toFixed(3)}).`);
    setHtml('intrinsic-finding', parts.join(' '));
  }

  // ---------- 07 throughput ----------
  const TP_SERIES = [
    { label: 'Laptop suite run (unpinned, 1 pass)', c: 'var(--fam-0)', shape: 'di' },
    { label: 'Pinned laptop core (mean of 2 runs)', c: 'var(--fam-1)', shape: '' },
    { label: 'Colab CPU (mean of 2 runs)', c: 'var(--fam-2)', shape: 'sq' },
  ];
  function renderThroughput() {
    const V = String(state.vocab);
    const key = state.tpMetric;
    const word = key.startsWith('encode') ? 'Encode' : 'Decode';
    const loc = R('throughput_local');
    const col = R('throughput_colab_cpu');
    const suite = R(`intrinsic_${V}`);
    const L = loc.available ? loc.data.models?.[V] || {} : {};
    const C = col.available ? col.data.models?.[V] || {} : {};
    const S = suite.available ? suite.data.models || {} : {};
    const names = [...new Set([...Object.keys(S), ...Object.keys(L), ...Object.keys(C)])];
    const rows = names.map((k) => ({ key: k, suite: S[k]?.[key], local: L[k]?.[key], colab: C[k]?.[key] }));
    const order = [...rows].sort((a, b) => (mean(b.local) ?? b.suite ?? 0) - (mean(a.local) ?? a.suite ?? 0));
    setHtml('tp-chart-title', `${word} speed in MB/s, V = ${vLabel(state.vocab)}.`);
    rowChart('tp-chart', {
      zero: true, label: `${word} MB/s per tokeniser in three settings`, title: `${word} MB/s`, series: TP_SERIES,
      rows: order.map((r) => ({
        key: r.key, marks: [{ v: r.suite, s: 0 }, { v: mean(r.local), s: 1 }, { v: mean(r.colab), s: 2 }], val: txt(mean(r.local), 1),
        tip: `<b class="mono">${esc(r.key)}</b><br>Laptop suite: ${txt(r.suite, 1)}<br>Pinned core: ${txt(mean(r.local), 1)}<br>Colab CPU: ${txt(mean(r.colab), 1)} MB/s`,
      })),
    });

    const sp = (M) => Object.values(M).map((m) => spread(m[key])).filter(isNum);
    const ratio = (a, b) => rows.map((r) => (isNum(a(r)) && isNum(b(r)) ? a(r) / b(r) : null)).filter(isNum);
    const facts = [];
    for (const [label, M, e] of [['Pinned laptop core', L, loc], ['Colab CPU', C, col]]) {
      const s = sp(M);
      if (!e.available) facts.push(`${label}: not run.`);
      else if (s.length) facts.push(`${label}: run-to-run spread median ${median(s).toFixed(2)}×, maximum ${Math.max(...s).toFixed(2)}×; ${s.filter((x) => x > 1.2).length} of ${s.length} tokenisers differ by more than 1.2× between runs.`);
    }
    if (suite.available) facts.push('Laptop suite run: one pass per tokeniser, so its spread is not recorded.');
    const r1 = ratio((r) => mean(r.local), (r) => r.suite);
    const r2 = ratio((r) => mean(r.colab), (r) => mean(r.local));
    if (r1.length) facts.push(`Pinned core ÷ laptop suite: median ${median(r1).toFixed(2)}× (${range(r1, 2, '×')}).`);
    if (r2.length) facts.push(`Colab CPU ÷ pinned core: median ${median(r2).toFixed(2)}× (${range(r2, 2, '×')}).`);
    setHtml('tp-facts', facts.map((f) => `<li>${esc(f)}</li>`).join(''));

    const passes = (v) => (Array.isArray(v) ? v.map((x) => x.toFixed(2)).join(' / ') : DASH);
    const sf = (v) => (isNum(v) ? `${v.toFixed(2)}×` : DASH);
    table('tp-table', [
      NAME_COL,
      { label: 'Laptop suite', dir: 1, get: (r) => r.suite, fmt: (v) => fmt(v, 2), tip: 'Intrinsic suite run: unpinned shared laptop, one pass' },
      { label: 'Pinned core', dir: 1, get: (r) => mean(r.local), fmt: (v, r) => passes(r.local), tip: 'One pinned performance core: run 1 / run 2' },
      { label: 'spread', dir: -1, get: (r) => spread(r.local), fmt: sf, tip: 'max / min over the pinned runs' },
      { label: 'Colab CPU', dir: 1, get: (r) => mean(r.colab), fmt: (v, r) => passes(r.colab), tip: 'Colab CPU runtime: run 1 / run 2' },
      { label: 'spread', dir: -1, get: (r) => spread(r.colab), fmt: sf, tip: 'max / min over the Colab runs' },
    ], rows, 2);
    setHtml('tp-prov', provHtml(suite, 'Laptop suite run') + provHtml(loc, 'Pinned laptop core') + provHtml(col, 'Colab CPU'));

    const pr = order.filter((r) => isNum(mean(r.local)));
    if (pr.length > 1) {
      const a = pr[0];
      const z = pr[pr.length - 1];
      const bpt = S[a.key]?.bytes_per_token;
      setHtml('tp-finding', `On the pinned core at V = ${vLabel(state.vocab)}, ${word.toLowerCase()} speed runs from ${mean(z.local).toFixed(1)} MB/s for ${code(z.key)} to ${mean(a.local).toFixed(1)} MB/s for ${code(a.key)}${isNum(bpt) ? `. ${code(a.key)} covers ${bpt.toFixed(2)} bytes per token, so its speed says little about its compression` : ''}.`);
    }
  }

  // ---------- 08 proxy LM ----------
  function renderProxy() {
    const e = R(state.tier);
    setHtml('proxy-prov', provHtml(e));
    renderCompare();
    if (!e.available) return;
    const meta = e.data.meta || {};
    const lm = meta.lm || {};
    const V = meta.requested_vocab_size;
    const params = isNum(lm.d_model) && isNum(V) ? V * lm.d_model + lm.n_layers * (4 * lm.d_model ** 2 + 2 * lm.d_model * lm.d_ffn) : null;
    const hw = Object.keys(meta.runs_by_hardware || {});
    setHtml('proxy-sub', esc(`${state.tier === 'proxy_big' ? 'Big' : 'Small'} LM: ${lm.n_layers ?? '?'} layers, d_model ${lm.d_model ?? '?'}${params ? `, about ${(params / 1e6).toFixed(1)}M parameters (${((V * lm.d_model) / 1e6).toFixed(1)}M in the embedding)` : ''}; ${isNum(meta.byte_budget) ? mb(meta.byte_budget) : '?'} of text per tokeniser; ${hw.length > 1 ? hw.join(' and ').replace(/`/g, '') : (meta.backend?.gpu ?? '').split(',')[0]}.`));

    const M = e.data.models || {};
    const models = Object.entries(M).map(([key, m]) => ({ key, ...m })).filter((m) => isNum(m.macro_val_bpb)).sort((a, b) => a.macro_val_bpb - b.macro_val_bpb);
    const vals = models.map((m) => m.macro_val_bpb);
    // A few variants sit far above the rest and would flatten every other difference: clip the scale there
    const cut = median(vals) + 4 * (median(vals) - vals[0]);
    const clip = Math.max(...vals) > cut ? cut : undefined;
    const ref = M.bpe_frequency?.macro_val_bpb;
    rowChart('proxy-chart', {
      label: 'Macro validation BPB per tokeniser with seed standard deviation', title: 'Macro validation BPB', max: clip, ref: isNum(ref) ? { v: ref, label: 'bpe_frequency' } : null,
      rows: models.map((m) => ({
        key: m.key, marks: [{ v: m.macro_val_bpb, lo: m.macro_val_bpb - (m.macro_val_bpb_sd ?? 0), hi: m.macro_val_bpb + (m.macro_val_bpb_sd ?? 0) }], val: m.macro_val_bpb.toFixed(3),
        tip: `<b class="mono">${esc(m.key)}</b><br>${m.macro_val_bpb.toFixed(4)} ± ${txt(m.macro_val_bpb_sd, 4)}<br>seeds: ${Object.values(m.macro_val_bpb_by_seed || {}).map((v) => v.toFixed(4)).join(', ')}`,
      })),
    });
    const off = clip ? models.filter((m) => m.macro_val_bpb > clip) : [];
    setHtml('proxy-offscale', off.length ? `Beyond the scale (arrow): ${off.map((m) => `${code(m.key)} ${m.macro_val_bpb.toFixed(3)}`).join(', ')}.` : '');

    const curves = models.filter((m) => m.curve?.length);
    const tail = curves.flatMap((m) => m.curve.filter((p, i) => i >= m.curve.length * 0.12).map((p) => p[1]));
    const inView = tail.filter((y) => !clip || y <= Math.max(...tail.filter((v) => v <= cut * 1.6), cut));
    const ymax = clip ? Math.min(Math.max(...tail), cut * 1.35) : Math.max(...tail);
    lineChart('proxy-curves', {
      label: 'Train BPB against megabytes of text consumed', xTitle: 'Text consumed (MB)', yTitle: 'Train BPB', yDomain: [Math.min(...inView) * 0.985, ymax],
      series: curves.map((m) => ({ key: m.key, pts: m.curve })),
      tipFn: (s, p) => `<b class="mono">${esc(s.key)}</b><br>${p[1].toFixed(3)} BPB at ${p[0].toFixed(1)} MB`,
    });

    const domains = meta.val_domains || Object.keys(models[0]?.val_domains || {});
    const pm = (v, sd) => (isNum(v) ? `${v.toFixed(3)}${isNum(sd) ? ` <span class="sd">± ${sd.toFixed(3)}</span>` : ''}` : DASH);
    table('proxy-table', [
      NAME_COL,
      { label: 'Macro BPB', dir: -1, get: (r) => r.macro_val_bpb, fmt: (v, r) => pm(v, r.macro_val_bpb_sd) },
      { label: 'Δ', tip: 'Macro BPB minus bpe_frequency’s', dir: -1, get: (r) => (isNum(ref) ? r.macro_val_bpb - ref : null), fmt: (v, r) => (r.key === 'bpe_frequency' ? '<span class="muted">reference</span>' : `<span class="${v <= 0 ? 'good' : 'bad'}">${signed(v)}</span>`) },
      ...domains.map((d) => ({ label: `${d} BPB`, dir: -1, get: (r) => r.val_domains?.[d]?.bpb, fmt: (v, r) => pm(v, r.val_domains?.[d]?.bpb_sd) })),
      { label: 'Train B/tok', dir: 0, get: (r) => r.train_bytes_per_token, fmt: (v) => fmt(v), tip: 'UTF-8 bytes per token of the LM training text' },
      { label: 'Tokens trained', dir: 0, get: (r) => r.tokens_consumed, fmt: int, tip: 'Tokens the LM trained on within the byte budget' },
    ], models, 1);

    const parts = [['proxy_small', 'Small LM'], ['proxy_big', 'Big LM']].map(([k, name]) => {
      const X = R(k).data?.models || {};
      const r = ranked(X, (m) => m.macro_val_bpb, -1);
      if (!r.length) return '';
      const z = seedZ(X, r);
      const beat = r.filter(([, v]) => v < (X.bpe_frequency?.macro_val_bpb ?? -Infinity)).length;
      return `${name}: ${code(r[0][0])} has the lowest macro BPB (${r[0][1].toFixed(3)})${z !== null ? `, ahead of ${code(r[1][0])} by ${resolved(z)}` : ''}; ${beat} of ${r.length} variants are below ${code('bpe_frequency')} (${txt(X.bpe_frequency?.macro_val_bpb)}).`;
    });
    setHtml('proxy-finding', parts.filter(Boolean).join(' '));
  }
  function renderCompare() {
    const e = R('proxy_compare');
    const vsRef = ['small LM:', 'big LM:'].map((p) => (e.provenance || []).find((l) => l.startsWith(p))).filter(Boolean);
    setHtml('proxy-margin', vsRef.length ? `Against ${code('bpe_frequency')}: ${vsRef.map(md).join('; ')}.` : '');
    if (!e.available) { setHtml('rank-chart', notRun({ path: `${R('proxy_small').path} and ${R('proxy_big').path}`, error: e.error })); return; }
    const { ref, names, tiers } = e.data;
    const s = tiers['small LM'];
    const b = tiers['big LM'];
    const order = [...names].sort((x, y) => b[x][0] - b[y][0]);
    const n = names.length;
    rowChart('rank-chart', {
      link: true, label: 'Rank of each tokeniser with the small and the big proxy LM', title: 'Rank (1 = lowest BPB)', domain: [0.5, n + 0.5], ticks: names.map((_, i) => i + 1),
      series: [{ label: 'Small LM', c: 'var(--fam-0)', shape: 'di' }, { label: 'Big LM', c: 'var(--fam-1)', shape: '' }],
      rows: order.map((k) => ({ key: k, marks: [{ v: s[k][0], s: 0 }, { v: b[k][0], s: 1 }], val: `${s[k][0]} → ${b[k][0]}`, tip: `<b class="mono">${esc(k)}</b><br>Small LM: rank ${s[k][0]}, Δ ${signed(s[k][1])} BPB<br>Big LM: rank ${b[k][0]}, Δ ${signed(b[k][1])} BPB` })),
    });
    const delta = (tier) => (v, r) => (r.key === ref ? '<span class="muted">reference</span>' : `${signed(v)}${tier[r.key][2] == null ? '' : ` <span class="sd">± ${tier[r.key][2].toFixed(3)}</span>`}`);
    table('compare-table', [
      NAME_COL,
      { label: 'Small LM rank', dir: -1, get: (r) => s[r.key][0] },
      { label: 'Small LM Δ BPB', dir: -1, get: (r) => s[r.key][1], fmt: delta(s), tip: 'Macro BPB minus bpe_frequency’s, ± standard error' },
      { label: 'Big LM rank', dir: -1, get: (r) => b[r.key][0] },
      { label: 'Big LM Δ BPB', dir: -1, get: (r) => b[r.key][1], fmt: delta(b), tip: 'Macro BPB minus bpe_frequency’s, ± standard error' },
      { label: 'Rank change', dir: 0, get: (r) => s[r.key][0] - b[r.key][0], fmt: (v) => (v === 0 ? '0' : `${v > 0 ? '+' : '−'}${Math.abs(v)}`), tip: 'Small LM rank minus big LM rank: positive = moved up with the big LM' },
    ], names.map((key) => ({ key })), 3);
  }

  // ---------- 09 vocabulary size ----------
  function renderVocab() {
    const c = INTRINSIC.find((x) => x.key === state.vocabMetric);
    const docs = VOCABS.map((v) => R(`intrinsic_${v}`).data?.models || {});
    const rows = KEYS.filter((k) => docs.some((d) => isNum(d[k]?.[c.key]))).map((key) => ({ key, v: docs.map((d) => d[key]?.[c.key]) }));
    setHtml('vocab-chart-title', `${esc(c.long)} at three vocabulary sizes. ${c.dir === 1 ? 'Higher is better.' : 'Lower is better.'}`);
    lineChart('vocab-chart', {
      label: `${c.long} at V = 16,384, 32,768 and 65,536`, dots: true, height: 420, xDomain: [-0.15, 2.15], xTicks: VOCABS.map((v, i) => ({ v: i, label: vLabel(v) })), xTitle: 'Vocabulary size', yTitle: c.long,
      series: rows.map((r) => ({ key: r.key, pts: r.v.map((y, i) => [i, y]).filter((p) => isNum(p[1])) })),
      tipFn: (s, p) => `<b class="mono">${esc(s.key)}</b><br>${p[1].toFixed(3)} at V = ${vLabel(VOCABS[p[0]])}`,
    });
    table('vocab-table', [
      NAME_COL,
      ...VOCABS.map((v, i) => ({ label: `V = ${vLabel(v)}`, dir: c.dir, get: (r) => r.v[i], fmt: c.fmt })),
      { label: 'Change', dir: 0, get: (r) => (isNum(r.v[0]) && isNum(r.v[2]) ? r.v[2] - r.v[0] : null), fmt: (v) => signed(v), tip: 'Value at 65,536 minus value at 16,384' },
    ], rows, 1);
    const d = rows.map((r) => (isNum(r.v[0]) && isNum(r.v[2]) ? r.v[2] - r.v[0] : null)).filter(isNum);
    if (d.length) setHtml('vocab-finding', `From 16,384 to 65,536 tokens, ${esc(c.long.toLowerCase())} rises for ${d.filter((x) => x > 0).length} and falls for ${d.filter((x) => x < 0).length} of ${d.length} variants; the median change is ${signed(median(d))}.`);
  }

  // ---------- 10 GPT-2 swap ----------
  const swapColor = (k) => (k === 'gpt2' ? 'var(--text)' : k.startsWith('tm_') ? 'var(--fam-0)' : undefined);
  const swapDash = (k) => (k === 'gpt2' ? 0 : k.startsWith('tm_') ? (k.endsWith('32000') ? 2 : 1) : undefined);
  function renderSwap() {
    const e = R('swap');
    renderSpacePrefix();
    renderAblation();
    renderLr();
    if (!e.available) { setHtml('swap-prov', notRun(e)); return; }
    const doc = e.data;
    const meta = doc.meta || {};
    const a = meta.recipe?.args || {};
    const M = doc.models || {};
    const eq = doc.gpt2_equivalent?.steps || {};
    const rows = Object.entries(M).filter(([, m]) => isNum(m.final_bpb)).map(([key, m]) => ({ key, ...m, eq: eq[key] })).sort((x, y) => x.final_bpb - y.final_bpb);
    const g = M.gpt2;
    setHtml('swap-sub', `${rows.length} tokenisers · ${esc(a.num_iterations ?? '?')} steps × ${int(Number(a.total_batch_size))} tokens per step · peak learning rate ${esc(a.learning_rate ?? '?')} · ${esc((meta.backend?.gpu || '?').split(',')[0])} · one run each.`);
    rowChart('swap-chart', {
      label: 'Final validation BPB of GPT-2 small per tokeniser', title: 'Final validation BPB', ref: g ? { v: g.final_bpb, label: 'gpt2' } : null,
      rows: rows.map((r) => ({ key: r.key, marks: [{ v: r.final_bpb }], val: r.final_bpb.toFixed(4), tip: `<b class="mono">${esc(r.key)}</b><br>${esc(dispName(r.key))}<br>${r.final_bpb.toFixed(4)} BPB · ${txt(r.val_bytes_per_token, 2)} val bytes per token` })),
    });
    table('swap-table', [
      NAME_COL,
      { label: 'Final BPB', dir: -1, get: (r) => r.final_bpb, fmt: (v) => fmt(v, 4) },
      { label: 'GPT-2-equiv. steps', dir: 1, get: (r) => r.eq, fmt: (v) => (isNum(v) ? String(v) : 'not reached'), tip: 'First step at which GPT-2’s smoothed train BPB reaches this final validation BPB' },
      { label: 'Val B/tok', dir: 1, get: (r) => r.val_bytes_per_token, fmt: (v) => fmt(v), tip: 'UTF-8 bytes per token on the validation text' },
      { label: 'Train B/tok', dir: 1, get: (r) => r.train_bytes_per_token, fmt: (v) => fmt(v) },
      { label: '|V|', dir: 0, get: (r) => r.vocab_size, fmt: int },
      { label: 'Params', dir: 0, get: (r) => r.params, fmt: (v) => (isNum(v) ? `${(v / 1e6).toFixed(1)} M` : DASH) },
      { label: 'ms/step', dir: -1, get: (r) => r.ms_per_step, fmt: (v) => fmt(v, 0) },
      { label: 'tok/s', dir: 1, get: (r) => r.tokens_per_s, fmt: int },
    ], rows, 1);

    const cav = [];
    if (g) cav.push(`Early training: ${esc(a.num_iterations)} steps leave GPT-2 at ${g.final_nats_per_token.toFixed(2)} nats per token on validation, far from the speedrun target of ${esc(meta.target?.nats_per_gpt2_token)}. The ranking need not hold for longer training. One run per tokeniser (llm.c’s fixed seed), so there is no seed spread.`);
    const ws = rows.filter((r) => r.spec?.kind === 'tokenisers' && r.key !== 'super_bpe');
    if (g && ws.length) {
      const less = ws.map((r) => 100 * (1 - r.train_bytes_per_token / g.train_bytes_per_token));
      cav.push(`Our ${ws.length} whitespace-isolating tokenisers give ${range(ws.map((r) => r.val_bytes_per_token))} validation bytes per token against GPT-2’s ${g.val_bytes_per_token.toFixed(2)}, so each step holds ${range(less, 0)}% less text. Their gap to GPT-2 mixes less text with different merges; the space-prefix rerun below separates the two.`);
    }
    cav.push('TokenMonster runs as its closest lossless variant (see provenance).');
    setHtml('swap-caveats', cav.map((c) => `<li>${c}</li>`).join(''));
    const notes = (e.notes || []).map((l) => `<li>${md(l)}</li>`).join('');
    setHtml('swap-prov', `${notes ? `<details class="fold"><summary>Method notes generated from the result file</summary><ul class="caveats">${notes}</ul></details>` : ''}${provHtml(e, 'Provenance, original run')}`);

    const ours = rows.filter((r) => r.spec?.kind === 'tokenisers');
    const best = comparableSwap();
    const bo = ranked(Object.fromEntries(Object.entries(best).filter(([, m]) => m.spec?.kind === 'tokenisers')), (m) => m.final_bpb, -1);
    if (g && ours.length) setHtml('swap-finding', `After ${esc(a.num_iterations)} steps, GPT-2’s own tokeniser has the lowest validation BPB in the original run (${g.final_bpb.toFixed(4)}). The closest of ours there is ${code(ours[0].key)} at ${ours[0].final_bpb.toFixed(4)} (${signed(ours[0].final_bpb - g.final_bpb, 4)}).${bo.length && bo[0][0] !== ours[0].key ? ` Over all comparable reruns, the lowest of ours is ${code(bo[0][0])} at ${bo[0][1].toFixed(4)} (${signed(bo[0][1] - g.final_bpb, 4)}).` : ''}`);
    renderSwapCurves();
  }
  function renderSwapCurves() {
    const e = R('swap');
    if (!e.available) return;
    const M = e.data.models || {};
    const train = e.train_bpb;
    const mode = state.swapMode === 'train' && train ? 'train' : 'val';
    const order = Object.keys(M).filter((k) => isNum(M[k].final_bpb) && (mode === 'val' ? M[k].curve?.length : train.bpb[k])).sort((x, y) => M[x].final_bpb - M[y].final_bpb);
    const series = order.map((k) => ({ key: k, c: swapColor(k), dash: swapDash(k), pts: mode === 'val' ? M[k].curve.map(([x, y]) => [x, y]) : train.steps.map((x, i) => [x, train.bpb[k][i]]) }));
    // The untrained start (about log2 V / bytes per token) would flatten the rest: scale to step 100 onwards
    const late = series.flatMap((s) => s.pts.filter((p) => p[0] >= 100).map((p) => p[1]));
    const every = e.data.meta?.recipe?.args?.val_loss_every;
    setHtml('swap-curve-sub', mode === 'val'
      ? `Validation BPB at every evaluation (every ${esc(every ?? '?')} steps). The scale covers step 100 onwards; the untrained step-0 values lie above it.`
      : `Train BPB per step, centred ${esc(train.window)}-step rolling mean. GPT-2’s curve is the one the GPT-2-equivalent steps are read from.`);
    lineChart('swap-curves', {
      label: 'BPB against training step per tokeniser', dots: mode === 'val', xDomain: [0, Math.max(...series.flatMap((s) => s.pts.map((p) => p[0])))], xTitle: 'Training step', yTitle: `${mode === 'val' ? 'Validation' : 'Train'} BPB`,
      yDomain: [Math.floor(Math.min(...late) * 20) / 20, Math.ceil(Math.max(...late) * 20) / 20], series,
      tipFn: (s, p) => `<b class="mono">${esc(s.key)}</b><br>${p[1].toFixed(4)} BPB at step ${p[0]}`,
    });
  }
  const notComparable = (e) => ((e.not_comparable || []).length ? `<span class="bad">Not comparable with the original run:</span> ${esc(e.not_comparable.join('; '))}.` : null);
  function renderSpacePrefix() {
    const e = R('swap_space_prefix');
    if (!e.available) return setHtml('sp-body', notRun(e));
    const p = e.pairs;
    if (!p || !p.rows.length) return setHtml('sp-body', `<p class="note">The file is present, but it could not be paired with the original run${e.error ? ` (${esc(e.error)})` : ''}.</p>${provHtml(e)}`);
    const g = p.gpt2;
    const rows = p.rows.map((r) => ({ key: r.variant, ...r })).sort((x, y) => (x.sp.final_bpb ?? 9) - (y.sp.final_bpb ?? 9));
    const deltas = rows.map((r) => r.delta_bpb).filter(isNum);
    const facts = [notComparable(e)].filter(Boolean);
    if (deltas.length) facts.push(`${deltas.filter((d) => d < 0).length} of ${deltas.length} tokenisers have a lower final BPB with the space prefix; the median change is ${signed(median(deltas), 4)} BPB.`);
    facts.push(`Validation bytes per token: ${range(rows.map((r) => r.old.val_bytes_per_token).filter(isNum))} before, ${range(rows.map((r) => r.sp.val_bytes_per_token).filter(isNum))} with the space prefix${g ? `; GPT-2 is at ${txt(g.val_bytes_per_token, 2)}` : ''}.`);
    if (rows[0] && g) facts.push(`Lowest with the space prefix: ${code(rows[0].key)} at ${txt(rows[0].sp.final_bpb, 4)}, ${signed(rows[0].sp.final_bpb - g.final_bpb, 4)} against GPT-2’s ${txt(g.final_bpb, 4)}.`);
    setHtml('sp-body', `<ul class="caveats">${facts.map((f) => `<li>${f}</li>`).join('')}</ul>
      <figure><figcaption><b>Final validation BPB, original run and space-prefix rerun.</b> Lower is better.</figcaption><div id="sp-chart" class="chart"></div></figure>
      <details class="fold"><summary>Table: both runs side by side</summary><div class="table-wrap"><table class="data" id="sp-table"></table></div></details>${provHtml(e, 'Provenance, space-prefix rerun')}`);
    rowChart('sp-chart', {
      link: true, label: 'Final validation BPB in the original run and the space-prefix rerun', title: 'Final validation BPB', ref: g ? { v: g.final_bpb, label: 'gpt2' } : null,
      series: [{ label: 'Original (whitespace)', c: 'var(--fam-0)', shape: 'di' }, { label: 'Space prefix', c: 'var(--fam-1)', shape: '' }],
      rows: rows.map((r) => ({ key: r.key, marks: [{ v: r.old.final_bpb, s: 0 }, { v: r.sp.final_bpb, s: 1 }], val: signed(r.delta_bpb, 3), tip: `<b class="mono">${esc(r.key)}</b><br>Original: ${txt(r.old.final_bpb, 4)}<br>Space prefix: ${txt(r.sp.final_bpb, 4)}<br>Change: ${signed(r.delta_bpb, 4)} BPB` })),
    });
    table('sp-table', [
      NAME_COL,
      { label: 'Val B/tok, orig.', dir: 1, get: (r) => r.old.val_bytes_per_token, fmt: (v) => fmt(v) },
      { label: 'Val B/tok, SP', dir: 1, get: (r) => r.sp.val_bytes_per_token, fmt: (v) => fmt(v) },
      { label: 'BPB, orig.', dir: -1, get: (r) => r.old.final_bpb, fmt: (v) => fmt(v, 4) },
      { label: 'BPB, SP', dir: -1, get: (r) => r.sp.final_bpb, fmt: (v) => fmt(v, 4) },
      { label: 'Δ BPB', dir: -1, get: (r) => r.delta_bpb, fmt: (v) => `<span class="${v <= 0 ? 'good' : 'bad'}">${signed(v, 4)}</span>` },
      { label: 'vs GPT-2', dir: -1, get: (r) => (g && isNum(r.sp.final_bpb) ? r.sp.final_bpb - g.final_bpb : null), fmt: (v) => signed(v, 4), tip: 'Space-prefix final BPB minus GPT-2’s' },
      { label: 'GPT-2-eq. steps, orig.', dir: 1, get: (r) => r.old.gpt2_equivalent_steps, fmt: (v) => (isNum(v) ? String(v) : DASH) },
      { label: 'GPT-2-eq. steps, SP', dir: 1, get: (r) => r.sp.gpt2_equivalent_steps, fmt: (v) => (isNum(v) ? String(v) : DASH) },
    ], rows, 4);
  }
  // [column label, short label, shape, colour, the arm it changes one thing of]
  const ARMS = {
    whitespace: ['whitespace, 50 MB', 'WS', 'di', 'var(--fam-0)'],
    space_prefix: ['space prefix, 50 MB', 'SP', '', 'var(--fam-1)', 'whitespace'],
    gpt2_split: ['GPT-2 regex split, 50 MB', 'G2', 'sq', 'var(--fam-2)', 'space_prefix'],
    space_prefix_300mb: ['space prefix, 300 MB', 'SP300', 'ho', 'var(--fam-3)', 'space_prefix'],
  };
  function renderAblation() {
    const e = R('swap_ablation');
    if (!e.available) return setHtml('abl-body', notRun(e));
    const a = e.ablation;
    if (!a || !a.rows.length) return setHtml('abl-body', `<p class="note">The file is present, but its arms could not be assembled${e.error ? ` (${esc(e.error)})` : ''}.</p>${provHtml(e)}`);
    const g = a.gpt2;
    const rows = a.rows.map((r) => ({ key: r.variant, ...r }));
    const bpb = (r, arm) => r[arm]?.final_bpb;
    const label = (arm) => ARMS[arm]?.[0] ?? arm;
    const short = (arm) => ARMS[arm]?.[1] ?? arm;
    const facts = [notComparable(e)].filter(Boolean);
    // Each arm against the arm it changes one thing of, so one arm's effect is not credited to another
    a.arms.forEach((arm) => {
      const base = ARMS[arm]?.[4];
      const d = rows.map((r) => (base && isNum(bpb(r, arm)) && isNum(bpb(r, base)) ? bpb(r, arm) - bpb(r, base) : null)).filter(isNum);
      if (d.length) facts.push(`${esc(label(arm))} against ${esc(label(base))}: lower final BPB for ${d.filter((x) => x < 0).length} of ${d.length} variants; median change ${signed(median(d), 4)} BPB.`);
    });
    const full = rows.filter((r) => a.arms.every((arm) => isNum(bpb(r, arm))));
    if (full.length) {
      const wins = Object.fromEntries(a.arms.map((arm) => [arm, 0]));
      full.forEach((r) => { wins[a.arms.reduce((x, y) => (bpb(r, y) < bpb(r, x) ? y : x))] += 1; });
      facts.push(`Lowest final BPB over the ${full.length} variants run in every arm: ${a.arms.map((arm) => `${esc(label(arm))} ${wins[arm]}`).join(', ')}.`);
    }
    setHtml('abl-body', `<ul class="caveats">${facts.map((f) => `<li>${f}</li>`).join('')}</ul>
      <figure><figcaption><b>Final validation BPB in each arm.</b> Lower is better.</figcaption><div id="abl-chart" class="chart"></div></figure>
      <details class="fold"><summary>Table: BPB and bytes per token in every arm</summary><div class="table-wrap"><table class="data" id="abl-table"></table></div></details>${provHtml(e, 'Provenance, ablation')}`);
    rowChart('abl-chart', {
      link: true, label: 'Final validation BPB per variant in each ablation arm', title: 'Final validation BPB', ref: g ? { v: g.final_bpb, label: 'gpt2' } : null,
      series: a.arms.map((arm) => ({ label: label(arm), c: ARMS[arm]?.[3], shape: ARMS[arm]?.[2] })),
      rows: full.concat(rows.filter((r) => !full.includes(r))).sort((x, y) => Math.min(...a.arms.map((arm) => bpb(x, arm) ?? 9)) - Math.min(...a.arms.map((arm) => bpb(y, arm) ?? 9))).map((r) => ({
        key: r.key, marks: a.arms.map((arm, i) => ({ v: bpb(r, arm), s: i })), val: txt(Math.min(...a.arms.map((arm) => bpb(r, arm) ?? Infinity)), 3),
        tip: `<b class="mono">${esc(r.key)}</b><br>${a.arms.map((arm) => `${esc(label(arm))}: ${txt(bpb(r, arm), 4)}`).join('<br>')}`,
      })),
    });
    table('abl-table', [
      NAME_COL,
      ...a.arms.map((arm) => ({ label: `BPB, ${short(arm)}`, dir: -1, get: (r) => bpb(r, arm), fmt: (v) => fmt(v, 4), tip: label(arm) })),
      ...a.arms.map((arm) => ({ label: `Val B/tok, ${short(arm)}`, dir: 1, get: (r) => r[arm]?.val_bytes_per_token, fmt: (v) => fmt(v), tip: label(arm) })),
    ], rows, 1);
  }
  function renderLr() {
    const e = R('swap_lr1p8e-3');
    const base = R('swap');
    if (!e.available) return setHtml('lr-body', notRun(e));
    const tr = e.data.trainer || {};
    const args = base.data?.meta?.recipe?.args || {};
    const ref = Object.fromEntries((base.data?.models?.gpt2?.curve || []).map(([s, b]) => [s, b]));
    const rows = (tr.evals || []).map((ev) => ({ step: ev.step, hi: ev.val_bpb, lo: ref[ev.step] }));
    const cmp = rows.filter((r) => r.step > 0 && isNum(r.lo));
    const last = rows[rows.length - 1];
    const peak = isNum(tr.peak_lr) ? tr.peak_lr.toExponential(1) : '?';
    const facts = [];
    if (cmp.length) facts.push(`With a peak learning rate of ${peak} in place of ${esc(args.learning_rate ?? '?')}, GPT-2’s validation BPB is higher at ${cmp.filter((r) => r.hi > r.lo).length} of ${cmp.length} evaluations after step 0. Final: ${txt(last.hi, 4)} against ${txt(last.lo, 4)} (${signed(last.hi - last.lo, 4)}).`);
    if (isNum(tr.max_norm_after_warmup)) facts.push(`Maximum gradient norm after the ${tr.warmup}-step warm-up: ${tr.max_norm_after_warmup.toFixed(2)}.`);
    const body = rows.map((r) => `<tr><td class="num">${esc(r.step)}</td><td class="r num">${fmt(r.lo, 4)}</td><td class="r num">${fmt(r.hi, 4)}</td><td class="r num"><span class="${r.hi - r.lo > 0 ? 'bad' : 'good'}">${isNum(r.lo) ? signed(r.hi - r.lo, 4) : '–'}</span></td></tr>`).join('');
    setHtml('lr-body', `<ul class="caveats">${facts.map((f) => `<li>${f}</li>`).join('')}</ul>
      <div class="table-wrap narrow"><table><thead><tr><th>Step</th><th class="r">Val BPB, lr ${esc(args.learning_rate ?? '?')} ↓</th><th class="r">Val BPB, lr ${peak} ↓</th><th class="r">Δ</th></tr></thead><tbody>${body}</tbody></table></div>
      <details class="prov"><summary>Provenance, learning-rate run: ${esc(`${tr.gpu ?? '?'} · torch ${tr.torch ?? '?'} · ${tr.n_steps} steps · one run`)}</summary><ul><li>file ${code(e.path)}; the lr ${esc(args.learning_rate ?? '?')} column is GPT-2 in ${code(base.path)}</li><li>${int(e.data.train?.tokens_per_step)} tokens per step; validation ${int(e.data.val?.docs)} documents, ${e.data.val?.bytes ? mb(e.data.val.bytes) : '?'}</li></ul></details>`);
  }

  // ---------- 11 FLORES+ ----------
  function renderFlores() {
    const e = R('flores');
    setHtml('flores-prov', provHtml(e));
    if (!e.available) return;
    const meta = e.data.meta || {};
    const L = meta.languages || {};
    const inMix = Object.values(L.in_mix || {});
    const outMix = L.out_of_mix || [];
    const thr = meta.parity_threshold ?? 1.5;
    const [tier, vocab] = state.flores.split('@');
    const rows = Object.entries(e.data.models || {})
      .filter(([, m]) => m.tier === tier && (!vocab || m.vocab_size === +vocab))
      .map(([key, m]) => ({ key, ...m, inMix: m.aggregates?.in_mix || {}, outMix: m.aggregates?.out_of_mix || {}, en: m.languages?.[L.english]?.tokens_per_sentence }))
      .sort((a, b) => (a.inMix.mean_premium ?? Infinity) - (b.inMix.mean_premium ?? Infinity));
    const shown = (k) => (tier === 'intrinsic' ? baseOf(k) : k);
    setHtml('flores-sub', `${rows.length} tokenisers, ${int(meta.flores?.sentences)} parallel sentences per language.`);
    rowChart('flores-chart', {
      link: true, zero: true, label: 'Mean tokenisation premium over in-mix and out-of-mix languages', title: 'Mean premium (tokens ÷ English tokens)', ref: { v: 1, label: 'parity with English' },
      series: [{ label: `In-mix (${inMix.length} languages)`, c: 'var(--fam-1)', shape: '' }, { label: `Out-of-mix (${outMix.length} languages)`, c: 'var(--fam-2)', shape: 'sq' }],
      rows: rows.map((r) => ({ key: r.key, shown: shown(r.key), marks: [{ v: r.inMix.mean_premium, s: 0 }, { v: r.outMix.mean_premium, s: 1 }], val: txt(r.inMix.mean_premium, 2), tip: `<b class="mono">${esc(r.key)}</b><br>In-mix mean: ${txt(r.inMix.mean_premium, 2)}<br>Out-of-mix mean: ${txt(r.outMix.mean_premium, 2)}<br>${txt(r.en, 1)} tokens per English sentence` })),
    });

    const T = meta.text || {};
    const ratio = (l) => (T[l]?.bytes && T[L.english]?.bytes ? T[l].bytes / T[L.english].bytes : null);
    const groupRatio = (ls) => mean(ls.map(ratio).filter(isNum));
    const short = (l) => l.split('_')[0];
    const cols = [
      ...inMix.map((l) => ({ label: short(l), full: l, get: (r) => r.languages?.[l]?.premium, bytes: ratio(l) })),
      { label: 'In-mix mean', avg: true, get: (r) => r.inMix.mean_premium, bytes: groupRatio(inMix) },
      ...outMix.map((l) => ({ label: short(l), full: l, get: (r) => r.languages?.[l]?.premium, bytes: ratio(l) })),
      { label: 'Out-of-mix mean', avg: true, get: (r) => r.outMix.mean_premium, bytes: groupRatio(outMix) },
    ];
    const sorted = cols.map((c) => rows.map(c.get).filter(isNum).sort((x, y) => x - y));
    const en = T[L.english];
    setHtml('flores-heat', `<thead><tr><th></th><th></th><th class="grp" colspan="${inMix.length + 1}">In-mix</th><th class="grp" colspan="${outMix.length + 1}">Out-of-mix</th></tr>
      <tr><th>Tokeniser</th><th data-tip="Tokens per English sentence: the premium’s denominator">EN tok/sent</th>${cols.map((c) => `<th class="${c.avg ? 'avg' : ''}"${c.full ? ` data-tip="${esc(c.full)}"` : ''}>${esc(c.label)}</th>`).join('')}</tr></thead>
      <tbody><tr><td class="muted">UTF-8 bytes</td><td class="cell muted">${txt(en ? en.bytes / en.sentences : null, 1)}</td>${cols.map((c) => `<td class="cell muted${c.avg ? ' avg' : ''}">${txt(c.bytes, 2)}</td>`).join('')}</tr>
      ${rows.map((r) => `<tr><td>${nameHtml(r.key, shown(r.key))}</td><td class="cell">${txt(r.en, 1)}</td>${cols.map((c, i) => `<td class="cell${c.avg ? ' avg' : ''}" style="${rankShade(sorted[i], c.get(r), -1)}">${txt(c.get(r), 2)}</td>`).join('')}</tr>`).join('')}</tbody>`);

    table('flores-table', [
      { label: 'Tokeniser', text: true, get: (r) => r.key, fmt: (k) => nameHtml(k, shown(k)) },
      { label: '|V|', dir: 0, get: (r) => r.vocab_size, fmt: int },
      { label: 'EN tok/sent', dir: 0, get: (r) => r.en, fmt: (v) => fmt(v, 1), tip: 'Tokens per English sentence: the premium is relative to it' },
      { label: 'In-mix mean', dir: -1, get: (r) => r.inMix.mean_premium, fmt: (v) => fmt(v, 2) },
      { label: 'In-mix max', dir: -1, get: (r) => r.inMix.max_premium, fmt: (v, r) => `${fmt(v, 2)} <span class="sd">${esc(r.inMix.worst_language ?? '')}</span>` },
      { label: `In-mix ≤ ${thr}`, dir: 1, get: (r) => r.inMix.share_premium_le_1_5, fmt: pct },
      { label: 'Out-of-mix mean', dir: -1, get: (r) => r.outMix.mean_premium, fmt: (v) => fmt(v, 2) },
      { label: 'Out-of-mix max', dir: -1, get: (r) => r.outMix.max_premium, fmt: (v, r) => `${fmt(v, 2)} <span class="sd">${esc(r.outMix.worst_language ?? '')}</span>` },
      { label: `Out-of-mix ≤ ${thr}`, dir: 1, get: (r) => r.outMix.share_premium_le_1_5, fmt: pct },
      { label: 'Byte fallback', dir: -1, get: (r) => r.outMix.byte_fallback_rate, fmt: pct, tip: 'Share of out-of-mix tokens that are one byte of a multi-byte character' },
    ], rows, 3);
    setHtml('flores-notes', (e.notes || []).map((l) => `<li>${md(l.replace('(column "UTF-8 bytes")', '(the “UTF-8 bytes” row of the table above)'))}</li>`).join(''));

    const ins = rows.map((r) => r.inMix.mean_premium).filter(isNum);
    const outs = rows.map((r) => r.outMix.mean_premium).filter(isNum);
    const best = rows.find((r) => isNum(r.inMix.mean_premium) && isNum(r.en));
    const med = median(rows.map((r) => r.en).filter(isNum));
    if (ins.length) setHtml('flores-finding', `In this group the mean in-mix premium runs from ${range(ins)} and the mean out-of-mix premium from ${range(outs)}. Languages absent from the training text cost ${(median(outs) / median(ins)).toFixed(1)}× the in-mix premium at the median.`);
    if (best && isNum(med)) setHtml('flores-caveat', `The premium is relative to each tokeniser’s own English cost. Here the lowest in-mix premium, ${code(best.key)} (${txt(best.inMix.mean_premium, 2)}), comes with ${txt(best.en, 1)} tokens per English sentence against the group’s median of ${txt(med, 1)}. A tokeniser that spends many tokens on English gets a low premium without being cheap on the other language, so read the premium next to the English tokens per sentence.`);
  }

  // ---------- 13 playground ----------
  const utf8 = new TextEncoder();
  const lossy = new TextDecoder('utf-8', { fatal: false });
  function renderPlayground() {
    const P = D.playground;
    if (!P) return setHtml('pg-list', '<p class="note">No playground data in data.js. Rebuild without --skip-playground.</p>');
    const sample = P.samples.find((s) => s.id === state.pgSample) || P.samples[0];
    const run = P.runs[state.pgVocab]?.[sample.id] || {};
    const bytes = utf8.encode(sample.text);
    $('pg-text').textContent = sample.text;
    const rows = Object.entries(run).map(([key, toks]) => ({ key, toks, n: toks.length })).sort((a, b) => a.n - b.n);
    rowChart('pg-chart', { mode: 'bar', label: 'Number of tokens per tokeniser for this sample', title: `Tokens for ${bytes.length} bytes`, rows: rows.map((r) => ({ key: r.key, marks: [{ v: r.n }], val: String(r.n) })) });
    const vis = (s) => esc(s).replace(/ /g, '<span class="ws">·</span>').replace(/\n/g, '<span class="ws">↵</span>\n');
    setHtml('pg-list', rows.map((r) => {
      let off = 0;
      const chips = r.toks.map(([id, len]) => {
        const piece = lossy.decode(bytes.subarray(off, off + len));
        off += len;
        return `<span data-tip="${esc(`id ${id} · ${len} byte${len === 1 ? '' : 's'}`)}">${vis(piece)}</span>`;
      }).join('');
      return `<div class="tok-row" style="--c:${famVar(r.key)}"><div class="tok-head">${nameHtml(r.key)}<span>${esc(dispName(r.key))}</span><span class="num">${r.n} tokens</span><span class="num">${(bytes.length / r.n).toFixed(2)} bytes/token</span></div><div class="toks">${chips}</div></div>`;
    }).join(''));
  }
  function initPlayground() {
    const P = D.playground;
    if (!P) return;
    state.pgSample = P.samples[0].id;
    state.pgVocab = String(P.vocabs[Math.min(1, P.vocabs.length - 1)]);
    setHtml('pg-sample', P.samples.map((s) => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join(''));
    setHtml('pg-vocab', P.vocabs.map((v) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === state.pgVocab}">V = ${vLabel(v)}</button>`).join(''));
    const c = P.corpus || {};
    setHtml('pg-note', `These tokenisers are trained on ${isNum(c.train_bytes) ? `${Math.round(c.train_bytes / 1000)} kB` : 'a small sample'} of the training text${c.synthetic ? ' (SYNTHETIC offline corpus)' : ''} at very small vocabularies, so the differences between scores show within one sentence. They are separate from the benchmark tokenisers. Here ${code('morpheme_bpe')} uses λ = 2.5 with a lexicon of ${int(c.lexicon_words)} words, and ${code('super_bpe')} switches at 0.75 V. A token is a byte string and can end inside a multi-byte character; such a piece shows as �. A space shows as · and a line break as ↵.`);
    sel('pg-sample', 'pgSample', renderPlayground);
    seg('pg-vocab', 'pgVocab', renderPlayground);
  }

  // ---------- navigation, theme, start ----------
  function initNav() {
    const sections = [...document.querySelectorAll('main section[data-nav]')];
    setHtml('toc', sections.map((s) => `<li><a href="#${s.id}">${esc(s.dataset.nav)}</a></li>`).join(''));
    setHtml('jump', sections.map((s, i) => `<option value="${s.id}">${String(i + 1).padStart(2, '0')}  ${esc(s.dataset.nav)}</option>`).join(''));
    const jump = $('jump');
    jump.addEventListener('change', () => { location.hash = jump.value; });
    const links = Object.fromEntries([...document.querySelectorAll('#toc a')].map((a) => [a.hash.slice(1), a]));
    const mark = (id) => {
      Object.entries(links).forEach(([k, a]) => (k === id ? a.setAttribute('aria-current', 'true') : a.removeAttribute('aria-current')));
      jump.value = id;
    };
    // The current section is the last one whose top has passed a line just under the top bar
    const onScroll = () => {
      const line = 120;
      let cur = sections[0];
      for (const s of sections) if (s.getBoundingClientRect().top <= line) cur = s;
      if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) cur = sections[sections.length - 1];
      mark(cur.id);
    };
    let queued = false;
    addEventListener('scroll', () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; onScroll(); }); }, { passive: true });
    onScroll();
  }
  $('theme').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('lernea-theme', next); } catch { /* private mode: the choice lasts for this page view */ }
  });

  function init() {
    document.querySelectorAll('.vocab-select').forEach((s) => {
      s.innerHTML = VOCABS.map((v) => `<option value="${v}">${vLabel(v)}</option>`).join('');
      s.addEventListener('change', () => {
        state.vocab = +s.value;
        document.querySelectorAll('.vocab-select').forEach((x) => { x.value = s.value; });
        safe([renderTokka, renderIntrinsic, renderThroughput]);
      });
    });
    const opts = CHARTABLE.map((c) => `<option value="${c.key}">${esc(c.long)} ${c.dir === 1 ? '↑' : '↓'}</option>`).join('');
    setHtml('intrinsic-metric', opts);
    setHtml('vocab-metric', opts);
    setHtml('flores-tier', VOCABS.map((v) => `<option value="intrinsic@${v}">Our variants, V = ${vLabel(v)}</option>`).join('') + '<option value="swap">GPT-2-swap tokenisers, V ≈ 50k</option>');
    sel('tokka-metric', 'tokkaMetric', renderTokka);
    sel('intrinsic-metric', 'intrinsicMetric', renderIntrinsic);
    sel('vocab-metric', 'vocabMetric', renderVocab);
    sel('flores-tier', 'flores', renderFlores);
    seg('tp-dir', 'tpMetric', renderThroughput);
    seg('proxy-tier', 'tier', renderProxy);
    seg('swap-mode', 'swapMode', renderSwapCurves);
    initPlayground();
    initNav();
    safe([renderAbout, renderVariants, renderOverall, renderTokka, renderIntrinsic, renderThroughput, renderProxy, renderVocab, renderSwap, renderFlores, renderPlayground]);
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
    }
    // A link to a section that only now has its full height
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }
  init();
})();
