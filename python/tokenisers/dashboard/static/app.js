/**
 * Tokenisers Lab — benchmark dashboard
 */

// Global Application State
const state = {
  data: null,
  activeTab: 'summary',
  activeFamilyFilter: 'ALL',
  sortColumn: 'macro_val_bpb',
  sortAscending: true,
  tokkaActiveMetric: 'bytes_per_token',
  vocab: 16384, // shared by the compression, intrinsic and throughput tabs
  tier: 'proxy_small',
  tpMetric: 'encode_mb_per_sec',
  swapMode: 'val',
  tables: {},
  selectedRadarModels: ['bpe_frequency', 'wp_bpe_ft', 'wp_bpe_pig', 'unigram_lm'],
  radarSlots: {},
  charts: {
    quadrant: null,
    extrinsicBar: null,
    extrinsicCurve: null,
    tokkaBar: null,
    radar: null,
    throughput: null,
    swap: null,
  },
  theme: readStoredTheme(),
};

const TABS = ['summary', 'overview', 'tokka', 'intrinsic', 'throughput', 'proxy', 'swap', 'concepts', 'playground'];
const VOCABS = [16384, 32768, 65536];
// Tokenisers in the GPT-2 swap that are not ours (no entry in the API's metadata)
const EXTERNAL = {
  gpt2: 'GPT-2 (tiktoken)',
  tm_englishcode_50256: 'TokenMonster englishcode 50,256',
  tm_englishcode_32000: 'TokenMonster englishcode 32,000',
};

// Families map to fixed categorical slots (--fam-1 … --fam-8 in index.html)
const FAMILIES = [
  'Information-Theoretic',
  'Frequency BPE',
  'EM / Probabilistic',
  'Structural / Multi-Stage',
  'Morphology / Guided',
  'Regularized BPE',
  'Information-Theoretic (Improved)',
  'Custom',
];
// Secondary encoding so families stay distinguishable without colour
// Only filled styles: line-only ones (star, cross) vanish behind the surface ring
const FAMILY_SHAPES = [['circle', 0], ['rect', 0], ['triangle', 0], ['rectRot', 0], ['triangle', 180], ['rectRounded', 45], ['rectRot', 0], ['circle', 0]];
const DASHES = [[], [6, 3], [2, 3], [10, 3, 2, 3]];

function familySlot(family) {
  const i = FAMILIES.indexOf(family);
  return i === -1 ? FAMILIES.length : i + 1;
}

function familyVar(family) {
  return `var(--fam-${familySlot(family)})`;
}

function getFamilyColor(family) {
  const hex = cssVar(`--fam-${familySlot(family)}`);
  const [shape, rotation] = FAMILY_SHAPES[familySlot(family) - 1];
  return { border: hex, bg: withAlpha(hex, 0.75), shape, rotation };
}

// Preset Texts for Live Playground
const PRESETS = {
  morphology: 'The internationalization of unhappily disproportionate misunderstandings requires robust subword tokenization.',
  code: `pub fn compute_pmi(freq_uv: u64, freq_u: u64, freq_v: u64, total: u64) -> f64 {
    let p_uv = (freq_uv as f64) / (total as f64);
    let p_u = (freq_u as f64) / (total as f64);
    let p_v = (freq_v as f64) / (total as f64);
    (p_uv / (p_u * p_v)).log2()
}`,
  multilingual: 'Tokenizers in Deutsch: Unabhängigkeitserklärung; Français: internationalisation; 汉语: 自然语言处理; 日本語: 機械学習モデル; Русский: сверхпроводимость; العربية: معالجة اللغات الطبيعية.',
  agglutinative: 'megszentségteleníthetetlenségeskedéseitekért epäjärjestelmällistyttämättömyydelläänsäkäänkö unconstitutionality hypercholesterolemia',
};

// Lifecycle Initialization
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  fetchBenchmarkData();
  setupEventListeners();
  loadPlaygroundPreset('morphology');
});

// Helpers
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function isNum(v) {
  return typeof v === 'number' && !Number.isNaN(v);
}

const missing = (v) => v === null || v === undefined || Number.isNaN(v);
const DASH = '<span class="muted">–</span>';

function fmt(v, digits = 3) {
  return isNum(v) ? v.toFixed(digits) : DASH;
}

function pct(v) {
  return isNum(v) ? (v * 100).toFixed(1) + '%' : DASH;
}

function signed(v, digits = 3) {
  return isNum(v) ? `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}` : DASH;
}

const int = (v) => (isNum(v) ? Math.round(v).toLocaleString('en-US') : DASH);
const mean = (a) => (Array.isArray(a) && a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
const spread = (a) => (Array.isArray(a) && a.length > 1 ? Math.max(...a) / Math.min(...a) : null);
const range = (a, d = 2, unit = '') => (a.length ? `${Math.min(...a).toFixed(d)}${unit}–${Math.max(...a).toFixed(d)}${unit}` : '–');

function median(vals) {
  const s = [...vals].sort((a, b) => a - b);
  const n = s.length;
  return n === 0 ? null : n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

const mb = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;

// One line of provenance from a result file's meta block (legacy files have none)
function provenance(m) {
  if (!m) return 'no provenance recorded';
  const b = m.backend || {};
  const parts = [];
  if (m.synthetic_data) parts.push('SYNTHETIC data');
  const hw = b.gpu ? b.gpu.split(',')[0] : b.kind === 'colab' ? `Colab CPU (${b.cpu ?? '?'})` : b.cpu || m.host?.cpu;
  if (hw) parts.push(hw);
  if (isNum(m.byte_budget)) parts.push(`${mb(m.byte_budget)} LM budget`);
  else if (isNum(m.train_bytes)) parts.push(`${mb(m.train_bytes)} train`);
  else if (isNum(m.prepared?.train_bytes)) parts.push(`${mb(m.prepared.train_bytes)} train`);
  if (m.recipe?.args?.num_iterations) parts.push(`${m.recipe.args.num_iterations} steps`);
  if (Array.isArray(m.seeds)) parts.push(`${m.seeds.length} seed${m.seeds.length === 1 ? '' : 's'}`);
  if (isNum(m.passes)) parts.push(`${m.passes} passes`);
  if (m.eval_split) parts.push(`${m.eval_split} split`);
  if (m.git_commit) parts.push(`@${m.git_commit.slice(0, 7)}${m.git_dirty ? '+dirty' : ''}`);
  return parts.join(' · ');
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function famLabel(family) {
  return `<span class="fam"><span class="dot" style="--c: ${familyVar(family)}"></span>${esc(family)}</span>`;
}

const algoMeta = (k) => state.data?.metadata?.[k];
const dispName = (k) => algoMeta(k)?.display_name ?? EXTERNAL[k] ?? k;
const famOf = (k) => algoMeta(k)?.family ?? 'Custom';

function nameCell(k) {
  return `<span class="fam" style="color: var(--text); align-items: baseline"><span class="dot" style="--c: ${familyVar(famOf(k))}"></span><span><span class="long-name"><span class="model-name">${esc(dispName(k))}</span><br></span><span class="model-key">${esc(k)}</span></span></span>`;
}

function setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function showBody(id, on) {
  document.getElementById(id)?.classList.toggle('hidden', !on);
}

// A result set from /api/benchmarks: {available, path, data, provenance, notes, ...}
function R(key) {
  return state.data?.results?.[key] ?? { available: false, path: key, data: null, provenance: [], notes: [] };
}

// Inline Markdown of report.py's lines: `code` and **bold** only
function md(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
}

function notRun(e) {
  return `<div class="empty"><b>Not yet run.</b> No results committed: <code>${esc(e.path)}</code> is missing${e.error ? ` (${esc(e.error)})` : ''}. This section fills in when it is.</div>`;
}

function provHtml(e, label = 'Provenance') {
  if (!e.available) return notRun(e);
  const lines = (e.provenance || []).map((l) => `<li>${md(l)}</li>`).join('');
  return `<details class="prov"><summary>${esc(label)}: ${esc(provenance(e.data?.meta))}</summary><ul>${lines}<li>file <code>${esc(e.path)}</code></li></ul></details>`;
}

function destroyChart(name) {
  if (state.charts[name]) state.charts[name].destroy();
  state.charts[name] = null;
}

// Sortable, ranked table. cols: {label, get(row), fmt?(value, row) -> html, dir: 1 higher is better,
// -1 lower is better, 0 none; text: left-aligned; tip}. Rows sort best-first; missing values sink.
function table(id, cols, rows, sortCol) {
  const prev = state.tables[id];
  const keep = prev && prev.sort.i < cols.length && prev.cols.length === cols.length;
  state.tables[id] = { cols, rows, sort: keep ? prev.sort : { i: sortCol, best: true } };
  drawTable(id);
}

function sortTable(id, i) {
  const t = state.tables[id];
  t.sort = t.sort.i === i ? { i, best: !t.sort.best } : { i, best: true };
  drawTable(id);
}

function drawTable(id) {
  const el = document.getElementById(id);
  const t = state.tables[id];
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
  const best = cols.map((col) => {
    const vals = col.dir ? rows.map((r) => col.get(r)).filter(isNum) : [];
    return vals.length > 1 ? (col.dir === 1 ? Math.max(...vals) : Math.min(...vals)) : null;
  });
  const arrow = (d) => (d === 1 ? '<span class="dir" title="higher is better">↑</span>' : d === -1 ? '<span class="dir" title="lower is better">↓</span>' : '');
  const head = cols
    .map((col, i) => {
      const s = i === sort.i ? ` aria-sort="${sign < 0 ? 'descending' : 'ascending'}"` : '';
      const tip = col.tip ? ` title="${esc(col.tip)}"` : '';
      return `<th class="${col.text ? '' : 'r'}"${s}><button type="button" class="th-btn"${tip} onclick="sortTable('${id}', ${i})">${col.label}${arrow(col.dir)}</button></th>`;
    })
    .join('');
  const body = sorted
    .map((r, k) => {
      const cells = cols
        .map((col, i) => {
          const v = col.get(r);
          const cls = `${col.text ? '' : 'r num'}${best[i] !== null && v === best[i] ? ' best' : ''}`;
          return `<td class="${cls}">${col.fmt ? col.fmt(v, r) : missing(v) ? DASH : esc(v)}</td>`;
        })
        .join('');
      return `<tr><td class="rank num">${k + 1}</td>${cells}</tr>`;
    })
    .join('');
  el.innerHTML = `<thead><tr><th class="rank">#</th>${head}</tr></thead><tbody>${body}</tbody>`;
}

const NAME_COL = { label: 'Tokeniser', text: true, get: (r) => r.key, fmt: (k) => nameCell(k) };

// Theme Management
function readStoredTheme() {
  try {
    const t = localStorage.getItem('tokenisers-theme');
    return t === 'light' || t === 'dark' ? t : null;
  } catch {
    return null;
  }
}

function initTheme() {
  if (state.theme) document.documentElement.setAttribute('data-theme', state.theme);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!state.theme) renderCharts();
  });
}

function isDarkTheme() {
  return state.theme ? state.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function setTheme(themeName) {
  state.theme = themeName;
  try {
    localStorage.setItem('tokenisers-theme', themeName);
  } catch {}
  document.documentElement.setAttribute('data-theme', themeName);
  renderCharts();
}

// Shared Chart.js styling, read from the CSS tokens so both themes stay in sync
function chartTheme() {
  const t = {
    text: cssVar('--text'),
    text2: cssVar('--text-2'),
    muted: cssVar('--muted'),
    grid: cssVar('--grid'),
    surface: cssVar('--surface'),
    border: cssVar('--border-strong'),
  };
  Chart.defaults.font.family = cssVar('--sans');
  Chart.defaults.font.size = 11.5;
  Chart.defaults.color = t.muted;
  Chart.defaults.borderColor = t.grid;
  return t;
}

function tooltipStyle(t) {
  return {
    backgroundColor: t.surface,
    titleColor: t.text,
    bodyColor: t.text2,
    borderColor: t.border,
    borderWidth: 1,
    padding: 10,
    cornerRadius: 8,
    boxPadding: 4,
    usePointStyle: true,
    titleFont: { weight: '600' },
  };
}

function legendStyle(t, position = 'top') {
  return {
    position,
    align: 'start',
    labels: { color: t.text2, usePointStyle: true, pointStyleWidth: 8, boxHeight: 7, padding: 14, font: { size: 11.5 } },
  };
}

// Line-chart legend: short dashes so dash styles stay visible
function lineLegend(t) {
  return { ...legendStyle(t, 'bottom'), labels: { ...legendStyle(t).labels, usePointStyle: false, boxWidth: 14, boxHeight: 2, padding: 8, font: { size: 10.5 } } };
}

function axis(t, title, extra = {}) {
  return {
    grid: { color: t.grid, drawTicks: false },
    border: { display: false },
    ticks: { color: t.muted, padding: 6 },
    title: title ? { display: true, text: title, color: t.text2, font: { size: 11.5, weight: '500' } } : { display: false },
    ...extra,
  };
}

// 1-2-5 ticks only, so log axes don't crowd
function logAxis(t) {
  return { type: 'logarithmic', ticks: { color: t.muted, padding: 6, callback: (v) => ([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].includes(v) ? v : '') } };
}

// Draws dashed median guides on the quadrant chart
const medianLinesPlugin = {
  id: 'medianLines',
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || !isNum(opts.x) || !isNum(opts.y)) return;
    const { ctx, chartArea: a, scales } = chart;
    const x = scales.x.getPixelForValue(opts.x);
    const y = scales.y.getPixelForValue(opts.y);
    ctx.save();
    ctx.strokeStyle = opts.color;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, a.top);
    ctx.lineTo(x, a.bottom);
    ctx.moveTo(a.left, y);
    ctx.lineTo(a.right, y);
    ctx.stroke();
    ctx.restore();
  },
};

// Data Fetching
async function fetchBenchmarkData() {
  const refreshBtn = document.getElementById('refresh-btn');
  const errEl = document.getElementById('load-error');
  if (refreshBtn) refreshBtn.classList.add('spinning');

  try {
    const res = await fetch('/api/benchmarks');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();
    errEl?.classList.add('hidden');
    renderAllViews();
  } catch (err) {
    console.error('Failed to load benchmark data:', err);
    if (errEl) {
      errEl.textContent = `Could not load benchmark data: ${err.message}`;
      errEl.classList.remove('hidden');
    }
  } finally {
    if (refreshBtn) refreshBtn.classList.remove('spinning');
  }
}

// Master Render Function
function renderAllViews() {
  if (!state.data) return;

  renderMeta();
  renderSummary();
  renderLeaderboard();
  renderIntrinsicTab();
  renderSwap();
  renderConceptFacts();
  renderCharts();

  renderMath();
}

// KaTeX auto-render skips single-dollar inline math unless asked
function renderMath() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
}

// Everything that draws a chart (re-run on theme change so colours follow the tokens)
function renderCharts() {
  if (!state.data) return;
  renderQuadrantChart();
  renderProxy();
  renderTokkaTab();
  renderRadarView();
  renderThroughput();
  renderSwapChart();
}

function renderMeta() {
  const sets = Object.values(state.data.results || {});
  const sub = document.getElementById('brand-sub');
  if (sub) sub.textContent = `${sets.filter((e) => e.available).length} of ${sets.length} result sets · compression, intrinsic, throughput, proxy LM, GPT-2 swap`;
  const foot = document.getElementById('footer-updated');
  if (foot && state.data.timestamp) foot.textContent = `Data loaded ${new Date(state.data.timestamp).toLocaleString()}`;
}

// Tab Switching: ARIA tabs with roving tabindex, arrow keys, and #hash deep links
function switchTab(tabName, focus = false) {
  if (!TABS.includes(tabName)) return;
  state.activeTab = tabName;

  TABS.forEach((id) => {
    const on = id === tabName;
    const btn = document.getElementById(`tab-nav-${id}`);
    btn?.setAttribute('aria-selected', String(on));
    if (btn) btn.tabIndex = on ? 0 : -1;
    if (on && focus) btn?.focus();
    document.getElementById(`tab-${id}`)?.classList.toggle('hidden', !on);
  });
  if (location.hash.slice(1) !== tabName) history.replaceState(null, '', `#${tabName}`);

  // Charts rendered while hidden have zero size; resize once visible
  setTimeout(() => {
    Object.values(state.charts).forEach((c) => c && c.resize());
  }, 50);
}

function initTabs() {
  document.querySelectorAll('[role="tab"]').forEach((btn) => btn.addEventListener('click', () => switchTab(btn.id.replace('tab-nav-', ''))));
  document.querySelector('[role="tablist"]')?.addEventListener('keydown', (e) => {
    const i = TABS.indexOf(state.activeTab);
    const j = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 }[e.key];
    if (j === undefined) return;
    e.preventDefault();
    switchTab(TABS[(j + TABS.length) % TABS.length], true);
  });
  const fromHash = () => {
    const h = location.hash.slice(1);
    if (TABS.includes(h)) switchTab(h);
  };
  window.addEventListener('hashchange', fromHash);
  fromHash();
}

// 1. Summary: category leaders and the result-set index
const RESULT_LABELS = {
  ...Object.fromEntries(VOCABS.flatMap((v) => [
    [`tokka_${v}`, ['Compression (Tokka-Bench)', v, 'tokka']],
    [`intrinsic_${v}`, ['Intrinsic metrics', v, 'intrinsic']],
  ])),
  throughput_local: ['Throughput, pinned laptop core', 'all three', 'throughput'],
  throughput_colab_cpu: ['Throughput, Colab CPU', 'all three', 'throughput'],
  proxy_small: ['Proxy LM, small', 16384, 'proxy'],
  proxy_big: ['Proxy LM, big', 16384, 'proxy'],
  proxy_compare: ['Proxy LM, small vs big', 16384, 'proxy'],
  swap: ['GPT-2 tokeniser swap', '≈ 50k', 'swap'],
  'swap_lr1p8e-3': ['GPT-2 swap, peak lr 1.8e-3 (GPT-2 only)', 50257, 'swap'],
  swap_space_prefix: ['GPT-2 swap, space-prefix rerun', '≈ 50k', 'swap'],
};

// Best-first [key, value] pairs
function ranked(models, get, dir) {
  return Object.entries(models || {})
    .map(([k, m]) => [k, get(m)])
    .filter(([, v]) => isNum(v))
    .sort((a, b) => (b[1] - a[1]) * dir);
}

// Margin between the two leaders of a seeded proxy-LM run, in standard errors of the difference
function seedMargin(models, a, b) {
  const se = (m) => (m.macro_val_bpb_sd ?? 0) ** 2 / Object.keys(m.macro_val_bpb_by_seed || {}).length;
  const d = models[b].macro_val_bpb - models[a].macro_val_bpb;
  const z = d / Math.sqrt(se(models[a]) + se(models[b]));
  return isNum(z) && Number.isFinite(z) ? `Lead over the runner-up: ${z.toFixed(1)} standard errors over the seeds, ${z < 2 ? 'not resolved' : 'resolved'}.` : '';
}

function leaderSpecs() {
  const tp = R('throughput_local');
  const spreads = tp.available ? Object.values(tp.data.models?.['16384'] || {}).map((m) => spread(m.encode_mb_per_sec)).filter(isNum) : [];
  return [
    { set: 'proxy_small', eyebrow: 'Proxy LM · small', unit: 'BPB', dir: -1, d: 3, tab: 'proxy', get: (m) => m.macro_val_bpb, what: 'Macro val bits per byte, mean of the seeds', caveat: (r, M) => seedMargin(M, r[0][0], r[1][0]) },
    { set: 'proxy_big', eyebrow: 'Proxy LM · big', unit: 'BPB', dir: -1, d: 3, tab: 'proxy', get: (m) => m.macro_val_bpb, what: 'Macro val bits per byte, mean of the seeds', caveat: (r, M) => seedMargin(M, r[0][0], r[1][0]) },
    {
      set: 'swap', eyebrow: 'GPT-2 swap', unit: 'BPB', dir: -1, d: 4, tab: 'swap', get: (m) => m.final_bpb, what: 'Final val bits per byte of GPT-2 small',
      caveat: (r, M) => {
        const ours = r.find(([k]) => M[k].spec?.kind === 'tokenisers');
        return `One seed per tokeniser, early training.${ours ? ` Best of ours: ${ours[0]} (${signed(ours[1] - r[0][1], 4)}).` : ''}`;
      },
    },
    { set: 'intrinsic_16384', eyebrow: 'Compression', unit: 'B/tok', dir: 1, d: 2, tab: 'intrinsic', get: (m) => m.bytes_per_token, what: 'Bytes per token on held-out text, V = 16,384', caveat: () => 'Compression is not LM quality; SuperBPE tokens can span words.' },
    { set: 'intrinsic_16384', eyebrow: 'Vocabulary use', unit: 'E₂.₅', dir: 1, d: 3, tab: 'intrinsic', get: (m) => m.renyi_e_2_5, what: 'Rényi efficiency α = 2.5, V = 16,384', caveat: () => 'How evenly tokens are used; not a quality score on its own.' },
    { set: 'intrinsic_16384', eyebrow: 'Morphology', unit: 'F1', dir: 1, d: 3, tab: 'intrinsic', get: (m) => m.morph_f1, what: 'Morpheme-boundary F1, mean over languages', caveat: () => 'morpheme_bpe is trained with a morpheme lexicon: supervision the others lack.' },
    { set: 'intrinsic_16384', eyebrow: 'Code identifiers', unit: 'F1', dir: 1, d: 3, tab: 'intrinsic', get: (m) => m.identifier_f1, what: 'Identifier-part boundary F1 on held-out code', caveat: (r) => `${r.filter(([, v]) => r[0][1] - v < 0.01).length - 1} others within 0.01 of the leader.` },
    {
      set: 'throughput_local', eyebrow: 'Encode speed', unit: 'MB/s', dir: 1, d: 1, tab: 'throughput', models: (doc) => doc.models?.['16384'], get: (m) => mean(m.encode_mb_per_sec), what: 'Encode MB/s on one pinned laptop core, V = 16,384, mean of the passes',
      caveat: (r) => {
        const bpt = R('intrinsic_16384').data?.models?.[r[0][0]]?.bytes_per_token;
        return `${isNum(bpt) ? `Speed is not efficiency: ${r[0][0]} covers ${bpt.toFixed(2)} bytes per token. ` : ''}${spreads.length ? `Pass-to-pass spread: median ${median(spreads).toFixed(2)}×, max ${Math.max(...spreads).toFixed(2)}×.` : ''}`;
      },
    },
  ];
}

function renderSummary() {
  const cards = leaderSpecs().map((s) => {
    const e = R(s.set);
    const head = `<div class="eyebrow">${esc(s.eyebrow)}</div>`;
    if (!e.available) return `<div class="kpi">${head}${notRun(e)}</div>`;
    const models = s.models ? s.models(e.data) || {} : e.data.models || {};
    const r = ranked(models, s.get, s.dir);
    if (!r.length) return `<div class="kpi">${head}<div class="kpi-note">No values in <code>${esc(e.path)}</code>.</div></div>`;
    const [k, v] = r[0];
    const runner = r[1] ? `<div class="kpi-ctx">Runner-up <b>${esc(r[1][0])}</b> ${r[1][1].toFixed(s.d)} (${signed(r[1][1] - v, s.d)})</div>` : '';
    return `<div class="kpi">
      ${head}
      <div class="kpi-value num">${v.toFixed(s.d)}<span class="kpi-unit">${esc(s.unit)}</span></div>
      <div class="kpi-model" title="${esc(dispName(k))}">${esc(dispName(k))} <span class="model-key">${esc(k)}</span></div>
      <div class="kpi-note">${esc(s.what)} · ${s.dir === 1 ? 'higher' : 'lower'} is better</div>
      ${runner}
      <div class="kpi-ctx">${esc(s.caveat(r, models))}</div>
      <div class="kpi-note">Run: ${esc(provenance(e.data.meta))}</div>
      <button type="button" class="link-btn" onclick="switchTab('${s.tab}')">Details →</button>
    </div>`;
  });
  setHtml('leaders', cards.join(''));

  const rows = Object.entries(state.data.results || {}).map(([key, e]) => {
    const [label, v, tab] = RESULT_LABELS[key] ?? [key, '', 'summary'];
    const status = e.available ? '<span class="badge ok">available</span>' : '<span class="badge missing">not yet run</span>';
    const run = e.available ? esc(e.data?.meta ? provenance(e.data.meta) : key === 'proxy_compare' ? 'computed from both proxy-LM files' : e.data?.trainer?.gpu ?? '') : '';
    return `<tr><td><button type="button" class="link-btn" onclick="switchTab('${tab}')">${esc(label)}</button></td><td class="num">${typeof v === 'number' ? v.toLocaleString('en-US') : esc(v)}</td><td>${status}</td><td class="muted">${run}</td><td><code>${esc(e.path)}</code></td></tr>`;
  });
  setHtml('results-index', `<thead><tr><th>Result set</th><th>V</th><th>Status</th><th>Run</th><th>File</th></tr></thead><tbody>${rows.join('')}</tbody>`);
}

// 2. Leaderboard Table (V = 16,384 unified view)
function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody || !state.data) return;

  const models = Object.values(state.data.models);
  const filtered = models.filter((m) => state.activeFamilyFilter === 'ALL' || m.family === state.activeFamilyFilter);

  filtered.sort((a, b) => {
    const valA = a[state.sortColumn];
    const valB = b[state.sortColumn];
    // Missing values always sink to the bottom
    if (!isNum(valA)) return isNum(valB) ? 1 : 0;
    if (!isNum(valB)) return -1;
    return state.sortAscending ? valA - valB : valB - valA;
  });

  document.querySelectorAll('#leaderboard-table th.sortable').forEach((th) => {
    if (th.dataset.sort === state.sortColumn) th.setAttribute('aria-sort', state.sortAscending ? 'ascending' : 'descending');
    else th.removeAttribute('aria-sort');
  });

  const countBadge = document.getElementById('leaderboard-count-badge');
  if (countBadge) countBadge.textContent = filtered.length;

  const bpbs = filtered.map((m) => m.macro_val_bpb).filter(isNum);
  const minBpb = Math.min(...bpbs);
  const maxBpb = Math.max(...bpbs);

  tbody.innerHTML = filtered
    .map((m, idx) => {
      // Bar inverted so the lowest BPB gets the longest bar
      const barWidth = isNum(m.macro_val_bpb) && maxBpb > minBpb ? Math.round(((maxBpb - m.macro_val_bpb) / (maxBpb - minBpb)) * 100) : 100;
      const gen2 = m.generation && m.generation.includes('Gen 2') ? ' <span class="tag">Gen 2</span>' : '';

      return `
      <tr>
        <td class="rank num">${idx + 1}</td>
        <td>
          <div class="model-name">${esc(m.display_name)}${gen2}</div>
          <div class="model-desc" title="${esc(m.description)}">${esc(m.description)}</div>
        </td>
        <td>${famLabel(m.family)}</td>
        <td class="r num">
          <div class="bar-cell">
            ${isNum(m.macro_val_bpb) ? `<span class="bar"><i style="width: ${Math.max(barWidth, 4)}%"></i></span>` : ''}
            <span class="hl">${fmt(m.macro_val_bpb)}</span>
          </div>
        </td>
        <td class="r num">${fmt(m.big_macro_val_bpb)}</td>
        <td class="r num">${fmt(m.bytes_per_token)}</td>
        <td class="r num">${fmt(m.renyi_e_2_5)}</td>
        <td class="r num">${pct(m.morph_f1)}</td>
        <td class="r num">${fmt(m.decode_mb_per_sec, 1)}</td>
        <td class="r"><button class="link-btn" onclick="inspectModelInPlayground('${esc(m.key)}')" title="Open in playground">Try →</button></td>
      </tr>`;
    })
    .join('');
}

function sortLeaderboard(col) {
  if (state.sortColumn === col) {
    state.sortAscending = !state.sortAscending;
  } else {
    state.sortColumn = col;
    // Lower is better for BPB, higher is better for the others
    state.sortAscending = col.includes('bpb');
  }
  renderLeaderboard();
}

// 3. Trade-off Quadrant Scatter Plot
const METRIC_LABELS = {
  bytes_per_token: 'Bytes / token',
  renyi_e_2_5: 'Rényi E₂.₅',
  morph_f1: 'Morphology F1',
  decode_mb_per_sec: 'Decode MB/s',
  macro_val_bpb: 'Small LM macro BPB',
  big_macro_val_bpb: 'Big LM macro BPB',
  english_bpb: 'Small LM English BPB',
  multilingual_bpb: 'Small LM multilingual BPB',
  code_bpb: 'Small LM code BPB',
};

function quadrantValue(m, metric) {
  if (metric === 'english_bpb') return m.val_domains?.english?.bpb;
  if (metric === 'multilingual_bpb') return m.val_domains?.multilingual?.bpb;
  if (metric === 'code_bpb') return m.val_domains?.code?.bpb;
  return m[metric];
}

function renderQuadrantChart() {
  const canvas = document.getElementById('quadrant-chart');
  if (!canvas || !state.data) return;

  const xMetric = document.getElementById('quadrant-x-select')?.value || 'bytes_per_token';
  const yMetric = document.getElementById('quadrant-y-select')?.value || 'macro_val_bpb';

  const points = Object.values(state.data.models)
    .map((m) => ({ x: quadrantValue(m, xMetric), y: quadrantValue(m, yMetric), model: m }))
    .filter((p) => isNum(p.x) && isNum(p.y));

  destroyChart('quadrant');
  if (points.length === 0) return;

  const families = FAMILIES.filter((f) => points.some((p) => p.model.family === f));
  const datasets = families.map((fam) => {
    const color = getFamilyColor(fam);
    return {
      label: fam,
      data: points.filter((p) => p.model.family === fam),
      backgroundColor: color.border,
      borderColor: cssVar('--surface'),
      borderWidth: 2,
      pointStyle: color.shape,
      rotation: color.rotation,
      pointRadius: 7,
      pointHoverRadius: 9,
    };
  });

  const t = chartTheme();
  // Invert BPB so lower (better) sits at the top
  const isYInverted = yMetric.includes('bpb');

  state.charts.quadrant = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    plugins: [medianLinesPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: legendStyle(t),
        medianLines: { x: median(points.map((p) => p.x)), y: median(points.map((p) => p.y)), color: t.border },
        tooltip: {
          ...tooltipStyle(t),
          callbacks: {
            title: (items) => items[0].raw.model.display_name,
            label: (ctx) => [
              `${METRIC_LABELS[xMetric]}: ${ctx.raw.x.toFixed(3)}`,
              `${METRIC_LABELS[yMetric]}: ${ctx.raw.y.toFixed(3)}`,
            ],
            footer: (items) => items[0].raw.model.family,
          },
          footerColor: t.muted,
          footerFont: { weight: '400', size: 11 },
        },
      },
      scales: {
        x: axis(t, `${METRIC_LABELS[xMetric]}  →  better`),
        y: axis(t, `${METRIC_LABELS[yMetric]}  →  better`, { reverse: isYInverted }),
      },
    },
  });
}

// 4. Proxy LM: validation BPB, convergence curves, domain table, small vs big
function renderProxy() {
  const e = R(state.tier);
  showBody('proxy-body', e.available);
  setHtml('proxy-prov', provHtml(e));
  renderCompare();
  if (!e.available) {
    destroyChart('extrinsicBar');
    destroyChart('extrinsicCurve');
    setHtml('proxy-sub', '');
    return;
  }

  const meta = e.data.meta || {};
  const lm = meta.lm || {};
  const V = meta.requested_vocab_size;
  // Embedding + per layer (4 d² attention + 2 d d_ffn FFN); biases and norms left out
  const params = isNum(lm.d_model) && isNum(V) ? V * lm.d_model + lm.n_layers * (4 * lm.d_model ** 2 + 2 * lm.d_model * lm.d_ffn) : null;
  setHtml(
    'proxy-sub',
    esc(
      `${state.tier === 'proxy_big' ? 'Big' : 'Small'} LM: ${lm.n_layers ?? '?'} layers, d_model ${lm.d_model ?? '?'}` +
        (params ? `, ≈ ${(params / 1e6).toFixed(1)}M parameters (${((V * lm.d_model) / 1e6).toFixed(1)}M embedding)` : '') +
        `; ${isNum(meta.byte_budget) ? mb(meta.byte_budget) : '?'} of text per tokeniser; seeds ${(meta.seeds || []).join(', ')}; ${meta.backend?.gpu ?? meta.backend?.kind ?? '?'}.`,
    ),
  );

  const domains = meta.val_domains || Object.keys(Object.values(e.data.models)[0]?.val_domains || {});
  const models = Object.entries(e.data.models)
    .map(([key, m]) => ({ key, display_name: dispName(key), family: famOf(key), ...m }))
    .filter((m) => isNum(m.macro_val_bpb))
    .sort((a, b) => a.macro_val_bpb - b.macro_val_bpb);
  const t = chartTheme();

  // Grouped bar chart
  const barCanvas = document.getElementById('extrinsic-bar-chart');
  destroyChart('extrinsicBar');
  if (barCanvas) {
    const bar = (label, data, color) => ({ label, data, backgroundColor: color, borderRadius: 2, borderSkipped: 'start', categoryPercentage: 0.75, barPercentage: 0.9 });
    state.charts.extrinsicBar = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels: models.map((m) => m.key),
        datasets: [
          ...domains.map((d, i) => bar(d, models.map((m) => m.val_domains?.[d]?.bpb ?? null), cssVar(`--fam-${i + 1}`))),
          bar('Macro avg', models.map((m) => m.macro_val_bpb), cssVar('--fam-8')),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        interaction: { mode: 'index', axis: 'y', intersect: false },
        plugins: {
          legend: legendStyle(t),
          tooltip: { ...tooltipStyle(t), callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x?.toFixed(3)}` } },
        },
        scales: {
          x: axis(t, 'Bits per byte (lower is better)'),
          y: axis(t, null, { grid: { display: false }, ticks: { color: t.text2, font: { size: 11 } } }),
        },
      },
    });
  }

  // Convergence line chart: same family shares a hue; dash pattern separates members
  const curveCanvas = document.getElementById('extrinsic-curve-chart');
  destroyChart('extrinsicCurve');
  if (curveCanvas) {
    const seen = {};
    const datasets = models
      .filter((m) => m.curve && m.curve.length > 0)
      .map((m) => {
        const color = getFamilyColor(m.family);
        const n = (seen[m.family] = (seen[m.family] ?? -1) + 1);
        return {
          label: m.key,
          data: rolling(m.curve.map((pt) => pt.train_bpb), 9).map((y, i) => ({ x: m.curve[i].bytes_consumed / 1e6, y, step: m.curve[i].step })),
          borderColor: color.border,
          backgroundColor: color.border,
          borderDash: DASHES[n % DASHES.length],
          borderWidth: 1.75,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.25,
        };
      });

    const ys = datasets.flatMap((d) => d.data.map((p) => p.y)).filter((y) => y > 0);
    const useLog = ys.length > 0 && Math.max(...ys) / Math.min(...ys) > 4;
    state.charts.extrinsicCurve = new Chart(curveCanvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: lineLegend(t),
          tooltip: { ...tooltipStyle(t), usePointStyle: false, callbacks: { title: (i) => `${i[0].parsed.x.toFixed(1)} MB · step ${i[0].raw.step}`, label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } },
        },
        scales: {
          x: axis(t, 'Bytes ingested (MB)', { type: 'linear', min: 0 }),
          // Log scale keeps high first steps and the ~2-3 BPB tail both readable; linear when the range is narrow
          y: axis(t, useLog ? 'Train BPB (log)' : 'Train BPB', useLog ? logAxis(t) : {}),
        },
      },
    });
  }

  // Domain table
  const ref = e.data.models.bpe_frequency;
  const pm = (v, sd) => (isNum(v) ? `${v.toFixed(3)}${isNum(sd) ? ` <span class="sd">± ${sd.toFixed(3)}</span>` : ''}` : DASH);
  table(
    'proxy-table',
    [
      NAME_COL,
      { label: 'Macro BPB', dir: -1, get: (r) => r.macro_val_bpb, fmt: (v, r) => `<b>${pm(v, r.macro_val_bpb_sd)}</b>` },
      { label: 'Δ', tip: 'Macro BPB minus bpe_frequency\'s', dir: -1, get: (r) => (ref ? r.macro_val_bpb - ref.macro_val_bpb : null), fmt: (v, r) => (r.key === 'bpe_frequency' ? '<span class="muted">reference</span>' : `<span class="${v <= 0 ? 'delta-good' : 'delta-bad'}">${signed(v)}</span>`) },
      ...domains.map((d) => ({ label: `${d} BPB`, dir: -1, get: (r) => r.val_domains?.[d]?.bpb, fmt: (v, r) => pm(v, r.val_domains?.[d]?.bpb_sd) })),
      { label: 'Train B/tok', dir: 0, get: (r) => r.train_bytes_per_token, fmt: (v) => fmt(v), tip: 'UTF-8 bytes per token of the LM training text' },
      { label: 'Tokens trained', dir: 0, get: (r) => r.tokens_consumed, fmt: int, tip: 'Tokens the LM trained on within the byte budget' },
    ],
    models,
    1,
  );
}

function renderCompare() {
  const e = R('proxy_compare');
  const cav = document.getElementById('proxy-caveat-small');
  if (!e.available) {
    setHtml('compare-facts', `<li>${notRun({ path: `${R('proxy_small').path} and ${R('proxy_big').path}` })}</li>`);
    setHtml('compare-table', '');
    return;
  }
  setHtml('compare-facts', e.provenance.map((l) => `<li>${md(l)}</li>`).join(''));
  const tierLine = (p) => e.provenance.find((l) => l.startsWith(p));
  if (cav && tierLine('small LM:')) cav.innerHTML = `The small LM cannot separate the top tokenisers; ${md(tierLine('small LM:'))}. With the ${md(tierLine('big LM:') ?? 'big LM: not run')}.`;

  const { ref, names, tiers } = e.data;
  const s = tiers['small LM'];
  const b = tiers['big LM'];
  const delta = (tier) => (v, r) => (r.key === ref ? '<span class="muted">reference</span>' : `${signed(v)} <span class="sd">± ${tier[r.key][2].toFixed(3)}</span>`);
  table(
    'compare-table',
    [
      NAME_COL,
      { label: 'Small LM rank', dir: -1, get: (r) => s[r.key][0] },
      { label: 'Small LM Δ BPB', dir: -1, get: (r) => s[r.key][1], fmt: delta(s) },
      { label: 'Big LM rank', dir: -1, get: (r) => b[r.key][0] },
      { label: 'Big LM Δ BPB', dir: -1, get: (r) => b[r.key][1], fmt: delta(b) },
      { label: 'Rank change', dir: 0, get: (r) => s[r.key][0] - b[r.key][0], fmt: (v) => (v === 0 ? '0' : `${v > 0 ? '+' : '−'}${Math.abs(v)}`), tip: 'Small LM rank minus big LM rank: positive = moved up with the big LM' },
    ],
    names.map((key) => ({ key })),
    3,
  );
}

// 5. Tokka-Bench Per-Language Heatmap & Grouped Chart
// +1: higher is better, -1: lower is better, 0: neutral magnitude
const TOKKA_DIRECTION = { bytes_per_token: 1, subword_fertility: -1, word_split_pct: -1, continued_word_rate: -1, unique_tokens: 0 };
// Natural languages show as upper-case ISO codes; code languages (and the legacy pooled 'code') by name
const LANG_LABELS = { code: 'Code', python: 'Python', javascript: 'JS', java: 'Java', cpp: 'C++', go: 'Go', rust: 'Rust' };
const langLabel = (l) => LANG_LABELS[l] ?? l.toUpperCase();

function renderTokkaTab() {
  if (!state.data) return;
  const e = R(`tokka_${state.vocab}`);
  showBody('tokka-body', e.available);
  setHtml('tokka-prov', provHtml(e));
  if (!e.available) {
    destroyChart('tokkaBar');
    setHtml('tokka-sub', '');
    setHtml('tokka-summary-table', '');
    return;
  }

  const matrix = e.data.models ?? e.data;
  const metric0 = state.tokkaActiveMetric;
  const avg = (k) => matrix[k]?.macro_avg?.[metric0] ?? (metric0 === 'unique_tokens' ? matrix[k]?.unique_tokens : null);
  // Rows ranked best-first on the shown metric's average (more first for the direction-less unique tokens)
  const rankKey = (k) => (isNum(avg(k)) ? avg(k) * (TOKKA_DIRECTION[metric0] || 1) : -Infinity);
  const models = Object.keys(matrix).sort((a, b) => rankKey(b) - rankKey(a) || 0);
  // Columns are the languages the results files have (the server derives them from the data)
  const langs = state.data.tokka_languages || [];
  const metric = state.tokkaActiveMetric;

  const tokkaMeta = e.data.meta;
  const sub = document.getElementById('tokka-sub');
  if (sub) {
    const macro = tokkaMeta?.macro_languages;
    const left = macro ? langs.filter((l) => !macro.includes(l)).map(langLabel) : [];
    sub.textContent =
      `V = ${state.vocab.toLocaleString('en-US')}. ${langs.length} languages, natural and code. Avg is the macro average` +
      (left.length ? ` and leaves out ${left.join(', ')} (too little eval text)` : '') +
      '.';
  }
  const head = document.getElementById('tokka-heatmap-head');
  if (head) head.innerHTML = `<th>Tokeniser</th>${langs.map((l) => `<th class="c">${esc(langLabel(l))}</th>`).join('')}<th class="c">Avg</th>`;
  const dir = TOKKA_DIRECTION[metric] ?? 0;

  const lo = document.getElementById('tokka-scale-lo');
  const hi = document.getElementById('tokka-scale-hi');
  if (lo) lo.textContent = dir === 0 ? 'fewer' : 'worse';
  if (hi) hi.textContent = dir === 0 ? 'more' : 'better';

  const isPct = metric === 'continued_word_rate' || metric === 'word_split_pct';
  const display = (v) => (isNum(v) ? (isPct ? v.toFixed(1) + '%' : metric === 'unique_tokens' ? String(Math.round(v)) : v.toFixed(2)) : '–');

  // Shade by rank within each column, so outliers (e.g. CJK) don't wash out the rest
  const cols = [...langs, 'macro_avg'];
  const sorted = {};
  cols.forEach((c) => {
    sorted[c] = models.map((k) => matrix[k]?.[c]?.[metric]).filter(isNum).sort((a, b) => a - b);
  });
  const shade = (v, c) => {
    if (!isNum(v)) return '';
    const vals = sorted[c];
    const lo = vals.indexOf(v);
    const hi = vals.lastIndexOf(v);
    let s = vals.length > 1 ? (lo + hi) / 2 / (vals.length - 1) : 1;
    if (dir === -1) s = 1 - s;
    return `background-color: color-mix(in srgb, var(--accent) ${Math.round(4 + s * 51)}%, var(--surface));`;
  };

  const tbody = document.getElementById('tokka-heatmap-tbody');
  if (tbody) {
    tbody.innerHTML = models
      .map((mKey) => {
        const row = matrix[mKey];
        const cells = langs
          .map((lang) => {
            const v = row?.[lang]?.[metric];
            return `<td class="cell" style="${shade(v, lang)}">${display(v)}</td>`;
          })
          .join('');
        const macroVal = row?.macro_avg?.[metric];

        return `
        <tr>
          <td class="model-name">${esc(dispName(mKey))}</td>
          ${cells}
          <td class="cell avg" style="${shade(macroVal, 'macro_avg')}">${display(macroVal)}</td>
        </tr>`;
      })
      .join('');
  }

  // Bar Chart: series are tokenisers, so each gets its own categorical slot
  const barCanvas = document.getElementById('tokka-bar-chart');
  destroyChart('tokkaBar');
  if (barCanvas) {
    const t = chartTheme();

    const datasets = models.slice(0, 6).map((mKey, i) => ({
      label: dispName(mKey),
      data: langs.map((lang) => matrix[mKey]?.[lang]?.[metric] ?? null),
      backgroundColor: cssVar(`--fam-${i + 1}`),
      borderRadius: 2,
      borderSkipped: 'start',
      categoryPercentage: 0.8,
      barPercentage: 0.9,
    }));

    // Some scripts are orders of magnitude apart; switch to log so the rest stay visible
    const all = datasets.flatMap((d) => d.data).filter((v) => isNum(v) && v > 0);
    const useLog = all.length > 0 && Math.max(...all) / Math.min(...all) > 20;
    const metricLabel = document.querySelector(`.tokka-metric-btn[data-metric="${metric}"]`)?.textContent || metric;

    state.charts.tokkaBar = new Chart(barCanvas, {
      type: 'bar',
      data: { labels: langs.map(langLabel), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: legendStyle(t), tooltip: tooltipStyle(t) },
        scales: {
          x: axis(t, null, { grid: { display: false } }),
          y: axis(t, useLog ? `${metricLabel} (log)` : metricLabel, useLog ? logAxis(t) : {}),
        },
      },
    });
  }

  const macro = (k) => (r) => r.macro_avg?.[k];
  table(
    'tokka-summary-table',
    [
      NAME_COL,
      { label: '|V|', dir: 0, get: (r) => r.vocab_size, fmt: int },
      { label: 'Macro B/tok', dir: 1, get: macro('bytes_per_token'), fmt: (v) => fmt(v), tip: 'Unweighted mean over the macro languages' },
      { label: 'Micro B/tok', dir: 1, get: (r) => r.micro_avg?.bytes_per_token, fmt: (v) => fmt(v), tip: 'Totals over all languages (token-weighted)' },
      { label: 'Fertility', dir: -1, get: macro('subword_fertility'), fmt: (v) => fmt(v) },
      { label: 'Word split %', dir: -1, get: macro('word_split_pct'), fmt: (v) => fmt(v, 1) },
      { label: 'Continued %', dir: -1, get: macro('continued_word_rate'), fmt: (v) => fmt(v, 1) },
      { label: 'Unique tokens used', dir: 0, get: (r) => r.unique_tokens, fmt: int, tip: 'Distinct tokens over all languages' },
    ],
    Object.entries(matrix).map(([key, m]) => ({ key, ...m })),
    2,
  );
}

// 6. Intrinsic & Morphology Table
// dir: +1 higher is better, -1 lower is better
const INTRINSIC_COLS = [
  { key: 'bytes_per_token', label: 'B/tok', group: 'Compression', dir: 1, fmt: (v) => fmt(v), tip: 'Bytes per token: how much raw text one token covers.' },
  { key: 'fertility', label: 'Fertility', group: 'Compression', dir: -1, fmt: (v) => fmt(v, 2), tip: 'Average tokens per word. Fewer splits per word is better.' },
  { key: 'renyi_e_1_0', label: 'E₁', group: 'Rényi efficiency', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi entropy efficiency at α = 1 (Shannon): how evenly the vocabulary is used, 0–1.' },
  { key: 'renyi_e_2_0', label: 'E₂', group: 'Rényi efficiency', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 2.' },
  { key: 'renyi_e_2_5', label: 'E₂.₅', group: 'Rényi efficiency', dir: 1, fmt: (v) => fmt(v), tip: 'Rényi efficiency at α = 2.5, the value Zouhar et al. (2023) proposed; weights frequent tokens more heavily.' },
  { key: 'gini_coefficient', label: 'Gini', group: 'Token usage', dir: -1, fmt: (v) => fmt(v), tip: 'Inequality of token frequencies. Lower means usage is spread more evenly.' },
  { key: 'active_fraction', label: 'Active', group: 'Token usage', dir: 1, fmt: pct, tip: 'Share of the vocabulary actually used on the corpus.' },
  { key: 'morph_f1', label: 'F1', group: 'Morpheme boundaries', dir: 1, fmt: pct, tip: 'SIGMORPHON 2022 morpheme-boundary F1, mean over languages: do token splits land on real morpheme boundaries?' },
  { key: 'morph_precision', label: 'Precision', group: 'Morpheme boundaries', dir: 1, fmt: pct, tip: 'Of the splits made, the share that are true morpheme boundaries.' },
  { key: 'morph_recall', label: 'Recall', group: 'Morpheme boundaries', dir: 1, fmt: pct, tip: 'Of the true morpheme boundaries, the share the tokeniser splits on.' },
  { key: 'identifier_f1', label: 'Ident F1', group: 'Code', dir: 1, fmt: pct, tip: 'Boundary F1 on code identifiers (snake_case / camelCase parts).' },
  { key: 'encode_mb_per_sec', label: 'Encode', group: 'Speed MB/s', dir: 1, fmt: (v) => fmt(v, 2), tip: 'Encoding throughput in MB/s (unpinned laptop run).' },
  { key: 'decode_mb_per_sec', label: 'Decode', group: 'Speed MB/s', dir: 1, fmt: (v) => fmt(v, 1), tip: 'Decoding throughput in MB/s (unpinned laptop run).' },
];

function sortIntrinsic(key) {
  // First click sorts best-first; clicking again flips
  state.intrinsicSort = state.intrinsicSort?.key === key ? { key, best: !state.intrinsicSort.best } : { key, best: true };
  renderIntrinsicTab();
}

function renderIntrinsicTab() {
  if (!state.data) return;

  const thead = document.getElementById('intrinsic-thead');
  const tbody = document.getElementById('intrinsic-tbody');
  if (!tbody || !thead) return;

  const e = R(`intrinsic_${state.vocab}`);
  showBody('intrinsic-body', e.available);
  setHtml('intrinsic-prov', provHtml(e));
  if (!e.available) {
    setHtml('intrinsic-vocab-note', '');
    setHtml('morph-table', '');
    return;
  }
  const docModels = e.data.models ?? e.data;
  const all = Object.entries(docModels).map(([key, m]) => ({ key, display_name: dispName(key), family: famOf(key), intrinsic: m }));
  const sort = state.intrinsicSort || (state.intrinsicSort = { key: 'renyi_e_2_5', best: true });

  // Vocab size is shared across runs; say it once instead of a constant column
  const vocabs = [...new Set(all.map((m) => m.intrinsic?.vocab_size).filter(isNum))];
  const note = document.getElementById('intrinsic-vocab-note');
  if (note) note.textContent = vocabs.length ? `Vocabulary size ${vocabs.map((v) => v.toLocaleString('en-US')).join(', ')}.` : '';

  const groups = [];
  INTRINSIC_COLS.forEach((c, i) => {
    c.gstart = i === 0 || INTRINSIC_COLS[i - 1].group !== c.group;
    if (c.gstart) groups.push({ name: c.group, span: 1 });
    else groups[groups.length - 1].span++;
  });

  thead.innerHTML = `
    <tr>
      <th class="group"></th>
      ${groups.map((g) => `<th class="group gstart" colspan="${g.span}">${g.name}</th>`).join('')}
    </tr>
    <tr>
      <th>Tokeniser</th>
      ${INTRINSIC_COLS.map((c) => {
        const sorted = sort.key === c.key ? ` aria-sort="${(c.dir === 1) === sort.best ? 'descending' : 'ascending'}"` : '';
        return `<th class="r sortable${c.gstart ? ' gstart' : ''}"${sorted}><button type="button" class="th-btn" title="${esc(c.tip)}" onclick="sortIntrinsic('${c.key}')">${c.label}<span class="dir">${c.dir === 1 ? '↑' : '↓'}</span></button></th>`;
      }).join('')}
    </tr>`;

  // Rank-based shading per column, direction-aware (same idea as the Tokka heatmap)
  const ranked = {};
  INTRINSIC_COLS.forEach((c) => {
    ranked[c.key] = all.map((m) => m.intrinsic?.[c.key]).filter(isNum).sort((a, b) => (a - b) * c.dir);
  });
  const cell = (c, v) => {
    const cls = `r num${c.gstart ? ' gstart' : ''}`;
    if (!isNum(v)) return `<td class="${cls}">${DASH}</td>`;
    const vals = ranked[c.key];
    const s = vals.length > 1 ? (vals.indexOf(v) + vals.lastIndexOf(v)) / 2 / (vals.length - 1) : 1;
    const best = v === vals[vals.length - 1] ? ' best' : '';
    return `<td class="${cls}${best}" style="background-color: color-mix(in srgb, var(--accent) ${Math.round(s * 32)}%, transparent)">${c.fmt(v)}</td>`;
  };

  const searchTerm = (document.getElementById('intrinsic-search')?.value || '').toLowerCase();
  const sortCol = INTRINSIC_COLS.find((c) => c.key === sort.key);
  const models = all
    .filter((m) => [m.display_name, m.family, m.key].some((s) => s.toLowerCase().includes(searchTerm)))
    .sort((a, b) => {
      const va = a.intrinsic?.[sort.key];
      const vb = b.intrinsic?.[sort.key];
      // Missing values always sink to the bottom
      if (!isNum(va)) return isNum(vb) ? 1 : 0;
      if (!isNum(vb)) return -1;
      return (vb - va) * sortCol.dir * (sort.best ? 1 : -1);
    });

  const colspan = INTRINSIC_COLS.length + 1;
  if (models.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${colspan}" class="muted" style="text-align: center; padding: 24px">No tokenisers match.</td></tr>`;
  } else {
    tbody.innerHTML = models
      .map((m) => {
        const intr = m.intrinsic || {};
        const name = `<td><span class="fam" style="color: var(--text)"><span class="dot" style="--c: ${familyVar(m.family)}"></span><span class="model-name">${esc(m.display_name)}</span></span></td>`;
        if (!INTRINSIC_COLS.some((c) => isNum(intr[c.key]))) {
          return `<tr>${name}<td class="na gstart" colspan="${INTRINSIC_COLS.length}">Not evaluated on intrinsic metrics</td></tr>`;
        }
        return `<tr>${name}${INTRINSIC_COLS.map((c) => cell(c, intr[c.key])).join('')}</tr>`;
      })
      .join('');
  }

  // Morph F1 per language
  const first = Object.values(docModels)[0] || {};
  const words = first.morph_words_by_lang || {};
  table(
    'morph-table',
    [
      NAME_COL,
      { label: 'Mean over languages', dir: 1, get: (r) => r.morph_f1, fmt: (v) => fmt(v), tip: 'The headline morph F1' },
      ...Object.keys(words).map((l) => ({ label: `${l} <span class="sd">${words[l].toLocaleString('en-US')}</span>`, dir: 1, get: (r) => r.morph_f1_by_lang?.[l], fmt: (v) => fmt(v), tip: `${words[l]} gold words` })),
      { label: 'Word-weighted', dir: 1, get: (r) => r.morph_f1_word_weighted, fmt: (v) => fmt(v) },
    ],
    Object.entries(docModels).map(([key, m]) => ({ key, ...m })),
    1,
  );
}

// 7. Throughput: the laptop suite run, a pinned laptop core and a Colab CPU, side by side
function renderThroughput() {
  if (!state.data) return;
  const V = String(state.vocab);
  const key = state.tpMetric;
  const loc = R('throughput_local');
  const col = R('throughput_colab_cpu');
  const suite = R(`intrinsic_${V}`);
  const L = loc.available ? loc.data.models?.[V] || {} : {};
  const C = col.available ? col.data.models?.[V] || {} : {};
  const S = suite.available ? suite.data.models || {} : {};
  const names = [...new Set([...Object.keys(S), ...Object.keys(L), ...Object.keys(C)])];
  const rows = names.map((k) => ({ key: k, suite: S[k]?.[key], local: L[k]?.[key], colab: C[k]?.[key] }));
  const dirWord = key.startsWith('encode') ? 'encode' : 'decode';

  setHtml('tp-sub', esc(`V = ${state.vocab.toLocaleString('en-US')}, ${dirWord} MB/s. Three runs of the same tokenisers on the same held-out documents.`));
  setHtml('tp-prov', provHtml(suite, 'Laptop suite run') + provHtml(loc, 'Pinned laptop core') + provHtml(col, 'Colab CPU'));

  // Measured spreads and ratios, from the files
  const sp = (M) => Object.values(M).map((m) => spread(m[key])).filter(isNum);
  const ratio = (a, b) => rows.map((r) => (isNum(a(r)) && isNum(b(r)) ? a(r) / b(r) : null)).filter(isNum);
  const facts = [];
  for (const [label, M, e] of [['Pinned laptop core', L, loc], ['Colab CPU', C, col]]) {
    const s = sp(M);
    if (!e.available) facts.push(`<b>${label}</b>: not yet run.`);
    else if (s.length) facts.push(`<b>${label}</b>: pass-to-pass spread median ${median(s).toFixed(2)}×, max ${Math.max(...s).toFixed(2)}×; ${s.filter((x) => x > 1.2).length} of ${s.length} tokenisers differ by more than 1.2×.`);
  }
  if (suite.available) facts.push('<b>Laptop suite run</b>: one pass per tokeniser, so its spread is not in the file.');
  const r1 = ratio((r) => mean(r.local), (r) => r.suite);
  const r2 = ratio((r) => mean(r.colab), (r) => mean(r.local));
  if (r1.length) facts.push(`<b>Pinned core ÷ laptop suite</b>: median ${median(r1).toFixed(2)}× (range ${range(r1, 2, '×')}).`);
  if (r2.length) facts.push(`<b>Colab CPU ÷ pinned core</b>: median ${median(r2).toFixed(2)}× (range ${range(r2, 2, '×')}).`);
  setHtml('tp-facts', facts.map((f) => `<li>${f}</li>`).join(''));

  const passes = (v) => (Array.isArray(v) ? v.map((x) => x.toFixed(2)).join(' / ') : DASH);
  const spreadFmt = (v) => (isNum(v) ? `${v.toFixed(2)}×` : DASH);
  table(
    'tp-table',
    [
      NAME_COL,
      { label: 'Laptop suite', dir: 1, get: (r) => r.suite, fmt: (v) => fmt(v, 2), tip: 'Intrinsic suite run: unpinned shared laptop, one pass' },
      { label: 'Pinned core', dir: 1, get: (r) => mean(r.local), fmt: (v, r) => passes(r.local), tip: 'One pinned performance core: pass 1 / pass 2' },
      { label: 'spread', dir: -1, get: (r) => spread(r.local), fmt: spreadFmt, tip: 'max / min over the pinned passes' },
      { label: 'Colab CPU', dir: 1, get: (r) => mean(r.colab), fmt: (v, r) => passes(r.colab), tip: 'Colab CPU runtime: pass 1 / pass 2' },
      { label: 'spread', dir: -1, get: (r) => spread(r.colab), fmt: spreadFmt, tip: 'max / min over the Colab passes' },
    ],
    rows,
    2,
  );

  const canvas = document.getElementById('tp-chart');
  destroyChart('throughput');
  if (!canvas || !rows.length) return;
  const t = chartTheme();
  const order = [...rows].sort((a, b) => (mean(b.local) ?? b.suite ?? 0) - (mean(a.local) ?? a.suite ?? 0));
  const bar = (label, data, color) => ({ label, data, backgroundColor: color, borderRadius: 2, borderSkipped: 'start', categoryPercentage: 0.8, barPercentage: 0.9 });
  state.charts.throughput = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: order.map((r) => r.key),
      datasets: [
        bar('Laptop suite (unpinned, 1 pass)', order.map((r) => r.suite ?? null), cssVar('--fam-8')),
        bar('Pinned laptop core (mean of passes)', order.map((r) => mean(r.local)), cssVar('--fam-1')),
        bar('Colab CPU (mean of passes)', order.map((r) => mean(r.colab)), cssVar('--fam-2')),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      interaction: { mode: 'index', axis: 'y', intersect: false },
      plugins: { legend: legendStyle(t), tooltip: { ...tooltipStyle(t), callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x?.toFixed(2)} MB/s` } } },
      scales: {
        x: axis(t, `${dirWord} MB/s (higher is better)`, { min: 0 }),
        y: axis(t, null, { grid: { display: false }, ticks: { color: t.text2, font: { size: 11 } } }),
      },
    },
  });
}

// 8. GPT-2 tokeniser swap: table, notes, BPB-vs-step chart, space-prefix rerun, lr 1.8e-3
function renderSwap() {
  const e = R('swap');
  showBody('swap-body', e.available);
  setHtml('swap-prov', provHtml(e));
  renderSpacePrefix();
  renderLr();
  if (!e.available) {
    setHtml('swap-sub', '');
    return;
  }
  const doc = e.data;
  const meta = doc.meta || {};
  const a = meta.recipe?.args || {};
  const models = doc.models || {};
  const eq = doc.gpt2_equivalent?.steps || {};
  const rows = Object.entries(models)
    .filter(([, m]) => isNum(m.final_bpb))
    .map(([key, m]) => ({ key, ...m, eq: eq[key] }));
  setHtml(
    'swap-sub',
    esc(`${rows.length} tokenisers · ${a.num_iterations ?? '?'} steps × ${int(Number(a.total_batch_size))} tokens per step · peak lr ${a.learning_rate ?? '?'} · ${(meta.backend?.gpu || '?').split(',')[0]} · one run each.`),
  );
  table(
    'swap-table',
    [
      NAME_COL,
      { label: 'Final BPB', dir: -1, get: (r) => r.final_bpb, fmt: (v) => `<b>${fmt(v, 4)}</b>` },
      { label: 'GPT-2-equiv. steps', dir: 1, get: (r) => r.eq, fmt: (v) => (isNum(v) ? String(v) : 'not reached'), tip: 'First step at which GPT-2\'s smoothed train BPB reaches this final val BPB' },
      { label: 'Val B/tok', dir: 1, get: (r) => r.val_bytes_per_token, fmt: (v) => fmt(v), tip: 'UTF-8 bytes per token on the val text: more text per step when higher' },
      { label: 'Train B/tok', dir: 1, get: (r) => r.train_bytes_per_token, fmt: (v) => fmt(v) },
      { label: '|V|', dir: 0, get: (r) => r.vocab_size, fmt: int },
      { label: 'Params', dir: 0, get: (r) => r.params, fmt: (v) => (isNum(v) ? `${(v / 1e6).toFixed(1)} M` : DASH) },
      { label: 'ms/step', dir: -1, get: (r) => r.ms_per_step, fmt: (v) => fmt(v, 0) },
      { label: 'tok/s', dir: 1, get: (r) => r.tokens_per_s, fmt: int },
    ],
    rows,
    1,
  );
  setHtml('swap-notes', (e.notes || []).map((l) => `<li>${md(l)}</li>`).join('') || '<li>No notes (the file lacks a field the README renderer needs).</li>');

  // Caveats with this run's numbers
  const g = models.gpt2;
  const regime = document.getElementById('swap-caveat-regime');
  if (regime && g) regime.textContent = `Early training: ${a.num_iterations} steps leave GPT-2 at ${g.final_nats_per_token.toFixed(2)} nats per token on val, far from the speedrun target of ${meta.target?.nats_per_gpt2_token}; the ranking need not hold for longer training. One run per tokeniser (llm.c's fixed seed): no seed spread.`;
  const ws = rows.filter((r) => r.spec?.kind === 'tokenisers' && r.key !== 'super_bpe');
  const space = document.getElementById('swap-caveat-space');
  if (space && g && ws.length) {
    const less = ws.map((r) => 100 * (1 - r.train_bytes_per_token / g.train_bytes_per_token));
    space.textContent = `Our ${ws.length} whitespace-isolating tokenisers give ${range(ws.map((r) => r.val_bytes_per_token))} val bytes per token against GPT-2's ${g.val_bytes_per_token.toFixed(2)}: every space is its own token, so each step holds ${range(less, 0)}% less text. Their gap to GPT-2 mixes less text with different merges until the space-prefix rerun separates the two.`;
  }
}

function rolling(a, w) {
  const h = Math.floor(w / 2);
  const c = [0];
  a.forEach((v) => c.push(c[c.length - 1] + v));
  return a.map((_, i) => {
    const lo = Math.max(0, i - h);
    const hi = Math.min(a.length, i + h + 1);
    return (c[hi] - c[lo]) / (hi - lo);
  });
}

function swapStyle(k, seen) {
  if (k === 'gpt2') return { color: cssVar('--text'), dash: [], width: 2.5 };
  if (k.startsWith('tm_')) return { color: cssVar('--fam-6'), dash: k.endsWith('32000') ? [6, 3] : [], width: 1.75 };
  const fam = famOf(k);
  const n = (seen[fam] = (seen[fam] ?? -1) + 1);
  return { color: getFamilyColor(fam).border, dash: DASHES[n % DASHES.length], width: 1.75 };
}

function renderSwapChart() {
  destroyChart('swap');
  const e = R('swap');
  const canvas = document.getElementById('swap-chart');
  if (!e.available || !canvas) return;
  const models = e.data.models || {};
  const train = e.train_bpb;
  const mode = state.swapMode === 'train' && train ? 'train' : 'val';
  const w = e.data.gpt2_equivalent?.window ?? 21;
  const every = e.data.meta?.recipe?.args?.val_loss_every;
  const order = Object.keys(models)
    .filter((k) => isNum(models[k].final_bpb) && (mode === 'val' ? models[k].curve?.length : train.bpb[k]))
    .sort((a, b) => models[a].final_bpb - models[b].final_bpb);
  const seen = {};
  const datasets = order.map((k) => {
    const st = swapStyle(k, seen);
    const data = mode === 'val' ? models[k].curve.map(([x, y]) => ({ x, y })) : rolling(train.bpb[k], w).map((y, i) => ({ x: train.steps[i], y }));
    return { label: k, data, borderColor: st.color, backgroundColor: st.color, borderDash: st.dash, borderWidth: st.width, pointRadius: mode === 'val' ? 2.5 : 0, pointHoverRadius: 4 };
  });
  // The untrained start (about log2 V / bytes per token BPB) would flatten the end: scale to step >= 100
  const late = datasets.flatMap((d) => d.data.filter((p) => p.x >= 100).map((p) => p.y));
  const yMax = late.length ? Math.ceil(Math.max(...late) * 10) / 10 : undefined;
  const yMin = late.length ? Math.floor(Math.min(...late) * 10) / 10 : undefined;
  setHtml(
    'swap-chart-sub',
    esc(
      mode === 'val'
        ? `Val BPB at every evaluation (every ${every ?? '?'} steps). The y axis covers step 100 onwards; the untrained step-0 values (about log₂ V ÷ bytes per token) lie above it.`
        : `Train BPB per step (train loss ÷ ln 2 ÷ train bytes per token, from nanogpt_swap_steps.csv), centred ${w}-step rolling mean: GPT-2's curve is the one the GPT-2-equivalent steps are read from.`,
    ),
  );
  const t = chartTheme();
  state.charts.swap = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: lineLegend(t),
        tooltip: { ...tooltipStyle(t), usePointStyle: false, callbacks: { title: (i) => `step ${i[0].parsed.x}`, label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(4)}` } },
      },
      scales: {
        x: axis(t, 'Step', { type: 'linear', min: 0 }),
        y: axis(t, `${mode === 'val' ? 'Val' : 'Train'} BPB (lower is better)`, { min: yMin, max: yMax }),
      },
    },
  });
}

function renderSpacePrefix() {
  const e = R('swap_space_prefix');
  const badge = document.getElementById('sp-status');
  if (badge) {
    badge.textContent = e.available ? 'results' : 'not yet run';
    badge.className = `badge ${e.available ? 'ok' : 'missing'}`;
  }
  if (!e.available) {
    setHtml('sp-body', `${notRun(e)}<p class="note">With the file, this section shows each tokeniser's original run next to its space-prefix run (bytes per token, final BPB, the change, the gap to GPT-2 and GPT-2-equivalent steps), with GPT-2 as the reference.</p>`);
    return;
  }
  const p = e.pairs;
  if (!p || !p.rows.length) {
    setHtml('sp-body', `<div class="empty">The space-prefix file is present, but ${p ? 'none of its tokenisers matches one in' : 'there is no'} <code>${esc(R('swap').path)}</code> to compare against.</div>${provHtml(e)}`);
    return;
  }
  const g = p.gpt2;
  const rows = p.rows.map((r) => ({ key: r.variant, ...r }));
  const deltas = rows.map((r) => r.delta_bpb).filter(isNum);
  const best = [...rows].filter((r) => isNum(r.sp.final_bpb)).sort((a, b) => a.sp.final_bpb - b.sp.final_bpb)[0];
  const facts = [];
  if (g) facts.push(`<b>GPT-2 reference</b> (${g.run === 'sp' ? 'from the space-prefix run' : 'from the original run'}): final BPB ${fmt(g.final_bpb, 4)}, ${fmt(g.val_bytes_per_token)} val bytes per token.`);
  if (deltas.length) facts.push(`<b>${deltas.filter((d) => d < 0).length} of ${deltas.length}</b> tokenisers have a lower final BPB with the space prefix; median change ${signed(median(deltas), 4)} BPB.`);
  facts.push(`Val bytes per token: ${range(rows.map((r) => r.old.val_bytes_per_token).filter(isNum))} before, ${range(rows.map((r) => r.sp.val_bytes_per_token).filter(isNum))} with the space prefix.`);
  if (best && g) facts.push(`Lowest space-prefix BPB: <code>${esc(best.key)}</code> ${fmt(best.sp.final_bpb, 4)}, ${signed(best.sp.final_bpb - g.final_bpb, 4)} against GPT-2.`);
  // No report notes here: report._swap_findings calls our tokenisers whitespace-isolating, which this run's are not
  setHtml(
    'sp-body',
    `<ul class="facts">${facts.map((f) => `<li>${f}</li>`).join('')}</ul>
     <div class="table-wrap" style="margin-top: 14px"><table class="data" id="sp-table"></table></div>
     <div class="table-foot"><span>orig. = original swap run · SP = space-prefix run · Δ = SP − orig. · ↑ higher is better · ↓ lower is better</span></div>
     ${provHtml(e)}`,
  );
  table(
    'sp-table',
    [
      NAME_COL,
      { label: 'Val B/tok, orig.', dir: 1, get: (r) => r.old.val_bytes_per_token, fmt: (v) => fmt(v) },
      { label: 'Val B/tok, SP', dir: 1, get: (r) => r.sp.val_bytes_per_token, fmt: (v) => fmt(v) },
      { label: 'BPB, orig.', dir: -1, get: (r) => r.old.final_bpb, fmt: (v) => fmt(v, 4) },
      { label: 'BPB, SP', dir: -1, get: (r) => r.sp.final_bpb, fmt: (v) => `<b>${fmt(v, 4)}</b>` },
      { label: 'Δ BPB', dir: -1, get: (r) => r.delta_bpb, fmt: (v) => `<span class="${v <= 0 ? 'delta-good' : 'delta-bad'}">${signed(v, 4)}</span>` },
      { label: 'vs GPT-2', dir: -1, get: (r) => (g && isNum(r.sp.final_bpb) ? r.sp.final_bpb - g.final_bpb : null), fmt: (v) => signed(v, 4), tip: 'Space-prefix final BPB minus GPT-2\'s' },
      { label: 'GPT-2-eq. steps, orig.', dir: 1, get: (r) => r.old.gpt2_equivalent_steps, fmt: (v) => (isNum(v) ? String(v) : DASH) },
      { label: 'GPT-2-eq. steps, SP', dir: 1, get: (r) => r.sp.gpt2_equivalent_steps, fmt: (v) => (isNum(v) ? String(v) : DASH) },
    ],
    rows,
    4,
  );
}

function renderLr() {
  const e = R('swap_lr1p8e-3');
  const base = R('swap');
  if (!e.available) {
    setHtml('lr-body', notRun(e));
    return;
  }
  const tr = e.data.trainer || {};
  const steps = tr.steps || [];
  const peak = steps.length ? Math.max(...steps.map((s) => s.lr)) : null;
  const args = base.data?.meta?.recipe?.args || {};
  const ref = Object.fromEntries((base.data?.models?.gpt2?.curve || []).map(([s, b]) => [s, b]));
  const rows = (tr.evals || []).map((ev) => ({ step: ev.step, hi: ev.val_bpb, lo: ref[ev.step] }));
  const cmp = rows.filter((r) => r.step > 0 && isNum(r.lo));
  const last = rows[rows.length - 1];
  const warm = Number(args.warmup_iters ?? 0);
  const norms = steps.filter((s) => s.step >= warm).map((s) => s.norm).filter(isNum);
  const facts = [];
  if (cmp.length) {
    const worse = cmp.filter((r) => r.hi > r.lo).length;
    facts.push(`Peak lr ${isNum(peak) ? peak.toExponential(1) : '?'} instead of ${args.learning_rate ?? '?'}: GPT-2's val BPB is higher at <b>${worse} of ${cmp.length}</b> evaluations after step 0; final ${fmt(last.hi, 4)} against ${fmt(last.lo, 4)} (${signed(last.hi - last.lo, 4)}).`);
  }
  if (norms.length) facts.push(`Max gradient norm after the ${warm}-step warmup: ${Math.max(...norms).toFixed(2)}.`);
  const body = rows
    .map((r) => `<tr><td class="num">${r.step}</td><td class="r num">${fmt(r.lo, 4)}</td><td class="r num">${fmt(r.hi, 4)}</td><td class="r num"><span class="${r.hi - r.lo > 0 ? 'delta-bad' : r.hi - r.lo < 0 ? 'delta-good' : ''}">${isNum(r.lo) ? signed(r.hi - r.lo, 4) : DASH}</span></td></tr>`)
    .join('');
  setHtml(
    'lr-body',
    `<ul class="facts">${facts.map((f) => `<li>${f}</li>`).join('')}</ul>
     <div class="table-wrap" style="margin-top: 14px"><table class="data"><thead><tr><th>Step</th><th class="r">Val BPB, lr ${esc(args.learning_rate ?? '?')} ↓</th><th class="r">Val BPB, lr ${isNum(peak) ? peak.toExponential(1) : '?'} ↓</th><th class="r">Δ ↓</th></tr></thead><tbody>${body}</tbody></table></div>
     <details class="prov"><summary>Provenance: ${esc(`${tr.gpu ?? '?'} · torch ${tr.torch ?? '?'} · ${steps.length} steps · one run (GPT-2 only; the sweep was stopped)`)}</summary><ul><li>file <code>${esc(e.path)}</code>; the lr 6e-4 column is GPT-2 in <code>${esc(base.path)}</code></li><li>${int(e.data.train?.tokens_per_step)} tokens per step; val ${int(e.data.val?.docs)} documents, ${e.data.val?.bytes ? mb(e.data.val.bytes) : '?'}</li></ul></details>`,
  );
}

// 9. Concepts: one measured line under the PMI explanation
function renderConceptFacts() {
  const e = R('tokka_16384');
  if (!e.available) return setHtml('fact-pmi', '');
  const M = e.data.models ?? e.data;
  const line = (k) => (M[k]?.macro_avg ? `<code>${k}</code> ${M[k].macro_avg.bytes_per_token.toFixed(2)} bytes per token, ${int(M[k].unique_tokens)} distinct tokens used` : null);
  const parts = ['wp_bpe_base', 'wp_bpe_ft', 'wordpiece_pmi', 'bpe_frequency'].map(line).filter(Boolean);
  setHtml('fact-pmi', parts.length ? `Measured (Tokka-Bench, V = ${int(e.data.meta?.requested_vocab_size)}, held-out text, macro average): ${parts.join('; ')}.` : '');
}

// 10. Multi-Axis Radar View
// Each selected model keeps its slot until deselected, so toggling never repaints the others
function radarSlot(mKey) {
  if (!state.radarSlots[mKey]) {
    const used = new Set(Object.values(state.radarSlots));
    const free = [1, 2, 3, 4, 5, 6, 7].find((s) => !used.has(s));
    state.radarSlots[mKey] = free || 8;
  }
  return state.radarSlots[mKey];
}

function renderRadarView() {
  if (!state.data) return;

  const canvas = document.getElementById('radar-chart');
  const togglesContainer = document.getElementById('radar-toggles');
  if (!canvas || !togglesContainer) return;

  const allModels = Object.values(state.data.models);
  state.selectedRadarModels.forEach(radarSlot);

  togglesContainer.innerHTML = allModels
    .map((m) => {
      const on = state.selectedRadarModels.includes(m.key);
      const dot = on ? `<span class="dot" style="--c: var(--fam-${state.radarSlots[m.key]})"></span>` : '';
      return `<button class="chip" aria-pressed="${on}" onclick="toggleRadarModel('${esc(m.key)}')">${dot}${esc(m.display_name)}</button>`;
    })
    .join('');

  destroyChart('radar');

  const getMinMax = (extractor) => {
    const vals = allModels.map(extractor).filter(isNum);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  };

  const axes = [
    (m) => m.bytes_per_token,
    (m) => m.renyi_e_2_5,
    (m) => m.morph_f1,
    (m) => m.identifier_f1,
    (m) => (m.macro_val_bpb ? 1 / m.macro_val_bpb : null),
    (m) => m.decode_mb_per_sec,
  ];
  const ranges = axes.map(getMinMax);

  const normalize = (val, mm) => {
    if (!isNum(val)) return 0.2;
    if (mm.max === mm.min) return 1.0;
    return Math.max(0.1, Math.min(1.0, (val - mm.min) / (mm.max - mm.min)));
  };

  const datasets = state.selectedRadarModels
    .map((mKey) => {
      const m = state.data.models[mKey];
      if (!m) return null;
      const hex = cssVar(`--fam-${state.radarSlots[mKey]}`);
      return {
        label: m.display_name,
        data: axes.map((f, i) => normalize(f(m), ranges[i])),
        backgroundColor: withAlpha(hex, 0.12),
        borderColor: hex,
        borderWidth: 2,
        pointBackgroundColor: hex,
        pointRadius: 3,
      };
    })
    .filter(Boolean);

  const t = chartTheme();

  state.charts.radar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: ['Compression', 'Rényi E₂.₅', 'Morphology F1', 'Identifier F1', 'Downstream (1/BPB)', 'Decode speed'],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: legendStyle(t, 'bottom'),
        tooltip: { ...tooltipStyle(t), callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.r.toFixed(2)}` } },
      },
      scales: {
        r: {
          min: 0,
          max: 1.0,
          ticks: { display: false, stepSize: 0.25 },
          pointLabels: { color: t.text2, font: { size: 11.5, weight: '500' } },
          grid: { color: t.grid },
          angleLines: { color: t.grid },
        },
      },
    },
  });
}

function toggleRadarModel(mKey) {
  if (state.selectedRadarModels.includes(mKey)) {
    if (state.selectedRadarModels.length > 1) {
      state.selectedRadarModels = state.selectedRadarModels.filter((k) => k !== mKey);
      delete state.radarSlots[mKey];
    }
  } else {
    state.selectedRadarModels.push(mKey);
  }
  renderRadarView();
}

// 11. Live Interactive Tokenizer Playground
function loadPlaygroundPreset(type) {
  const input = document.getElementById('playground-input');
  if (input && PRESETS[type]) {
    input.value = PRESETS[type];
    runLiveTokenize();
  }
}

function inspectModelInPlayground(modelKey) {
  switchTab('playground');
  const target = document.getElementById(`tok-card-${modelKey}`);
  if (target) {
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.add('flash');
    setTimeout(() => target.classList.remove('flash'), 1600);
  }
}

async function runLiveTokenize() {
  const inputEl = document.getElementById('playground-input');
  const vocabEl = document.getElementById('playground-vocab-input');
  const loadingEl = document.getElementById('playground-loading');
  const resultsContainer = document.getElementById('playground-results-container');
  const btn = document.getElementById('playground-tokenize-btn');

  const text = inputEl?.value || '';
  const vocabSize = parseInt(vocabEl?.value || '400', 10);

  if (loadingEl) loadingEl.classList.remove('hidden');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/tokenize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, vocab_size: vocabSize }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    const data = await res.json();

    renderPlaygroundResults(data.results, data.raw_bytes);
    if (data.corpus?.synthetic && resultsContainer) {
      resultsContainer.insertAdjacentHTML(
        'afterbegin',
        '<div class="error">Trained on the SYNTHETIC offline test corpus (dashboard started with --offline), not real data.</div>',
      );
    }
  } catch (err) {
    console.error('Tokenize error:', err);
    if (resultsContainer) {
      resultsContainer.innerHTML = `<div class="error">Tokenisation failed: ${esc(err.message)}</div>`;
    }
  } finally {
    if (loadingEl) loadingEl.classList.add('hidden');
    if (btn) btn.disabled = false;
  }
}

function renderPlaygroundResults(results, rawBytes) {
  const container = document.getElementById('playground-results-container');
  if (!container) return;

  if (!results || results.length === 0) {
    container.innerHTML = '<div class="loading">No tokeniser results available.</div>';
    return;
  }

  container.innerHTML = results
    .map((res) => {
      // Make whitespace visible without changing the token's width much
      const pillHtml = res.token_details
        .map((tok) => {
          const piece = esc(tok.piece).replace(/ /g, '<span class="ws">␣</span>').replace(/\n/g, '<span class="ws">↵</span>');
          return `<span class="tk" title="ID ${tok.id} · ${tok.byte_len} bytes">${piece}</span>`;
        })
        .join('');

      return `
      <div id="tok-card-${esc(res.name)}" class="tok-row">
        <div class="tok-head">
          <div class="left">
            <span class="model-name">${esc(res.display_name)}</span>
            ${famLabel(res.family)}
          </div>
          <div class="tok-stats">
            <span><b>${esc(res.num_tokens)}</b> tokens</span>
            <span><b>${esc(res.bytes_per_token)}</b> B/tok</span>
            <span><b>${esc(res.encode_ms)}</b> ms</span>
          </div>
        </div>
        <div class="tokens">${pillHtml}</div>
      </div>`;
    })
    .join('');
}

// Event Listeners Setup
function segmented(selector, pick) {
  document.querySelectorAll(selector).forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll(selector).forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      pick(btn.dataset);
    });
  });
}

function setupEventListeners() {
  // Family Filter Chips
  document.querySelectorAll('.family-filter-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.family-filter-btn').forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      state.activeFamilyFilter = btn.dataset.family;
      renderLeaderboard();
    });
  });

  // Quadrant Selectors
  document.getElementById('quadrant-x-select')?.addEventListener('change', renderQuadrantChart);
  document.getElementById('quadrant-y-select')?.addEventListener('change', renderQuadrantChart);

  // One vocabulary-size switch per tab, kept in sync
  document.querySelectorAll('.vocab-group').forEach((g) => {
    g.innerHTML = VOCABS.map((v) => `<button class="vocab-btn" data-vocab="${v}" aria-pressed="${v === state.vocab}">V = ${v.toLocaleString('en-US')}</button>`).join('');
  });
  segmented('.vocab-btn', (d) => {
    state.vocab = Number(d.vocab);
    document.querySelectorAll('.vocab-btn').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.vocab === d.vocab)));
    renderTokkaTab();
    renderIntrinsicTab();
    renderThroughput();
  });
  segmented('.tokka-metric-btn', (d) => {
    state.tokkaActiveMetric = d.metric;
    renderTokkaTab();
  });
  segmented('.tp-metric-btn', (d) => {
    state.tpMetric = d.metric;
    renderThroughput();
  });
  segmented('.tier-btn', (d) => {
    state.tier = d.tier;
    renderProxy();
  });
  segmented('.swap-mode-btn', (d) => {
    state.swapMode = d.mode;
    renderSwapChart();
  });

  // Intrinsic Search
  document.getElementById('intrinsic-search')?.addEventListener('input', renderIntrinsicTab);

  // Playground Vocab Slider sync
  const vocabSlider = document.getElementById('playground-vocab-slider');
  const vocabInput = document.getElementById('playground-vocab-input');
  if (vocabSlider && vocabInput) {
    vocabSlider.addEventListener('input', (e) => {
      vocabInput.value = e.target.value;
    });
    vocabInput.addEventListener('input', (e) => {
      vocabSlider.value = e.target.value;
    });
  }

  // Ctrl/Cmd+Enter tokenises
  document.getElementById('playground-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runLiveTokenize();
  });

  document.getElementById('refresh-btn')?.addEventListener('click', fetchBenchmarkData);
  document.getElementById('theme-btn')?.addEventListener('click', () => setTheme(isDarkTheme() ? 'light' : 'dark'));
}
