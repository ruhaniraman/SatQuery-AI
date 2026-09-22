// Renders the real dashboard components to HTML (server-side, via Vite's module loader) with
// realistic workspace state. There is no DOM here, so interactions are covered by the browser test
// harness; this catches crashes and wrong content for every state a panel can be in.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

let vite;
const load = (path) => vite.ssrLoadModule(path);
const html = (element) => renderToStaticMarkup(element);
const h = React.createElement;

before(async () => {
  vite = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'silent', configFile: false, plugins: [(await import('@vitejs/plugin-react')).default()] });
});
after(async () => { await vite.close(); });

const file = (name, size = 2048) => new File([new Uint8Array(size)], name);
const slot = (name, modality = 'optical', size, previewState = 'ready') => ({
  file: name ? file(name, size) : null,
  preview: name && previewState === 'ready' ? 'blob:preview' : null,
  previewState: name ? previewState : 'none',
  previewError: previewState === 'failed' ? 'Could not read the file' : null,
  modality,
});

const emptySlots = () => ({ a: slot(null), before: slot(null), after: slot(null), optical: slot(null, 'optical'), sar: slot(null, 'sar') });

function makeWs(overrides = {}) {
  const base = {
    slots: emptySlots(),
    latest: { scan: null, change: null, fusion: null, annotated: null },
    runs: [], reportRun: null, reportRunId: null, selectReportRun() {}, async prepareReport() { return {}; },
    setImage() {}, removeImage() {}, setModality() {},
    viewMode: 'map', setViewMode() {}, activeLayer: 'a', setActiveLayer() {},
    chat: [], isExecuting: false, busy: null,
    sendMessage() {}, runScan() {}, runChange() {}, runFusion() {}, analyzeView() {}, dismissError() {}, stopAnalysis() {},
  };
  const ws = { ...base, ...overrides };
  ws.errorFor = (scope) => (ws.error && ws.error.scope === scope ? ws.error.message : null);
  return ws;
}
const withSlots = (patch) => ({ slots: { ...emptySlots(), ...patch } });
const single = () => withSlots({ a: slot('scene.tif', 'optical', 3 * 1024 * 1024) });

const CHANGE_ANSWER = [
  'WARNING: Image B could not be registered to Image A (only 5 inliers); it was only resized to fit.',
  '',
  'Changed areas',
  '- Likely change, centre (~6% of the scene). Also flagged by the deforestation scan.',
  '    Before: Dense dark green canopy.',
  '    After: A mix of brown and green ground.',
  '    Measured: green-dominant pixels 0% -> 18%; average brightness 63 -> 85 (0-255).',
].join('\n');

const response = (overrides = {}) => ({
  answer: 'Two changes found.',
  visual_evidence_url: '/reports/abc/evidence.png',
  report_download_url: '/reports/abc/report.pdf',
  report_error: null,
  agent_execution_trace: {
    pipeline_id: 'abc-123', task: 'CHANGE_DETECTION', routing: 'rule-based',
    routing_reason: '2 images of the same sensor type (optical + optical)',
    nodes_traversed: ['RuleBasedRouter', 'InputPreprocessor', 'SpatialAlignment', 'TemporalOrdering', 'Hybrid-CDVQA-Engine', 'EvidenceReport'],
    telemetry: {
      model_used: 'Qwen/Qwen2-VL-2B-Instruct (base weights, LoRA adapters disabled)',
      active_adapter: null,
      sar_despeckled: true,
      major_regions_detected: 2,
      loader_notes: ['a', 'b'],
      comparison_details: { 'How this was produced': ['Assembled by code from measurements.'] },
      temporal_order: { basis: 'as-uploaded', reordered: false, summary: 'Capture dates are unknown; treated Image A as BEFORE.', warnings: [] },
    },
    warnings: ['Band order is not declared in the file; assumed the first three bands are R, G, B.'],
    validation_status: 'image payload checks passed', execution_status: 'completed',
  },
  ...overrides,
});

async function runOf(kind, { title, answer, at = Date.now(), adapter } = {}) {
  const { makeRun } = await load('/src/dashboard/utils/runs.js');
  return makeRun({ kind, title, adapter, at, data: response(answer ? { answer } : {}) });
}

// ------------------------------------------------------------------ feature registry

test('the feature registry is well formed, grouped, and drives the nav', async () => {
  const { FEATURES, featureById, GROUP_LABELS } = await load('/src/dashboard/features.js');
  assert.deepEqual(FEATURES.map((f) => f.id), ['imagery', 'assistant', 'scans', 'change', 'fusion', 'report']);
  assert.equal(new Set(FEATURES.map((f) => f.id)).size, FEATURES.length, 'ids are unique');
  for (const f of FEATURES) {
    for (const key of ['id', 'group', 'label', 'title', 'summary']) assert.ok(f[key], `${f.id} has ${key}`);
    assert.ok(f.icon && f.Panel, `${f.id} has icon and Panel`);
    assert.equal(typeof f.layers, 'function', `${f.id} says what the viewer can show`);
    assert.equal(typeof f.evidenceRun, 'function', `${f.id} says whose evidence it shows`);
  }
  assert.deepEqual(FEATURES.filter((f) => f.group === 'single').map((f) => f.id), ['imagery', 'assistant', 'scans']);
  assert.deepEqual(FEATURES.filter((f) => f.group === 'pair').map((f) => f.id), ['change', 'fusion'], 'change detection and fusion are separate features');
  assert.deepEqual(GROUP_LABELS, { single: 'Single', pair: 'Pairs', output: '' });
  assert.equal(featureById('nope').id, 'imagery', 'unknown id falls back to the first feature');

  // Every registered panel renders in its empty state without throwing
  for (const f of FEATURES) assert.doesNotThrow(() => html(h(f.Panel, { ws: makeWs(), goTo() {} })), f.id);
});

test('each feature offers its own viewer layers, and the viewer never lands on an empty one', async () => {
  const { featureById, effectiveLayer } = await load('/src/dashboard/features.js');
  const ws = makeWs({ ...withSlots({ a: slot('a.png'), before: slot('b.png'), after: slot('c.png'), optical: slot('o.png'), sar: slot('s.png', 'sar') }) });
  const labels = (id, run) => featureById(id).layers(ws, run).map((l) => l.label);
  const run = await runOf('change');
  assert.deepEqual(labels('imagery'), ['Image · Optical', 'Evidence', 'Compare']);
  // a scan's evidence is the image with the findings drawn on it: Highlights, and a swipe against the original
  const scanRun = await runOf('scan', { adapter: 'mining' });
  assert.deepEqual(labels('imagery', scanRun), ['Image · Optical', 'Highlights', 'Compare']);
  const compare = (r, w = ws) => featureById('scans').layers(w, r).find((l) => l.value === 'compare').disabled;
  assert.equal(compare(scanRun), false);
  assert.equal(compare(run), true, 'only a scan has highlights to compare');
  assert.equal(compare(scanRun, makeWs(withSlots({ a: slot('a.png', 'optical', 2048, 'loading') }))), true, 'the original must be drawable');
  assert.deepEqual(labels('change', run), ['Before · Optical', 'After · Optical', 'Swipe', 'Changes']);
  assert.deepEqual(labels('fusion', run), ['Optical', 'SAR', 'Composite']);
  assert.deepEqual(labels('report', run), ['Evidence']);

  // evidence is only offered when there is a run with an evidence image
  assert.equal(featureById('change').layers(ws, null).find((l) => l.value === 'evidence').disabled, true);
  assert.equal(featureById('change').layers(ws, run).find((l) => l.value === 'evidence').disabled, false);
  // swipe needs both images
  const oneImage = makeWs(withSlots({ before: slot('b.png') }));
  assert.equal(featureById('change').layers(oneImage, null).find((l) => l.value === 'swipe').disabled, true);

  // layer resolution: keep a usable request, otherwise fall back to the first usable layer
  const layers = featureById('fusion').layers(ws, null);
  assert.equal(effectiveLayer(layers, 'sar'), 'sar');
  assert.equal(effectiveLayer(layers, 'before'), 'optical', 'a layer from another feature falls back');
  assert.equal(effectiveLayer(layers, 'evidence'), 'optical', 'an empty layer falls back');

  // an empty slot layer is still offered (it shows an upload prompt); only evidence/swipe are ever disabled
  const emptyWs = makeWs();
  assert.equal(featureById('change').layers(emptyWs, null)[0].empty, true);
  assert.equal(featureById('change').layers(emptyWs, null)[0].disabled, undefined, 'clickable');
  assert.equal(effectiveLayer(featureById('fusion').layers(emptyWs, null), 'sar'), 'sar', 'an empty slot layer is still shown, as an upload prompt');
  assert.equal(effectiveLayer(featureById('change').layers(emptyWs, null), 'a'), 'before', 'a foreign layer falls back to the first upload');
  assert.equal(effectiveLayer(featureById('report').layers(emptyWs, null), 'a'), null, 'nothing usable at all -> null');

  // each feature says which slots it owns, so the Inputs viewer can upload into them
  const ids = (id) => featureById(id).uploadTargets(emptyWs).map((t) => t.id);
  assert.deepEqual([ids('imagery'), ids('assistant'), ids('scans')], [['a'], ['a'], ['a']]);
  assert.deepEqual(ids('change'), ['before', 'after']);
  assert.deepEqual(ids('fusion'), ['optical', 'sar']);
  assert.deepEqual(ids('report'), []);
});

test('NavRail lists every feature, marks the open one, groups them and shows status badges', async () => {
  const { FEATURES } = await load('/src/dashboard/features.js');
  const NavRail = (await load('/src/dashboard/components/NavRail.jsx')).default;
  const ws = makeWs({ ...single(), runs: [await runOf('scan')] });
  const out = html(h(NavRail, { features: FEATURES, activeId: 'assistant', panelOpen: true, onSelect() {}, ws }));
  for (const f of FEATURES) assert.match(out, new RegExp(`>${f.label}<`));
  assert.equal((out.match(/aria-current="page"/g) || []).length, 1);
  assert.match(out, />Single</);
  assert.match(out, />Pairs</);
  assert.equal((out.match(/aria-hidden="true" class="mx-2 my-1 border-t/g) || []).length, 2, 'a divider before Pairs and before Report');
  assert.match(out, /rounded-full bg-blue-400/, 'ready badge on Imagery and Report');

  const collapsed = html(h(NavRail, { features: FEATURES, activeId: 'assistant', panelOpen: false, onSelect() {}, ws }));
  assert.equal((collapsed.match(/aria-current/g) || []).length, 0, 'nothing is "current" when the panel is collapsed');

  const busy = html(h(NavRail, { features: FEATURES, activeId: 'imagery', panelOpen: true, onSelect() {}, ws: makeWs({ isExecuting: true, busy: 'scan:mining' }) }));
  assert.match(busy, /animate-spin/);
  assert.match(busy, /aria-label="Working"/, 'the running feature\'s own icon is replaced by the spinner, not just badged');
  const busyChange = html(h(NavRail, { features: FEATURES, activeId: 'imagery', panelOpen: true, onSelect() {}, ws: makeWs({ isExecuting: true, busy: 'change' }) }));
  assert.match(busyChange, /animate-spin/);
  // only the busy feature's icon becomes the spinner; a non-busy feature (Report) keeps its own icon
  assert.equal((busy.match(/aria-label="Working"/g) || []).length, 1);
});

// ------------------------------------------------------------------ imagery

test('ImageryPanel: one image slot with a sensor toggle, and pointers to the two-image features', async () => {
  const ImageryPanel = (await load('/src/dashboard/components/panels/ImageryPanel.jsx')).default;
  const out = html(h(ImageryPanel, { ws: makeWs(), goTo() {} }));
  assert.match(out, />Image</);
  assert.match(out, /Drop an image or click to browse/);
  assert.match(out, /data-slot="a"/);
  assert.doesNotMatch(out, /data-slot="b"/, 'the second image belongs to the pair features now');
  assert.match(out, /aria-label="Image sensor type"/);
  assert.match(out, />Change detection</);
  assert.match(out, />Optical–SAR fusion</);
});

test('ImageryPanel: shows file details, GeoTIFF badge and the chosen sensor', async () => {
  const ImageryPanel = (await load('/src/dashboard/components/panels/ImageryPanel.jsx')).default;
  const out = html(h(ImageryPanel, { ws: makeWs(withSlots({ a: slot('scene.tif', 'sar', 3 * 1024 * 1024) })), goTo() {} }));
  assert.match(out, /scene\.tif/);
  assert.match(out, /3\.0 MB · GeoTIFF/);
  assert.match(out, /aria-label="Replace Image"/);
  assert.match(out, /aria-label="Remove Image"/);
  assert.match(out, /aria-pressed="true"[^>]*>SAR</);
});

test('ImageSlot: a GeoTIFF shows a spinner while it renders, and a readable note if that fails', async () => {
  const ImageSlot = (await load('/src/dashboard/components/ImageSlot.jsx')).default;
  const props = { id: 'a', title: 'Image', onFile() {}, onRemove() {} };
  const loading = html(h(ImageSlot, { ...props, slot: slot('scene.tif', 'optical', 10, 'loading') }));
  assert.match(loading, /aria-label="Rendering preview"/);
  assert.match(loading, /animate-spin/);
  const failed = html(h(ImageSlot, { ...props, slot: slot('scene.tif', 'optical', 10, 'failed') }));
  assert.match(failed, /Preview unavailable: Could not read the file/);
  assert.match(failed, /can still be analysed/);
  const png = html(h(ImageSlot, { ...props, slot: slot('photo.png') }));
  assert.match(png, /<img[^>]*src="blob:preview"/);
  assert.doesNotMatch(png, /Preview unavailable/);
  const fixed = html(h(ImageSlot, { ...props, sensor: 'fixed', slot: slot(null, 'sar') }));
  assert.match(fixed, />SAR</);
  assert.doesNotMatch(fixed, /aria-pressed/);
});

// ------------------------------------------------------------------ assistant

test('AssistantPanel: guides the user to add imagery first', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const out = html(h(AssistantPanel, { ws: makeWs(), goTo() {} }));
  assert.match(out, /Add an image to start/);
  assert.match(out, /Open Imagery/);
  assert.match(out, /<button[^>]*disabled[^>]*aria-label="Send"/s);
});

test('AssistantPanel: suggestions when ready, and every message role renders', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const ready = html(h(AssistantPanel, { ws: makeWs(single()), goTo() {} }));
  assert.match(ready, /Hi! I&#x27;m the SatQuery-AI assistant/, 'the greeting shows before any real conversation');
  assert.match(ready, /What land cover is visible\?/);

  const chat = [
    { role: 'system', content: 'Initiating mining scan...' },
    { role: 'ai', content: 'Scan complete. Found 2 potential region(s).' },
    { role: 'user', content: 'and the north side?' },
  ];
  const out = html(h(AssistantPanel, { ws: makeWs({ ...single(), chat }), goTo() {} }));
  assert.match(out, /Initiating mining scan/);
  assert.match(out, /Found 2 potential region/);
  assert.match(out, /and the north side\?/);
  assert.doesNotMatch(out, /Hi! I&#x27;m the SatQuery-AI assistant/, 'the greeting disappears once there is a real conversation');
  assert.doesNotMatch(out, /What land cover is visible\?/, 'suggestions disappear once there is a conversation');
});

test('AssistantPanel keeps two-image results out of the chat (they have their own panels)', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const chat = [
    { role: 'user', content: 'what is here?' },
    { role: 'user', content: 'Summarize the key changes.', scope: 'pair' },
    { role: 'ai', content: 'Changed areas - Major change, right', scope: 'pair' },
  ];
  const out = html(h(AssistantPanel, { ws: makeWs({ ...single(), chat }), goTo() {} }));
  assert.match(out, /what is here\?/);
  assert.doesNotMatch(out, /Summarize the key changes|Major change/);
});

test('AssistantPanel: busy state, and only ITS errors are shown', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const busy = html(h(AssistantPanel, { ws: makeWs({ ...single(), isExecuting: true }), goTo() {} }));
  assert.match(busy, /Analysing/);
  assert.match(busy, /<textarea[^>]*disabled/);
  assert.match(busy, /aria-label="Stop analysing"/, 'the send button becomes a stop button while running');
  assert.doesNotMatch(busy, /aria-label="Send"/);
  const idle = html(h(AssistantPanel, { ws: makeWs(single()), goTo() {} }));
  assert.match(idle, /aria-label="Send"/);
  assert.doesNotMatch(idle, /aria-label="Stop analysing"/);
  const err = html(h(AssistantPanel, { ws: makeWs({ ...single(), error: { message: 'The AI model is not available: X', scope: 'assistant' } }), goTo() {} }));
  assert.match(err, /role="alert"/);
  assert.match(err, /The AI model is not available: X/);
  const other = html(h(AssistantPanel, { ws: makeWs({ ...single(), error: { message: 'A scan failed', scope: 'scans' } }), goTo() {} }));
  assert.doesNotMatch(other, /A scan failed/, 'a scan error does not leak into the chat');
});

test('chat bubbles format assistant answers but leave the user\'s own text alone', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const chat = [
    { role: 'user', content: 'ASSESSMENT: is this mine?' },
    { role: 'ai', content: 'OBSERVATIONS: Farmland. ASSESSMENT: Mostly agricultural.' },
  ];
  const out = html(h(AssistantPanel, { ws: makeWs({ ...single(), chat }), goTo() {} }));
  assert.equal((out.match(/uppercase tracking-wider text-blue-300/g) || []).length, 2, 'only the assistant message is reformatted');
  assert.match(out, /ASSESSMENT: is this mine\?/);
});

// ------------------------------------------------------------------ scans

test('ScansPanel: three scans, disabled without imagery, latest result with evidence link', async () => {
  const { default: ScansPanel, SCANS } = await load('/src/dashboard/components/panels/ScansPanel.jsx');
  assert.deepEqual(SCANS.map((s) => s.id), ['mining', 'agriculture', 'deforestation']);
  const empty = html(h(ScansPanel, { ws: makeWs(), goTo() {} }));
  assert.match(empty, /Add an image first/);
  assert.equal((empty.match(/disabled=""/g) || []).length, 3);

  const scan = await runOf('scan', { adapter: 'mining', answer: 'Scan complete. Found 2 potential region(s).' });
  scan.data.answer = 'Scan complete. Found 2 potential region(s).';
  const ws = makeWs({ ...single(), latest: { scan, change: null, fusion: null, annotated: scan }, runs: [scan] });
  const out = html(h(ScansPanel, { ws, goTo() {} }));
  assert.match(out, /Mining scan/);
  assert.match(out, /Found 2 potential region/);
  assert.match(out, /View highlights/);
  assert.equal((out.match(/disabled=""/g) || []).length, 0);

  const running = html(h(ScansPanel, { ws: makeWs({ ...single(), isExecuting: true, busy: 'scan:agriculture' }), goTo() {} }));
  assert.match(running, /Scanning/);
  const failed = html(h(ScansPanel, { ws: makeWs({ ...single(), error: { message: 'Scan blew up', scope: 'scans' } }), goTo() {} }));
  assert.match(failed, /Scan blew up/);
});

// ------------------------------------------------------------------ change detection


// ------------------------------------------------------------------ scan results

const SCAN = {
  adapter: 'mining', subject: 'surface mining activity', level: 'high', threshold: 0.8, coverage_pct: 50, faint_areas: 2,
  headline: 'Surface mining activity found in 2 areas (high confidence), covering about 50% of the view.',
  note: 'The scan answered yes for 10 of 16 areas, which is more than one feature usually covers, so it may be over-reporting.',
  findings: [
    { id: 1, label: 'High confidence', where: 'upper-left', coverage_pct: 25, confidence: 89, peak: 91, box: [0, 0, 1, 0.25], tiles: [[0, 0]] },
    { id: 2, label: 'Likely', where: 'lower-right', coverage_pct: 25, confidence: 82, peak: 85, box: [0.25, 0.5, 1, 1], tiles: [[1, 2]] },
  ],
  grid: [[0.84, 0.41, 0.15, 0.12], [0.91, 0.25, 0.88, 0.27], [0.91, 0.59, 0.97, 0.35], [0.82, 0.62, 0.88, 0.85]],
};

test('ScanResult: answer first, then numbered areas, then how it was produced', async () => {
  const ScanResult = (await load('/src/dashboard/components/ScanResult.jsx')).default;
  const out = html(h(ScanResult, { scan: SCAN, link: 'https://www.openstreetmap.org/#map=15/1/2', onShow() {} }));
  assert.ok(out.indexOf('High confidence') < out.indexOf('Surface mining activity found in 2 areas'), 'confidence chip, then headline');
  assert.ok(out.indexOf('Surface mining activity found') < out.indexOf('upper-left'), 'headline before the areas');
  assert.ok(out.indexOf('lower-right') < out.indexOf('How this was produced'), 'areas before the details');
  assert.match(out, /Check this one by eye/);                     // the over-reporting warning
  assert.match(out, /over-reporting/);
  assert.equal((out.match(/Show area \d on the image/g) || []).length, 2);
  assert.match(out, /Copy summary/);
  assert.match(out, /href="https:\/\/www\.openstreetmap\.org\/#map=15\/1\/2"[^>]*>.*Open on map/s);
  assert.equal((out.match(/<span[^>]*rounded text-\[10px\]/g) || []).length, 16, 'all 16 scores are shown');
  assert.match(out, /Areas at 80% or more are outlined/);
  assert.match(out, /not measured against ground truth/);
  assert.doesNotMatch(out, /adapter|LoRA|logit|tile/i, 'no internal jargon');
  assert.doesNotMatch(out, /\[object Object\]|NaN|undefined/);
});

test('ScanResult: no map link without a map capture, no Show buttons without a viewer, nothing found reads plainly', async () => {
  const ScanResult = (await load('/src/dashboard/components/ScanResult.jsx')).default;
  const plain = html(h(ScanResult, { scan: SCAN }));
  assert.doesNotMatch(plain, /Open on map/);
  assert.doesNotMatch(plain, /Show area/);
  const none = { ...SCAN, level: 'none', findings: [], note: '', headline: 'No confident surface mining activity found. 2 faint areas scored above chance and are tinted lightly.' };
  const out = html(h(ScanResult, { scan: none }));
  assert.match(out, /Nothing confident/);
  assert.match(out, /No confident surface mining activity found/);
  assert.doesNotMatch(out, /Areas found|Check this one by eye/);
  assert.equal(html(h(ScanResult, { scan: null })), '');
});

test('ScanResult compact (the map card) is short: chip, headline, warning, one line per area, no buttons', async () => {
  const ScanResult = (await load('/src/dashboard/components/ScanResult.jsx')).default;
  const out = html(h(ScanResult, { scan: SCAN, compact: true, onShow() {}, link: 'https://x.test' }));
  assert.match(out, /High confidence/);
  assert.match(out, /Surface mining activity found in 2 areas/);
  assert.match(out, /Likely · lower-right/);
  assert.doesNotMatch(out, /<button|Copy summary|How this was produced/);
});

test('ScansPanel shows the structured result when the backend sent one, the plain answer when it did not', async () => {
  const { default: ScansPanel } = await load('/src/dashboard/components/panels/ScansPanel.jsx');
  const scan = await runOf('scan', { adapter: 'mining' });
  scan.data.scan = SCAN;
  scan.data.answer = SCAN.headline;
  scan.meta = { bounds: [[30.7, 76.75], [30.72, 76.8]] };
  const ws = makeWs({ ...single(), latest: { scan, change: null, fusion: null, annotated: scan }, runs: [scan], focusArea() {} });
  const out = html(h(ScansPanel, { ws, goTo() {} }));
  assert.match(out, /Mining · latest result|Mining &middot; latest result/);
  assert.match(out, /Show area 1 on the image/);
  assert.match(out, /Open on map/);                                // a map capture knows where it was taken
  const older = await runOf('scan', { adapter: 'mining', answer: 'Scan complete. Found 2 potential region(s).' });
  const oldOut = html(h(ScansPanel, { ws: makeWs({ ...single(), latest: { scan: older, change: null, fusion: null, annotated: older }, runs: [older] }), goTo() {} }));
  assert.match(oldOut, /Found 2 potential region/);
  assert.doesNotMatch(oldOut, /Show area/);
});

test('InputsCanvas: the Compare layer swipes original against highlights, the Evidence layer takes the focus', async () => {
  const InputsCanvas = (await load('/src/dashboard/components/InputsCanvas.jsx')).default;
  const run = await runOf('scan', { adapter: 'mining' });
  const ws = makeWs({ ...single(), focus: { box: [0, 0, 0.5, 0.5], token: 1 } });
  const compare = html(h(InputsCanvas, { ws, layer: 'compare', evidenceRun: run, targets: [] }));
  assert.match(compare, /Swipe to compare the original with the highlights/);
  assert.match(compare, /Original/);
  assert.match(compare, /Highlights/);
  const evidence = html(h(InputsCanvas, { ws, layer: 'evidence', evidenceRun: run, targets: [] }));
  assert.match(evidence, /Analysis evidence/);
  const none = html(h(InputsCanvas, { ws: makeWs(), layer: 'compare', evidenceRun: run, targets: [] }));
  assert.doesNotMatch(none, /Swipe to compare the original/);        // no original to compare with
});

test('ChangePanel: its own Before / After inputs, a sensor toggle, and a disabled run button until both exist', async () => {
  const ChangePanel = (await load('/src/dashboard/components/panels/ChangePanel.jsx')).default;
  const empty = html(h(ChangePanel, { ws: makeWs(), goTo() {} }));
  assert.match(empty, /data-slot="before"/);
  assert.match(empty, /data-slot="after"/);
  assert.match(empty, /aria-label="Sensor type of both images"/);
  assert.match(empty, /Add Before and After to compare/);
  assert.match(empty, /<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Run change detection/s);

  const ready = html(h(ChangePanel, { ws: makeWs(withSlots({ before: slot('a.png'), after: slot('b.png') })), goTo() {} }));
  assert.doesNotMatch(ready, /Add both the Before/);
  assert.doesNotMatch(ready, /<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Run change detection/s);

  const running = html(h(ChangePanel, { ws: makeWs({ ...withSlots({ before: slot('a.png'), after: slot('b.png') }), isExecuting: true, busy: 'change' }), goTo() {} }));
  assert.match(running, /aria-busy="true"/);
  assert.match(running, />Analysing</);
});

test('ChangePanel: mixing sensor types is explained, not silently allowed', async () => {
  const ChangePanel = (await load('/src/dashboard/components/panels/ChangePanel.jsx')).default;
  const out = html(h(ChangePanel, { ws: makeWs(withSlots({ before: slot('a.png', 'optical'), after: slot('b.png', 'sar') })), goTo() {} }));
  assert.match(out, /different sensor types/);
  assert.match(out, /use Optical–SAR fusion/);
  assert.match(out, /<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Run change detection/s);
});

test('ChangePanel: the latest result is laid out as warning + area cards, and only its own errors show', async () => {
  const ChangePanel = (await load('/src/dashboard/components/panels/ChangePanel.jsx')).default;
  const run = await runOf('change', { answer: CHANGE_ANSWER });
  const ws = makeWs({
    ...withSlots({ before: slot('a.png'), after: slot('b.png') }), runs: [run], latest: { scan: null, change: run, fusion: null, annotated: run },
    error: { message: 'Scan blew up', scope: 'scans' },
  });
  const out = html(h(ChangePanel, { ws, goTo() {} }));
  assert.match(out, /could not be registered/);
  assert.match(out, />Likely change</);
  assert.match(out, /~6% of the scene/);
  assert.match(out, /deforestation<\/span> scan/);
  assert.match(out, />Before</);
  assert.match(out, /Dense dark green canopy\./);
  assert.match(out, /green-dominant pixels 0% -&gt; 18%/);
  assert.match(out, /View evidence/);
  assert.match(out, /Swipe before \/ after/);
  assert.doesNotMatch(out, /Scan blew up/);
  const own = html(h(ChangePanel, { ws: { ...ws, errorFor: () => 'Comparison exploded' }, goTo() {} }));
  assert.match(own, /Comparison failed/);
  assert.match(own, /Comparison exploded/);
});

test('ChangeAnswer: an empty result, a warned-empty result and an old model-written answer', async () => {
  const ChangeAnswer = (await load('/src/dashboard/components/ChangeAnswer.jsx')).default;
  const none = html(h(ChangeAnswer, { text: 'Changed areas\n- No changed area was found.' }));
  assert.match(none, /No changed area was found\./);
  const warned = html(h(ChangeAnswer, { text: 'WARNING: The two images differ strongly in level of detail.\n\nChanged areas\n- No changed area was flagged, but the warning above means this is NOT evidence that nothing changed.' }));
  assert.match(warned, /differ strongly in level of detail/);
  assert.match(warned, /NOT evidence that nothing changed/);
  const legacy = html(h(ChangeAnswer, { text: 'OBSERVATIONS: A pit grew. ASSESSMENT: More excavation.' }));
  assert.match(legacy, />OBSERVATIONS</);
  assert.doesNotThrow(() => html(h(ChangeAnswer, { text: null })));
  const major = html(h(ChangeAnswer, { text: 'Changed areas\n- Major change, right (~18% of the scene). Also flagged by the mining and agriculture scans.' }));
  assert.match(major, /text-red-300/);
  assert.match(major, /mining, agriculture<\/span> scans/);
});

// ------------------------------------------------------------------ fusion

test('FusionPanel: an Optical and a SAR input with fixed sensor types, disabled until both exist', async () => {
  const FusionPanel = (await load('/src/dashboard/components/panels/FusionPanel.jsx')).default;
  const empty = html(h(FusionPanel, { ws: makeWs(), goTo() {} }));
  assert.match(empty, /data-slot="optical"/);
  assert.match(empty, /data-slot="sar"/);
  assert.match(empty, />Optical image</);
  assert.match(empty, />SAR image</);
  assert.match(empty, /Add both images to fuse them/);
  assert.match(empty, /<button[^>]*disabled=""[^>]*>(?:(?!<\/button>).)*Run fusion/s);
  assert.doesNotMatch(empty, /aria-pressed/, 'no sensor toggle: the slot decides');

  const run = await runOf('fusion', { answer: 'A river crosses the scene.' });
  run.data.answer = 'A river crosses the scene.';
  const ws = makeWs({ ...withSlots({ optical: slot('o.png'), sar: slot('s.png', 'sar') }), runs: [run], latest: { scan: null, change: null, fusion: run, annotated: run } });
  const out = html(h(FusionPanel, { ws, goTo() {} }));
  assert.match(out, /A river crosses the scene\./);
  assert.match(out, /View composite/);
  assert.doesNotMatch(out, /Add both an optical/);
});

// ------------------------------------------------------------------ report

test('ReportPanel: empty state', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  assert.match(html(await signedIn(null, h(ReportPanel, { ws: makeWs() }))), /No report yet/);
});

test('ReportPanel renders a realistic trace, including objects, booleans and lists, without crashing', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  const run = await runOf('change', { title: 'Change detection' });
  const out = html(await signedIn(null, h(ReportPanel, { ws: makeWs({ runs: [run], reportRun: run }) })));
  assert.match(out, /Change detection/);                                   // task, humanised
  assert.match(out, /2 images of the same sensor type/);
  assert.match(out, /Rule Based Router/);                                  // stages, humanised
  assert.match(out, /Hybrid CDVQA Engine/);
  assert.match(out, /Capture dates are unknown; treated Image A as BEFORE\./); // dict telemetry -> its summary
  assert.match(out, /Sar despeckled/);                                     // boolean key shown...
  assert.match(out, />yes</);                                              // ...with a visible value
  assert.match(out, /a; b/);                                               // list telemetry
  assert.match(out, /assumed the first three bands/);                      // warning surfaced
  assert.match(out, /Download PDF report/);
  assert.doesNotMatch(out, /\[object Object\]/);
  assert.doesNotMatch(out, /Comparison details|How this was produced/i, 'the supporting data is printed in the PDF, not in this table');
});

test('ReportPanel shows which evidence the report uses and lets any run be picked', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  const scan = await runOf('scan', { title: 'Mining scan', at: 1000 });
  const chat = await runOf('chat', { title: 'What is here?', at: 2000 });
  const runs = [chat, scan];                                                // newest first
  const auto = html(await signedIn(null, h(ReportPanel, { ws: makeWs({ runs, reportRun: scan, reportRunId: null }) })));
  assert.match(auto, /Using the latest annotated evidence: “Mining scan”/);
  assert.match(auto, /Plain questions never replace it/);
  assert.match(auto, /src="http:\/\/localhost:8000\/reports\/abc\/evidence\.png"/);
  assert.match(auto, />Mining scan</);
  assert.match(auto, />What is here\?</);
  assert.match(auto, /annotated evidence/);
  assert.match(auto, /plain answer/);
  assert.equal((auto.match(/aria-label="Used in the report"/g) || []).length, 1, 'exactly one run is marked as used');
  assert.doesNotMatch(auto, /Automatic/, 'no reset link while nothing was picked');

  const picked = html(await signedIn(null, h(ReportPanel, { ws: makeWs({ runs, reportRun: chat, reportRunId: chat.id }) })));
  assert.match(picked, /Using the evidence of “What is here\?”, as you chose\./);
  assert.match(picked, /Automatic/);
});

// ------------------------------------------------------------------ inputs canvas

test('InputsCanvas: empty, image, TIFF states, evidence and swipe layers', async () => {
  const InputsCanvas = (await load('/src/dashboard/components/InputsCanvas.jsx')).default;
  const noTargets = html(h(InputsCanvas, { ws: makeWs(), layer: null, evidenceRun: null }));
  assert.match(noTargets, /Nothing to show yet/);
  assert.doesNotMatch(noTargets, /Add an image|data-canvas-upload/);
  const empty = html(h(InputsCanvas, { ws: makeWs(), layer: 'a', evidenceRun: null, targets: [{ id: 'a', label: 'Image' }], onFile() {} }));
  assert.match(empty, /Add an image/);
  assert.match(empty, /aria-label="Add Image"/);
  assert.match(empty, /drop it anywhere on this view/);
  assert.match(empty, /data-canvas-upload/);

  const png = html(h(InputsCanvas, { ws: makeWs(withSlots({ a: slot('photo.png') })), layer: 'a', evidenceRun: null, onAddImagery() {} }));
  assert.match(png, /<img[^>]*src="blob:preview"/);
  assert.match(png, /title="photo\.png"/);
  assert.match(png, /aria-label="Zoom in"/, 'images can be zoomed');

  const rendering = html(h(InputsCanvas, { ws: makeWs(withSlots({ a: slot('scene.tif', 'optical', 10, 'loading') })), layer: 'a', evidenceRun: null, onAddImagery() {} }));
  assert.match(rendering, /Rendering a preview of this GeoTIFF/);
  assert.doesNotMatch(rendering, /<img/);
  const failed = html(h(InputsCanvas, { ws: makeWs(withSlots({ a: slot('scene.tif', 'optical', 10, 'failed') })), layer: 'a', evidenceRun: null, onAddImagery() {} }));
  assert.match(failed, /Preview unavailable/);
  assert.match(failed, /Could not read the file/);

  const run = await runOf('scan', { title: 'Mining scan' });
  const ev = html(h(InputsCanvas, { ws: makeWs(single()), layer: 'evidence', evidenceRun: run, onAddImagery() {} }));
  assert.match(ev, /src="http:\/\/localhost:8000\/reports\/abc\/evidence\.png"/);
  assert.match(ev, /title="Mining scan"/);

  const both = withSlots({ before: slot('a.png'), after: slot('b.png') });
  const swipe = html(h(InputsCanvas, { ws: makeWs(both), layer: 'swipe', evidenceRun: null, onAddImagery() {} }));
  assert.match(swipe, /role="slider"/);
  assert.match(swipe, /aria-valuenow="50"/);
  assert.match(swipe, />Before</);
  assert.match(swipe, />After</);
  const waiting = html(h(InputsCanvas, { ws: makeWs(withSlots({ before: slot('a.png'), after: slot('b.tif', 'optical', 10, 'loading') })), layer: 'swipe', evidenceRun: null, onAddImagery() {} }));
  assert.doesNotMatch(waiting, /role="slider"/, 'no slider until both previews are ready');
  assert.match(waiting, /Rendering a preview/);
});

test('InputsCanvas: images can be added from the Inputs view itself', async () => {
  const InputsCanvas = (await load('/src/dashboard/components/InputsCanvas.jsx')).default;
  const pair = [{ id: 'before', label: 'Before' }, { id: 'after', label: 'After' }];
  const one = makeWs(withSlots({ before: slot('a.png') }));

  // the empty After layer offers ONLY its own upload
  const afterLayer = html(h(InputsCanvas, { ws: one, layer: 'after', evidenceRun: null, targets: pair, onFile() {} }));
  assert.match(afterLayer, /aria-label="Add After"/);
  assert.doesNotMatch(afterLayer, /aria-label="Add Before"/);

  // with nothing on screen but the feature has empty slots, every empty one gets a button
  const both = html(h(InputsCanvas, { ws: makeWs(), layer: null, evidenceRun: null, targets: pair, onFile() {} }));
  assert.match(both, /aria-label="Add Before"/);
  assert.match(both, /aria-label="Add After"/);

  // a shown image can be replaced; evidence and swipe are not replaceable
  const shown = html(h(InputsCanvas, { ws: one, layer: 'before', evidenceRun: null, targets: pair, onFile() {} }));
  assert.match(shown, />Replace</);
  assert.match(shown, /data-canvas-upload/);
  const run = await runOf('change');
  const evidence = html(h(InputsCanvas, { ws: one, layer: 'evidence', evidenceRun: run, targets: pair, onFile() {} }));
  assert.doesNotMatch(evidence, />Replace</);
  const noUpload = html(h(InputsCanvas, { ws: one, layer: 'before', evidenceRun: null, targets: [], onFile() {} }));
  assert.doesNotMatch(noUpload, />Replace<|data-canvas-upload/, 'a feature without slots (Report) offers no upload');
});

// ------------------------------------------------------------------ viewer widgets

test('ZoomPan starts fitted: zoom-out and fit are disabled, the level is 100%', async () => {
  const ZoomPan = (await load('/src/dashboard/components/ZoomPan.jsx')).default;
  const out = html(h(ZoomPan, null, h('div', null, 'content')));
  assert.match(out, /content/);
  assert.match(out, />100%</);
  assert.match(out, /<button[^>]*disabled=""[^>]*aria-label="Zoom out"/);
  assert.match(out, /<button[^>]*disabled=""[^>]*aria-label="Fit to view"/);
  assert.match(out, /aria-label="Zoom in"/);
  assert.doesNotMatch(out, /<button[^>]*disabled=""[^>]*aria-label="Zoom in"/);
});

test('StatusPill says what the backend is doing', async () => {
  const { default: StatusPill, describeStatus } = await load('/src/dashboard/components/StatusPill.jsx');
  assert.deepEqual(describeStatus({ state: 'ready' }, false), { tone: 'ready', text: 'Agent ready' });
  assert.deepEqual(describeStatus({ state: 'ready' }, true), { tone: 'busy', text: 'Analysing' });
  assert.deepEqual(describeStatus({ state: 'loading' }, false), { tone: 'warn', text: 'Model loading' });
  assert.deepEqual(describeStatus({ state: 'model-error' }, false), { tone: 'error', text: 'Model unavailable' });
  assert.deepEqual(describeStatus({ state: 'offline' }, true), { tone: 'error', text: 'Backend offline' }, 'a dead backend beats "analysing"');
  assert.deepEqual(describeStatus({ state: 'checking' }, false), { tone: 'idle', text: 'Checking' });
  const out = html(h(StatusPill, { health: { state: 'offline', message: 'Cannot reach the backend at http://localhost:8000.' }, executing: false, onRefresh() {} }));
  assert.match(out, /Backend offline/);
  assert.match(out, /title="Cannot reach the backend at http:\/\/localhost:8000\. Click to check again\."/);
  assert.match(out, /border-red-800\/40/);
});

test('ThemeToggle names the theme it switches to', async () => {
  const ThemeToggle = (await load('/src/dashboard/components/ThemeToggle.jsx')).default;
  assert.match(html(h(ThemeToggle, { theme: 'dark', onChange() {} })), /aria-label="Switch to light theme"/);
  assert.match(html(h(ThemeToggle, { theme: 'light', onChange() {} })), /aria-label="Switch to dark theme"/);
});

test('CoordinateBox: a labelled input and Go, and a remove button only once there is a marker', async () => {
  const CoordinateBox = (await load('/src/dashboard/components/CoordinateBox.jsx')).default;
  const idle = html(h(CoordinateBox, { onGo() {}, onClear() {}, hasPin: false }));
  assert.match(idle, /aria-label="Search a place or coordinates"/);
  assert.match(idle, /placeholder="Search a place or Lat, Lng"/);
  assert.match(idle, /aria-label="Search"/);
  assert.doesNotMatch(idle, /aria-label="Clear the search/);
  assert.match(html(h(CoordinateBox, { onGo() {}, onClear() {}, hasPin: true })), /aria-label="Clear the search and remove the marker"/);
});

test('FormattedAnswer labels OBSERVATIONS / ASSESSMENT and copes with run-on model output', async () => {
  const { FormattedAnswer } = await load('/src/dashboard/components/ui.jsx');
  const split = html(h(FormattedAnswer, { text: 'OBSERVATIONS: A river.\nASSESSMENT: About a quarter is water.' }));
  assert.equal((split.match(/uppercase tracking-wider text-blue-300/g) || []).length, 2, 'both sections labelled');
  assert.match(split, />OBSERVATIONS<.*A river\./s);
  assert.match(split, />ASSESSMENT<.*About a quarter is water\./s);

  // A model that puts everything on one line still gets two sections
  const runOn = html(h(FormattedAnswer, { text: 'OBSERVATIONS: A river. ASSESSMENT: Mostly water.' }));
  assert.equal((runOn.match(/<p>/g) || []).length, 2);
  assert.equal((runOn.match(/uppercase tracking-wider text-blue-300/g) || []).length, 2);

  // Plain text (a scan summary, an error-ish reply) is left alone; empty input is safe
  const plain = html(h(FormattedAnswer, { text: 'Scan complete. Found 2 potential region(s).' }));
  assert.match(plain, /Scan complete\. Found 2 potential region\(s\)\./);
  assert.doesNotMatch(plain, /text-blue-300/);
  assert.doesNotThrow(() => html(h(FormattedAnswer, { text: null })));
  assert.doesNotThrow(() => html(h(FormattedAnswer, { text: '' })));
});

// ------------------------------------------------------------------ buttons

test('a disabled or busy Button never gets a hover colour (it used to turn black on hover)', async () => {
  const { Button } = await load('/src/dashboard/components/ui.jsx');
  const primary = html(h(Button, { variant: 'primary', disabled: true }, 'Go'));
  assert.doesNotMatch(primary, /disabled:hover:bg-inherit/);
  assert.match(primary, /enabled:hover:bg-blue-500/, 'hover styles are limited to enabled buttons');
  assert.doesNotMatch(primary, /(^|\s)hover:bg-/);
  for (const variant of ['subtle', 'ghost', 'amber', 'emerald', 'rose']) {
    const out = html(h(Button, { variant, disabled: true }, 'x'));
    assert.match(out, /enabled:hover:bg-/, variant);
    assert.doesNotMatch(out, /(^|\s)hover:bg-/, variant);
  }
  const busy = html(h(Button, { loading: true, disabled: true }, 'Analysing'));
  assert.match(busy, /aria-busy="true"/);
  assert.match(busy, /cursor-progress/);
  assert.doesNotMatch(busy, /disabled:opacity-40/, 'a busy button is not dimmed like an unavailable one');
  const idle = html(h(Button, null, 'Go'));
  assert.doesNotMatch(idle, /aria-busy/);
  assert.match(idle, /cursor-pointer/);
});

test('Segmented options only get a hover colour when enabled', async () => {
  const { Segmented } = await load('/src/dashboard/components/ui.jsx');
  const out = html(h(Segmented, { label: 'x', value: 'a', onChange() {}, options: [{ value: 'a', label: 'A' }, { value: 'b', label: 'B', disabled: true }] }));
  assert.match(out, /enabled:hover:bg-white\/5/);
  assert.doesNotMatch(out, /(^|\s)hover:bg-white\/5/);
});

// ------------------------------------------------------------------ on-map ask bar

const mapAskProps = (over = {}) => ({ ws: makeWs(), layer: 'optical', capture() {}, overlay: null, onOverlay() {}, onClearOverlay() {}, onToggleOverlay() {}, onOpenFeature() {}, ...over });

test('MapAsk: idle bar has a prompt box, one scan chip per feature scan, and Save view as', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const { SCANS } = await load('/src/dashboard/components/panels/ScansPanel.jsx');
  const out = html(h(MapAsk, mapAskProps()));
  assert.match(out, /aria-label="Ask about this map view"/);
  assert.match(out, /<button[^>]*disabled[^>]*aria-label="Ask about this view"/s, 'send is disabled until something is typed');
  assert.match(out, /Scan this view:/);
  for (const scan of SCANS) assert.match(out, new RegExp(`aria-label="Scan this view for ${scan.title}"`));
  assert.doesNotMatch(out, /disabled=""[^>]*aria-label="Scan this view for/);
  assert.match(out, /Save view as/);
  assert.doesNotMatch(out, /role="menu"/, 'the menu is closed until opened');
});

test('MapAsk: while a request is running the controls lock and the status says why', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const out = html(h(MapAsk, mapAskProps({ ws: makeWs({ isExecuting: true }) })));
  assert.match(out, /Analysing/);
  assert.match(out, /<input[^>]*disabled/);
  assert.doesNotMatch(out, /Scan this view for/, 'scan chips are replaced by the status while busy');
});

test('MapAsk on the SAR layer: asks are tagged SAR and the optical-only scans are disabled', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const out = html(h(MapAsk, mapAskProps({ layer: 'sar' })));
  assert.match(out, /placeholder="Ask about this SAR view…"/);
  assert.match(out, /Feature scans are optical-only/);
  assert.equal((out.match(/disabled=""[^>]*title="The feature scans were built for optical imagery"/g) || []).length, 3, 'all three scans are disabled');
  assert.match(out, /Save view as/);
});

// ------------------------------------------------------------------ round 9: map captures, timeline, SAR dialog

const RELEASES = [
  { id: '31144', date: '2014-06-11', template: 'https://t/31144/{z}/{y}/{x}' },
  { id: '32337', date: '2018-05-16', template: 'https://t/32337/{z}/{y}/{x}' },
  { id: '26334', date: '2026-08-05', template: 'https://t/26334/{z}/{y}/{x}' },
];
const capture = (date, bounds = [[23.7, 86.3], [23.9, 86.5]]) => ({ label: `Wayback ${date}`, date, bounds, zoom: 12 });
const slotWithMeta = (name, meta) => ({ ...slot(name), meta });

test('a map capture shows its date and zoom on its slot, an uploaded file does not', async () => {
  const ImageSlot = (await load('/src/dashboard/components/ImageSlot.jsx')).default;
  const props = { id: 'before', title: 'Before', onFile() {}, onRemove() {} };
  const captured = html(h(ImageSlot, { ...props, slot: slotWithMeta('map-view-wayback-2017-05-17.png', capture('2017-05-17')) }));
  assert.match(captured, /Wayback 2017-05-17 · zoom 12/);
  assert.doesNotMatch(html(h(ImageSlot, { ...props, slot: slot('photo.png') })), /Wayback|zoom/);
});

test('ChangePanel warns when the two map captures show different ground, and only then', async () => {
  const ChangePanel = (await load('/src/dashboard/components/panels/ChangePanel.jsx')).default;
  const far = capture('2026-08-05', [[10, 10], [10.2, 10.2]]);
  const apart = html(h(ChangePanel, { ws: makeWs(withSlots({ before: slotWithMeta('a.png', capture('2017-05-17')), after: slotWithMeta('b.png', far) })), goTo() {} }));
  assert.match(apart, /show different areas \(0% overlap\)/);
  assert.match(apart, /Capture both from the same map view/);

  const same = html(h(ChangePanel, { ws: makeWs(withSlots({ before: slotWithMeta('a.png', capture('2017-05-17')), after: slotWithMeta('b.png', capture('2026-08-05')) })), goTo() {} }));
  assert.doesNotMatch(same, /different areas/);
  const uploads = html(h(ChangePanel, { ws: makeWs(withSlots({ before: slot('a.png'), after: slot('b.png') })), goTo() {} }));
  assert.doesNotMatch(uploads, /different areas/, 'uploaded files carry no area, so nothing is compared');
});

test('Timeline: loading, error, and the ready card with its date, slider, ticks and capture buttons', async () => {
  const Timeline = (await load('/src/dashboard/components/Timeline.jsx')).default;
  const base = { releases: RELEASES, index: 2, ready: true, busy: false, note: null, before: slot(null), after: slot(null), onIndex() {}, onRetry() {}, onUse() {}, onOpenChange() {} };

  assert.match(html(h(Timeline, { ...base, status: 'loading', releases: [] })), /Loading available dates/);
  const failed = html(h(Timeline, { ...base, status: 'error', releases: [], error: 'Could not reach the historical imagery service.' }));
  assert.match(failed, /role="alert"/);
  assert.match(failed, /Could not reach the historical imagery service/);
  assert.match(failed, />Try again</);

  const out = html(h(Timeline, { ...base, status: 'ready' }));
  assert.match(out, />5 Aug 2026</);
  assert.match(out, /<input[^>]*type="range"[^>]*max="2"[^>]*value="2"/);
  assert.match(out, /aria-valuetext="5 Aug 2026"/);
  assert.match(out, />2014</);
  assert.match(out, />2026</);
  assert.match(out, />Use as Before</);
  assert.match(out, />Use as After</);
  assert.match(out, /published each version of its basemap/, 'the caveat about release dates is available on the info icon');
  assert.doesNotMatch(out, /Open Change detection|not set/, 'nothing captured yet, so no chips');
  assert.match(out, /<button[^>]*disabled=""[^>]*aria-label="Later version"/, 'the newest date has nothing later');
  assert.doesNotMatch(out, /<button[^>]*disabled=""[^>]*aria-label="Earlier version"/);
});

test('Timeline: buttons wait for the map, and the date is locked while a capture runs', async () => {
  const Timeline = (await load('/src/dashboard/components/Timeline.jsx')).default;
  const base = { status: 'ready', releases: RELEASES, index: 1, busy: false, note: null, before: slot(null), after: slot(null), onIndex() {}, onRetry() {}, onUse() {}, onOpenChange() {} };
  const waiting = html(h(Timeline, { ...base, ready: false }));
  assert.match(waiting, /<button[^>]*disabled=""[^>]*>Use as Before/);
  assert.match(waiting, /Loading<\/span>/);
  const busy = html(h(Timeline, { ...base, ready: true, busy: true }));
  assert.match(busy, /<input[^>]*type="range"[^>]*disabled=""/, 'the slider cannot move mid-capture (it would capture the wrong date)');
  assert.match(busy, /<button[^>]*disabled=""[^>]*aria-label="Earlier version"/);
  assert.match(busy, /<button[^>]*disabled=""[^>]*aria-label="Later version"/);
  assert.match(busy, /Capturing<\/span>/);
  const idle = html(h(Timeline, { ...base, ready: true }));
  assert.doesNotMatch(idle, /<input[^>]*type="range"[^>]*disabled=""/);
  assert.match(html(h(Timeline, { ...base, ready: true, note: 'The map is still loading. Try again in a moment.' })), /role="alert"[^>]*>The map is still loading/);
});

test('Timeline: chips for what was captured, the Open button only for a matching pair, a warning otherwise', async () => {
  const Timeline = (await load('/src/dashboard/components/Timeline.jsx')).default;
  const base = { status: 'ready', releases: RELEASES, index: 1, ready: true, busy: false, note: null, onIndex() {}, onRetry() {}, onUse() {}, onOpenChange() {} };
  const one = html(h(Timeline, { ...base, before: slotWithMeta('a.png', capture('2014-06-11')), after: slot(null) }));
  assert.match(one, /Before<\/span><span>11 Jun 2014/);
  assert.match(one, /After<\/span><span>not set/);
  assert.doesNotMatch(one, /Open Change detection/);

  const pair = html(h(Timeline, { ...base, before: slotWithMeta('a.png', capture('2014-06-11')), after: slotWithMeta('b.png', capture('2026-08-05')) }));
  assert.match(pair, /Open Change detection/);
  assert.doesNotMatch(pair, /different areas/);

  const apart = html(h(Timeline, { ...base, before: slotWithMeta('a.png', capture('2014-06-11')), after: slotWithMeta('b.png', capture('2026-08-05', [[50, 5], [50.2, 5.2]])) }));
  assert.match(apart, /different areas \(0% overlap\)/);
  assert.match(apart, /Keep the map still between them/);
  assert.doesNotMatch(apart, /Open Change detection/, 'a mismatched pair is not offered for comparison');
});

test('SarDialog: Copernicus steps and fields, a custom-URL tab, and Disconnect only when something is connected', async () => {
  const SarDialog = (await load('/src/dashboard/components/SarDialog.jsx')).default;
  const props = { onSave() {}, onDisconnect() {}, onClose() {} };
  const fresh = html(h(SarDialog, { ...props, source: null }));
  assert.match(fresh, /role="dialog"[^>]*aria-label="Connect SAR imagery"/);
  assert.match(fresh, /Create a free account at/);
  assert.match(fresh, /href="https:\/\/dataspace\.copernicus\.eu"/);
  assert.match(fresh, /href="https:\/\/dataspace\.copernicus\.eu\/analyse\/apis\/sentinel-hub"/);
  assert.match(fresh, /Configuration Utility/);
  assert.match(fresh, /id="sar-instance"/);
  assert.match(fresh, /id="sar-layer"/);
  assert.match(fresh, /id="sar-time"/);
  assert.match(fresh, /Kept only in this browser/);
  assert.doesNotMatch(fresh, /Disconnect|Connected:/);
  assert.doesNotMatch(fresh, /id="sar-url"/, 'the URL field belongs to the other tab');

  const cdse = { kind: 'cdse', template: 'https://sh.dataspace.copernicus.eu/ogc/wmts/abc12345?LAYER=S1&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}', instanceId: 'abc12345-aaaa', layer: 'S1-VV', timeRange: '' };
  const connected = html(h(SarDialog, { ...props, source: cdse }));
  assert.match(connected, /Connected: Copernicus · S1-VV/);
  assert.match(connected, />Disconnect</);
  assert.match(connected, /value="abc12345-aaaa"/, 'editing starts from what is saved');
  assert.match(connected, /value="S1-VV"/);

  const custom = html(h(SarDialog, { ...props, source: { kind: 'custom', template: 'https://t.example/{z}/{x}/{y}.png', attribution: 'My SAR' } }));
  assert.match(custom, /id="sar-url"[^>]*value="https:\/\/t\.example\/\{z\}\/\{x\}\/\{y\}\.png"/);
  assert.match(custom, /Connected: Custom tile URL/);
});

test('MapAsk on the Past layer is optical: scans stay available and saves go to the optical slots', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const out = html(h(MapAsk, mapAskProps({ layer: 'past', layerLabel: 'Wayback 2014-06-11' })));
  assert.match(out, /placeholder="Ask about what&#x27;s on the map…"/);
  assert.doesNotMatch(out, /optical-only|SAR view/);
  assert.equal((out.match(/disabled=""[^>]*title="The feature scans were built for optical imagery"/g) || []).length, 0, 'the scans work on historical optical imagery');
});

test('the nav buttons fill the rail on desktop, so the icons sit at its centre', async () => {
  const { FEATURES } = await load('/src/dashboard/features.js');
  const NavRail = (await load('/src/dashboard/components/NavRail.jsx')).default;
  const out = html(h(NavRail, { features: FEATURES, activeId: 'imagery', panelOpen: true, onSelect() {}, ws: makeWs() }));
  assert.equal((out.match(/lg:w-full lg:min-w-0/g) || []).length, FEATURES.length);
});

test('the header uses the new logo, sized for the nav column, with a larger title', async () => {
  const { MemoryRouter } = await import('react-router-dom');
  const Dashboard = (await load('/src/Dashboard.jsx')).default;
  const { AuthProvider } = await load('/src/auth/AuthContext.jsx');
  const out = html(h(MemoryRouter, null, h(AuthProvider, null, h(Dashboard))));
  assert.match(out, /<span role="img" aria-hidden="true" class="sq-logo-mark h-10 w-10" style="--sq-logo:url\([^)]*logo-mark[^)]*\)"/);
  assert.match(out, /class="flex w-\[76px\] shrink-0 justify-center"/, 'the logo column is as wide as the nav rail');
  assert.match(out, /text-2xl font-bold[^>]*>SatQuery-AI</);
  assert.match(out, /aria-label="SatQuery-AI home"/);
});

// ------------------------------------------------------------------ the whole dashboard

test('the Dashboard renders as a whole: header, status, theme toggle and every nav item', async () => {
  const { MemoryRouter } = await import('react-router-dom');
  const Dashboard = (await load('/src/Dashboard.jsx')).default;
  const { FEATURES } = await load('/src/dashboard/features.js');
  const { AuthProvider } = await load('/src/auth/AuthContext.jsx');
  const out = html(h(MemoryRouter, null, h(AuthProvider, null, h(Dashboard))));
  assert.match(out, /SatQuery-AI/);
  assert.match(out, /data-theme="dark"/);
  assert.match(out, /aria-label="Switch to light theme"/);
  assert.match(out, /Checking/, 'the status pill starts by checking the backend');
  for (const f of FEATURES) assert.match(out, new RegExp(`>${f.label}<`));
  assert.match(out, /aria-label="Imagery"/, 'the open panel is labelled');
  assert.match(out, /aria-label="Viewer"/);
});


// ------------------------------------------------------------------ accounts, search, header layout

const signedIn = async (user, element) => {
  const { AuthContext } = await load('/src/auth/authState.js');
  return h(AuthContext.Provider, { value: { status: user ? 'authed' : 'anon', user, signIn() {}, signOut() {}, recheck() {} } }, element);
};

test('UserMenu shows the person and a sign-out button, and nothing when signed out', async () => {
  const UserMenu = (await load('/src/dashboard/components/UserMenu.jsx')).default;
  const out = html(await signedIn({ name: 'Ada Lovelace', email: 'ada@example.com' }, h(UserMenu)));
  assert.match(out, />Ada Lovelace</);
  assert.match(out, />A</, 'the initial');
  assert.match(out, /aria-label="Sign out"/);
  assert.equal(html(await signedIn(null, h(UserMenu))), '');
});

test('AccountReports shows nothing when signed out, and its section when signed in', async () => {
  const AccountReports = (await load('/src/dashboard/components/AccountReports.jsx')).default;
  assert.equal(html(await signedIn(null, h(AccountReports))), '', 'no account, nothing to show');
  assert.equal(html(h(AccountReports)), '', 'no <AuthProvider> at all (e.g. a standalone panel render) is just as safe');
  const out = html(await signedIn({ name: 'Ada', email: 'ada@example.com' }, h(AccountReports)));
  assert.match(out, /Your saved reports/);
});

test('the login and sign-up pages have the right fields, a Google slot and a link to each other', async () => {
  const { MemoryRouter } = await import('react-router-dom');
  const AuthPage = (await load('/src/auth/AuthPage.jsx')).default;
  const login = html(await signedIn(null, h(MemoryRouter, null, h(AuthPage, { mode: 'login' }))));
  assert.match(login, /Welcome back/);
  assert.match(login, /autoComplete="current-password"|autocomplete="current-password"/);
  assert.doesNotMatch(login, /Confirm password/);
  assert.match(login, /href="\/signup"/);
  assert.match(login, /Google sign-in is off/, 'no client id known yet: the setup note, not a broken button');
  const signup = html(await signedIn(null, h(MemoryRouter, null, h(AuthPage, { mode: 'signup' }))));
  assert.match(signup, /Create your account/);
  assert.match(signup, /id="auth-name"/);
  assert.match(signup, /Confirm password/);
  assert.match(signup, /href="\/login"/);
});

// A redirect (<Navigate>) only fires in an effect, so static rendering shows the page it LEAVES: nothing, not the form.
test('a signed-in visitor is not shown the login form', async () => {
  const { MemoryRouter, Routes, Route } = await import('react-router-dom');
  const AuthPage = (await load('/src/auth/AuthPage.jsx')).default;
  const out = html(await signedIn({ name: 'A', email: 'a@b.co' }, h(MemoryRouter, { initialEntries: ['/login'] },
    h(Routes, null, h(Route, { path: '/login', element: h(AuthPage) }), h(Route, { path: '/dashboard', element: h('p', null, 'DASHBOARD HERE') })))));
  assert.doesNotMatch(out, /Welcome back/);
  assert.doesNotMatch(out, /auth-password/);
});

test('RequireAuth shows the page to a signed-in person and never to anyone else', async () => {
  const { MemoryRouter, Routes, Route } = await import('react-router-dom');
  const RequireAuth = (await load('/src/auth/RequireAuth.jsx')).default;
  const tree = () => h(MemoryRouter, { initialEntries: ['/dashboard'] }, h(Routes, null,
    h(Route, { path: '/login', element: h('p', null, 'LOGIN PAGE') }),
    h(Route, { path: '/dashboard', element: h(RequireAuth, null, h('p', null, 'SECRET')) })));
  const okOut = html(await signedIn({ name: 'A', email: 'a@b.co' }, tree()));
  assert.match(okOut, /SECRET/);
  const anonOut = html(await signedIn(null, tree()));
  assert.doesNotMatch(anonOut, /SECRET/);
  const { AuthContext } = await load('/src/auth/authState.js');
  for (const status of ['loading', 'offline']) {
    const held = html(h(AuthContext.Provider, { value: { status, user: null, recheck() {} } }, tree()));
    assert.doesNotMatch(held, /SECRET/, status);
    assert.match(held, status === 'offline' ? /Cannot reach the server/ : /Checking your session/);
  }
});

test('CoordinateBox shows a list of matching places and a search error', async () => {
  const CoordinateBox = (await load('/src/dashboard/components/CoordinateBox.jsx')).default;
  assert.doesNotMatch(html(h(CoordinateBox, { onGo() {}, onClear() {}, hasPin: false })), /Matching places/);
});

test('the viewer header holds a slot for the timeline between the Live map switch and the layer switch', async () => {
  const Viewer = (await load('/src/dashboard/components/Viewer.jsx')).default;
  const { featureById } = await load('/src/dashboard/features.js');
  const out = html(h(Viewer, { ws: makeWs({ viewMode: 'map' }), feature: featureById('imagery'), onOpenFeature() {} }));
  const live = out.indexOf('Live map');
  const slot = out.indexOf('flex-1 basis-[22rem]');
  assert.ok(live > -1 && slot > live, 'the timeline slot follows the Live map / Inputs switch');
  const inputs = html(h(Viewer, { ws: makeWs({ viewMode: 'inputs' }), feature: featureById('fusion'), onOpenFeature() {} }));
  assert.ok(inputs.indexOf('flex-1 basis-[22rem]') < inputs.indexOf('>Composite<'), 'and comes before the Optical / SAR / Composite switch');
});
