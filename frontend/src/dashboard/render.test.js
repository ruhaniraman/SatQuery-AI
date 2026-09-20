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
const slot = (name, modality = 'optical', size) => ({ file: name ? file(name, size) : null, preview: name ? 'blob:preview' : null, modality });

function makeWs(overrides = {}) {
  const base = {
    slots: { a: slot(null), b: slot(null) },
    images: [],
    mode: { id: 'none', label: 'No imagery yet', hint: 'Add an image to begin.' },
    setImage() {}, setModality() {}, removeImageB() {},
    viewMode: 'map', setViewMode() {}, activeLayer: 'imageA', setActiveLayer() {},
    chat: [], result: null, latest: { scan: null, compare: null },
    error: null, isExecuting: false, busy: null,
    sendMessage() {}, runScan() {}, runCompare() {}, dismissError() {},
  };
  return { ...base, ...overrides };
}
const withImages = (n, modalityB = 'optical') => {
  const a = slot('scene.tif', 'optical', 3 * 1024 * 1024);
  const b = n > 1 ? slot('later.png', modalityB) : slot(null);
  const images = [{ file: a.file, modality: 'optical' }].concat(n > 1 ? [{ file: b.file, modality: modalityB }] : []);
  const mode = n === 1
    ? { id: 'single', label: 'Single-image analysis', hint: 'Ask questions or run a feature scan.' }
    : modalityB === 'optical'
      ? { id: 'change', label: 'Change detection', hint: 'Same sensor type on both images.' }
      : { id: 'fusion', label: 'Optical–SAR fusion', hint: 'Different sensor types.' };
  return { slots: { a, b }, images, mode };
};

const REALISTIC_RESULT = {
  answer: 'Two changes found.',
  visual_evidence_url: '/reports/abc/evidence.png',
  report_download_url: '/reports/abc/report.pdf',
  report_error: null,
  agent_execution_trace: {
    pipeline_id: 'abc-123', task: 'CHANGE_DETECTION', routing: 'rule-based',
    routing_reason: '2 images of the same sensor type (optical + optical)',
    nodes_traversed: ['RuleBasedRouter', 'InputPreprocessor', 'SpatialAlignment', 'TemporalOrdering', 'Hybrid-CDVQA-Engine', 'Qwen2-VL-Bridge'],
    telemetry: {
      model_used: 'Qwen/Qwen2-VL-2B-Instruct (base weights, LoRA adapters disabled)',
      active_adapter: null,
      sar_despeckled: true,
      major_regions_detected: 2,
      loader_notes: ['a', 'b'],
      temporal_order: { basis: 'as-uploaded', reordered: false, summary: 'Capture dates are unknown; treated Image A as BEFORE.', warnings: [] },
    },
    warnings: ['Band order is not declared in the file; assumed the first three bands are R, G, B.'],
    validation_status: 'image payload checks passed', execution_status: 'completed',
  },
};

// ------------------------------------------------------------------ feature registry

test('the feature registry is well formed and drives the nav', async () => {
  const { FEATURES, featureById } = await load('/src/dashboard/features.js');
  assert.ok(FEATURES.length >= 5);
  assert.equal(new Set(FEATURES.map((f) => f.id)).size, FEATURES.length, 'ids are unique');
  for (const f of FEATURES) {
    for (const key of ['id', 'label', 'title', 'summary']) assert.ok(f[key], `${f.id} has ${key}`);
    assert.ok(f.icon && f.Panel, `${f.id} has icon and Panel`);
  }
  assert.equal(featureById('nope').id, FEATURES[0].id, 'unknown id falls back to the first feature');

  // Every registered panel renders in its empty state without throwing
  for (const f of FEATURES) assert.doesNotThrow(() => html(h(f.Panel, { ws: makeWs(), goTo() {} })), f.id);
});

test('NavRail lists every feature, marks the open one and shows status badges', async () => {
  const { FEATURES } = await load('/src/dashboard/features.js');
  const NavRail = (await load('/src/dashboard/components/NavRail.jsx')).default;
  const ws = makeWs({ ...withImages(1), result: REALISTIC_RESULT });
  const out = html(h(NavRail, { features: FEATURES, activeId: 'assistant', panelOpen: true, onSelect() {}, ws }));
  for (const f of FEATURES) assert.match(out, new RegExp(`>${f.label}<`));
  assert.equal((out.match(/aria-current="page"/g) || []).length, 1);
  assert.match(out, /rounded-full bg-blue-400/, 'ready badge on Imagery and Report');

  const collapsed = html(h(NavRail, { features: FEATURES, activeId: 'assistant', panelOpen: false, onSelect() {}, ws }));
  assert.equal((collapsed.match(/aria-current/g) || []).length, 0, 'nothing is "current" when the panel is collapsed');

  const busy = html(h(NavRail, { features: FEATURES, activeId: 'imagery', panelOpen: true, onSelect() {}, ws: makeWs({ isExecuting: true, busy: 'scan:mining' }) }));
  assert.match(busy, /animate-spin/);
});

// ------------------------------------------------------------------ imagery

test('ImageryPanel: empty state locks Image B until Image A exists', async () => {
  const ImageryPanel = (await load('/src/dashboard/components/panels/ImageryPanel.jsx')).default;
  const out = html(h(ImageryPanel, { ws: makeWs() }));
  assert.match(out, /Image A/);
  assert.match(out, /Drop an image or click to browse/);
  assert.match(out, /Add Image A first/);
  assert.match(out, /No imagery yet/);
  assert.match(out, /data-slot="a"/);
  assert.match(out, /data-slot="b"/);
});

test('ImageryPanel: shows file details, GeoTIFF badge, sensor selection and the mode', async () => {
  const ImageryPanel = (await load('/src/dashboard/components/panels/ImageryPanel.jsx')).default;
  const out = html(h(ImageryPanel, { ws: makeWs(withImages(2, 'sar')) }));
  assert.match(out, /scene\.tif/);
  assert.match(out, /3\.0 MB · GeoTIFF/);
  assert.match(out, /later\.png/);
  assert.match(out, /Optical–SAR fusion/);
  assert.match(out, /aria-label="Remove Image B"/);
  assert.match(out, /aria-pressed="true"/);
  assert.doesNotMatch(out, /Add Image A first/);
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
  const ready = html(h(AssistantPanel, { ws: makeWs(withImages(1)), goTo() {} }));
  assert.match(ready, /Ask about your imagery/);
  assert.match(ready, /What land cover is visible\?/);

  const chat = [
    { role: 'system', content: 'Initiating mining scan...' },
    { role: 'ai', content: 'Scan complete. Found 2 potential region(s).' },
    { role: 'user', content: 'and the north side?' },
  ];
  const out = html(h(AssistantPanel, { ws: makeWs({ ...withImages(1), chat }), goTo() {} }));
  assert.match(out, /Initiating mining scan/);
  assert.match(out, /Found 2 potential region/);
  assert.match(out, /and the north side\?/);
  assert.doesNotMatch(out, /Ask about your imagery/, 'suggestions disappear once there is a conversation');
});

test('AssistantPanel: busy and error states', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const busy = html(h(AssistantPanel, { ws: makeWs({ ...withImages(1), isExecuting: true }), goTo() {} }));
  assert.match(busy, /Analysing/);
  assert.match(busy, /<textarea[^>]*disabled/);
  const err = html(h(AssistantPanel, { ws: makeWs({ ...withImages(1), error: 'The AI model is not available: X' }), goTo() {} }));
  assert.match(err, /role="alert"/);
  assert.match(err, /The AI model is not available: X/);
});

// ------------------------------------------------------------------ scans

test('ScansPanel: three scans, disabled without imagery, latest result with evidence link', async () => {
  const { default: ScansPanel, SCANS } = await load('/src/dashboard/components/panels/ScansPanel.jsx');
  assert.deepEqual(SCANS.map((s) => s.id), ['mining', 'agriculture', 'deforestation']);
  const empty = html(h(ScansPanel, { ws: makeWs(), goTo() {} }));
  assert.match(empty, /Add an image first/);
  assert.equal((empty.match(/disabled=""/g) || []).length, 3);

  const ws = makeWs({ ...withImages(1), latest: { scan: { adapter: 'mining', answer: 'Scan complete. Found 2 potential region(s).' }, compare: null }, result: REALISTIC_RESULT });
  const out = html(h(ScansPanel, { ws, goTo() {} }));
  assert.match(out, /Mining scan/);
  assert.match(out, /Found 2 potential region/);
  assert.match(out, /View evidence/);
  assert.equal((out.match(/disabled=""/g) || []).length, 0);

  const running = html(h(ScansPanel, { ws: makeWs({ ...withImages(1), isExecuting: true, busy: 'scan:agriculture' }), goTo() {} }));
  assert.match(running, /Scanning/);
});

// ------------------------------------------------------------------ compare

test('ComparePanel: needs two images; wording follows the mode', async () => {
  const ComparePanel = (await load('/src/dashboard/components/panels/ComparePanel.jsx')).default;
  const one = html(h(ComparePanel, { ws: makeWs(withImages(1)), goTo() {} }));
  assert.match(one, /Add a second image/);
  assert.match(one, /<button[^>]*disabled[^>]*>.*Run change detection/s);

  const change = html(h(ComparePanel, { ws: makeWs(withImages(2, 'optical')), goTo() {} }));
  assert.match(change, /Run change detection/);
  assert.doesNotMatch(change, /Add a second image/);
  const fusion = html(h(ComparePanel, { ws: makeWs(withImages(2, 'sar')), goTo() {} }));
  assert.match(fusion, /Run fusion/);

  const withResult = html(h(ComparePanel, { ws: makeWs({ ...withImages(2), latest: { scan: null, compare: { mode: 'change', answer: 'A new road appears.' } }, result: REALISTIC_RESULT }), goTo() {} }));
  assert.match(withResult, /A new road appears\./);
});

// ------------------------------------------------------------------ report

test('ReportPanel: empty state', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  assert.match(html(h(ReportPanel, { ws: makeWs() })), /No report yet/);
});

test('ReportPanel renders a realistic trace, including objects, booleans and lists, without crashing', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  const out = html(h(ReportPanel, { ws: makeWs({ result: REALISTIC_RESULT }) }));
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
});

test('ReportPanel: failed PDF is shown as unavailable with the reason', async () => {
  const ReportPanel = (await load('/src/dashboard/components/panels/ReportPanel.jsx')).default;
  const result = { ...REALISTIC_RESULT, report_download_url: null, report_error: 'PDF generation failed: RuntimeError' };
  const out = html(h(ReportPanel, { ws: makeWs({ result }) }));
  assert.match(out, /Report unavailable/);
  assert.match(out, /PDF generation failed: RuntimeError/);
  assert.doesNotMatch(out, /Download PDF report/);
});

// ------------------------------------------------------------------ inputs canvas

test('InputsCanvas: empty, image, GeoTIFF placeholder and evidence layers', async () => {
  const InputsCanvas = (await load('/src/dashboard/components/InputsCanvas.jsx')).default;
  const empty = html(h(InputsCanvas, { ws: makeWs(), onAddImagery() {} }));
  assert.match(empty, /No imagery loaded/);
  assert.match(empty, /Add imagery/);

  const png = html(h(InputsCanvas, { ws: makeWs({ slots: { a: slot('photo.png'), b: slot(null) }, activeLayer: 'imageA' }), onAddImagery() {} }));
  assert.match(png, /<img[^>]*src="blob:preview"/);
  assert.match(png, /Image A · Optical/);

  const tif = html(h(InputsCanvas, { ws: makeWs({ slots: { a: slot('scene.tif'), b: slot(null) }, activeLayer: 'imageA' }), onAddImagery() {} }));
  assert.match(tif, /GeoTIFF loaded/);
  assert.doesNotMatch(tif, /<img/);

  const ev = html(h(InputsCanvas, { ws: makeWs({ ...withImages(2), activeLayer: 'evidence', result: REALISTIC_RESULT }), onAddImagery() {} }));
  assert.match(ev, /src="http:\/\/localhost:8000\/reports\/abc\/evidence\.png"/);

  const noEvidenceYet = html(h(InputsCanvas, { ws: makeWs({ ...withImages(1), activeLayer: 'evidence' }), onAddImagery() {} }));
  assert.match(noEvidenceYet, /scene\.tif/, 'falls back to Image A rather than showing nothing');

  // The caption names the file (the layer tabs already say which layer this is); evidence has none
  assert.match(png, /title="photo\.png"/);
  assert.doesNotMatch(ev, /max-w-\[60%\]/);
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

test('chat bubbles format assistant answers but leave the user\'s own text alone', async () => {
  const AssistantPanel = (await load('/src/dashboard/components/panels/AssistantPanel.jsx')).default;
  const chat = [
    { role: 'user', content: 'ASSESSMENT: is this mine?' },
    { role: 'ai', content: 'OBSERVATIONS: Farmland. ASSESSMENT: Mostly agricultural.' },
  ];
  const out = html(h(AssistantPanel, { ws: makeWs({ ...withImages(1), chat }), goTo() {} }));
  assert.equal((out.match(/uppercase tracking-wider text-blue-300/g) || []).length, 2, 'only the assistant message is reformatted');
  assert.match(out, /ASSESSMENT: is this mine\?/);
});

// ------------------------------------------------------------------ on-map ask bar

test('MapAsk: idle bar has a prompt box and one scan chip per feature scan', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const { SCANS } = await load('/src/dashboard/components/panels/ScansPanel.jsx');
  const out = html(h(MapAsk, { ws: makeWs(), capture() {}, overlay: null, onOverlay() {}, onClearOverlay() {}, onToggleOverlay() {}, onOpenAssistant() {} }));
  assert.match(out, /aria-label="Ask about this map view"/);
  assert.match(out, /<button[^>]*disabled[^>]*aria-label="Ask about this view"/s, 'send is disabled until something is typed');
  assert.match(out, /Scan this view:/);
  for (const scan of SCANS) assert.match(out, new RegExp(`aria-label="Scan this view for ${scan.title}"`));
  assert.doesNotMatch(out, /disabled=""[^>]*aria-label="Scan this view for/);
});

test('MapAsk: while a request is running the controls lock and the status says why', async () => {
  const MapAsk = (await load('/src/dashboard/components/MapAsk.jsx')).default;
  const out = html(h(MapAsk, { ws: makeWs({ isExecuting: true }), capture() {}, overlay: null, onOverlay() {}, onClearOverlay() {}, onToggleOverlay() {}, onOpenAssistant() {} }));
  assert.match(out, /Analysing/);
  assert.match(out, /<input[^>]*disabled/);
  assert.doesNotMatch(out, /Scan this view for/, 'scan chips are replaced by the status while busy');
});
