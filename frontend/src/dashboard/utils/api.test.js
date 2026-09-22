import test from 'node:test';
import assert from 'node:assert/strict';
import {
  analysisMode, buildAnalyzeFormData, DEFAULT_QUERIES, evidenceUrl, formatBytes, formatErrorDetail,
  formatTelemetryValue, humanizeIdentifier, isTiffFile, readErrorMessage, reportFileName, reportUrl,
  requestAnalysis, checkHealth, requestPreview, requestReport,
} from './api.js';
import { formatLatLng, MAP_CONFIG } from './mapConfig.js';

const file = (name, size = 10) => new File([new Uint8Array(size)], name);
const response = (status, body, { textThrows = false } = {}) => ({
  ok: status >= 200 && status < 300,
  status,
  text: async () => { if (textThrows) throw new Error('network'); return body; },
  json: async () => JSON.parse(body),
});

// ------------------------------------------------------------------ telemetry / errors

test('formatTelemetryValue turns every JSON shape into text (React throws on objects)', () => {
  const order = { basis: 'as-uploaded', summary: 'Capture dates are unknown', warnings: [] };
  assert.equal(formatTelemetryValue(order), 'Capture dates are unknown');
  assert.equal(formatTelemetryValue(true), 'yes');
  assert.equal(formatTelemetryValue(false), 'no');
  assert.equal(formatTelemetryValue(null), 'n/a');
  assert.equal(formatTelemetryValue(undefined), 'n/a');
  assert.equal(formatTelemetryValue([]), 'none');
  assert.equal(formatTelemetryValue(['a', true, null]), 'a; yes; n/a');
  assert.equal(formatTelemetryValue(0), '0');
  assert.equal(formatTelemetryValue({ r: 16 }), '{"r":16}');
});

test('formatErrorDetail handles FastAPI 422 lists instead of printing [object Object]', () => {
  assert.equal(formatErrorDetail('Invalid adapter'), 'Invalid adapter');
  assert.equal(formatErrorDetail([{ loc: ['body', 'query'], msg: 'Field required' }]), 'query: Field required');
  assert.equal(
    formatErrorDetail([{ loc: ['body', 'query'], msg: 'Field required' }, { loc: ['body', 'images', 0], msg: 'bad file' }]),
    'query: Field required; images.0: bad file',
  );
  assert.equal(formatErrorDetail([{ msg: 'oops' }]), 'oops');
  assert.equal(formatErrorDetail({ msg: 'boom' }), 'boom');
  assert.equal(formatErrorDetail(null), null);
  assert.equal(formatErrorDetail(''), null);
  assert.equal(formatErrorDetail(42), '42');
});

test('readErrorMessage never throws, whatever the error body is', async () => {
  assert.equal(await readErrorMessage(response(400, JSON.stringify({ detail: 'Cannot combine a TIFF' }))), 'Cannot combine a TIFF');
  assert.equal(await readErrorMessage(response(503, JSON.stringify({ detail: 'The AI model is not available: X' }))), 'The AI model is not available: X');
  assert.equal(await readErrorMessage(response(502, '<html><body>Bad Gateway</body></html>')), 'Analysis failed (HTTP 502).');
  assert.equal(await readErrorMessage(response(500, '')), 'Analysis failed (HTTP 500).');
  assert.equal(await readErrorMessage(response(500, '{"other":1}')), 'Analysis failed (HTTP 500).');
  assert.equal(await readErrorMessage(response(500, 'null')), 'Analysis failed (HTTP 500).');
  assert.equal(await readErrorMessage(response(504, '', { textThrows: true })), 'Analysis failed (HTTP 504).');
});

// ------------------------------------------------------------------ routing mirror

test('analysisMode mirrors the server routing rule', () => {
  const img = (modality) => ({ file: file('x.png'), modality });
  assert.equal(analysisMode([]).id, 'none');
  assert.equal(analysisMode(undefined).id, 'none');
  assert.equal(analysisMode([img('optical')]).id, 'single');
  assert.equal(analysisMode([img('optical'), img('optical')]).id, 'change');
  assert.equal(analysisMode([img('sar'), img('sar')]).id, 'change');
  assert.equal(analysisMode([img('optical'), img('sar')]).id, 'fusion');
  assert.equal(analysisMode([img('sar'), img('optical')]).id, 'fusion');
  assert.match(analysisMode([img('optical'), img('optical')]).hint, /BEFORE/);
});

// ------------------------------------------------------------------ request building

test('buildAnalyzeFormData sends exactly the fields the API expects', () => {
  const single = buildAnalyzeFormData({ query: 'q', adapter: 'mining', images: [{ file: file('a.png'), modality: 'sar' }], history: [{ role: 'user', content: 'q' }] });
  assert.equal(single.get('query'), 'q');
  assert.equal(single.get('adapter'), 'mining');
  assert.equal(single.get('modality_a'), 'sar');
  assert.equal(single.get('modality_b'), null);
  assert.equal(single.getAll('images').length, 1);
  assert.deepEqual(JSON.parse(single.get('chat_history')), [{ role: 'user', content: 'q' }]);

  const pair = buildAnalyzeFormData({
    query: 'q', images: [{ file: file('a.tif'), modality: 'optical' }, { file: file('b.tif'), modality: 'sar' }],
  });
  assert.equal(pair.get('adapter'), 'general');
  assert.equal(pair.get('modality_a'), 'optical');
  assert.equal(pair.get('modality_b'), 'sar');
  assert.deepEqual(pair.getAll('images').map((f) => f.name), ['a.tif', 'b.tif']);
  assert.equal(pair.get('chat_history'), '[]');
});

test('the default queries are never empty (the API rejects an empty query)', () => {
  for (const q of Object.values(DEFAULT_QUERIES)) assert.ok(q.trim().length > 0);
});

test('requestAnalysis posts to /analyze and returns the parsed result', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return response(200, JSON.stringify({ answer: 'ok' })); };
  const out = await requestAnalysis({ query: 'q', images: [{ file: file('a.png'), modality: 'optical' }], fetchImpl, baseUrl: 'http://api.test' });
  assert.deepEqual(out, { answer: 'ok' });
  assert.equal(seen.url, 'http://api.test/analyze');
  assert.equal(seen.init.method, 'POST');
  assert.ok(seen.init.body instanceof FormData);
});

test('requestAnalysis surfaces server errors, 422 lists and gateway pages as readable messages', async () => {
  const images = [{ file: file('a.png'), modality: 'optical' }];
  const ask = (res) => requestAnalysis({ query: 'q', images, fetchImpl: async () => res, baseUrl: 'http://api.test' });
  await assert.rejects(ask(response(400, JSON.stringify({ detail: 'Invalid adapter' }))), { message: 'Invalid adapter' });
  await assert.rejects(ask(response(422, JSON.stringify({ detail: [{ loc: ['body', 'query'], msg: 'Field required' }] }))), { message: 'query: Field required' });
  await assert.rejects(ask(response(502, '<html>Bad Gateway</html>')), { message: 'Analysis failed (HTTP 502).' });
});

test('requestAnalysis explains an unreachable server instead of "Failed to fetch"', async () => {
  const fetchImpl = async () => { throw new TypeError('Failed to fetch'); };
  await assert.rejects(
    requestAnalysis({ query: 'q', images: [{ file: file('a.png'), modality: 'optical' }], fetchImpl, baseUrl: 'http://api.test' }),
    (err) => /Cannot reach the analysis server at http:\/\/api\.test/.test(err.message),
  );
});

test('requestAnalysis passes the signal through and lets an abort pass through as-is (the Stop button)', async () => {
  let seenSignal;
  const abort = Object.assign(new Error('aborted'), { name: 'AbortError' });
  const fetchImpl = async (url, init) => { seenSignal = init.signal; throw abort; };
  const controller = new AbortController();
  await assert.rejects(
    requestAnalysis({ query: 'q', images: [{ file: file('a.png'), modality: 'optical' }], signal: controller.signal, fetchImpl, baseUrl: 'http://api.test' }),
    (err) => err.name === 'AbortError',
  );
  assert.equal(seenSignal, controller.signal);
});

// ------------------------------------------------------------------ small helpers

test('URL helpers tolerate results without evidence or report', () => {
  assert.equal(evidenceUrl(null, 'http://x'), null);
  assert.equal(evidenceUrl({ visual_evidence_url: '/reports/1/evidence.png' }, 'http://x'), 'http://x/reports/1/evidence.png');
  assert.equal(reportUrl({ report_download_url: null }, 'http://x'), null);
  assert.equal(reportUrl({ report_download_url: '/reports/1/report.pdf' }, 'http://x'), 'http://x/reports/1/report.pdf');
  assert.equal(reportFileName({ agent_execution_trace: { pipeline_id: 'abc' } }), 'satquery-report-abc.pdf');
  assert.equal(reportFileName(null), 'satquery-report-report.pdf');
});

test('formatBytes and isTiffFile', () => {
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(2048), '2 KB');
  assert.equal(formatBytes(5 * 1024 * 1024), '5.0 MB');
  assert.equal(formatBytes(NaN), '');
  assert.equal(isTiffFile(file('a.TIF')), true);
  assert.equal(isTiffFile(file('a.tiff')), true);
  assert.equal(isTiffFile(file('a.png')), false);
  assert.equal(isTiffFile(null), false);
});

test('humanizeIdentifier makes backend identifiers readable', () => {
  assert.equal(humanizeIdentifier('RuleBasedRouter'), 'Rule Based Router');
  assert.equal(humanizeIdentifier('Hybrid-CDVQA-Engine'), 'Hybrid CDVQA Engine');
  assert.equal(humanizeIdentifier('CHANGE_DETECTION'), 'CHANGE DETECTION');
  assert.equal(humanizeIdentifier('major_regions_detected'), 'major regions detected');
  assert.equal(humanizeIdentifier(null), '');
});

// ------------------------------------------------------------------ map config

test('formatLatLng gives hemispheres and wraps longitudes past the date line', () => {
  assert.equal(formatLatLng({ lat: 12.9716, lng: 77.5946 }), '12.9716°N  77.5946°E');
  assert.equal(formatLatLng({ lat: -33.8688, lng: -70.6693 }), '33.8688°S  70.6693°W');
  assert.equal(formatLatLng({ lat: 0, lng: 190 }), '0.0000°N  170.0000°W');
  assert.equal(formatLatLng({ lat: 0, lng: -190 }), '0.0000°N  170.0000°E');
  assert.equal(formatLatLng(null), '');
});

test('map defaults use a keyless imagery source with attribution', () => {
  assert.match(MAP_CONFIG.imageryUrl, /\{z\}.*\{y\}.*\{x\}/);
  assert.ok(MAP_CONFIG.imageryAttribution.length > 0);
  assert.ok(MAP_CONFIG.minZoom < MAP_CONFIG.maxZoom);
});

// ------------------------------------------------------------------ preview / report / health

test('requestPreview posts the file and modality and resolves to the PNG blob', async () => {
  let seen;
  const png = new Blob(['x'], { type: 'image/png' });
  const fetchImpl = async (url, init) => { seen = { url, init }; return { ok: true, status: 200, blob: async () => png }; };
  const out = await requestPreview({ file: file('scene.tif'), modality: 'sar', fetchImpl, baseUrl: 'http://api' });
  assert.equal(out, png);
  assert.equal(seen.url, 'http://api/preview');
  assert.equal(seen.init.method, 'POST');
  assert.equal(seen.init.body.get('modality'), 'sar');
  assert.equal(seen.init.body.get('image').name, 'scene.tif');
});

test('requestPreview gives readable errors', async () => {
  const down = () => Promise.reject(new TypeError('Failed to fetch'));
  await assert.rejects(requestPreview({ file: file('a.tif'), fetchImpl: down, baseUrl: 'http://api' }), /Cannot reach the server at http:\/\/api to render a preview/);
  const bad = async () => response(400, JSON.stringify({ detail: "Could not read 'a.tif' as a raster image: boom" }));
  await assert.rejects(requestPreview({ file: file('a.tif'), fetchImpl: bad }), /Could not read 'a.tif'/);
  const proxy = async () => response(502, '<html>Bad Gateway</html>');
  await assert.rejects(requestPreview({ file: file('a.tif'), fetchImpl: proxy }), /Preview failed \(HTTP 502\)/);
});

test('requestReport sends the session and the whole conversation', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return { ok: true, status: 200, json: async () => ({ report_download_url: '/reports/x/report_chat.pdf', report_error: null }) }; };
  const history = [{ role: 'user', content: 'hi' }, { role: 'ai', content: 'hello' }];
  const out = await requestReport({ sessionId: 'abc', history, fetchImpl, baseUrl: 'http://api' });
  assert.equal(out.report_download_url, '/reports/x/report_chat.pdf');
  assert.equal(seen.url, 'http://api/report');
  assert.equal(seen.init.body.get('session_id'), 'abc');
  assert.deepEqual(JSON.parse(seen.init.body.get('chat_history')), history);
});

test('requestReport gives readable errors', async () => {
  await assert.rejects(requestReport({ sessionId: 'x', fetchImpl: async () => response(404, JSON.stringify({ detail: 'No such analysis run.' })) }), /No such analysis run/);
  await assert.rejects(requestReport({ sessionId: 'x', fetchImpl: () => Promise.reject(new Error('down')), baseUrl: 'http://api' }), /Cannot reach the server at http:\/\/api to build the report/);
});

test('checkHealth maps every backend state, and never throws', async () => {
  const health = (body, status = 200) => async () => ({ ok: status < 300, status, json: async () => body });
  assert.equal((await checkHealth({ fetchImpl: health({ status: 'ok', model_loaded: true, model_error: null }) })).state, 'ready');
  assert.equal((await checkHealth({ fetchImpl: health({ status: 'ok', model_loaded: false, model_error: null }) })).state, 'loading');
  const failed = await checkHealth({ fetchImpl: health({ status: 'ok', model_loaded: false, model_error: 'RuntimeError: no weights' }) });
  assert.equal(failed.state, 'model-error');
  assert.match(failed.message, /no weights/);
  assert.equal((await checkHealth({ fetchImpl: health({}, 500) })).state, 'offline');
  assert.equal((await checkHealth({ fetchImpl: () => Promise.reject(new TypeError('Failed to fetch')) })).state, 'offline');
  assert.equal((await checkHealth({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => { throw new Error('not json'); } }) })).state, 'offline');
});

test('checkHealth gives up on a hung backend instead of waiting forever', async () => {
  const hung = (_url, init) => new Promise((_res, rej) => init.signal.addEventListener('abort', () => rej(new DOMException('aborted', 'AbortError'))));
  const started = Date.now();
  const out = await checkHealth({ fetchImpl: hung, timeoutMs: 30 });
  assert.equal(out.state, 'offline');
  assert.ok(Date.now() - started < 1000);
});
