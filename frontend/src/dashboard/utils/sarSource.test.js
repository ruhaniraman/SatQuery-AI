import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CDSE_WMTS, SAR_STORAGE_KEY, buildCdseTemplate, clearSarSource, describeSarSource, loadSarSource, normaliseTimeRange,
  resolveSarConfig, sampleTileUrl, saveSarSource, testSarTemplate, validateTemplate,
} from './sarSource.js';

const ID = '1a2b3c4d-1111-2222-3333-444455556666';
const memory = (initial = {}) => {
  const data = { ...initial };
  return { getItem: (k) => (k in data ? data[k] : null), setItem: (k, v) => { data[k] = String(v); }, removeItem: (k) => { delete data[k]; }, data };
};

test('the Copernicus WMTS URL is built in the documented format', () => {
  const built = buildCdseTemplate({ instanceId: ID, layer: 'S1-VV' });
  assert.equal(built.ok, true);
  assert.equal(built.template, `${CDSE_WMTS}/${ID}?SERVICE=WMTS&VERSION=1.0.0&REQUEST=GetTile&LAYER=S1-VV&TILEMATRIXSET=PopularWebMercator256&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/png`);
  assert.ok(built.template.includes('{z}') && built.template.includes('{y}') && built.template.includes('{x}'), 'placeholders stay literal for Leaflet');
  assert.equal(validateTemplate(built.template).ok, true, 'what we build is a valid template');
});

test('an optional time range becomes an ISO interval; names and spaces are encoded', () => {
  const built = buildCdseTemplate({ instanceId: `  ${ID}  `, layer: ' Sentinel 1 VV ', timeRange: '2024-01-01/2024-03-31' });
  assert.equal(built.ok, true);
  assert.match(built.template, /LAYER=Sentinel%201%20VV&/);
  assert.match(built.template, /&TIME=2024-01-01T00%3A00%3A00Z%2F2024-03-31T23%3A59%3A59Z$/);
  assert.doesNotMatch(buildCdseTemplate({ instanceId: ID, layer: 'L' }).template, /TIME=/, 'no time means the latest data');
});

test('bad instance ids, layers and dates get a reason, never a broken URL', () => {
  const bad = (input, pattern) => { const r = buildCdseTemplate(input); assert.equal(r.ok, false); assert.match(r.error, pattern); assert.equal(r.template, undefined); };
  bad({ instanceId: '', layer: 'L' }, /instance ID/);
  bad({ instanceId: 'short', layer: 'L' }, /instance ID/);
  bad({ instanceId: 'has spaces in it here', layer: 'L' }, /instance ID/);
  bad({ instanceId: ID, layer: '   ' }, /layer name/);
  bad({ instanceId: ID, layer: 'a&b=c' }, /layer name/);
  bad({ instanceId: ID, layer: 'L', timeRange: 'last month' }, /date like/);
  bad({ instanceId: ID, layer: 'L', timeRange: '2024-03-31/2024-01-01' }, /after the end/);
  bad({ instanceId: ID, layer: 'L', timeRange: '2024-02-31' }, /does not exist|after the end/);
});

test('time ranges: empty, one day, a range', () => {
  assert.deepEqual(normaliseTimeRange(''), { ok: true, value: '' });
  assert.deepEqual(normaliseTimeRange(undefined), { ok: true, value: '' });
  assert.equal(normaliseTimeRange('2024-05-01').value, '2024-05-01T00:00:00Z/2024-05-01T23:59:59Z');
  assert.equal(normaliseTimeRange('2024-05-01/2024-05-09').value, '2024-05-01T00:00:00Z/2024-05-09T23:59:59Z');
  assert.equal(normaliseTimeRange('2024-05-01/2024-05-09/2024-05-10').ok, false);
});

test('a custom tile URL needs http(s) and all three placeholders', () => {
  assert.equal(validateTemplate('https://tiles.example.com/{z}/{x}/{y}.png').ok, true);
  assert.equal(validateTemplate('http://localhost:8080/{z}/{y}/{x}').ok, true);
  assert.match(validateTemplate('ftp://x/{z}/{x}/{y}').error, /https/);
  assert.match(validateTemplate('').error, /https/);
  assert.match(validateTemplate('https://tiles.example.com/{z}/{x}.png').error, /needs \{y\}/);
  assert.match(validateTemplate('https://tiles.example.com/tile.png').error, /needs \{z\}, \{x\}, \{y\}/);
  assert.equal(validateTemplate(null).ok, false);
});

test('the test tile URL has real numbers in place of the placeholders', () => {
  assert.equal(sampleTileUrl('https://t.example/{z}/{x}/{y}.png'), 'https://t.example/3/4/3.png');
  assert.equal(sampleTileUrl('https://t.example/?z={z}&r={y}&c={x}'), 'https://t.example/?z=3&r=3&c=4');
});

const reply = (status, contentType) => async () => ({ ok: status >= 200 && status < 300, status, headers: { get: () => contentType } });

test('the connection test tells success, a wrong id/layer, a refusal and an unreadable answer apart', async () => {
  const T = 'https://t.example/{z}/{x}/{y}';
  assert.equal((await testSarTemplate(T, { fetchImpl: reply(200, 'image/png') })).ok, true);
  assert.equal((await testSarTemplate(T, { fetchImpl: reply(200, undefined) })).ok, true, 'a missing content type is not held against it');
  const html = await testSarTemplate(T, { fetchImpl: reply(200, 'text/html; charset=utf-8') });
  assert.equal(html.ok, false);
  assert.match(html.message, /not with an image \(text\/html\)/);
  assert.match((await testSarTemplate(T, { fetchImpl: reply(400, 'application/xml') })).message, /does not know this instance or layer/);
  assert.match((await testSarTemplate(T, { fetchImpl: reply(404) })).message, /does not know/);
  assert.match((await testSarTemplate(T, { fetchImpl: reply(403) })).message, /refused/);
  assert.match((await testSarTemplate(T, { fetchImpl: reply(500) })).message, /HTTP 500/);
  const blocked = await testSarTemplate(T, { fetchImpl: () => Promise.reject(new TypeError('Failed to fetch')) });
  assert.deepEqual([blocked.ok, blocked.unreadable], [false, true]);
  assert.match(blocked.message, /CORS/);
});

test('a saved source is read back; corrupt or unsafe values are ignored', () => {
  const storage = memory();
  const source = { kind: 'cdse', template: buildCdseTemplate({ instanceId: ID, layer: 'L' }).template, layer: 'L', instanceId: ID };
  assert.equal(saveSarSource(source, storage), true);
  assert.deepEqual(loadSarSource(storage), source);
  clearSarSource(storage);
  assert.equal(loadSarSource(storage), null);
  assert.equal(SAR_STORAGE_KEY in storage.data, false);

  assert.equal(saveSarSource({ template: 'not a template' }, storage), false, 'invalid sources are never saved');
  assert.equal(saveSarSource(null, storage), false);
  for (const junk of ['{not json', 'null', '"text"', JSON.stringify({ template: 42 }), JSON.stringify({ template: 'javascript:{z}{x}{y}' })]) {
    assert.equal(loadSarSource(memory({ [SAR_STORAGE_KEY]: junk })), null, junk);
  }
});

test('blocked storage never breaks the page', () => {
  const throwing = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); }, removeItem() { throw new Error('denied'); } };
  assert.equal(loadSarSource(throwing), null);
  assert.equal(saveSarSource({ template: 'https://t/{z}/{x}/{y}' }, throwing), false);
  assert.doesNotThrow(() => clearSarSource(throwing));
  assert.equal(loadSarSource(null), null);
});

test('a connected source overrides the environment settings', () => {
  const env = { sarUrl: '', hasSar: false, sarAttribution: 'SAR imagery', sarMaxNativeZoom: 14, sarCrossOrigin: false, sarCanExportView: false, other: 1 };
  assert.equal(resolveSarConfig(env, null), env, 'nothing connected: the environment is used untouched');
  const merged = resolveSarConfig(env, { template: 'https://t/{z}/{x}/{y}', attribution: 'My SAR', maxZoom: 12 });
  assert.deepEqual([merged.sarUrl, merged.hasSar, merged.sarAttribution, merged.sarMaxNativeZoom, merged.sarCanExportView, merged.other], ['https://t/{z}/{x}/{y}', true, 'My SAR', 12, true, 1]);
  assert.equal(resolveSarConfig(env, { template: 'https://t/{z}/{x}/{y}' }).sarAttribution, 'SAR imagery', 'attribution falls back');
});

test('a source is described in one line', () => {
  assert.equal(describeSarSource(null), '');
  assert.equal(describeSarSource({ kind: 'custom', template: 'x' }), 'Custom tile URL');
  assert.equal(describeSarSource({ kind: 'cdse', layer: 'S1-VV' }), 'Copernicus · S1-VV');
  assert.equal(describeSarSource({ kind: 'cdse', layer: 'S1-VV', timeRange: '2024-01-01/2024-03-31' }), 'Copernicus · S1-VV · 2024-01-01/2024-03-31');
});
