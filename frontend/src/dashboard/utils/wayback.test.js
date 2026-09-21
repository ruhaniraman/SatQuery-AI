import test from 'node:test';
import assert from 'node:assert/strict';
import { formatReleaseDate, loadWaybackReleases, nearestReleaseIndex, parseWaybackConfig, yearTicks } from './wayback.js';
import { SAME_AREA_MIN, boundsOverlap, captureFileName, describeCapture } from './views.js';

const TILE = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile';
const entry = (id, date) => ({ itemID: `x${id}`, itemTitle: `World Imagery (Wayback ${date})`, itemURL: `${TILE}/${id}/{level}/{row}/{col}` });
// The shape of the real file (order is NOT chronological)
const CONFIG = {
  26334: entry(26334, '2026-08-05'),
  31144: entry(31144, '2014-06-11'),
  32337: entry(32337, '2018-05-16'),
  31026: entry(31026, '2017-02-27'),
};

test('the release list becomes an oldest-first list with Leaflet placeholders', () => {
  const releases = parseWaybackConfig(CONFIG);
  assert.deepEqual(releases.map((r) => r.date), ['2014-06-11', '2017-02-27', '2018-05-16', '2026-08-05']);
  assert.equal(releases[0].id, '31144');
  assert.equal(releases[0].template, `${TILE}/31144/{z}/{y}/{x}`);
  assert.ok(releases.every((r) => !r.template.includes('{level}') && !r.template.includes('{row}') && !r.template.includes('{col}')));
});

test('one release per date (the newest id wins) and broken entries are skipped', () => {
  const releases = parseWaybackConfig({
    100: entry(100, '2020-01-01'),
    200: entry(200, '2020-01-01'),
    300: { itemTitle: 'World Imagery (no date here)', itemURL: `${TILE}/300/{level}/{row}/{col}` },
    400: { itemTitle: 'World Imagery (Wayback 2021-01-01)', itemURL: 'https://example.com/no-placeholders' },
    500: { itemTitle: 'World Imagery (Wayback 2022-01-01)' },
    600: null,
  });
  assert.deepEqual(releases.map((r) => [r.date, r.id]), [['2020-01-01', '200']]);
  assert.deepEqual(parseWaybackConfig(null), []);
  assert.deepEqual(parseWaybackConfig(undefined), []);
  assert.deepEqual(parseWaybackConfig('nonsense'), []);
});

test('dates are shown as calendar dates without timezone drift', () => {
  assert.equal(formatReleaseDate('2014-06-11'), '11 Jun 2014');
  assert.equal(formatReleaseDate('2026-01-01'), '1 Jan 2026');
  assert.equal(formatReleaseDate('2020-12-31'), '31 Dec 2020');
  assert.equal(formatReleaseDate('garbage'), 'garbage');
  assert.equal(formatReleaseDate(undefined), '');
});

test('the nearest release to a date is found, ties going to the older one', () => {
  const releases = parseWaybackConfig(CONFIG);
  assert.equal(nearestReleaseIndex(releases, '2014-01-01'), 0);
  assert.equal(nearestReleaseIndex(releases, '2017-05-01'), 1);
  assert.equal(nearestReleaseIndex(releases, '2030-01-01'), 3);
  assert.equal(nearestReleaseIndex(releases, '2018-05-16'), 2);
  assert.equal(nearestReleaseIndex([], '2020-01-01'), -1);
  const tie = [{ date: '2020-01-01' }, { date: '2020-01-03' }];
  assert.equal(nearestReleaseIndex(tie, '2020-01-02'), 0);
});

test('year ticks are the first release of each year and never crowd', () => {
  const releases = parseWaybackConfig(CONFIG);
  assert.deepEqual(yearTicks(releases), [
    { year: '2014', index: 0 }, { year: '2017', index: 1 }, { year: '2018', index: 2 }, { year: '2026', index: 3 },
  ]);
  const many = Array.from({ length: 40 }, (_, i) => ({ date: `${2000 + i}-06-01` }));
  const ticks = yearTicks(many, 6);
  assert.equal(ticks.length, 6);
  assert.deepEqual([ticks[0].year, ticks[5].year], ['2000', '2039'], 'the first and last years are always labelled');
  assert.ok(ticks.every((t, i) => i === 0 || t.index > ticks[i - 1].index), 'in order, no duplicates');
  assert.deepEqual(yearTicks([]), []);
});

test('loadWaybackReleases returns parsed releases or a readable error', async () => {
  const ok = async () => ({ ok: true, status: 200, json: async () => CONFIG });
  assert.equal((await loadWaybackReleases({ fetchImpl: ok })).length, 4);
  await assert.rejects(loadWaybackReleases({ fetchImpl: () => Promise.reject(new TypeError('Failed to fetch')) }), /Could not reach the historical imagery service/);
  await assert.rejects(loadWaybackReleases({ fetchImpl: async () => ({ ok: false, status: 503 }) }), /HTTP 503/);
  await assert.rejects(loadWaybackReleases({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => { throw new Error('bad json'); } }) }), /could not be read/);
  await assert.rejects(loadWaybackReleases({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }) }), /empty/);
});

// ------------------------------------------------------------------ views

const BOX = [[23.7, 86.3], [23.9, 86.5]];

test('identical, shifted and unrelated captures are told apart by overlap', () => {
  assert.equal(boundsOverlap(BOX, BOX), 1);
  const nudged = [[23.7, 86.31], [23.9, 86.51]];                         // 5% of the width
  const overlap = boundsOverlap(BOX, nudged);
  assert.ok(overlap > SAME_AREA_MIN && overlap < 1, `${overlap}`);
  const halfway = [[23.7, 86.4], [23.9, 86.6]];
  assert.ok(boundsOverlap(BOX, halfway) < SAME_AREA_MIN);
  assert.equal(boundsOverlap(BOX, [[10, 10], [11, 11]]), 0, 'a different place shares nothing');
});

test('a zoom difference lowers the overlap even when the centre is the same', () => {
  const zoomedIn = [[23.75, 86.35], [23.85, 86.45]];                     // a quarter of the area
  assert.ok(Math.abs(boundsOverlap(BOX, zoomedIn) - 0.25) < 1e-9);
});

test('missing or malformed bounds give null, never a wrong number', () => {
  for (const bad of [undefined, null, [], [[1, 2]], [[NaN, 2], [3, 4]], [[5, 5], [1, 1]], 'x']) {
    assert.equal(boundsOverlap(BOX, bad), null);
    assert.equal(boundsOverlap(bad, BOX), null);
  }
});

test('capture file names carry the date and stay filesystem-safe', () => {
  assert.equal(captureFileName('Wayback 2014-06-11'), 'map-view-wayback-2014-06-11.png');
  assert.equal(captureFileName('Optical'), 'map-view-optical.png');
  assert.equal(captureFileName('  A/B: c?  '), 'map-view-a-b-c.png');
  assert.equal(captureFileName(''), 'map-view.png');
  assert.equal(captureFileName(null), 'map-view.png');
});

test('a capture is described in one line, or not at all', () => {
  assert.equal(describeCapture({ label: 'Wayback 2014-06-11', zoom: 12 }), 'Wayback 2014-06-11 · zoom 12');
  assert.equal(describeCapture({ label: 'SAR' }), 'SAR');
  assert.equal(describeCapture(null), '');
  assert.equal(describeCapture({ zoom: 5 }), '');
});
