import test from 'node:test';
import assert from 'node:assert/strict';
import { captureMapView, CAPTURE_FILE_NAME, currentTiles, isSettled, planTileDraws, whenSettled } from './mapCapture.js';

const rect = (left, top, width = 256, height = 256) => ({ rect: { left, top, width, height } });
const noWait = async () => {};

// ------------------------------------------------------------------ planTileDraws

test('adjacent tiles share edges exactly, even with fractional positions (no seams)', () => {
  // Leaflet positions tiles at fractional pixels during/after zooming
  const tiles = [rect(10.4, 20.6, 256.3), rect(266.7, 20.6, 256.3), rect(10.4, 276.9, 256.3)];
  const [a, b, c] = planTileDraws(tiles, { left: 0, top: 0 }, { width: 800, height: 800 });
  assert.equal(a.x + a.w, b.x, 'right edge of A is left edge of B');
  assert.equal(a.y + a.h, c.y, 'bottom edge of A is top edge of C');
});

test('positions are relative to the map container, not the page', () => {
  const [t] = planTileDraws([rect(500, 300)], { left: 480, top: 280 }, { width: 800, height: 600 });
  assert.deepEqual([t.x, t.y, t.w, t.h], [20, 20, 256, 256]);
});

test('tiles fully outside the canvas are skipped; partially visible ones are kept', () => {
  const size = { width: 500, height: 400 };
  const plan = planTileDraws(
    [rect(-300, 0), rect(-100, 0), rect(480, 100), rect(500, 0), rect(0, 400), rect(100, -300)],
    { left: 0, top: 0 }, size,
  );
  assert.deepEqual(plan.map((p) => p.index), [1, 2], 'left-of, right-of, below and above tiles are dropped');
});

test('a degenerate sliver still draws at least one pixel', () => {
  const [t] = planTileDraws([rect(10, 10, 0.2, 0.2)], { left: 0, top: 0 }, { width: 100, height: 100 });
  assert.ok(t.w >= 1 && t.h >= 1);
});

// ------------------------------------------------------------------ tile selection / settling

test('only loaded tiles of the current zoom are used (old zoom tiles linger while new ones load)', () => {
  const el = (n) => ({ id: n });
  const layer = { _tiles: {
    a: { current: true, loaded: 123, el: el('a') },
    b: { current: false, loaded: 123, el: el('b') },      // previous zoom level
    c: { current: true, loaded: undefined, el: el('c') }, // still loading
    d: { current: true, loaded: 5, el: null },            // no element
  } };
  assert.deepEqual(currentTiles(layer).map((t) => t.el.id), ['a']);
  assert.deepEqual(currentTiles({}), []);
  assert.deepEqual(currentTiles(null), []);
});

test('isSettled is false while tiles load or a zoom animates', () => {
  assert.equal(isSettled({}, {}), true);
  assert.equal(isSettled({}, { _loading: true }), false);
  assert.equal(isSettled({ _animatingZoom: true }, {}), false);
});

test('whenSettled waits for the map, then resolves', async () => {
  const layer = { _loading: true };
  let polls = 0;
  await whenSettled({}, layer, { wait: async () => { if (++polls === 3) layer._loading = false; } });
  assert.equal(polls, 3);
});

test('whenSettled gives up with a user-facing message instead of hanging', async () => {
  await assert.rejects(
    whenSettled({}, { _loading: true }, { timeout: 500, interval: 100, wait: noWait }),
    { message: 'The map is still loading. Try again in a moment.' },
  );
});

// ------------------------------------------------------------------ captureMapView (fake DOM / Leaflet)

function fakeScene({ tiles, size = { width: 400, height: 300 }, toBlob, layer = {} } = {}) {
  const draws = [];
  const ctx = {
    fillStyle: '', fillRect() {},
    drawImage: (...args) => draws.push(args),
  };
  const canvas = {
    width: 0, height: 0,
    getContext: () => ctx,
    toBlob: toBlob || ((cb) => cb(new Blob(['png-bytes'], { type: 'image/png' }))),
  };
  const map = {
    getContainer: () => ({ getBoundingClientRect: () => ({ left: 100, top: 50 }), clientWidth: size.width, clientHeight: size.height }),
    getBounds: () => ({ getSouth: () => 12.5, getWest: () => 77.1, getNorth: () => 13.5, getEast: () => 78.2 }),
    getZoom: () => 14,
  };
  const tileList = (tiles || []).map((t, i) => ({ current: true, loaded: 1, el: { name: `tile${i}`, getBoundingClientRect: () => t } }));
  const layerObj = { _tiles: Object.fromEntries(tileList.map((t, i) => [String(i), t])), ...layer };
  return { canvas, map, layer: layerObj, draws, doc: { createElement: (tag) => { assert.equal(tag, 'canvas'); return canvas; } } };
}

test('captureMapView draws each tile where it sits and returns a PNG file with the view bounds', async () => {
  const scene = fakeScene({ tiles: [
    { left: 100, top: 50, width: 256, height: 256 },
    { left: 356, top: 50, width: 256, height: 256 },
  ] });
  const out = await captureMapView(scene.map, scene.layer, { doc: scene.doc });

  assert.equal(scene.canvas.width, 400);
  assert.equal(scene.canvas.height, 300);
  assert.deepEqual(scene.draws.map((d) => [d[0].name, d[1], d[2], d[3], d[4]]), [['tile0', 0, 0, 256, 256], ['tile1', 256, 0, 256, 256]]);
  assert.ok(out.file instanceof File);
  assert.equal(out.file.name, CAPTURE_FILE_NAME);
  assert.equal(out.file.type, 'image/png');
  assert.deepEqual(out.bounds, [[12.5, 77.1], [13.5, 78.2]]);
  assert.deepEqual([out.zoom, out.width, out.height], [14, 400, 300]);
});

test('captureMapView refuses to produce a blank picture when nothing is loaded', async () => {
  const scene = fakeScene({ tiles: [] });
  await assert.rejects(captureMapView(scene.map, scene.layer, { doc: scene.doc }), { message: 'No map imagery is loaded yet.' });
});

test('captureMapView explains a CORS-blocked (tainted) canvas', async () => {
  const scene = fakeScene({
    tiles: [{ left: 100, top: 50, width: 256, height: 256 }],
    toBlob: () => { const e = new Error('Tainted canvases may not be exported.'); e.name = 'SecurityError'; throw e; },
  });
  await assert.rejects(
    captureMapView(scene.map, scene.layer, { doc: scene.doc }),
    (err) => /does not allow its imagery to be exported/.test(err.message) && /CORS/.test(err.message),
  );
});

test('captureMapView reports an encoding failure, and other errors pass through unchanged', async () => {
  const empty = fakeScene({ tiles: [{ left: 100, top: 50 }], toBlob: (cb) => cb(null) });
  await assert.rejects(captureMapView(empty.map, empty.layer, { doc: empty.doc }), { message: 'Could not encode the map view.' });
  const odd = fakeScene({ tiles: [{ left: 100, top: 50 }], toBlob: () => { throw new RangeError('boom'); } });
  await assert.rejects(captureMapView(odd.map, odd.layer, { doc: odd.doc }), { name: 'RangeError', message: 'boom' });
});
