import test from 'node:test';
import assert from 'node:assert/strict';
import { IDENTITY, MAX_SCALE, MIN_SCALE, clampView, focusView, isZoomed, panBy, wheelFactor, zoomAt, zoomLabel } from './viewMath.js';

const box = { width: 800, height: 400 };
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-6, `${a} ~ ${b}`);

test('the identity view is not zoomed and cannot be panned', () => {
  assert.equal(isZoomed(IDENTITY), false);
  assert.deepEqual(panBy(IDENTITY, 100, -50, box), { scale: 1, x: 0, y: 0 });
});

test('zooming keeps the point under the cursor where it was', () => {
  const view = zoomAt(IDENTITY, 2, 600, 100, box);
  near(view.scale, 2);
  // the image point that was under (600, 100) must still be under (600, 100): (600 - x) / scale is unchanged
  near((600 - view.x) / view.scale, 600);
  near((100 - view.y) / view.scale, 100);
});

test('zooming at a corner stays anchored to that corner', () => {
  assert.deepEqual(zoomAt(IDENTITY, 4, 0, 0, box), { scale: 4, x: 0, y: 0 });
  const bottomRight = zoomAt(IDENTITY, 4, 800, 400, box);
  near(bottomRight.x, 800 - 3200);
  near(bottomRight.y, 400 - 1600);
});

test('scale is limited to the allowed range', () => {
  assert.equal(zoomAt(IDENTITY, 0.1, 10, 10, box).scale, MIN_SCALE);
  assert.equal(zoomAt(IDENTITY, 1000, 10, 10, box).scale, MAX_SCALE);
  let view = IDENTITY;
  for (let i = 0; i < 50; i += 1) view = zoomAt(view, 1.5, 400, 200, box);
  assert.equal(view.scale, MAX_SCALE);
});

test('zooming out fully returns to the identity view', () => {
  let view = zoomAt(IDENTITY, 8, 123, 321, box);
  view = zoomAt(view, 1 / 8, 123, 321, box);
  near(view.scale, 1);
  near(view.x, 0);
  near(view.y, 0);
});

test('panning never lets the image leave the viewport', () => {
  const zoomed = zoomAt(IDENTITY, 2, 400, 200, box);
  const farRight = panBy(zoomed, 5000, 5000, box);
  assert.equal(farRight.x, 0);
  assert.equal(farRight.y, 0);
  const farLeft = panBy(zoomed, -5000, -5000, box);
  assert.equal(farLeft.x, 800 - 1600);
  assert.equal(farLeft.y, 400 - 800);
});

test('clampView repairs an out-of-range view', () => {
  assert.deepEqual(clampView({ scale: 0.2, x: 50, y: -50 }, box), { scale: 1, x: 0, y: 0 });
  const v = clampView({ scale: 2, x: 999, y: -999 }, box);
  assert.equal(v.x, 0);
  assert.equal(v.y, 400 - 800);
});

test('the wheel factor is symmetric and bounded', () => {
  near(wheelFactor(100) * wheelFactor(-100), 1);
  assert.ok(wheelFactor(-100) > 1 && wheelFactor(100) < 1);
  assert.equal(wheelFactor(0), 1);
  assert.equal(wheelFactor(1e9), wheelFactor(240), 'a huge wheel delta cannot produce a huge jump');
});

test('the zoom label is a percentage', () => {
  assert.equal(zoomLabel(IDENTITY), '100%');
  assert.equal(zoomLabel({ scale: 2.5, x: 0, y: 0 }), '250%');
  assert.equal(isZoomed({ scale: 1.0000001, x: 0, y: 0 }), false);
  assert.equal(isZoomed({ scale: 1.5, x: 0, y: 0 }), true);
});


test('focusing a region centres it and zooms in, never past the limits', () => {
  // a square image in a wide box sits in the middle 400 px; the top-left quarter of it
  const view = focusView([0, 0, 0.25, 0.25], 1, box, 0);
  assert.ok(view.scale > 1 && view.scale <= MAX_SCALE);
  // region centre in container px: image spans x 200..600, y 0..400 -> (250, 50); it must land at the container centre or be clamped to the edge
  const cx = (250 * view.scale + view.x);
  const cy = (50 * view.scale + view.y);
  assert.ok(cx >= 0 && cx <= 800 && cy >= 0 && cy <= 400);
  assert.ok(Math.abs(cy - 200) < 200);
  assert.ok(view.x <= 0 && view.y <= 0);
});

test('focusing the whole image, or with bad input, gives the fitted view', () => {
  assert.deepEqual(focusView([0, 0, 1, 1], 2, box), IDENTITY);
  assert.deepEqual(focusView(null, 2, box), IDENTITY);
  assert.deepEqual(focusView([0, 0, 0.5, 0.5], 0, box), IDENTITY);
  assert.deepEqual(focusView([0, 0, 0.5, 0.5], 1, { width: 0, height: 0 }), IDENTITY);
});

test('a region in the middle lands in the middle of the container', () => {
  const view = focusView([0.4, 0.4, 0.6, 0.6], 2, box, 0.2);       // image fills the box exactly (2:1)
  near(400 * view.scale + view.x, 400);
  near(200 * view.scale + view.y, 200);
});
