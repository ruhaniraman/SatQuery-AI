// Turns what is currently visible on the Leaflet map into an image file. No dependencies: the loaded
// imagery tiles are composited onto a canvas. The planning/selection helpers are pure (and tested in
// Node); captureMapView() needs a real browser and Leaflet objects.

export const CAPTURE_FILE_NAME = 'map-view.png';
const BACKGROUND = '#070b14';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Where each tile lands on the canvas. Edges are rounded (not widths) so neighbouring tiles share an
// edge exactly and no hairline gaps appear. Tiles completely outside the canvas are skipped.
// tiles: [{ rect: { left, top, width, height } }] in viewport pixels; container: { left, top }.
export function planTileDraws(tiles, container, size) {
  const plan = [];
  tiles.forEach((tile, index) => {
    const x0 = Math.round(tile.rect.left - container.left);
    const y0 = Math.round(tile.rect.top - container.top);
    const x1 = Math.round(tile.rect.left + tile.rect.width - container.left);
    const y1 = Math.round(tile.rect.top + tile.rect.height - container.top);
    if (x1 <= 0 || y1 <= 0 || x0 >= size.width || y0 >= size.height) return;
    plan.push({ index, x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) });
  });
  return plan;
}

// Leaflet keeps tiles from the previous zoom level around while the new ones load. Only tiles of the
// current zoom that have finished loading belong in the picture. (`_tiles` is Leaflet's own registry;
// the version is pinned by the lockfile.)
export function currentTiles(layer) {
  return Object.values(layer?._tiles || {}).filter((tile) => tile.current && tile.loaded && tile.el);
}

export const isSettled = (map, layer) => !layer?._loading && !map?._animatingZoom;

// Resolves once the imagery has finished loading and no zoom animation is running, so the capture
// is not half-blank. Rejects (with a message fit for the user) if that takes too long.
export async function whenSettled(map, layer, { timeout = 8000, interval = 100, wait = sleep } = {}) {
  let waited = 0;
  while (!isSettled(map, layer)) {
    if (waited >= timeout) throw new Error('The map is still loading. Try again in a moment.');
    await wait(interval);
    waited += interval;
  }
}

export async function captureMapView(map, layer, { doc = document } = {}) {
  await whenSettled(map, layer);
  const tiles = currentTiles(layer);
  if (tiles.length === 0) throw new Error('No map imagery is loaded yet.');

  const container = map.getContainer();
  const box = container.getBoundingClientRect();
  const size = { width: container.clientWidth, height: container.clientHeight };
  const canvas = doc.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, size.width, size.height);

  const plan = planTileDraws(tiles.map((t) => ({ rect: t.el.getBoundingClientRect() })), box, size);
  for (const draw of plan) ctx.drawImage(tiles[draw.index].el, draw.x, draw.y, draw.w, draw.h);

  let blob;
  try {
    blob = await new Promise((resolve, reject) => {
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode the map view.'))), 'image/png');
    });
  } catch (err) {
    if (err?.name === 'SecurityError') {
      throw new Error("This map provider does not allow its imagery to be exported (missing CORS headers).");
    }
    throw err;
  }

  const bounds = map.getBounds();
  return {
    file: new File([blob], CAPTURE_FILE_NAME, { type: 'image/png' }),
    // [[south, west], [north, east]]: exactly what an image overlay needs to sit on the same ground
    bounds: [[bounds.getSouth(), bounds.getWest()], [bounds.getNorth(), bounds.getEast()]],
    zoom: map.getZoom(),
    width: size.width,
    height: size.height,
  };
}
