// Zoom/pan arithmetic for the image viewer. Pure: tested in plain Node.
//
// The stage (the fitted image area) is the same size as its container, and the view is a CSS transform
// { scale, x, y } applied to it with transform-origin at the top-left. scale >= 1 always, so the stage
// always covers the container once clamped and the image never drifts out of sight.

export const MIN_SCALE = 1;
export const MAX_SCALE = 16;
export const IDENTITY = Object.freeze({ scale: 1, x: 0, y: 0 });

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// Keep the stage covering the container
export function clampView(view, container) {
  const scale = clamp(view.scale, MIN_SCALE, MAX_SCALE);
  return {
    scale,
    x: clamp(view.x, container.width - container.width * scale, 0),
    y: clamp(view.y, container.height - container.height * scale, 0),
  };
}

// Zoom by `factor` keeping the point (px, py), given in container coordinates, fixed on screen
export function zoomAt(view, factor, px, py, container) {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const ratio = scale / view.scale;
  return clampView({ scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio }, container);
}

export const panBy = (view, dx, dy, container) => clampView({ ...view, x: view.x + dx, y: view.y + dy }, container);

// Mouse-wheel / trackpad delta to a zoom factor. Exponential, so zooming in and back out is symmetric.
export const wheelFactor = (deltaY) => Math.exp(clamp(-deltaY, -240, 240) * 0.0018);

export const isZoomed = (view) => view.scale > MIN_SCALE + 1e-6;

export const zoomLabel = (view) => `${Math.round(view.scale * 100)}%`;

// The view that fills the container with a region of the image. `box` is [ymin, xmin, ymax, xmax] as
// fractions of the image; `aspect` is the image's width / height. The image is drawn object-contain, so
// first find where it really sits in the container. Zooms to the region plus `margin` on each side, but
// never out past the fitted view (and never in past MAX_SCALE).
export function focusView(box, aspect, container, margin = 0.2) {
  const { width: cw, height: ch } = container;
  if (!box || !(aspect > 0) || !(cw > 0) || !(ch > 0)) return IDENTITY;
  const fitsWidth = cw / ch <= aspect;
  const dw = fitsWidth ? cw : ch * aspect;
  const dh = fitsWidth ? cw / aspect : ch;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;
  const [ymin, xmin, ymax, xmax] = box;
  const rw = Math.max((xmax - xmin) * dw, 1);
  const rh = Math.max((ymax - ymin) * dh, 1);
  const scale = clamp(Math.min(cw / (rw * (1 + 2 * margin)), ch / (rh * (1 + 2 * margin))), MIN_SCALE, MAX_SCALE);
  const cx = ox + ((xmin + xmax) / 2) * dw;
  const cy = oy + ((ymin + ymax) / 2) * dh;
  return clampView({ scale, x: cw / 2 - cx * scale, y: ch / 2 - cy * scale }, container);
}
