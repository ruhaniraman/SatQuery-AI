// Helpers about captured map views (pure, tested in Node).

// Two captures only make a change-detection pair if they show the same ground. bounds are
// [[south, west], [north, east]]. Returns intersection / union of the two boxes, 0 (nothing shared)
// to 1 (identical), or null when either box is missing or malformed.
export function boundsOverlap(a, b) {
  const box = (bounds) => {
    if (!Array.isArray(bounds) || bounds.length !== 2) return null;
    const [[south, west], [north, east]] = bounds;
    return [south, west, north, east].every(Number.isFinite) && north > south && east > west ? { south, west, north, east } : null;
  };
  const p = box(a);
  const q = box(b);
  if (!p || !q) return null;
  const h = Math.max(0, Math.min(p.north, q.north) - Math.max(p.south, q.south));
  const w = Math.max(0, Math.min(p.east, q.east) - Math.max(p.west, q.west));
  const inter = h * w;
  const union = (p.north - p.south) * (p.east - p.west) + (q.north - q.south) * (q.east - q.west) - inter;
  return union > 0 ? inter / union : null;
}

// Below this, a Before/After pair is flagged: the two captures do not cover the same area
export const SAME_AREA_MIN = 0.85;

// "Wayback 2014-06-11" -> "map-view-wayback-2014-06-11.png"
export function captureFileName(label) {
  const slug = String(label || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return slug ? `map-view-${slug}.png` : 'map-view.png';
}

// One line for a slot's capture: "Wayback 2014-06-11 · zoom 12"
export const describeCapture = (meta) => (meta && meta.label ? `${meta.label}${Number.isFinite(meta.zoom) ? ` · zoom ${meta.zoom}` : ''}` : '');
