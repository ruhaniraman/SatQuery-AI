// Place search for the map's search box. Uses OpenStreetMap Nominatim (no API key, CORS enabled). Its usage
// policy allows about one request per second and discourages true search-as-you-type, so the box (CoordinateBox.jsx)
// only fires this while typing after a debounce (AUTOSEARCH_DEBOUNCE_MS) and a minimum length (AUTOSEARCH_MIN_CHARS),
// cancelling the previous request first; pressing Go/Enter still searches immediately. Point it at another geocoder
// with VITE_GEOCODE_URL (same response shape as Nominatim's `jsonv2`).
// Pure apart from the injected fetch: tested in plain Node.
import { parseCoordinates } from './coordinates.js';

export const GEOCODE_URL = import.meta.env?.VITE_GEOCODE_URL || 'https://nominatim.openstreetmap.org/search';

// A gentle compromise, not a green light: Nominatim's policy is written for deliberate lookups, not
// keystroke traffic. Debouncing and a minimum length keep it to roughly one request per pause in typing,
// which is fine for this app's own light/dev usage but would need a dedicated autocomplete provider (e.g.
// Photon) before relying on it at real production volume.
export const AUTOSEARCH_DEBOUNCE_MS = 500;
export const AUTOSEARCH_MIN_CHARS = 3;

// Decide what a typed string is. Anything made only of digits, N/S/E/W and separators (and that has a digit)
// is a coordinate, so a bad one reports what is wrong with the numbers; everything else is a place name.
export function interpretSearch(input) {
  const text = String(input ?? '').trim();
  if (!text) return { kind: 'empty' };
  const parsed = parseCoordinates(text);
  if (parsed.ok) return { kind: 'coordinates', lat: parsed.lat, lng: parsed.lng };
  const stripped = text.replace(/latitude|longitude|lat|long|lon|lng/gi, ' ');
  if (/\d/.test(stripped) && /^[\d\sNSEWnsew.,;:+\-−–—°º′’'″”"]+$/.test(stripped)) return { kind: 'bad-coordinates', error: parsed.error };
  return { kind: 'place', query: text };
}

// A short label for a result: the first parts of Nominatim's long display name
export function shortName(displayName) {
  const parts = String(displayName ?? '').split(',').map((p) => p.trim()).filter(Boolean);
  if (parts.length <= 3) return parts.join(', ');
  return `${parts.slice(0, 2).join(', ')}, ${parts[parts.length - 1]}`;
}

// [south, north, west, east] strings -> Leaflet bounds [[south, west], [north, east]] (null if unusable)
function toBounds(box) {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const [s, n, w, e] = box.map(Number);
  if (![s, n, w, e].every(Number.isFinite) || n < s || e < w) return null;
  return [[s, w], [n, e]];
}

export async function searchPlaces(query, { fetchImpl = fetch, baseUrl = GEOCODE_URL, limit = 5, signal } = {}) {
  const url = `${baseUrl}?${new URLSearchParams({ q: query, format: 'jsonv2', limit: String(limit), addressdetails: '0' })}`;
  let response;
  try {
    response = await fetchImpl(url, { headers: { Accept: 'application/json' }, signal });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new Error('Could not reach the place search service. Check your connection and try again.', { cause: err });
  }
  if (!response.ok) throw new Error(`The place search service answered ${response.status}. Try again in a moment.`);
  let rows;
  try {
    rows = await response.json();
  } catch {
    throw new Error('The place search service sent an unreadable answer.');
  }
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => ({
      lat: Number(row.lat),
      lng: Number(row.lon),
      name: shortName(row.display_name),
      fullName: row.display_name || '',
      bounds: toBounds(row.boundingbox),
    }))
    .filter((r) => Number.isFinite(r.lat) && Number.isFinite(r.lng));
}
