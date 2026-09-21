// Connecting a SAR tile provider from a personal account, without editing files. Pure (no DOM, no React):
// tested in plain Node. The choice is stored in this browser's localStorage and overrides the
// VITE_SAR_* environment settings.
//
// Two ways in:
//   'cdse'   Copernicus Data Space (free account): an INSTANCE ID and a LAYER name from its Sentinel Hub
//            Configuration Utility. The WMTS tile URL is built from them (format from its documentation:
//            SERVICE/VERSION/REQUEST/LAYER/TILEMATRIXSET/TILEMATRIX/TILEROW/TILECOL/FORMAT/TIME).
//   'custom' any tile URL containing {z}, {x} and {y} (Sentinel Hub proper, a self-hosted server, ...).
//
// The instance id acts like a password for that account's layers, so it is only ever kept in this browser.

export const SAR_STORAGE_KEY = 'sq-sar-source';
export const CDSE_WMTS = 'https://sh.dataspace.copernicus.eu/ogc/wmts';
// Pages confirmed to exist (from the Copernicus Data Space documentation)
export const CDSE_SIGNUP = 'https://dataspace.copernicus.eu';
export const CDSE_SENTINEL_HUB = 'https://dataspace.copernicus.eu/analyse/apis/sentinel-hub';
export const CDSE_HELP = 'https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/OGC.html';
const INSTANCE_ID = /^[A-Za-z0-9][A-Za-z0-9-]{7,}$/;
const LAYER_NAME = /^[A-Za-z0-9][A-Za-z0-9_. -]{0,79}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;

const fail = (error) => ({ ok: false, error });

// Date.parse is lenient ("2024-02-31" becomes 2 March), so check the calendar day really exists
const isRealDay = (text) => {
  const [year, month, day] = text.split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
};

// "2024-01-01/2024-03-31" (or one day) -> "2024-01-01T00:00:00Z/2024-03-31T23:59:59Z". "" means "latest".
export function normaliseTimeRange(text) {
  const raw = String(text ?? '').trim();
  if (!raw) return { ok: true, value: '' };
  const [from, to = from, extra] = raw.split('/').map((part) => part.trim());
  if (extra !== undefined || !DAY.test(from) || !DAY.test(to)) return fail('Use a date like 2024-03-31, or a range like 2024-01-01/2024-03-31.');
  if (!isRealDay(from) || !isRealDay(to)) return fail('That date does not exist.');
  if (from > to) return fail('The start date is after the end date.');
  return { ok: true, value: `${from}T00:00:00Z/${to}T23:59:59Z` };
}

// Leaflet-style template ({z}/{y}/{x} are left for the map to fill in)
export function buildCdseTemplate({ instanceId, layer, timeRange = '' }) {
  const id = String(instanceId ?? '').trim();
  const name = String(layer ?? '').trim();
  if (!INSTANCE_ID.test(id)) return fail('The instance ID looks wrong. It is a long code like 1a2b3c4d-… from the Configuration Utility.');
  if (!name) return fail('Enter the layer name exactly as it appears in your configuration.');
  if (!LAYER_NAME.test(name)) return fail('The layer name has characters a layer name cannot have.');
  const time = normaliseTimeRange(timeRange);
  if (!time.ok) return time;
  const query = [
    'SERVICE=WMTS', 'VERSION=1.0.0', 'REQUEST=GetTile', `LAYER=${encodeURIComponent(name)}`, 'TILEMATRIXSET=PopularWebMercator256',
    'TILEMATRIX={z}', 'TILEROW={y}', 'TILECOL={x}', 'FORMAT=image/png', ...(time.value ? [`TIME=${encodeURIComponent(time.value)}`] : []),
  ].join('&');
  return { ok: true, template: `${CDSE_WMTS}/${id}?${query}` };
}

// A custom URL must be http(s) and carry all three placeholders
export function validateTemplate(url) {
  const text = String(url ?? '').trim();
  if (!/^https?:\/\//i.test(text)) return fail('The URL must start with https:// (or http://).');
  const missing = ['{z}', '{x}', '{y}'].filter((token) => !text.includes(token));
  if (missing.length) return fail(`The URL needs ${missing.join(', ')} where the zoom level and tile position go.`);
  return { ok: true, template: text };
}

// The URL for one real tile (zoom 3), used to ask the provider "does this work?" before saving
export const sampleTileUrl = (template) => template.replace('{z}', '3').replace('{x}', '4').replace('{y}', '3');

// Probe the provider. A readable answer that is an image passes; a readable error says what is wrong; a fetch
// that cannot be read at all is usually CORS (the map can still show tiles, but they cannot be captured).
export async function testSarTemplate(template, { fetchImpl = fetch } = {}) {
  let response;
  try {
    response = await fetchImpl(sampleTileUrl(template));
  } catch {
    return { ok: false, unreadable: true, message: 'The browser could not read a test tile. The address may be wrong, offline, or the provider blocks browser access (CORS). The map may still display it, but captures would fail.' };
  }
  if (response.ok) {
    const type = response.headers?.get?.('content-type') || '';
    return /^image\//i.test(type) || !type
      ? { ok: true, message: 'The provider answered with an image.' }
      : { ok: false, message: `The provider answered, but not with an image (${type.split(';')[0]}). Check the layer name.` };
  }
  if (response.status === 401 || response.status === 403) return { ok: false, message: 'The provider refused the request. Check the instance ID and that your account is active.' };
  if (response.status === 400 || response.status === 404) return { ok: false, message: 'The provider does not know this instance or layer. Check the instance ID and the layer name.' };
  return { ok: false, message: `The provider answered HTTP ${response.status}.` };
}

const browserStorage = () => {
  try { return globalThis.localStorage ?? null; } catch { return null; }
};

export function loadSarSource(storage = browserStorage()) {
  try {
    const source = JSON.parse(storage?.getItem(SAR_STORAGE_KEY) || 'null');
    return source && typeof source.template === 'string' && validateTemplate(source.template).ok ? source : null;
  } catch {
    return null;
  }
}

export function saveSarSource(source, storage = browserStorage()) {
  if (!source || !validateTemplate(source.template).ok) return false;
  try {
    storage?.setItem(SAR_STORAGE_KEY, JSON.stringify(source));
    return true;
  } catch {
    return false;
  }
}

export function clearSarSource(storage = browserStorage()) {
  try { storage?.removeItem(SAR_STORAGE_KEY); } catch { /* nothing stored */ }
}

// The map settings to use: a connected source wins over the VITE_SAR_* environment values
export function resolveSarConfig(config, source) {
  if (!source) return config;
  return {
    ...config,
    sarUrl: source.template,
    hasSar: true,
    sarAttribution: source.attribution || config.sarAttribution,
    sarMaxNativeZoom: Number(source.maxZoom) || config.sarMaxNativeZoom,
    sarCrossOrigin: 'anonymous',
    sarCanExportView: true,
  };
}

export function describeSarSource(source) {
  if (!source) return '';
  return source.kind === 'cdse' ? `Copernicus · ${source.layer}${source.timeRange ? ` · ${source.timeRange}` : ''}` : 'Custom tile URL';
}
