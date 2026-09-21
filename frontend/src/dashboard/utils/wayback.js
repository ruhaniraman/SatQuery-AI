// Esri "World Imagery Wayback": the free archive of past releases of the basemap. The list of releases is a
// public JSON file (CORS-enabled, like the tiles), so the app needs no key. Pure helpers, tested in Node.
//
// CAUTION shown to the user: a release date is when Esri published that version of the basemap, not when the
// ground was photographed, and where nothing new was captured a newer release simply repeats older imagery.
// Two releases can therefore look identical; that is "no change visible", not an error.

export const WAYBACK_CONFIG_URL = 'https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json';

const TITLE_DATE = /(\d{4}-\d{2}-\d{2})/;
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// The config maps release id -> { itemTitle: "World Imagery (Wayback 2014-06-11)", itemURL: ".../tile/31144/{level}/{row}/{col}" }.
// Returns [{ id, date: 'YYYY-MM-DD', template }] oldest first, with Leaflet's {z}/{y}/{x} in the template.
export function parseWaybackConfig(config) {
  const byDate = new Map();
  for (const [id, entry] of Object.entries(config || {})) {
    const date = String(entry?.itemTitle || '').match(TITLE_DATE)?.[1];
    const url = entry?.itemURL;
    if (!date || typeof url !== 'string' || !url.includes('{level}') || !url.includes('{row}') || !url.includes('{col}')) continue;
    const template = url.replace('{level}', '{z}').replace('{row}', '{y}').replace('{col}', '{x}');
    const known = byDate.get(date);
    if (!known || Number(id) > Number(known.id)) byDate.set(date, { id, date, template });   // one release per date
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

// "2014-06-11" -> "11 Jun 2014" (no timezone maths: it is a calendar date, not an instant)
export function formatReleaseDate(date) {
  const match = String(date || '').match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return String(date || '');
  return `${Number(match[3])} ${MONTHS[Number(match[2]) - 1] || '?'} ${match[1]}`;
}

// Index of the release closest to an ISO date (ties go to the older one)
export function nearestReleaseIndex(releases, isoDate) {
  if (!releases.length) return -1;
  const target = Date.parse(`${isoDate}T00:00:00Z`);
  let best = 0;
  let bestGap = Infinity;
  releases.forEach((release, i) => {
    const gap = Math.abs(Date.parse(`${release.date}T00:00:00Z`) - target);
    if (gap < bestGap) { best = i; bestGap = gap; }
  });
  return best;
}

// Labels for under the slider: the first release of each year, thinned so they never crowd
export function yearTicks(releases, maxTicks = 6) {
  const firsts = [];
  releases.forEach((release, index) => {
    const year = release.date.slice(0, 4);
    if (!firsts.length || firsts[firsts.length - 1].year !== year) firsts.push({ year, index });
  });
  if (firsts.length <= maxTicks) return firsts;
  const step = (firsts.length - 1) / (maxTicks - 1);
  const picked = Array.from({ length: maxTicks }, (_, i) => firsts[Math.round(i * step)]);
  return picked.filter((tick, i) => picked.findIndex((t) => t.index === tick.index) === i);
}

export async function loadWaybackReleases({ fetchImpl = fetch, url = WAYBACK_CONFIG_URL } = {}) {
  let response;
  try {
    response = await fetchImpl(url);
  } catch {
    throw new Error('Could not reach the historical imagery service. Check your internet connection.');
  }
  if (!response.ok) throw new Error(`The historical imagery service answered HTTP ${response.status}.`);
  let releases;
  try {
    releases = parseWaybackConfig(await response.json());
  } catch {
    throw new Error('The historical imagery list could not be read.');
  }
  if (!releases.length) throw new Error('The historical imagery list was empty.');
  return releases;
}
