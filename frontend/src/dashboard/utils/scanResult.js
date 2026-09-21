// Pure helpers for showing a feature scan's result (no React, no DOM: tested in plain Node).
// The backend does the reading of the scan (utils are only for wording and links): `scan` is the
// `scan` block of the /analyze response: { adapter, level, headline, note, findings, grid, threshold, ... }.

export const LEVELS = {
  high: { label: 'High confidence', chip: 'border-blue-400/40 bg-blue-500/20 text-blue-200' },
  likely: { label: 'Likely', chip: 'border-sky-400/30 bg-sky-500/15 text-sky-200' },
  possible: { label: 'Possible', chip: 'border-white/15 bg-white/5 text-slate-300' },
  none: { label: 'Nothing confident', chip: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300' },
};
export const levelOf = (scan) => LEVELS[scan?.level] || LEVELS.none;

// Tailwind needs whole class names, so the pin colours (matching the highlight colours drawn server-side) are listed.
export const PIN_CLASS = { mining: 'bg-amber-500', deforestation: 'bg-rose-500', agriculture: 'bg-emerald-500' };
export const pinClass = (scan) => PIN_CLASS[scan?.adapter] || 'bg-amber-500';

export const scanOf = (run) => run?.data?.scan || null;

const percent = (p) => `${Math.round(p * 100)}%`;
export const thresholdPercent = (scan) => percent(scan?.threshold ?? 0.8);

// A map link for the ground a capture covered. `meta` is a capture's { bounds: [[south, west], [north, east]] }.
// The zoom is estimated from the width of the view (a screen about 512 px wide), so the link opens on
// roughly what was scanned. null when the image was not a map capture.
export function mapLink(meta) {
  const b = meta?.bounds;
  if (!Array.isArray(b) || b.length !== 2 || b.flat().some((v) => !Number.isFinite(v))) return null;
  const [[south, west], [north, east]] = b;
  const lat = (south + north) / 2;
  const lng = (west + east) / 2;
  const span = Math.abs(east - west);
  const zoom = span > 0 ? Math.min(19, Math.max(1, Math.round(Math.log2(360 / span) + 1))) : 15;
  const fix = (v) => v.toFixed(5);
  return `https://www.openstreetmap.org/?mlat=${fix(lat)}&mlon=${fix(lng)}#map=${zoom}/${fix(lat)}/${fix(lng)}`;
}

// The result as plain text to paste into a message or an email
export function summaryText(scan, link = null) {
  if (!scan) return '';
  const lines = [scan.headline];
  if (scan.note) lines.push(`Note: ${scan.note}`);
  (scan.findings || []).forEach((f) => {
    lines.push(`${f.id}. ${f.label}: ${f.where}, about ${f.coverage_pct}% of the view (average score ${f.confidence}%)`);
  });
  lines.push('Automated screening: check the highlighted areas by eye.');
  if (link) lines.push(`Map: ${link}`);
  return lines.join('\n');
}

// One line for the map card: "1 Likely · upper-left"
export const findingLine = (f) => `${f.label} · ${f.where}`;
