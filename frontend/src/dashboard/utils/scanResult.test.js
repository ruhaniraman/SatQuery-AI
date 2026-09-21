import test from 'node:test';
import assert from 'node:assert/strict';
import { findingLine, levelOf, mapLink, pinClass, scanOf, summaryText, thresholdPercent } from './scanResult.js';

const scan = {
  adapter: 'mining', level: 'high', threshold: 0.8, note: 'It may be over-reporting.',
  headline: 'Surface mining activity found in 2 areas (high confidence), covering about 50% of the view.',
  findings: [
    { id: 1, label: 'High confidence', where: 'upper-left', coverage_pct: 25, confidence: 89, box: [0, 0, 1, 0.25] },
    { id: 2, label: 'Likely', where: 'lower-right', coverage_pct: 25, confidence: 80, box: [0.25, 0.5, 1, 1] },
  ],
};

test('levels have words, and an unknown level reads as nothing confident', () => {
  assert.equal(levelOf(scan).label, 'High confidence');
  assert.equal(levelOf({ level: 'weird' }).label, 'Nothing confident');
  assert.equal(levelOf(null).label, 'Nothing confident');
});

test('scanOf reads the scan block only when the run has one', () => {
  assert.equal(scanOf({ data: { scan } }), scan);
  assert.equal(scanOf({ data: { answer: 'x' } }), null);
  assert.equal(scanOf(null), null);
});

test('the threshold and the pin colour follow the scan', () => {
  assert.equal(thresholdPercent(scan), '80%');
  assert.equal(thresholdPercent({}), '80%');
  assert.equal(pinClass(scan), 'bg-amber-500');
  assert.equal(pinClass({ adapter: 'deforestation' }), 'bg-rose-500');
});

test('a map link centres on the capture and picks a sensible zoom', () => {
  const link = mapLink({ bounds: [[30.70, 76.75], [30.72, 76.80]] });
  assert.match(link, /^https:\/\/www\.openstreetmap\.org\/\?mlat=30\.71000&mlon=76\.77500#map=\d+\/30\.71000\/76\.77500$/);
  const zoom = Number(link.split('#map=')[1].split('/')[0]);
  assert.ok(zoom >= 12 && zoom <= 17, `zoom ${zoom}`);
  assert.equal(mapLink(null), null);
  assert.equal(mapLink({ bounds: [[1, 2], [NaN, 4]] }), null);
  assert.equal(mapLink({ bounds: 'nope' }), null);
});

test('the summary text has the headline, the note, every finding and the caveat', () => {
  const text = summaryText(scan, 'https://example.org/map');
  const lines = text.split('\n');
  assert.equal(lines[0], scan.headline);
  assert.ok(lines.includes('Note: It may be over-reporting.'));
  assert.ok(lines.some((l) => l.startsWith('1. High confidence: upper-left')));
  assert.ok(lines.some((l) => l.startsWith('2. Likely: lower-right')));
  assert.ok(text.includes('check the highlighted areas by eye'));
  assert.ok(lines.at(-1) === 'Map: https://example.org/map');
  assert.equal(summaryText(null), '');
  assert.ok(!summaryText({ ...scan, note: '' }).includes('Note:'));
});

test('a finding reads as one short line', () => {
  assert.equal(findingLine(scan.findings[0]), 'High confidence · upper-left');
});
