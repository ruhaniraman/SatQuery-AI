import test from 'node:test';
import assert from 'node:assert/strict';
import { parseChangeAnswer } from './changeAnswer.js';

const WARNING = 'WARNING: Image B could not be registered to Image A (only 5 inliers); it was only resized to fit.';

const REAL = [
  WARNING,
  '',
  'Changed areas',
  '- Likely change, centre (~6% of the scene). Also flagged by the deforestation scan.',
  '    Before: Dense dark green canopy. Some bare soil is visible.',
  '    After: A mix of brown and green ground.',
  '    Measured: green-dominant pixels 0% -> 18%; average brightness 63 -> 85 (0-255). This may reflect the two images differing in source, not the ground.',
  '- Major change, right (~18% of the scene). Also flagged by the mining and agriculture scans.',
  '    Measured: bright pixels 10% -> 13%.',
  '- Possible smaller change, lower-right (~3% of the scene).',
].join('\n');

test('warnings, areas and their Before / After / Measured rows are separated', () => {
  const r = parseChangeAnswer(REAL);
  assert.equal(r.legacy, false);
  assert.deepEqual(r.warnings, [WARNING.slice(9)]);
  assert.equal(r.areas.length, 3);
  const [first, second, third] = r.areas;
  assert.deepEqual([first.kind, first.where, first.sizePercent], ['likely', 'centre', 6]);
  assert.deepEqual(first.scans, ['deforestation']);
  assert.equal(first.before, 'Dense dark green canopy. Some bare soil is visible.');
  assert.equal(first.after, 'A mix of brown and green ground.');
  assert.match(first.measured, /^green-dominant pixels 0% -> 18%.*not the ground\.$/);
  assert.deepEqual([second.kind, second.where, second.sizePercent], ['major', 'right', 18]);
  assert.deepEqual(second.scans, ['mining', 'agriculture'], 'several scans are split into names');
  assert.equal(second.before, '');
  assert.equal(second.measured, 'bright pixels 10% -> 13%.');
  assert.deepEqual([third.kind, third.scans], ['possible', []]);
});

test('three scans are all named', () => {
  const r = parseChangeAnswer('Changed areas\n- Likely change, centre (~5% of the scene). Also flagged by the mining, deforestation and agriculture scans.');
  assert.deepEqual(r.areas[0].scans, ['mining', 'deforestation', 'agriculture']);
});

test('the empty and the warned-empty answers become notes', () => {
  const empty = parseChangeAnswer('Changed areas\n- No changed area was found.');
  assert.deepEqual([empty.areas, empty.notes, empty.warnings], [[], ['No changed area was found.'], []]);
  const warned = parseChangeAnswer(`${WARNING}\n\nChanged areas\n- No changed area was flagged, but the warning above means this is NOT evidence that nothing changed.`);
  assert.equal(warned.warnings.length, 1);
  assert.match(warned.notes[0], /NOT evidence that nothing changed/);
});

test('several warnings and the unanswered-question footer are kept', () => {
  const r = parseChangeAnswer(`WARNING: one\nWARNING: two\n\nChanged areas\n- No changed area was found.\n\nYour question (how many trucks?) is not answered separately in comparison mode. Ask about one image in the Assistant panel for follow-up questions.`);
  assert.deepEqual(r.warnings, ['one', 'two']);
  assert.equal(r.footer.length, 1);
  assert.match(r.footer[0], /^Your question \(how many trucks\?\)/);
});

test('an older model-written answer is reported as legacy', () => {
  const r = parseChangeAnswer('CHANGES: the pit grew. ASSESSMENT: more excavation.');
  assert.equal(r.legacy, true);
  assert.deepEqual([r.areas, r.warnings], [[], []]);
});

test('missing or odd input never throws', () => {
  for (const input of [undefined, null, '', 42, '\r\n', 'Changed areas']) {
    const r = parseChangeAnswer(input);
    assert.ok(Array.isArray(r.areas) && Array.isArray(r.warnings));
  }
  assert.equal(parseChangeAnswer('Changed areas').legacy, false);
  assert.equal(parseChangeAnswer('Changed areas').areas.length, 0);
});

test('Windows line endings are handled', () => {
  const r = parseChangeAnswer(REAL.replace(/\n/g, '\r\n'));
  assert.equal(r.areas.length, 3);
  assert.equal(r.areas[0].after, 'A mix of brown and green ground.');
});
