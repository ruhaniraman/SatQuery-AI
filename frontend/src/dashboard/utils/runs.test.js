import test from 'node:test';
import assert from 'node:assert/strict';
import {
  addRun, affectedKinds, describeReportChoice, dropRunsFor, emptySlot, isAnnotated, latestRun, makeRun, pickReportRun,
} from './runs.js';

const data = (id) => ({ agent_execution_trace: { pipeline_id: id }, answer: id });
const run = (kind, id, at = 1) => makeRun({ kind, data: data(id), at });
const push = (...specs) => specs.reduce((runs, [kind, id]) => addRun(runs, run(kind, id)), []);   // newest first

test('scans, change detection and fusion are annotated; plain questions are not', () => {
  assert.deepEqual(['chat', 'scan', 'change', 'fusion'].map(isAnnotated), [false, true, true, true]);
  assert.equal(isAnnotated('nonsense'), false);
});

test('a run remembers its server session and gets a readable default title', () => {
  const r = run('scan', 'sess-1');
  assert.equal(r.sessionId, 'sess-1');
  assert.equal(r.title, 'Feature scan');
  assert.equal(r.annotated, true);
  assert.equal(makeRun({ kind: 'chat', title: 'What is here?', data: null }).sessionId, null);
  assert.equal(makeRun({ kind: 'scan', adapter: 'mining', data: null }).adapter, 'mining');
  assert.equal(r.adapter, null);
  assert.notEqual(run('chat', 'x').id, run('chat', 'x').id, 'ids are unique even at the same instant');
});

test('the report defaults to the newest ANNOTATED run, so a later plain question never replaces it', () => {
  const runs = push(['scan', 'scan-1'], ['chat', 'chat-1'], ['chat', 'chat-2']);      // chat-2 is newest
  assert.equal(pickReportRun(runs, null).sessionId, 'scan-1');
  const two = push(['scan', 'scan-1'], ['change', 'change-1'], ['chat', 'chat-1']);
  assert.equal(pickReportRun(two, null).sessionId, 'change-1', 'newest of the annotated ones');
});

test('with only plain questions the newest run is used, and with no runs there is none', () => {
  assert.equal(pickReportRun(push(['chat', 'a'], ['chat', 'b']), null).sessionId, 'b');
  assert.equal(pickReportRun([], null), null);
});

test('the user can pick any run; a pick that no longer exists falls back to the default', () => {
  const runs = push(['scan', 'scan-1'], ['chat', 'chat-1']);
  const chat = runs.find((r) => r.sessionId === 'chat-1');
  assert.equal(pickReportRun(runs, chat.id).sessionId, 'chat-1');
  assert.equal(pickReportRun(runs, 'gone').sessionId, 'scan-1');
});

test('latestRun looks only at the kinds asked for', () => {
  const runs = push(['scan', 's1'], ['change', 'c1'], ['scan', 's2'], ['chat', 'q']);
  assert.equal(latestRun(runs, ['scan']).sessionId, 's2');
  assert.equal(latestRun(runs, ['change']).sessionId, 'c1');
  assert.equal(latestRun(runs, ['fusion']), null);
  assert.equal(latestRun(runs).sessionId, 'q');
});

test('replacing an image invalidates only the runs that used it', () => {
  const runs = push(['chat', 'q'], ['scan', 's'], ['change', 'c'], ['fusion', 'f']);
  const ids = (list) => list.map((r) => r.sessionId).sort();
  assert.deepEqual(ids(dropRunsFor(runs, 'a')), ['c', 'f']);
  assert.deepEqual(ids(dropRunsFor(runs, 'before')), ['f', 'q', 's']);
  assert.deepEqual(ids(dropRunsFor(runs, 'after')), ['f', 'q', 's']);
  assert.deepEqual(ids(dropRunsFor(runs, 'optical')), ['c', 'q', 's']);
  assert.deepEqual(ids(dropRunsFor(runs, 'sar')), ['c', 'q', 's']);
  assert.deepEqual(affectedKinds('unknown'), []);
});

test('empty slots start optical with no preview', () => {
  assert.deepEqual(emptySlot(), { file: null, preview: null, previewState: 'none', modality: 'optical', meta: null });
  assert.equal(emptySlot('sar').modality, 'sar');
});

test('the report choice is explained in one sentence', () => {
  assert.match(describeReportChoice([], null), /Run an analysis/);
  const runs = push(['scan', 's'], ['chat', 'q']);
  assert.match(describeReportChoice(runs, null), /latest annotated evidence: “Feature scan”.*never replace/);
  const chat = runs.find((r) => r.kind === 'chat');
  assert.match(describeReportChoice(runs, chat.id), /as you chose/);
  assert.match(describeReportChoice(push(['chat', 'q']), null), /Only plain questions/);
});
