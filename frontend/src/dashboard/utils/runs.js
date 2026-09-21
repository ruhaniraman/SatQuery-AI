// Pure rules for the workspace's slots and analysis runs (no React, no DOM: tested in plain Node).
//
// A chat produces many analysis runs, and each has its own evidence image. A scan or a comparison
// draws something on its evidence (boxes, a change overlay, a fused composite); a plain question about
// one image just gets the raw image back. The report should show the annotated one, so:
//   - runs are kept in a list, newest first;
//   - a run is "annotated" when it is a scan, change detection or fusion;
//   - by default the report uses the newest annotated run, so a later plain question can never replace
//     it with a raw picture; the user can pick any other run.

// Image slots. 'a' is the single image used by Imagery / Assistant / Scans; the others belong to one
// two-image feature each, so change detection and fusion never share inputs.
export const SLOT_IDS = ['a', 'before', 'after', 'optical', 'sar'];

// meta: what a map capture remembers about itself ({ label, bounds, zoom, date }), null for uploaded files
export const emptySlot = (modality = 'optical') => ({ file: null, preview: null, previewState: 'none', modality, meta: null });

export const RUN_KINDS = {
  chat: { label: 'Question', annotated: false },
  scan: { label: 'Feature scan', annotated: true },
  change: { label: 'Change detection', annotated: true },
  fusion: { label: 'Optical–SAR fusion', annotated: true },
};

export const isAnnotated = (kind) => Boolean(RUN_KINDS[kind]?.annotated);

let counter = 0;
export function makeRun({ kind, title, data, adapter = null, meta = null, at = Date.now() }) {
  counter += 1;
  return {
    id: `${at}-${counter}`,
    kind,
    title: title || RUN_KINDS[kind]?.label || 'Analysis',
    at,
    adapter,                                          // which LoRA scan ('mining', ...) for scans, else null
    data,                                             // the /analyze response
    meta,                                             // where a map capture was taken ({ bounds, zoom }), else null
    sessionId: data?.agent_execution_trace?.pipeline_id || null,
    annotated: isAnnotated(kind),
  };
}

export const addRun = (runs, run) => [run, ...runs];

// The run the report is built on: the user's pick if it still exists, else the newest annotated run,
// else (only plain questions so far) the newest run.
export function pickReportRun(runs, selectedId) {
  if (!runs.length) return null;
  const chosen = selectedId && runs.find((r) => r.id === selectedId);
  if (chosen) return chosen;
  return runs.find((r) => r.annotated) || runs[0];
}

export const latestRun = (runs, kinds) => runs.find((r) => !kinds || kinds.includes(r.kind)) || null;

// Which runs stop making sense when a slot's image is replaced or removed
const AFFECTED = {
  a: ['chat', 'scan'],
  before: ['change'],
  after: ['change'],
  optical: ['fusion'],
  sar: ['fusion'],
};
export const affectedKinds = (slotId) => AFFECTED[slotId] || [];
export const dropRunsFor = (runs, slotId) => runs.filter((r) => !affectedKinds(slotId).includes(r.kind));

export const runTime = (run, locale) =>
  new Date(run.at).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });

// What the report will show, as one sentence for the UI
export function describeReportChoice(runs, selectedId) {
  const run = pickReportRun(runs, selectedId);
  if (!run) return 'Run an analysis to build a report.';
  const picked = selectedId && runs.some((r) => r.id === selectedId);
  if (picked) return `Using the evidence of “${run.title}”, as you chose.`;
  if (run.annotated) return `Using the latest annotated evidence: “${run.title}”. Plain questions never replace it.`;
  return 'Only plain questions so far, so the report shows the analysed image itself.';
}
