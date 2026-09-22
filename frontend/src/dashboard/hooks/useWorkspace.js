import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BACKEND_URL, DEFAULT_QUERIES, isTiffFile, reportFileName, requestAnalysis, requestPreview, requestReport,
} from '../utils/api';
import { readToken } from '../utils/auth';
import { mapLink } from '../utils/scanResult';
import {
  addRun, dropRunsFor, emptySlot, isAnnotated, latestRun, makeRun, pickReportRun,
} from '../utils/runs';

const revoke = (url) => {
  if (url) URL.revokeObjectURL(url);   // each createObjectURL holds the file in memory until revoked
};

const initialSlots = () => ({
  a: emptySlot(),                 // the single image: Imagery, Assistant, Scans
  before: emptySlot(),            // Change detection
  after: emptySlot(),
  optical: emptySlot('optical'),  // Optical-SAR fusion (sensor types are fixed by the slot)
  sar: emptySlot('sar'),
});
const FIXED_MODALITY = { optical: 'optical', sar: 'sar' };

const short = (text, n = 48) => (text.length > n ? `${text.slice(0, n - 1).trimEnd()}…` : text);
const asImage = (slot) => ({ file: slot.file, modality: slot.modality });

// Messages from two-image runs live in the same conversation (and the PDF), but must not be fed back to
// the model as context for a question about the single image.
const forSingleImageModel = (chat) => chat.filter((entry) => entry.scope !== 'pair');

// Everything the panels share: the image slots, the conversation, the analysis runs and the viewer
// state. Panels stay presentational; this is the only place that talks to the API.
export function useWorkspace() {
  const [slots, setSlots] = useState(initialSlots);

  // Viewer: the live map is the default; the inputs view shows the user's images and the evidence.
  const [viewMode, setViewMode] = useState('map');           // 'map' | 'inputs'
  const [activeLayer, setActiveLayer] = useState('a');       // a slot id | 'evidence' | 'swipe'

  const [focus, setFocus] = useState(null);                  // { box, token }: a region of the evidence for the viewer to zoom to
  const [chat, setChat] = useState([]);
  const [runs, setRuns] = useState([]);                      // newest first, see utils/runs.js
  const [reportRunId, setReportRunId] = useState(null);      // null = automatic (newest annotated run)
  const [error, setError] = useState(null);                  // { message, scope } | null
  const [isExecuting, setIsExecuting] = useState(false);
  const [busy, setBusy] = useState(null);                    // which control started the run

  // Latest preview URLs and request counters, so replacing/removing an image (and unmounting) can free
  // the old URL and ignore a slow preview that arrives after the image was replaced.
  const previews = useRef({});
  const tokens = useRef({});
  useEffect(() => () => Object.values(previews.current).forEach(revoke), []);

  const patchSlot = useCallback((id, changes) => {
    setSlots((prev) => ({ ...prev, [id]: { ...prev[id], ...changes } }));
  }, []);

  // Browsers cannot draw a TIFF, so the backend renders it to a PNG with the loader the analysis uses.
  const renderTiff = useCallback(async (id, file, modality) => {
    tokens.current[id] = (tokens.current[id] || 0) + 1;
    const token = tokens.current[id];
    patchSlot(id, { previewState: 'loading', previewError: null });
    try {
      const url = URL.createObjectURL(await requestPreview({ file, modality }));
      if (tokens.current[id] !== token) { revoke(url); return; }         // replaced while rendering
      revoke(previews.current[id]);
      previews.current[id] = url;
      patchSlot(id, { preview: url, previewState: 'ready' });
    } catch (err) {
      if (tokens.current[id] !== token) return;
      patchSlot(id, { preview: null, previewState: 'failed', previewError: err.message });
    }
  }, [patchSlot]);

  const setImage = useCallback((id, file, { reveal = true, modality, meta = null } = {}) => {
    if (!file) return;
    tokens.current[id] = (tokens.current[id] || 0) + 1;      // any preview still rendering is now stale
    revoke(previews.current[id]);
    previews.current[id] = null;
    const tiff = isTiffFile(file);
    const chosen = FIXED_MODALITY[id] ?? modality ?? slots[id].modality;
    let preview = null;
    if (!tiff) {
      preview = URL.createObjectURL(file);
      previews.current[id] = preview;
    }
    setSlots((prev) => ({ ...prev, [id]: { file, preview, previewState: tiff ? 'loading' : 'ready', previewError: null, modality: chosen, meta } }));
    if (tiff) renderTiff(id, file, chosen);
    setRuns((prev) => dropRunsFor(prev, id));                // old answers describe the old imagery
    if (id === 'a') {
      setChat([]);
      setError(null);
    }
    setActiveLayer(id);
    if (reveal) setViewMode('inputs');                       // a map capture stays on the map
  }, [renderTiff, slots]);

  const removeImage = useCallback((id) => {
    tokens.current[id] = (tokens.current[id] || 0) + 1;
    revoke(previews.current[id]);
    previews.current[id] = null;
    setSlots((prev) => ({ ...prev, [id]: emptySlot(FIXED_MODALITY[id] ?? prev[id].modality) }));
    setRuns((prev) => dropRunsFor(prev, id));
    if (id === 'a') setChat([]);
    setActiveLayer((layer) => (layer === id || layer === 'evidence' || layer === 'swipe' ? id : layer));
  }, []);

  // Sensor type of the single image (Imagery), or of BOTH images of a change-detection pair
  const setModality = useCallback((id, modality) => {
    const ids = id === 'before' || id === 'after' ? ['before', 'after'] : [id];
    ids.forEach((which) => {
      if (FIXED_MODALITY[which]) return;
      const slot = slots[which];
      if (slot.modality === modality) return;
      patchSlot(which, { modality });
      if (slot.file && isTiffFile(slot.file)) renderTiff(which, slot.file, modality);   // SAR is despeckled in the preview
    });
  }, [slots, patchSlot, renderTiff]);

  // The in-flight /analyze request, so the Stop button can give up on it. Only one run is ever active
  // (execute() below refuses to start a second one), so a single ref is enough. Note this only stops the
  // BROWSER from waiting: main_api.py's endpoints are sync/threadpool, so the backend's own GPU work for
  // an already-started request keeps running to completion and still holds the model lock meanwhile.
  const abortRef = useRef(null);
  const stopAnalysis = useCallback(() => { abortRef.current?.abort(); }, []);

  // Runs one analysis and records it. `pair` marks two-image runs so their messages stay out of the
  // single-image model context. Resolves to { data, run }, or null when it failed, was stopped, or one
  // is already running.
  const execute = useCallback(async ({
    kind, title, query, adapter = 'general', images, history, userEntry, errorScope, pair = false, revealEvidence = true, meta = null,
  }) => {
    if (isExecuting) return null;
    const scope = pair ? { scope: 'pair' } : {};
    setChat((prev) => [...prev, { ...userEntry, ...scope }]);
    setError(null);
    setFocus(null);
    setIsExecuting(true);
    setBusy(kind === 'scan' ? `scan:${adapter}` : kind);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const data = await requestAnalysis({ query, adapter, images, history, token: readToken(), signal: controller.signal });
      const run = makeRun({ kind, title, data, adapter: kind === 'scan' ? adapter : null, meta });
      setRuns((prev) => addRun(prev, run));
      setChat((prev) => [...prev, { role: 'ai', content: data.answer, ...scope }]);
      // The backend draws detected regions / the change overlay / the fused composite into the evidence
      // image, so show it for scans and two-image runs.
      if (revealEvidence && data.visual_evidence_url && isAnnotated(kind)) {
        setViewMode('inputs');
        setActiveLayer('evidence');
      }
      return { data, run };
    } catch (err) {
      const message = err?.name === 'AbortError' ? 'Response was interrupted.' : err.message;
      setError({ message, scope: errorScope });
      return null;
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setIsExecuting(false);
      setBusy(null);
    }
  }, [isExecuting]);

  const needImage = (id, scope, message) => {
    if (slots[id].file) return true;
    setError({ message, scope });
    return false;
  };

  const sendMessage = useCallback(async (text) => {
    if (!needImage('a', 'assistant', 'Add an image in the Imagery panel first.')) return null;
    const out = await execute({
      kind: 'chat', title: short(text), query: text, images: [asImage(slots.a)], errorScope: 'assistant',
      history: [...forSingleImageModel(chat), { role: 'user', content: text }], userEntry: { role: 'user', content: text },
    });
    return out?.data ?? null;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [execute, slots, chat]);

  const runScan = useCallback(async (adapter) => {
    if (!needImage('a', 'scans', 'Add an image in the Imagery panel first.')) return null;
    const entry = { role: 'system', content: `Initiating ${adapter} scan...` };
    const out = await execute({
      kind: 'scan', adapter, title: `${adapter[0].toUpperCase()}${adapter.slice(1)} scan`, query: DEFAULT_QUERIES.scan,
      images: [asImage(slots.a)], errorScope: 'scans', history: [...forSingleImageModel(chat), entry], userEntry: entry,
      meta: slots.a.meta,
    });
    return out?.data ?? null;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [execute, slots, chat]);

  const runChange = useCallback(async (text) => {
    if (!slots.before.file || !slots.after.file) {
      setError({ message: 'Add both the Before and the After image first.', scope: 'change' });
      return null;
    }
    if (slots.before.modality !== slots.after.modality) {
      setError({ message: 'Before and After must be the same sensor type. To combine optical and SAR, use Optical–SAR fusion.', scope: 'change' });
      return null;
    }
    const query = (text || '').trim() || DEFAULT_QUERIES.change;
    const out = await execute({
      kind: 'change', title: 'Change detection', query, images: [asImage(slots.before), asImage(slots.after)],
      errorScope: 'change', pair: true, history: [], userEntry: { role: 'user', content: query },
    });
    return out?.data ?? null;
  }, [execute, slots]);

  const runFusion = useCallback(async (text) => {
    if (!slots.optical.file || !slots.sar.file) {
      setError({ message: 'Add both the optical and the SAR image first.', scope: 'fusion' });
      return null;
    }
    const query = (text || '').trim() || DEFAULT_QUERIES.fusion;
    const out = await execute({
      kind: 'fusion', title: 'Optical–SAR fusion', query, images: [asImage(slots.optical), asImage(slots.sar)],
      errorScope: 'fusion', pair: true, history: [], userEntry: { role: 'user', content: query },
    });
    return out?.data ?? null;
  }, [execute, slots]);

  // Ask about (or scan) a captured map view. The view replaces the single image, the answer lands in
  // the shared conversation, and the viewer stays on the map. `modality` is 'sar' for the SAR layer.
  const analyzeView = useCallback(async (file, { question, adapter = 'general', modality = 'optical', bounds = null, zoom = null }) => {
    setImage('a', file, { reveal: false, modality });
    const isScan = adapter !== 'general';
    const entry = isScan ? { role: 'system', content: `Initiating ${adapter} scan...` } : { role: 'user', content: question };
    const out = await execute({
      kind: isScan ? 'scan' : 'chat', adapter, title: isScan ? `${adapter[0].toUpperCase()}${adapter.slice(1)} scan` : short(question),
      query: isScan ? DEFAULT_QUERIES.scan : question, images: [{ file, modality }], errorScope: 'map',
      history: [entry], userEntry: entry, revealEvidence: false, meta: bounds ? { bounds, zoom } : null,
    });
    return out?.data ?? null;
  }, [execute, setImage]);

  // ---- report ----
  const reportRun = useMemo(() => pickReportRun(runs, reportRunId), [runs, reportRunId]);

  // Build the PDF for the chosen run with the whole conversation. Resolves to { url, filename, error }:
  // if the rebuild fails the run's own PDF (its conversation up to that point) is offered instead.
  const prepareReport = useCallback(async (base = BACKEND_URL) => {
    if (!reportRun) return { url: null, filename: null, error: 'Run an analysis first.' };
    const filename = reportFileName(reportRun.data);
    try {
      const out = await requestReport({ sessionId: reportRun.sessionId, history: chat, mapLink: mapLink(reportRun.meta), baseUrl: base });
      if (out.report_download_url) return { url: `${base}${out.report_download_url}`, filename, error: null };
      throw new Error(out.report_error || 'The report could not be built.');
    } catch (err) {
      const own = reportRun.data?.report_download_url;
      if (own) return { url: `${base}${own}`, filename, error: null, fallback: err.message };
      return { url: null, filename, error: err.message };
    }
  }, [reportRun, chat]);

  const latest = useMemo(() => ({
    scan: latestRun(runs, ['scan']),
    change: latestRun(runs, ['change']),
    fusion: latestRun(runs, ['fusion']),
    annotated: latestRun(runs, ['scan', 'change', 'fusion']),
  }), [runs]);

  // Show one numbered area of a scan on the image: open the highlights and zoom to it
  const focusArea = useCallback((box) => {
    setViewMode('inputs');
    setActiveLayer('evidence');
    setFocus({ box, token: Date.now() });
  }, []);

  const errorFor = useCallback((scope) => (error && error.scope === scope ? error.message : null), [error]);

  return {
    slots, latest, runs, reportRun, reportRunId, selectReportRun: setReportRunId, prepareReport,
    setImage, removeImage, setModality,
    viewMode, setViewMode, activeLayer, setActiveLayer, focus, focusArea,
    chat, error, errorFor, isExecuting, busy, stopAnalysis,
    sendMessage, runScan, runChange, runFusion, analyzeView, dismissError: () => setError(null),
  };
}
