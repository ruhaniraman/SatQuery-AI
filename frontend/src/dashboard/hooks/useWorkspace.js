import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { analysisMode, DEFAULT_QUERIES, requestAnalysis } from '../utils/api';

const emptySlot = (modality = 'optical') => ({ file: null, preview: null, modality });
const revoke = (url) => {
  if (url) URL.revokeObjectURL(url);   // each createObjectURL holds the file in memory until revoked
};

// Everything the panels share: the two image slots, the conversation, the latest result and the
// viewer state. Panels stay presentational; this is the only place that talks to the API.
export function useWorkspace() {
  const [a, setA] = useState(() => emptySlot());
  const [b, setB] = useState(() => emptySlot());

  // Viewer: the live map is the default; the inputs view shows the user's images and the evidence.
  const [viewMode, setViewMode] = useState('map');          // 'map' | 'inputs'
  const [activeLayer, setActiveLayer] = useState('imageA'); // 'imageA' | 'imageB' | 'evidence'

  const [chat, setChat] = useState([]);
  const [result, setResult] = useState(null);
  // True when the current result came from a two-image run (so removing Image B invalidates it)
  const [resultWasTwoImage, setResultWasTwoImage] = useState(false);
  const [latest, setLatest] = useState({ scan: null, compare: null });
  const [error, setError] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [busy, setBusy] = useState(null);                   // which control started the run

  // Latest preview URLs, so replacing/removing an image (and unmounting) can free the old one
  // without doing side effects inside a state updater (StrictMode runs those twice).
  const previews = useRef({ a: null, b: null });
  useEffect(() => () => {
    revoke(previews.current.a);
    revoke(previews.current.b);
  }, []);

  const images = useMemo(() => {
    if (!a.file) return [];
    const list = [{ file: a.file, modality: a.modality }];
    if (b.file) list.push({ file: b.file, modality: b.modality });
    return list;
  }, [a, b]);
  const mode = useMemo(() => analysisMode(images), [images]);

  const clearResults = useCallback(() => {
    setChat([]);
    setResult(null);
    setResultWasTwoImage(false);
    setLatest({ scan: null, compare: null });
    setError(null);
  }, []);

  const setImage = useCallback((which, file, { reveal = true } = {}) => {
    if (!file) return;
    revoke(previews.current[which]);
    const preview = URL.createObjectURL(file);
    previews.current[which] = preview;
    (which === 'a' ? setA : setB)((prev) => ({ ...prev, file, preview }));
    clearResults();                                          // old answers describe the old imagery
    setActiveLayer(which === 'a' ? 'imageA' : 'imageB');
    if (reveal) setViewMode('inputs');                       // a map capture stays on the map
  }, [clearResults]);

  const setModality = useCallback((which, modality) => {
    (which === 'a' ? setA : setB)((prev) => ({ ...prev, modality }));
  }, []);

  const removeImageB = useCallback(() => {
    revoke(previews.current.b);
    previews.current.b = null;
    setB((prev) => emptySlot(prev.modality));
    if (activeLayer === 'imageB' || activeLayer === 'evidence') setActiveLayer('imageA');
    // A change-detection/fusion result describes a pairing that no longer exists: clear it, and
    // leave a note in the conversation record saying so (the earlier messages stay as history).
    if (resultWasTwoImage) {
      setResult(null);
      setResultWasTwoImage(false);
      setLatest((prev) => ({ ...prev, compare: null }));
      setChat((prev) => [...prev, { role: 'system', content: 'Image B was removed; the previous two-image result was cleared.' }]);
    }
  }, [activeLayer, resultWasTwoImage]);

  // imagesOverride / historyOverride let a caller run on an image that has only just been set (state
  // has not re-rendered yet); revealEvidence=false keeps the viewer where it is.
  const run = useCallback(async ({ kind, query, adapter = 'general', userEntry, useBoth, imagesOverride, historyOverride, revealEvidence = true }) => {
    if (isExecuting) return null;
    if (!a.file && !imagesOverride) {
      setError('Add Image A in the Imagery panel first.');
      return null;
    }
    const runImages = imagesOverride ?? (useBoth ? images : images.slice(0, 1));
    const history = [...(historyOverride ?? chat), userEntry];
    setChat(history);
    setError(null);
    setIsExecuting(true);
    setBusy(kind);
    try {
      const data = await requestAnalysis({ query, adapter, images: runImages, history });
      setResult(data);
      setResultWasTwoImage(runImages.length === 2);
      setChat((prev) => [...prev, { role: 'ai', content: data.answer }]);
      // The backend draws detected regions / the change overlay / the fused composite into the
      // evidence image, so show it for scans and two-image runs.
      if (revealEvidence && data.visual_evidence_url && (adapter !== 'general' || runImages.length === 2)) {
        setViewMode('inputs');
        setActiveLayer('evidence');
      }
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setIsExecuting(false);
      setBusy(null);
    }
  }, [a.file, images, chat, isExecuting]);

  const sendMessage = useCallback((text) => run({
    kind: 'chat', query: text, adapter: 'general', userEntry: { role: 'user', content: text }, useBoth: images.length === 2,
  }), [run, images.length]);

  const runScan = useCallback(async (adapter) => {
    const data = await run({
      kind: `scan:${adapter}`, query: DEFAULT_QUERIES.scan, adapter,
      userEntry: { role: 'system', content: `Initiating ${adapter} scan...` }, useBoth: false,
    });
    if (data) setLatest((prev) => ({ ...prev, scan: { adapter, answer: data.answer } }));
    return data;
  }, [run]);

  // Ask about (or scan) a captured map view. The view replaces Image A, the answer lands in the shared
  // conversation, and the viewer stays on the map. Returns the API response, or null on failure.
  const analyzeView = useCallback(async (file, { question, adapter = 'general' }) => {
    setImage('a', file, { reveal: false });
    const isScan = adapter !== 'general';
    const query = isScan ? DEFAULT_QUERIES.scan : question;
    const data = await run({
      kind: isScan ? `scan:${adapter}` : 'chat', query, adapter,
      userEntry: isScan ? { role: 'system', content: `Initiating ${adapter} scan...` } : { role: 'user', content: question },
      useBoth: false, imagesOverride: [{ file, modality: 'optical' }], historyOverride: [], revealEvidence: false,
    });
    if (data && isScan) setLatest((prev) => ({ ...prev, scan: { adapter, answer: data.answer } }));
    return data;
  }, [run, setImage]);

  const runCompare = useCallback(async (text) => {
    const query = (text || '').trim() || (mode.id === 'fusion' ? DEFAULT_QUERIES.fusion : DEFAULT_QUERIES.change);
    const data = await run({ kind: 'compare', query, adapter: 'general', userEntry: { role: 'user', content: query }, useBoth: true });
    if (data) setLatest((prev) => ({ ...prev, compare: { mode: mode.id, answer: data.answer } }));
    return data;
  }, [run, mode.id]);

  return {
    slots: { a, b }, images, mode,
    setImage, setModality, removeImageB,
    viewMode, setViewMode, activeLayer, setActiveLayer,
    chat, result, latest, error, isExecuting, busy,
    sendMessage, runScan, runCompare, analyzeView, dismissError: () => setError(null),
  };
}
