// Pure, framework-free helpers for talking to the backend. No JSX, no assets, no DOM: everything
// here runs (and is tested) in plain Node.

// Point the build at another backend with VITE_BACKEND_URL (e.g. in frontend/.env)
export const BACKEND_URL = (import.meta.env?.VITE_BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, '');

// A backend reached through an ngrok tunnel (the free plan): ngrok answers every browser request with its own
// "You are about to visit" HTML page unless this header is sent, so every call to such a backend carries it
// (and its images are fetched with it, see hooks/useBackendImage.js). Other backends get no extra header, so
// local use does not pay for a CORS preflight.
export function needsTunnelHeader(url = BACKEND_URL) {
  try {
    return /(^|\.)ngrok(-free)?\.(app|dev|io)$/.test(new URL(url).hostname);
  } catch {
    return false;
  }
}
export const tunnelHeaders = (url = BACKEND_URL) => (needsTunnelHeader(url) ? { 'ngrok-skip-browser-warning': '1' } : {});

// Headers for a backend call, or undefined when there are none.
export function backendHeaders(baseUrl = BACKEND_URL, extra = {}) {
  const headers = { ...tunnelHeaders(baseUrl), ...extra };
  return Object.keys(headers).length ? headers : undefined;
}

// Telemetry values are arbitrary JSON (strings, numbers, booleans, null, lists, objects). React
// throws on an object child and renders nothing for booleans, so turn everything into text.
export function formatTelemetryValue(val) {
  if (val === null || val === undefined) return 'n/a';
  if (typeof val === 'boolean') return val ? 'yes' : 'no';
  if (Array.isArray(val)) return val.length ? val.map(formatTelemetryValue).join('; ') : 'none';
  if (typeof val === 'object') {
    if (typeof val.summary === 'string') return val.summary;
    return JSON.stringify(val);
  }
  return String(val);
}

// FastAPI returns `detail` as a string for our own errors but as a list of {loc, msg} objects for
// request-validation (422) errors. Anything else is stringified rather than shown as [object Object].
export function formatErrorDetail(detail) {
  if (detail === null || detail === undefined || detail === '') return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      if (typeof d === 'string') return d;
      if (d && typeof d.msg === 'string') {
        const where = Array.isArray(d.loc) ? d.loc.filter((part) => part !== 'body').join('.') : '';
        return where ? `${where}: ${d.msg}` : d.msg;
      }
      return JSON.stringify(d);
    }).join('; ');
  }
  if (typeof detail === 'object') return typeof detail.msg === 'string' ? detail.msg : JSON.stringify(detail);
  return String(detail);
}

// Never throws: a proxy/gateway error page or an empty body is not JSON, and must not turn into an
// unhandled promise rejection instead of the error banner.
export async function readErrorMessage(response, what = 'Analysis') {
  const fallback = `${what} failed (HTTP ${response.status}).`;
  try {
    const text = await response.text();
    try {
      return formatErrorDetail(JSON.parse(text)?.detail) || fallback;
    } catch {
      return fallback;
    }
  } catch {
    return fallback;
  }
}

// What the backend will do with the current images. It mirrors the server's routing rule (image
// count + the sensor type the user tagged each image with); the server stays the authority.
export function analysisMode(images) {
  if (!images || images.length === 0) {
    return { id: 'none', label: 'No imagery yet', hint: 'Add an image to begin.' };
  }
  if (images.length === 1) {
    return { id: 'single', label: 'Single-image analysis', hint: 'Ask questions or run a feature scan.' };
  }
  if (images[0].modality === images[1].modality) {
    return {
      id: 'change',
      label: 'Change detection',
      hint: 'Same sensor type on both images. Image A is treated as BEFORE and Image B as AFTER unless their capture dates say otherwise.',
    };
  }
  return {
    id: 'fusion',
    label: 'Optical–SAR fusion',
    hint: 'Different sensor types. The two images are fused into one composite and described together.',
  };
}

// Queries the API requires (an empty query is rejected) when the user has not typed one.
export const DEFAULT_QUERIES = {
  scan: 'Extract features.',
  change: 'Summarize the key changes.',
  fusion: 'Describe the scene.',
};

export function buildAnalyzeFormData({ query, adapter = 'general', images, history = [] }) {
  const form = new FormData();
  form.append('query', query);
  form.append('adapter', adapter);
  images.forEach((img, i) => {
    form.append('images', img.file);
    form.append(i === 0 ? 'modality_a' : 'modality_b', img.modality);
  });
  form.append('chat_history', JSON.stringify(history));
  return form;
}

// The model endpoints (/analyze, /preview, /report) need the signed-in user's session token, sent as a
// Bearer header (a missing or expired one is a 401 whose message asks to sign in again).
const authHeaders = (token, baseUrl) => backendHeaders(baseUrl, token ? { Authorization: `Bearer ${token}` } : {});

// An Error fit to show the user, carrying the HTTP status (401 = signed out: the dashboard then re-checks the session).
async function httpError(response, what) {
  const err = new Error(await readErrorMessage(response, what));
  err.status = response.status;
  return err;
}

// `images` is [{ file, modality }] with Image A first. The run is added to the signed-in account's report
// history (GET /auth/reports). `signal`, when given (an AbortController's), lets the
// caller stop the request; that rejection is an AbortError, passed through as-is rather than turned
// into the generic "cannot reach the server" message. Resolves to the parsed response, or throws an
// Error whose message is fit to show the user.
export async function requestAnalysis({ query, adapter, images, history, token, signal, fetchImpl = fetch, baseUrl = BACKEND_URL }) {
  const body = buildAnalyzeFormData({ query, adapter, images, history });
  let response;
  try {
    response = await fetchImpl(`${baseUrl}/analyze`, { method: 'POST', body, headers: authHeaders(token, baseUrl), signal });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    throw new Error(`Cannot reach the analysis server at ${baseUrl}. Is the backend running?`, { cause: err });
  }
  if (!response.ok) throw await httpError(response, 'Analysis');
  return response.json();
}

// Browsers cannot draw TIFF/GeoTIFF, so the backend renders one to a PNG using the same loader the
// analysis uses (percentile stretch, SAR despeckle). Resolves to the PNG as a Blob.
export async function requestPreview({ file, modality = 'optical', token, fetchImpl = fetch, baseUrl = BACKEND_URL }) {
  const form = new FormData();
  form.append('image', file);
  form.append('modality', modality);
  let response;
  try {
    response = await fetchImpl(`${baseUrl}/preview`, { method: 'POST', body: form, headers: authHeaders(token, baseUrl) });
  } catch {
    throw new Error(`Cannot reach the server at ${baseUrl} to render a preview.`);
  }
  if (!response.ok) throw await httpError(response, 'Preview');
  return response.blob();
}

// Rebuild the PDF of one finished run with the whole conversation so far. Resolves to
// { report_download_url, report_error }.
export async function requestReport({ sessionId, history = [], mapLink = null, token, fetchImpl = fetch, baseUrl = BACKEND_URL }) {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('chat_history', JSON.stringify(history));
  if (mapLink) form.append('map_link', mapLink);
  let response;
  try {
    response = await fetchImpl(`${baseUrl}/report`, { method: 'POST', body: form, headers: authHeaders(token, baseUrl) });
  } catch {
    throw new Error(`Cannot reach the server at ${baseUrl} to build the report.`);
  }
  if (!response.ok) throw await httpError(response, 'Report');
  return response.json();
}

// What the header shows. Never throws: an unreachable backend is a state, not an error.
//   ready | loading (up, model still loading) | model-error | offline
export async function checkHealth({ fetchImpl = fetch, baseUrl = BACKEND_URL, timeoutMs = 4000 } = {}) {
  const controller = typeof AbortController === 'function' ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const init = { headers: backendHeaders(baseUrl), signal: controller?.signal };
    const response = await fetchImpl(`${baseUrl}/health`, init.headers || init.signal ? init : undefined);
    if (!response.ok) return { state: 'offline', message: `The backend answered HTTP ${response.status}.` };
    const body = await response.json();
    if (body.model_loaded) return { state: 'ready', message: 'Backend and model are ready.' };
    if (body.model_error) return { state: 'model-error', message: `The model failed to load: ${body.model_error}` };
    return { state: 'loading', message: 'The model is still loading.' };
  } catch {
    return { state: 'offline', message: `Cannot reach the backend at ${baseUrl}.` };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export const evidenceUrl = (result, baseUrl = BACKEND_URL) =>
  result?.visual_evidence_url ? `${baseUrl}${result.visual_evidence_url}` : null;

export const reportUrl = (result, baseUrl = BACKEND_URL) =>
  result?.report_download_url ? `${baseUrl}${result.report_download_url}` : null;

export const reportFileName = (result) =>
  `satquery-report-${result?.agent_execution_trace?.pipeline_id || 'report'}.pdf`;

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Save a PDF from a same-origin blob: <a download> is ignored for cross-origin URLs (backend :8000
// vs UI :5173), so the PDF would just open in a tab instead of downloading. Falls back to opening it.
export async function savePdf(url, filename) {
  try {
    const res = await fetch(url, { headers: backendHeaders(url) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const objectUrl = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  } catch {
    window.open(url, '_blank', 'noopener');
  }
}

export const isTiffFile = (file) => Boolean(file && /\.tiff?$/i.test(file.name));

export const isTwoImageRun = (images) => images.length === 2;

// Nicer labels for the backend's stage/task identifiers
export function humanizeIdentifier(value) {
  return String(value ?? '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
