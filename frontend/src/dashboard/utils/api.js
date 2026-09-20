// Pure, framework-free helpers for talking to the backend. No JSX, no assets, no DOM: everything
// here runs (and is tested) in plain Node.

// Point the build at another backend with VITE_BACKEND_URL (e.g. in frontend/.env)
export const BACKEND_URL = import.meta.env?.VITE_BACKEND_URL || 'http://localhost:8000';

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
export async function readErrorMessage(response) {
  const fallback = `Analysis failed (HTTP ${response.status}).`;
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

// `images` is [{ file, modality }] with Image A first. Resolves to the parsed response, or throws an
// Error whose message is fit to show the user.
export async function requestAnalysis({ query, adapter, images, history, fetchImpl = fetch, baseUrl = BACKEND_URL }) {
  const body = buildAnalyzeFormData({ query, adapter, images, history });
  let response;
  try {
    response = await fetchImpl(`${baseUrl}/analyze`, { method: 'POST', body });
  } catch {
    throw new Error(`Cannot reach the analysis server at ${baseUrl}. Is the backend running?`);
  }
  if (!response.ok) throw new Error(await readErrorMessage(response));
  return response.json();
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
