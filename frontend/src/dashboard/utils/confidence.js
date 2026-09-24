// What the answer card shows about the model's confidence, read from the /analyze trace. Pure.
//
// The backend's `telemetry.confidence` is the probability the model gave its own answer tokens (for a yes/no,
// count or rural/urban question: the remote-sensing VQA adapter's short answer). It is NOT a calibrated chance
// of being right; the benchmark run found the base model overconfident, so the card says "model certainty".

export const LEVELS = [
  { min: 0.85, level: 'high', label: 'High' },
  { min: 0.6, level: 'medium', label: 'Medium' },
  { min: 0, level: 'low', label: 'Low' },
];

export function confidenceInfo(data) {
  const telemetry = data?.agent_execution_trace?.telemetry;
  const value = telemetry?.confidence;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const clamped = Math.min(1, Math.max(0, value));
  const { level, label } = LEVELS.find((l) => clamped >= l.min);
  return {
    pct: Math.round(clamped * 100),
    level,
    label,
    adapted: Boolean(telemetry.short_answer),
    method: telemetry.confidence_method || '',
  };
}
