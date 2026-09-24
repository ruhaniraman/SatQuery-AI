import { test } from 'node:test';
import assert from 'node:assert/strict';
import { confidenceInfo } from './confidence.js';

const withTelemetry = (telemetry) => ({ agent_execution_trace: { telemetry } });

test('reads the trace confidence as a percentage and a level', () => {
  const info = confidenceInfo(withTelemetry({ confidence: 0.934, confidence_method: 'probability ...', short_answer: 'Yes' }));
  assert.deepEqual(info, { pct: 93, level: 'high', label: 'High', adapted: true, method: 'probability ...' });
  assert.equal(confidenceInfo(withTelemetry({ confidence: 0.7 })).level, 'medium');
  assert.equal(confidenceInfo(withTelemetry({ confidence: 0.3 })).label, 'Low');
  assert.equal(confidenceInfo(withTelemetry({ confidence: 0.85 })).level, 'high');   // boundary is inclusive
  assert.equal(confidenceInfo(withTelemetry({ confidence: 0.7 })).adapted, false);
});

test('no number means nothing to show (scans, change detection, fusion, old runs)', () => {
  assert.equal(confidenceInfo(withTelemetry({})), null);
  assert.equal(confidenceInfo(withTelemetry({ confidence: null })), null);
  assert.equal(confidenceInfo(withTelemetry({ confidence: 'high' })), null);
  assert.equal(confidenceInfo({}), null);
  assert.equal(confidenceInfo(null), null);
});

test('out-of-range values are clamped, never shown above 100%', () => {
  assert.equal(confidenceInfo(withTelemetry({ confidence: 1.2 })).pct, 100);
  assert.equal(confidenceInfo(withTelemetry({ confidence: -0.1 })).pct, 0);
});
