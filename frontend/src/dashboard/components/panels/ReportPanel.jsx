import React, { useState } from 'react';
import { ArrowRight, Download, FileText, ShieldCheck, TriangleAlert } from 'lucide-react';
import { Button, Card, EmptyState, KeyValue, Label, Notice } from '../ui';
import { formatTelemetryValue, humanizeIdentifier, reportFileName, reportUrl } from '../../utils/api';

const sentence = (value) => {
  const text = humanizeIdentifier(value).toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
};

// Save the PDF from a same-origin blob: <a download> is ignored for cross-origin URLs (backend :8000
// vs UI :5173), so the PDF would just open in a tab. Falls back to opening it.
export async function saveReport(result) {
  const url = reportUrl(result);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const objectUrl = URL.createObjectURL(await res.blob());
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = reportFileName(result);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  } catch {
    window.open(url, '_blank', 'noopener');
  }
}

export default function ReportPanel({ ws }) {
  const { result } = ws;
  const [downloading, setDownloading] = useState(false);
  const trace = result?.agent_execution_trace;

  if (!trace) {
    return (
      <EmptyState icon={FileText} title="No report yet">
        Run an analysis to see exactly what was executed, which model produced the answer, and download the audit PDF.
      </EmptyState>
    );
  }

  const telemetry = Object.entries(trace.telemetry || {});
  const download = async () => {
    setDownloading(true);
    try { await saveReport(result); } finally { setDownloading(false); }
  };

  return (
    <div className="space-y-3.5">
      <Label icon={ShieldCheck}>Audit trace</Label>

      <Card className="space-y-3 p-3.5">
        <KeyValue rows={[
          ['Task', sentence(trace.task)],
          ['Routing', `${sentence(trace.routing)} — ${trace.routing_reason}`],
          ['Status', sentence(trace.execution_status)],
          ['Validation', sentence(trace.validation_status)],
          ['Run ID', <span key="id" className="font-mono text-[11px] text-blue-300">{trace.pipeline_id}</span>],
        ]} />
      </Card>

      <div>
        <Label className="mb-2">Stages run</Label>
        <ol className="flex flex-wrap items-center gap-1.5">
          {trace.nodes_traversed.map((node, i) => (
            <li key={`${node}-${i}`} className="flex items-center gap-1.5">
              <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-slate-200">{humanizeIdentifier(node)}</span>
              {i < trace.nodes_traversed.length - 1 && <ArrowRight size={12} className="text-slate-600" aria-hidden="true" />}
            </li>
          ))}
        </ol>
      </div>

      {(trace.warnings || []).length > 0 && (
        <div className="space-y-2">
          {trace.warnings.map((w, i) => <Notice key={i} tone="warn" icon={TriangleAlert}>{w}</Notice>)}
        </div>
      )}

      {telemetry.length > 0 && (
        <div>
          <Label className="mb-2">Details</Label>
          <Card className="p-3.5">
            <KeyValue rows={telemetry.map(([key, value]) => [sentence(key), formatTelemetryValue(value)])} />
          </Card>
        </div>
      )}

      {result.report_download_url ? (
        <Button icon={Download} loading={downloading} onClick={download} className="w-full">Download PDF report</Button>
      ) : (
        <div className="space-y-2">
          <Button icon={Download} disabled className="w-full" title={result.report_error || undefined}>Report unavailable</Button>
          {result.report_error && <p className="text-xs text-slate-400">{result.report_error}</p>}
        </div>
      )}
    </div>
  );
}
