import { useState } from 'react';
import {
  ArrowRight, Check, Download, FileText, GitCompareArrows, Merge, MessageSquare, RotateCcw, ScanSearch, TriangleAlert,
} from 'lucide-react';
import { Button, Card, EmptyState, KeyValue, Label, Notice, PreviewImage } from '../ui';
import { cx } from '../uiHelpers';
import { evidenceUrl, formatTelemetryValue, humanizeIdentifier, savePdf } from '../../utils/api';
import { describeReportChoice, runTime } from '../../utils/runs';
import AccountReports from '../AccountReports';

const sentence = (value) => {
  const text = humanizeIdentifier(value).toLowerCase();
  return text.charAt(0).toUpperCase() + text.slice(1);
};

const KIND_ICON = { chat: MessageSquare, scan: ScanSearch, change: GitCompareArrows, fusion: Merge };

// Telemetry that is printed in the PDF instead of in this table
const HIDDEN_TELEMETRY = new Set(['comparison_details']);

function RunRow({ run, chosen, onPick }) {
  const Icon = KIND_ICON[run.kind] || FileText;
  return (
    <button
      type="button"
      onClick={() => onPick(run.id)}
      aria-pressed={chosen}
      className={cx(
        'flex w-full cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
        chosen ? 'border-blue-500/50 bg-blue-500/10' : 'border-white/10 bg-black/20 hover:bg-white/5',
      )}
    >
      <span className={cx('flex h-7 w-7 shrink-0 items-center justify-center rounded-md', run.annotated ? 'bg-blue-500/15 text-blue-300' : 'bg-white/5 text-slate-400')}>
        <Icon size={14} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] text-slate-100" title={run.title}>{run.title}</span>
        <span className="block text-[11px] text-slate-500">{runTime(run)} · {run.annotated ? 'annotated evidence' : 'plain answer'}</span>
      </span>
      {chosen && <Check size={14} className="shrink-0 text-blue-300" aria-label="Used in the report" />}
    </button>
  );
}

export default function ReportPanel({ ws }) {
  const { runs, reportRun, reportRunId, selectReportRun, prepareReport } = ws;
  const [state, setState] = useState({ busy: false, error: null, note: null });

  if (!reportRun) {
    return (
      <div className="space-y-3.5">
        <EmptyState icon={FileText} title="No report yet">
          Run an analysis to see what was done, which model produced the answer, and download the PDF report.
        </EmptyState>
        <AccountReports />
      </div>
    );
  }

  const trace = reportRun.data.agent_execution_trace;
  const evidence = evidenceUrl(reportRun.data);
  const telemetry = Object.entries(trace.telemetry || {}).filter(([key]) => !HIDDEN_TELEMETRY.has(key));

  const download = async () => {
    setState({ busy: true, error: null, note: null });
    const out = await prepareReport();
    if (out.url) {
      await savePdf(out.url, out.filename);
      setState({ busy: false, error: null, note: out.fallback ? `Saved the run's own report, because the full one could not be built (${out.fallback}).` : null });
    } else {
      setState({ busy: false, error: out.error, note: null });
    }
  };

  return (
    <div className="space-y-3.5">
      <Card className="space-y-2.5 p-3.5">
        <div className="aspect-video overflow-hidden rounded-lg border border-white/10 bg-black/40">
          <PreviewImage src={evidence} alt={`Evidence of ${reportRun.title}`} className="h-full w-full object-contain" />
        </div>
        <p className="text-xs leading-relaxed text-slate-400">{describeReportChoice(runs, reportRunId)}</p>
        <Button icon={Download} loading={state.busy} onClick={download} className="w-full">Download PDF report</Button>
        {state.error && <Notice tone="error" title="Report unavailable">{state.error}</Notice>}
        {state.note && <Notice tone="warn" icon={TriangleAlert}>{state.note}</Notice>}
      </Card>

      <div>
        <Label
          className="mb-2"
          right={reportRunId && (
            <button type="button" onClick={() => selectReportRun(null)} className="inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium text-blue-300 hover:underline">
              <RotateCcw size={11} /> Automatic
            </button>
          )}
        >
          Analyses in this session
        </Label>
        <div className="space-y-1.5">
          {runs.map((run) => <RunRow key={run.id} run={run} chosen={run.id === reportRun.id} onPick={selectReportRun} />)}
        </div>
      </div>

      <Label className="!mt-5">About this analysis &middot; {reportRun.title}</Label>
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
          <Label className="mb-2">Technical details</Label>
          <Card className="p-3.5">
            <KeyValue rows={telemetry.map(([key, value]) => [sentence(key), formatTelemetryValue(value)])} />
          </Card>
        </div>
      )}

      <AccountReports className="!mt-5" />
    </div>
  );
}
