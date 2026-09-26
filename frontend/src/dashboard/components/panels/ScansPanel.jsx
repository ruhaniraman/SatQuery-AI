import { CircleAlert, Eye } from 'lucide-react';
import { Button, Card, Label, Notice } from '../ui';
import { cx, useScrollToNew } from '../uiHelpers';
import ScanResult from '../ScanResult';
import { mapLink } from '../../utils/scanResult';
import { SCANS } from '../../constants';

// Tailwind needs the class names written out in full, so each tone lists them explicitly.
const TONES = {
  amber: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
  emerald: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300',
  rose: 'border-rose-500/30 bg-rose-500/15 text-rose-300',
};

function ScanCard({ scan, ws }) {
  const { runScan, isExecuting, busy, slots } = ws;
  const Icon = scan.icon;
  const running = busy === `scan:${scan.id}`;
  return (
    <Card className="flex items-center gap-3 p-3.5">
      <div className={cx('flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border', TONES[scan.tone])}>
        <Icon size={18} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-slate-100">{scan.title}</p>
        <p className="text-xs leading-snug text-slate-400">{scan.description}</p>
      </div>
      <Button
        variant={scan.tone} size="sm" loading={running}
        disabled={isExecuting || !slots.a.file}
        onClick={() => runScan(scan.id)}
        aria-label={`Run ${scan.title} scan`}
      >
        {running ? 'Scanning' : 'Run scan'}
      </Button>
    </Card>
  );
}

export default function ScansPanel({ ws, goTo }) {
  const { latest, errorFor, dismissError, slots, setViewMode, setActiveLayer, focusArea } = ws;
  const error = errorFor('scans');
  const scan = latest.scan && SCANS.find((s) => s.id === latest.scan.adapter);
  const resultRef = useScrollToNew(latest.scan?.id);

  return (
    <div className="space-y-3">
      {!slots.a.file && (
        <Notice tone="info">
          Add an image first.{' '}
          <button type="button" className="cursor-pointer font-semibold underline" onClick={() => goTo('imagery')}>Open Imagery</button>
        </Notice>
      )}

      {SCANS.map((s) => <ScanCard key={s.id} scan={s} ws={ws} />)}

      {error && <Notice tone="error" icon={CircleAlert} title="Scan failed" onDismiss={dismissError}>{error}</Notice>}

      {scan && (
        <Card ref={resultRef} className="scroll-mt-2 space-y-2 p-3.5">
          <Label>{scan.title} &middot; latest result</Label>
          {latest.scan.data.scan
            ? <ScanResult scan={latest.scan.data.scan} link={mapLink(latest.scan.meta)} onShow={(finding) => focusArea(finding.box)} />
            : <p className="text-[13px] leading-relaxed text-slate-100">{latest.scan.data.answer}</p>}
          {latest.scan.data.visual_evidence_url && (
            <Button variant="subtle" size="sm" icon={Eye} onClick={() => { setViewMode('inputs'); setActiveLayer('evidence'); }}>
              View highlights
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}
