import React from 'react';
import { CircleAlert, Eye, Pickaxe, ScanSearch, Sprout, Trees } from 'lucide-react';
import { Button, Card, Label, Notice, cx } from '../ui';

// Tailwind needs the class names written out in full, so each tone lists them explicitly.
const TONES = {
  amber: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
  emerald: 'border-emerald-500/30 bg-emerald-500/15 text-emerald-300',
  rose: 'border-rose-500/30 bg-rose-500/15 text-rose-300',
};

export const SCANS = [
  { id: 'mining', title: 'Mining', icon: Pickaxe, tone: 'amber', description: 'Surface extraction pits, quarries and open-cast workings.' },
  { id: 'agriculture', title: 'Agriculture', icon: Sprout, tone: 'emerald', description: 'Cultivated fields and crop rows.' },
  { id: 'deforestation', title: 'Deforestation', icon: Trees, tone: 'rose', description: 'Clear-cut land, active logging and canopy loss.' },
];

function ScanCard({ scan, ws }) {
  const { runScan, isExecuting, busy, images } = ws;
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
        disabled={isExecuting || images.length === 0}
        onClick={() => runScan(scan.id)}
        aria-label={`Run ${scan.title} scan`}
      >
        {running ? 'Scanning' : 'Run scan'}
      </Button>
    </Card>
  );
}

export default function ScansPanel({ ws, goTo }) {
  const { latest, error, dismissError, images, setViewMode, setActiveLayer, result } = ws;
  const scan = latest.scan && SCANS.find((s) => s.id === latest.scan.adapter);

  return (
    <div className="space-y-3">
      <Label icon={ScanSearch}>Feature scans</Label>
      <p className="text-xs leading-relaxed text-slate-400">
        Each scan examines Image A on a 4&times;4 grid and outlines the matching regions on the evidence layer.
      </p>

      {images.length === 0 && (
        <Notice tone="info">
          Add an image first.{' '}
          <button type="button" className="cursor-pointer font-semibold underline" onClick={() => goTo('imagery')}>Open Imagery</button>
        </Notice>
      )}

      {SCANS.map((s) => <ScanCard key={s.id} scan={s} ws={ws} />)}

      {error && <Notice tone="error" icon={CircleAlert} title="Scan failed" onDismiss={dismissError}>{error}</Notice>}

      {scan && (
        <Card className="space-y-2 p-3.5">
          <Label>{scan.title} scan &middot; latest result</Label>
          <p className="text-[13px] leading-relaxed text-slate-100">{latest.scan.answer}</p>
          {result?.visual_evidence_url && (
            <Button variant="subtle" size="sm" icon={Eye} onClick={() => { setViewMode('inputs'); setActiveLayer('evidence'); }}>
              View evidence
            </Button>
          )}
        </Card>
      )}
    </div>
  );
}
