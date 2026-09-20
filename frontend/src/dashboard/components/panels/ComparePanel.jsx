import React, { useState } from 'react';
import { CircleAlert, Eye, GitCompare, Play } from 'lucide-react';
import { Button, Card, FormattedAnswer, Label, Notice } from '../ui';
import { ModeCard } from './ImageryPanel';

export default function ComparePanel({ ws, goTo }) {
  const { mode, images, isExecuting, busy, error, dismissError, runCompare, latest, result, setViewMode, setActiveLayer } = ws;
  const [question, setQuestion] = useState('');
  const ready = images.length === 2;
  const running = busy === 'compare';
  const action = mode.id === 'fusion' ? 'Run fusion' : 'Run change detection';

  return (
    <div className="space-y-3">
      <Label icon={GitCompare}>Compare two images</Label>
      <p className="text-xs leading-relaxed text-slate-400">
        Two images of the same sensor type are compared for change. An optical image and a SAR image are fused
        into one composite. Set each image&rsquo;s sensor type in Imagery.
      </p>

      <ModeCard mode={mode} />

      {!ready && (
        <Notice tone="info">
          {images.length === 0 ? 'Add two images to compare them.' : 'Add a second image (Image B) to compare.'}{' '}
          <button type="button" className="cursor-pointer font-semibold underline" onClick={() => goTo('imagery')}>Open Imagery</button>
        </Notice>
      )}

      <div>
        <label htmlFor="compare-question" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Question <span className="font-normal normal-case tracking-normal text-slate-500">(optional)</span>
        </label>
        <textarea
          id="compare-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          disabled={!ready || isExecuting}
          placeholder="e.g. Has any new construction appeared?"
          className="w-full resize-none rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
        />
      </div>

      <Button icon={Play} loading={running} disabled={!ready || isExecuting} onClick={() => runCompare(question)} className="w-full">
        {running ? 'Analysing' : action}
      </Button>

      {error && <Notice tone="error" icon={CircleAlert} title="Comparison failed" onDismiss={dismissError}>{error}</Notice>}

      {latest.compare && (
        <Card className="space-y-2 p-3.5">
          <Label>{latest.compare.mode === 'fusion' ? 'Fusion' : 'Change detection'} &middot; latest result</Label>
          <FormattedAnswer text={latest.compare.answer} className="text-[13px] leading-relaxed text-slate-100" />
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
