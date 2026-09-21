import React, { useState } from 'react';
import { CircleAlert, Eye, Play } from 'lucide-react';
import { Button, Card, FormattedAnswer, Label, Notice, useScrollToNew } from '../ui';
import ImageSlot from '../ImageSlot';
import { runTime } from '../../utils/runs';

// Optical-SAR fusion: one optical and one SAR image of the same place, fused into a single composite
// (optical colour, SAR brightness) and described. Own inputs; the sensor type of each is fixed by its slot.
export default function FusionPanel({ ws }) {
  const { slots, setImage, removeImage, isExecuting, busy, runFusion, errorFor, dismissError, latest, setViewMode, setActiveLayer } = ws;
  const [question, setQuestion] = useState('');
  const ready = Boolean(slots.optical.file && slots.sar.file);
  const running = busy === 'fusion';
  const error = errorFor('fusion');
  const run = latest.fusion;
  const resultRef = useScrollToNew(run?.id);

  return (
    <div className="space-y-3">
      <ImageSlot id="optical" title="Optical image" slot={slots.optical} onFile={setImage} onRemove={removeImage} sensor="fixed" />
      <ImageSlot id="sar" title="SAR image" slot={slots.sar} onFile={setImage} onRemove={removeImage} sensor="fixed" />

      {!ready && <Notice tone="info">Add both images to fuse them.</Notice>}

      <div>
        <label htmlFor="fusion-question" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Question <span className="font-normal normal-case tracking-normal text-slate-500">(optional)</span>
        </label>
        <textarea
          id="fusion-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={2}
          disabled={!ready || isExecuting}
          placeholder="e.g. Where is open water?"
          className="w-full resize-none rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
        />
      </div>

      <Button icon={Play} loading={running} disabled={!ready || isExecuting} onClick={() => runFusion(question)} className="w-full">
        {running ? 'Analysing' : 'Run fusion'}
      </Button>

      {error && <Notice tone="error" icon={CircleAlert} title="Fusion failed" onDismiss={dismissError}>{error}</Notice>}

      {run && (
        <Card ref={resultRef} className="scroll-mt-2 space-y-2.5 p-3.5">
          <Label right={<span className="text-[11px] text-slate-500">{runTime(run)}</span>}>Latest result</Label>
          <FormattedAnswer text={run.data.answer} className="text-[13px] leading-relaxed text-slate-100" />
          {run.data.visual_evidence_url && (
            <Button variant="subtle" size="sm" icon={Eye} onClick={() => { setViewMode('inputs'); setActiveLayer('evidence'); }}>View composite</Button>
          )}
        </Card>
      )}
    </div>
  );
}
