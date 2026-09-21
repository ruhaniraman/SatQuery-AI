import React, { useState } from 'react';
import { CircleAlert, Columns2, Eye, Play, TriangleAlert } from 'lucide-react';
import { Button, Card, Label, Notice, Segmented, useScrollToNew } from '../ui';
import ImageSlot, { SENSORS } from '../ImageSlot';
import ChangeAnswer from '../ChangeAnswer';
import { runTime } from '../../utils/runs';
import { SAME_AREA_MIN, boundsOverlap } from '../../utils/views';

// Change detection: two images of the SAME sensor type, compared for change. Has its own inputs, so it
// never shares images with the Assistant or with Optical-SAR fusion.
export default function ChangePanel({ ws }) {
  const { slots, setImage, removeImage, setModality, isExecuting, busy, runChange, errorFor, dismissError, latest, setViewMode, setActiveLayer } = ws;
  const [question, setQuestion] = useState('');
  const bothLoaded = Boolean(slots.before.file && slots.after.file);
  const sameSensor = slots.before.modality === slots.after.modality;
  const ready = bothLoaded && sameSensor;
  const overlap = boundsOverlap(slots.before.meta?.bounds, slots.after.meta?.bounds);   // only for map captures
  const areasDiffer = overlap !== null && overlap < SAME_AREA_MIN;
  const running = busy === 'change';
  const error = errorFor('change');
  const run = latest.change;
  const resultRef = useScrollToNew(run?.id);

  const show = (layer) => { setViewMode('inputs'); setActiveLayer(layer); };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Sensor type</span>
        <Segmented size="sm" label="Sensor type of both images" value={slots.before.modality} onChange={(value) => setModality('before', value)} options={SENSORS} />
      </div>

      <ImageSlot id="before" title="Before" slot={slots.before} onFile={setImage} onRemove={removeImage} />
      <ImageSlot id="after" title="After" slot={slots.after} onFile={setImage} onRemove={removeImage} />

      {!bothLoaded && <Notice tone="info">Add Before and After to compare.</Notice>}
      {bothLoaded && !sameSensor && (
        <Notice tone="warn" icon={TriangleAlert}>
          The two images have different sensor types. Change detection needs the same type on both; use Optical–SAR fusion to combine them.
        </Notice>
      )}

      {bothLoaded && areasDiffer && (
        <Notice tone="warn" icon={TriangleAlert}>
          The two map captures show different areas ({Math.round(overlap * 100)}% overlap). Capture both from the same map view.
        </Notice>
      )}

      <div>
        <label htmlFor="change-question" className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Question <span className="font-normal normal-case tracking-normal text-slate-500">(optional)</span>
        </label>
        <textarea
          id="change-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={2}
          disabled={!ready || isExecuting}
          placeholder="e.g. Has any new construction appeared?"
          className="w-full resize-none rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
        />
      </div>

      <Button icon={Play} loading={running} disabled={!ready || isExecuting} onClick={() => runChange(question)} className="w-full">
        {running ? 'Analysing' : 'Run change detection'}
      </Button>

      {error && <Notice tone="error" icon={CircleAlert} title="Comparison failed" onDismiss={dismissError}>{error}</Notice>}

      {run && (
        <Card ref={resultRef} className="scroll-mt-2 space-y-3 p-3.5">
          <Label right={<span className="text-[11px] text-slate-500">{runTime(run)}</span>}>Latest result</Label>
          <ChangeAnswer text={run.data.answer} />
          <div className="flex flex-wrap gap-2">
            {run.data.visual_evidence_url && <Button variant="subtle" size="sm" icon={Eye} onClick={() => show('evidence')}>View evidence</Button>}
            <Button variant="subtle" size="sm" icon={Columns2} onClick={() => show('swipe')} disabled={!bothLoaded}>Swipe before / after</Button>
          </div>
        </Card>
      )}
    </div>
  );
}
