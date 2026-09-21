import React from 'react';
import { CircleCheck, TriangleAlert } from 'lucide-react';
import { FormattedAnswer, Notice, cx } from './ui';
import { parseChangeAnswer } from '../utils/changeAnswer';

// Tailwind needs the class names written out in full.
const KIND_STYLE = {
  major: 'border-red-500/30 bg-red-500/15 text-red-300',
  likely: 'border-amber-500/30 bg-amber-500/15 text-amber-300',
  possible: 'border-white/15 bg-white/5 text-slate-300',
};

function Row({ label, children }) {
  return (
    <div className="grid grid-cols-[3.75rem,1fr] gap-x-2 text-[12px] leading-relaxed" style={{ gridTemplateColumns: '3.75rem 1fr' }}>
      <span className="pt-px text-[10px] font-semibold uppercase tracking-wider text-slate-500">{label}</span>
      <span className="min-w-0 break-words text-slate-200">{children}</span>
    </div>
  );
}

function AreaCard({ area }) {
  return (
    <div className="space-y-2 rounded-lg border border-white/10 bg-black/30 p-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className={cx('rounded-md border px-2 py-0.5 text-[11px] font-semibold', KIND_STYLE[area.kind])}>{area.label}</span>
        <span className="text-[13px] font-medium capitalize text-slate-100">{area.where}</span>
        <span className="text-[11px] text-slate-400">~{area.sizePercent}% of the scene</span>
      </div>
      {area.scans.length > 0 && (
        <p className="text-[11px] text-slate-400">
          Also flagged by the <span className="font-medium text-slate-200">{area.scans.join(', ')}</span> scan{area.scans.length > 1 ? 's' : ''}
        </p>
      )}
      {(area.before || area.after || area.measured) && (
        <div className="space-y-1.5 border-t border-white/10 pt-2">
          {area.before && <Row label="Before">{area.before}</Row>}
          {area.after && <Row label="After">{area.after}</Row>}
          {area.measured && <Row label="Measured"><span className="text-slate-400">{area.measured}</span></Row>}
        </div>
      )}
    </div>
  );
}

// The comparison answer as warnings, one card per changed area (Before / After / Measured), and any
// plain notes. An older, model-written answer is shown as paragraphs instead.
export default function ChangeAnswer({ text }) {
  const parsed = parseChangeAnswer(text);
  if (parsed.legacy) return <FormattedAnswer text={text} className="text-[13px] leading-relaxed text-slate-100" />;
  return (
    <div className="space-y-2.5">
      {parsed.warnings.map((warning, i) => <Notice key={i} tone="warn" icon={TriangleAlert}>{warning}</Notice>)}
      {parsed.areas.length === 0 && parsed.notes.map((note, i) => (
        <p key={i} className="flex items-start gap-2 text-[13px] leading-relaxed text-slate-200">
          <CircleCheck size={15} className="mt-0.5 shrink-0 text-blue-400" />{note}
        </p>
      ))}
      {parsed.areas.map((area, i) => <AreaCard key={i} area={area} />)}
      {parsed.footer.map((line, i) => <p key={i} className="text-[11px] leading-relaxed text-slate-500">{line}</p>)}
    </div>
  );
}
