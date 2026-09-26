import { cx } from './uiHelpers';

const TONE = {
  high: { bar: 'bg-emerald-400', text: 'text-emerald-300' },
  medium: { bar: 'bg-amber-400', text: 'text-amber-300' },
  low: { bar: 'bg-rose-400', text: 'text-rose-300' },
};

// One line under an answer: "Confidence 93% · High", a small meter, and where the number comes from.
// `info` is utils/confidence.confidenceInfo(). Shown ONLY for the VQA adapter's short answers (yes/no,
// rural/urban), whose calibration was measured on RSVQA-LR. A free-text answer's token probability is ~0.95
// even when the text is generic, so the badge would nearly always read "High"; that number stays in the
// trace, data.json and PDF instead.
export default function ConfidenceBadge({ info, className }) {
  if (!info?.adapted) return null;
  const tone = TONE[info.level];
  const source = 'remote-sensing VQA adapter';
  return (
    <div className={cx('mt-2 flex items-center gap-2 text-[11px] text-slate-400', className)} title={info.method}>
      <span>
        Confidence <span className={cx('font-semibold', tone.text)}>{info.pct}%</span>
        <span className={tone.text}> · {info.label}</span>
      </span>
      <span className="h-1 w-14 overflow-hidden rounded-full bg-white/10" role="img" aria-label={`Confidence ${info.pct} percent, ${info.label}`}>
        <span className={cx('block h-full rounded-full', tone.bar)} style={{ width: `${info.pct}%` }} />
      </span>
      <span className="truncate text-slate-500">{source}</span>
    </div>
  );
}
