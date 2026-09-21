import React from 'react';
import { ArrowRight, Check, ChevronLeft, ChevronRight, CircleAlert, Info, Loader2, RefreshCw } from 'lucide-react';
import { Button, cx } from './ui';
import { formatReleaseDate, yearTicks } from '../utils/wayback';
import { SAME_AREA_MIN, boundsOverlap } from '../utils/views';

const NOTE = 'Dates are when Esri published each version of its basemap, not when the ground was photographed. '
  + 'Where nothing new was captured, a newer version repeats older imagery, so two dates can look identical.';

function Chip({ label, meta }) {
  const set = Boolean(meta?.date);
  return (
    <span className={cx('inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px]', set ? 'border-blue-500/40 bg-blue-500/10 text-blue-200' : 'border-white/10 bg-white/5 text-slate-500')}>
      {set && <Check size={11} aria-hidden="true" />}
      <span className="font-semibold">{label}</span>
      <span>{set ? formatReleaseDate(meta.date) : 'not set'}</span>
    </span>
  );
}

// A date slider over the imagery archive. The map shows the chosen date; "Use as Before / After" captures what is
// on screen into Change detection, remembering the date and the area so a mismatch can be flagged. Keep the
// map still between the two captures: both must show the same ground.
export default function Timeline({ status, error, releases, index, ready, busy, note, before, after, onIndex, onRetry, onUse, onOpenChange }) {
  const shell = 'pointer-events-auto absolute left-3 top-[3.25rem] z-10 w-[min(27rem,calc(100%-23rem))] rounded-xl border border-white/15 bg-black/70 p-3 backdrop-blur-md';

  if (status === 'loading' || status === 'idle') {
    return <div className={cx(shell, 'flex items-center gap-2 text-xs text-slate-300')}><Loader2 size={14} className="animate-spin text-blue-400" /> Loading available dates…</div>;
  }
  if (status === 'error') {
    return (
      <div className={cx(shell, 'space-y-2')} role="alert">
        <p className="flex items-start gap-2 text-xs text-red-200"><CircleAlert size={14} className="mt-0.5 shrink-0" />{error}</p>
        <Button variant="subtle" size="sm" icon={RefreshCw} onClick={onRetry}>Try again</Button>
      </div>
    );
  }

  const release = releases[index];
  const last = releases.length - 1;
  const overlap = boundsOverlap(before.meta?.bounds, after.meta?.bounds);
  const differ = overlap !== null && overlap < SAME_AREA_MIN;
  const both = before.meta?.date && after.meta?.date;
  const step = (delta) => onIndex(Math.min(last, Math.max(0, index + delta)));

  return (
    <div className={cx(shell, 'space-y-2.5')}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Imagery date</span>
          <span title={NOTE} className="cursor-help text-slate-500" aria-label={NOTE} role="img"><Info size={13} /></span>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" onClick={() => step(-1)} disabled={index <= 0 || busy} aria-label="Earlier version"
            className="inline-flex h-6 w-6 cursor-pointer items-center justify-center rounded-md text-slate-300 transition enabled:hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"><ChevronLeft size={15} /></button>
          <span className="min-w-[6.5rem] text-center text-sm font-semibold text-slate-100" aria-live="polite">{formatReleaseDate(release.date)}</span>
          <button type="button" onClick={() => step(1)} disabled={index >= last || busy} aria-label="Later version"
            className="inline-flex h-6 w-6 cursor-pointer items-center justify-center rounded-md text-slate-300 transition enabled:hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"><ChevronRight size={15} /></button>
        </div>
      </div>

      <div>
        <input
          type="range" min={0} max={last} step={1} value={index}
          onChange={(e) => onIndex(Number(e.target.value))}
          disabled={busy}
          aria-label="Imagery date" aria-valuetext={formatReleaseDate(release.date)}
          className="h-1.5 w-full cursor-pointer accent-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <div className="relative mt-1 h-3.5 text-[10px] text-slate-500" aria-hidden="true">
          {yearTicks(releases).map((tick) => (
            <span key={tick.index} className="absolute -translate-x-1/2" style={{ left: `${last ? (tick.index / last) * 100 : 0}%` }}>{tick.year}</span>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="sm" variant="subtle" disabled={!ready || busy} onClick={() => onUse('before')}>Use as Before</Button>
        <Button size="sm" variant="subtle" disabled={!ready || busy} onClick={() => onUse('after')}>Use as After</Button>
        {!ready && <span className="inline-flex items-center gap-1 text-[11px] text-slate-400"><Loader2 size={11} className="animate-spin" /> Loading</span>}
        {busy && <span className="inline-flex items-center gap-1 text-[11px] text-slate-400"><Loader2 size={11} className="animate-spin" /> Capturing</span>}
      </div>

      {note && <p role="alert" className="text-[11px] leading-snug text-red-300">{note}</p>}

      {(before.meta?.date || after.meta?.date) && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Chip label="Before" meta={before.meta} />
          <Chip label="After" meta={after.meta} />
          {both && !differ && (
            <Button size="sm" variant="primary" icon={ArrowRight} onClick={onOpenChange} className="ml-auto">Open Change detection</Button>
          )}
        </div>
      )}
      {differ && (
        <p role="status" className="rounded-lg border border-amber-700/40 bg-amber-950/30 px-2.5 py-1.5 text-[11px] leading-snug text-amber-100">
          The two captures show different areas ({Math.round(overlap * 100)}% overlap). Keep the map still between them.
        </p>
      )}
    </div>
  );
}
