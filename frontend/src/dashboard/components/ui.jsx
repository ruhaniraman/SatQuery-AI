import React, { useState } from 'react';
import { ImageIcon, Loader2 } from 'lucide-react';
import { cx } from './uiHelpers';

// Shared building blocks. They encode the dashboard's look (dark glass, blue accent, small
// uppercase labels) once, so every panel stays consistent.

export function Card({ className = '', children, ...rest }) {
  return (
    <div className={cx('rounded-xl border border-white/10 bg-black/40', className)} {...rest}>
      {children}
    </div>
  );
}

export function Label({ icon: Icon, children, right, className = '' }) {
  return (
    <div className={cx('flex items-center justify-between gap-2', className)}>
      <h3 className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        {Icon && <Icon size={13} className="text-blue-400" />}
        {children}
      </h3>
      {right}
    </div>
  );
}

// Hover colours apply to ENABLED buttons only (enabled:hover:). A disabled or busy button (for example
// "Analysing") used to inherit the panel's dark background on hover and turn black; now it never changes.
const BUTTON_VARIANTS = {
  primary: 'bg-blue-600/90 enabled:hover:bg-blue-500 text-slate-950 border-white/20',
  subtle: 'bg-white/5 enabled:hover:bg-white/10 text-slate-200 border-white/10',
  ghost: 'bg-transparent enabled:hover:bg-white/5 text-slate-300 border-transparent',
  // The scan colours from the original dashboard (amber / emerald / rose)
  amber: 'bg-amber-600/80 enabled:hover:bg-amber-500 text-white border-white/10',
  emerald: 'bg-emerald-600/80 enabled:hover:bg-emerald-500 text-white border-white/10',
  rose: 'bg-rose-600/80 enabled:hover:bg-rose-500 text-white border-white/10',
};
const BUTTON_SIZES = { sm: 'text-xs px-2.5 py-1.5', md: 'text-[13px] px-3.5 py-2' };

export function Button({ variant = 'primary', size = 'md', icon: Icon, loading = false, className = '', children, ...rest }) {
  return (
    <button
      type="button"
      aria-busy={loading || undefined}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg border font-semibold transition',
        // busy = working (readable, progress cursor); disabled = unavailable (dimmed)
        loading ? 'cursor-progress opacity-80' : 'cursor-pointer disabled:cursor-not-allowed disabled:opacity-40',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
        BUTTON_VARIANTS[variant], BUTTON_SIZES[size], className,
      )}
      {...rest}
    >
      {loading ? <Loader2 size={14} className="animate-spin" /> : Icon && <Icon size={14} />}
      {children}
    </button>
  );
}

// Compact toggle group (single choice). Each option: { value, label, icon, disabled, title }
export function Segmented({ value, onChange, options, label, size = 'md', className = '' }) {
  return (
    <div role="group" aria-label={label} className={cx('inline-flex rounded-lg border border-white/10 bg-black/40 p-0.5', className)}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={active}
            disabled={opt.disabled}
            title={opt.title}
            onClick={() => onChange(opt.value)}
            className={cx(
              'inline-flex items-center gap-1.5 rounded-md font-medium transition cursor-pointer',
              size === 'sm' ? 'px-2 py-1 text-[11px]' : 'px-3 py-1.5 text-xs',
              'disabled:cursor-not-allowed disabled:opacity-40',
              'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
              active ? 'bg-blue-600/90 text-slate-950' : 'text-slate-300 enabled:hover:bg-white/5',
            )}
          >
            {opt.icon && <opt.icon size={13} />}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

const NOTICE_TONES = {
  info: 'border-blue-700/40 bg-blue-950/30 text-blue-100',
  warn: 'border-amber-700/40 bg-amber-950/30 text-amber-100',
  error: 'border-red-800/40 bg-red-950/30 text-red-200',
};

export function Notice({ tone = 'info', icon: Icon, title, children, onDismiss, className = '' }) {
  return (
    <div role={tone === 'error' ? 'alert' : undefined} className={cx('flex items-start gap-2.5 rounded-lg border p-2.5 text-xs leading-relaxed', NOTICE_TONES[tone], className)}>
      {Icon && <Icon size={14} className="mt-0.5 shrink-0" />}
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        <div className="break-words">{children}</div>
      </div>
      {onDismiss && (
        <button type="button" onClick={onDismiss} aria-label="Dismiss" className="shrink-0 text-current/70 hover:text-current cursor-pointer">
          &times;
        </button>
      )}
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action, className = '' }) {
  return (
    <div className={cx('flex flex-col items-center justify-center gap-2 px-6 py-8 text-center', className)}>
      {Icon && (
        <div className="mb-1 flex h-11 w-11 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-blue-400">
          <Icon size={20} />
        </div>
      )}
      <p className="text-sm font-semibold text-slate-200">{title}</p>
      {children && <p className="max-w-xs text-xs leading-relaxed text-slate-400">{children}</p>}
      {action}
    </div>
  );
}

// An <img> that degrades gracefully: a file that is not really an image (or a broken URL) shows a
// placeholder instead of the browser's broken-image icon.
export function PreviewImage({ src, alt, className, fallback }) {
  // Remember WHICH src failed, so a new src gets a fresh try without resetting state in an effect.
  const [failedSrc, setFailedSrc] = useState(null);
  if (!src || failedSrc === src) {
    return fallback ?? (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-slate-500">
        <ImageIcon size={32} />
        <p className="text-xs">Preview unavailable</p>
      </div>
    );
  }
  return <img src={src} alt={alt} className={className} onError={() => setFailedSrc(src)} />;
}

// The model answers as "OBSERVATIONS: ... ASSESSMENT: ..."; give those sections a visible label
// instead of one run-on paragraph. Any other text is shown as plain paragraphs.
const SECTION = /^(OBSERVATIONS|ASSESSMENT):\s*/i;
export function FormattedAnswer({ text, className = '' }) {
  const lines = String(text ?? '')
    .replace(/\s+(ASSESSMENT:)/g, '\n$1')
    .split('\n')
    .filter((line) => line.trim());
  return (
    <div className={cx('space-y-1.5', className)}>
      {lines.map((line, i) => {
        const match = line.match(SECTION);
        return match ? (
          <p key={i}>
            <span className="mr-1.5 text-[10px] font-semibold uppercase tracking-wider text-blue-300">{match[1]}</span>
            {line.slice(match[0].length)}
          </p>
        ) : <p key={i}>{line}</p>;
      })}
    </div>
  );
}

// Label/value rows for the run summary
export function KeyValue({ rows, className = '' }) {
  return (
    <dl className={cx('grid grid-cols-[auto,1fr] gap-x-4 gap-y-1.5 text-xs', className)} style={{ gridTemplateColumns: 'max-content 1fr' }}>
      {rows.map(([key, value]) => (
        <React.Fragment key={key}>
          <dt className="text-slate-400">{key}</dt>
          <dd className="min-w-0 break-words text-slate-100">{value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}
