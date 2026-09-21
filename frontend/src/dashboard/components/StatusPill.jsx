import React from 'react';
import { cx } from './ui';

// Tailwind needs the class names written out in full.
const TONES = {
  ready: { box: 'border-blue-700/40 bg-blue-950/40 text-blue-300', dot: 'bg-blue-400' },
  busy: { box: 'border-blue-700/40 bg-blue-950/40 text-blue-300', dot: 'bg-blue-400 animate-pulse' },
  warn: { box: 'border-amber-700/40 bg-amber-950/40 text-amber-100', dot: 'bg-amber-400 animate-pulse' },
  error: { box: 'border-red-800/40 bg-red-950/40 text-red-200', dot: 'bg-red-400' },
  idle: { box: 'border-white/15 bg-white/5 text-slate-300', dot: 'bg-slate-400 animate-pulse' },
};

// What the header pill says, from the backend's health and whether an analysis is running.
export function describeStatus(health, executing) {
  switch (health.state) {
    case 'offline': return { tone: 'error', text: 'Backend offline' };
    case 'model-error': return { tone: 'error', text: 'Model unavailable' };
    case 'loading': return { tone: 'warn', text: 'Model loading' };
    case 'ready': return executing ? { tone: 'busy', text: 'Analysing' } : { tone: 'ready', text: 'Agent ready' };
    default: return executing ? { tone: 'busy', text: 'Analysing' } : { tone: 'idle', text: 'Checking' };
  }
}

// Clicking re-checks straight away, which is what you want right after starting the backend.
export default function StatusPill({ health, executing, onRefresh }) {
  const { tone, text } = describeStatus(health, executing);
  const styles = TONES[tone];
  return (
    <button
      type="button"
      onClick={onRefresh}
      title={`${health.message} Click to check again.`}
      className={cx('flex cursor-pointer items-center rounded-full border px-3 py-1 text-xs transition focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70', styles.box)}
    >
      <span className={cx('mr-2 h-2 w-2 rounded-full', styles.dot)} aria-hidden="true" />
      <span role="status">{text}</span>
    </button>
  );
}
