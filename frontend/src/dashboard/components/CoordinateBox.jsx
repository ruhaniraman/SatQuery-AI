import React, { useState } from 'react';
import { Crosshair, Navigation, X } from 'lucide-react';
import { parseCoordinates } from '../utils/coordinates';
import { cx } from './ui';

// "Go to coordinates" for the live map, at its top right. Accepts decimal degrees, hemisphere letters and
// degrees-minutes-seconds (see utils/coordinates.js); a bad entry says what is wrong instead of failing silently.
export default function CoordinateBox({ onGo, onClear, hasPin, footer = null }) {
  const [text, setText] = useState('');
  const [error, setError] = useState(null);

  const submit = (e) => {
    e.preventDefault();
    const parsed = parseCoordinates(text);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    setError(null);
    onGo({ lat: parsed.lat, lng: parsed.lng });
  };

  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 flex w-[min(20rem,calc(100%-6rem))] flex-col items-end gap-1.5">
      <form onSubmit={submit} className="pointer-events-auto flex w-full items-center gap-1 rounded-lg border border-white/15 bg-black/60 p-1 backdrop-blur-md" role="search">
        <Crosshair size={14} className="ml-2 shrink-0 text-blue-400" aria-hidden="true" />
        <input
          type="text"
          value={text}
          onChange={(e) => { setText(e.target.value); if (error) setError(null); }}
          aria-label="Go to coordinates"
          aria-invalid={error ? 'true' : undefined}
          placeholder="Lat, Lng  e.g. 23.75, 86.42"
          spellCheck={false}
          autoComplete="off"
          className={cx('min-w-0 flex-1 bg-transparent px-1.5 py-1 text-[12px] text-slate-100 placeholder:text-slate-500 focus:outline-none', error && 'text-red-300')}
        />
        {hasPin && (
          <button type="button" onClick={() => { onClear(); setText(''); setError(null); }} aria-label="Remove the marker" title="Remove the marker"
            className="inline-flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
            <X size={13} />
          </button>
        )}
        <button type="submit" aria-label="Go to these coordinates" title="Go"
          className="inline-flex h-7 shrink-0 cursor-pointer items-center gap-1 rounded-md bg-blue-600/90 px-2 text-[11px] font-semibold text-slate-950 transition hover:bg-blue-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300">
          <Navigation size={12} /> Go
        </button>
      </form>
      {error && (
        <p role="alert" className="pointer-events-auto w-full rounded-lg border border-red-800/40 bg-red-950/60 px-2.5 py-1.5 text-[11px] leading-snug text-red-200 backdrop-blur-md">
          {error}
        </p>
      )}
      {footer}
    </div>
  );
}
