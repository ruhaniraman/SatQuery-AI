import React, { useEffect, useRef, useState } from 'react';
import { Loader2, MapPin, Navigation, Search, X } from 'lucide-react';
import { interpretSearch, searchPlaces, AUTOSEARCH_DEBOUNCE_MS, AUTOSEARCH_MIN_CHARS } from '../utils/geocode';
import { COORDINATE_HELP } from '../utils/coordinates';
import { cx } from './ui';

// One search box for the live map, at its top right. Type a place ("Chandigarh", "Jharia coalfield") or coordinates
// (decimal degrees, hemisphere letters, degrees-minutes-seconds: see utils/coordinates.js). Coordinates fly straight
// there; a place is looked up when submitted and, if several places match, offered as a short list. A bad entry
// says what is wrong instead of failing silently.
export default function CoordinateBox({ onGo, onClear, hasPin, footer = null, search = searchPlaces }) {
  const [text, setText] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState([]);
  const abortRef = useRef(null);
  const skipAutoRef = useRef(false);
  useEffect(() => () => abortRef.current?.abort(), []);

  const choose = (place) => {
    skipAutoRef.current = true;
    abortRef.current?.abort();
    setResults([]);
    setText(place.name || text);
    onGo({ lat: place.lat, lng: place.lng, label: place.name, bounds: place.bounds });
  };

  const runSearch = async (query, { silent = false } = {}) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    try {
      const found = await search(query, { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (found.length === 0) { if (!silent) setError(`No place found for “${query}”. Try a bigger area, a different spelling, or coordinates.`); }
      else if (found.length === 1 && !silent) choose(found[0]);
      else setResults(found); // a single match while still typing is shown, not auto-selected
    } catch (err) {
      if (err?.name !== 'AbortError' && !silent) setError(err.message);
    } finally {
      if (abortRef.current === controller) setBusy(false);
    }
  };

  // Search as you type, gently: debounced and only once there is enough to look up, so this stays a
  // light trickle of requests rather than one per keystroke (see utils/geocode.js). The input's own
  // onChange already clears any stale result list, so this effect only ever schedules the fetch.
  useEffect(() => {
    if (skipAutoRef.current) { skipAutoRef.current = false; return undefined; }
    const what = interpretSearch(text);
    if (what.kind !== 'place' || what.query.length < AUTOSEARCH_MIN_CHARS) return undefined;
    const timer = setTimeout(() => { runSearch(what.query, { silent: true }); }, AUTOSEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  const submit = async (e) => {
    e.preventDefault();
    setResults([]);
    const what = interpretSearch(text);
    if (what.kind === 'empty') { setError('Type a place name or coordinates. ' + COORDINATE_HELP); return; }
    if (what.kind === 'bad-coordinates') { setError(what.error); return; }
    setError(null);
    if (what.kind === 'coordinates') { onGo({ lat: what.lat, lng: what.lng }); return; }
    await runSearch(what.query);
  };

  const clear = () => {
    skipAutoRef.current = true;
    abortRef.current?.abort();
    setBusy(false);
    onClear();
    setText('');
    setError(null);
    setResults([]);
  };

  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 flex w-[min(22rem,calc(100%-6rem))] flex-col items-end gap-1.5">
      <form onSubmit={submit} className="pointer-events-auto flex w-full items-center gap-1 rounded-lg border border-white/15 bg-black/60 p-1 backdrop-blur-md" role="search">
        <Search size={14} className="ml-2 shrink-0 text-blue-400" aria-hidden="true" />
        <input
          type="text"
          value={text}
          onChange={(e) => { setText(e.target.value); if (error) setError(null); if (results.length) setResults([]); }}
          aria-label="Search a place or coordinates"
          aria-invalid={error ? 'true' : undefined}
          placeholder="Search a place or Lat, Lng"
          spellCheck={false}
          autoComplete="off"
          className={cx('min-w-0 flex-1 bg-transparent px-1.5 py-1 text-[12px] text-slate-100 placeholder:text-slate-500 focus:outline-none', error && 'text-red-300')}
        />
        {(hasPin || text) && (
          <button type="button" onClick={clear} aria-label="Clear the search and remove the marker" title="Clear"
            className="inline-flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
            <X size={13} />
          </button>
        )}
        <button type="submit" disabled={busy} aria-label="Search" title="Search"
          className="inline-flex h-7 shrink-0 cursor-pointer items-center gap-1 rounded-md bg-blue-600/90 px-2 text-[11px] font-semibold text-slate-950 transition enabled:hover:bg-blue-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 disabled:cursor-progress">
          {busy ? <Loader2 size={12} className="animate-spin" /> : <Navigation size={12} />} Go
        </button>
      </form>
      {results.length > 0 && (
        <ul aria-label="Matching places" className="pointer-events-auto w-full overflow-hidden rounded-lg border border-white/15 bg-black/70 backdrop-blur-md">
          {results.map((place, i) => (
            <li key={`${place.lat},${place.lng},${i}`}>
              <button type="button" onClick={() => choose(place)} title={place.fullName}
                className="flex w-full cursor-pointer items-start gap-2 px-2.5 py-2 text-left text-[12px] text-slate-200 transition hover:bg-white/10 focus:bg-white/10 focus:outline-none">
                <MapPin size={13} className="mt-0.5 shrink-0 text-blue-400" aria-hidden="true" />
                <span className="min-w-0 break-words">{place.name}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {error && (
        <p role="alert" className="pointer-events-auto w-full rounded-lg border border-red-800/40 bg-red-950/60 px-2.5 py-1.5 text-[11px] leading-snug text-red-200 backdrop-blur-md">
          {error}
        </p>
      )}
      {footer}
    </div>
  );
}
