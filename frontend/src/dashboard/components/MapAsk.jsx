import React, { useState } from 'react';
import { CircleAlert, Eye, EyeOff, MessageSquare, SendHorizontal, X } from 'lucide-react';
import { Button, FormattedAnswer, Notice } from './ui';
import { SCANS } from './panels/ScansPanel';
import { evidenceUrl } from '../utils/api';

// The prompt bar on the live map. "Ask" and the scan chips first capture exactly what is on screen
// (via `capture`, supplied by MapView), then send that picture as Image A. The answer is shown on the
// map; scan results come back as an overlay pinned to the captured ground.
export default function MapAsk({ ws, capture, overlay, onOverlay, onClearOverlay, onToggleOverlay, onOpenAssistant }) {
  const [text, setText] = useState('');
  const [capturing, setCapturing] = useState(false);
  const [card, setCard] = useState(null);          // { title, answer }
  const [captureError, setCaptureError] = useState(null);
  const [failed, setFailed] = useState(false);      // the last analysis failed; ws.error says why

  const busy = capturing || ws.isExecuting;
  const status = capturing ? 'Capturing view…' : ws.isExecuting ? 'Analysing…' : null;

  const go = async ({ question, adapter = 'general', title }) => {
    if (busy) return;
    setCard(null);
    setCaptureError(null);
    setFailed(false);
    onClearOverlay();

    setCapturing(true);
    let shot;
    try {
      shot = await capture();
    } catch (err) {
      setCaptureError(err.message);
      return;
    } finally {
      setCapturing(false);
    }

    const data = await ws.analyzeView(shot.file, { question, adapter });
    if (!data) {
      setFailed(true);
      return;
    }
    setCard({ title, answer: data.answer });
    if (adapter !== 'general' && data.visual_evidence_url) {
      onOverlay({ url: evidenceUrl(data), bounds: shot.bounds });
    }
  };

  const submit = (e) => {
    e.preventDefault();
    const question = text.trim();
    if (!question || busy) return;
    setText('');
    go({ question, title: question });
  };

  const error = captureError || (failed ? ws.error : null);

  return (
    <div className="pointer-events-none absolute inset-x-3 bottom-9 z-10 flex flex-col items-center gap-2">
      {(card || error) && (
        <div className="pointer-events-auto w-full max-w-xl space-y-2 rounded-xl border border-white/15 bg-black/75 p-3.5 backdrop-blur-md">
          {error ? (
            <Notice tone="error" icon={CircleAlert} title="Couldn't analyse this view" onDismiss={() => { setCaptureError(null); setFailed(false); }}>
              {error}
            </Notice>
          ) : (
            <>
              <div className="flex items-start justify-between gap-3">
                <p className="min-w-0 truncate text-[11px] font-semibold uppercase tracking-wider text-slate-400" title={card.title}>{card.title}</p>
                <button type="button" onClick={() => setCard(null)} aria-label="Close answer" className="shrink-0 cursor-pointer rounded p-0.5 text-slate-400 hover:text-slate-100">
                  <X size={14} />
                </button>
              </div>
              <div className="max-h-48 overflow-y-auto pr-1 sq-scroll">
                <FormattedAnswer text={card.answer} className="text-[13px] leading-relaxed text-slate-100" />
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="subtle" size="sm" icon={MessageSquare} onClick={onOpenAssistant}>Continue in Assistant</Button>
                {overlay && (
                  <Button variant="subtle" size="sm" icon={overlay.visible ? EyeOff : Eye} onClick={onToggleOverlay}>
                    {overlay.visible ? 'Hide overlay' : 'Show overlay'}
                  </Button>
                )}
              </div>
            </>
          )}
        </div>
      )}

      <div className="pointer-events-auto w-full max-w-xl rounded-xl border border-white/15 bg-black/70 p-2 backdrop-blur-md">
        <form onSubmit={submit} className="flex items-center gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={busy}
            aria-label="Ask about this map view"
            placeholder="Ask about what's on the map…"
            className="min-w-0 flex-1 rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
          <Button type="submit" size="sm" icon={SendHorizontal} disabled={busy || !text.trim()} aria-label="Ask about this view" className="h-9 w-9 shrink-0 !p-0" />
        </form>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[11px] text-slate-400">{status || 'Scan this view:'}</span>
          {!status && SCANS.map((scan) => (
            <Button
              key={scan.id}
              variant={scan.tone}
              size="sm"
              disabled={busy}
              onClick={() => go({ adapter: scan.id, title: `${scan.title} scan` })}
              aria-label={`Scan this view for ${scan.title}`}
            >
              {scan.title}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
