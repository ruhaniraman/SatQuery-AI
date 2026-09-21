import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, ChevronDown, CircleAlert, Eye, EyeOff, ImageDown, MessageSquare, SendHorizontal, X } from 'lucide-react';
import { Button, FormattedAnswer, Notice, cx } from './ui';
import ScanResult from './ScanResult';
import { mapLink } from '../utils/scanResult';
import { SCANS } from './panels/ScansPanel';
import { evidenceUrl } from '../utils/api';
import { captureFileName } from '../utils/views';

// Where a captured view can be saved. Fusion's two slots are fixed to one sensor type each, so only the
// matching one is offered for the imagery type on screen.
const SAVE_TARGETS = [
  { slot: 'a', label: 'Single image', feature: 'imagery', featureLabel: 'Imagery' },
  { slot: 'before', label: 'Change detection · Before', feature: 'change', featureLabel: 'Change detection' },
  { slot: 'after', label: 'Change detection · After', feature: 'change', featureLabel: 'Change detection' },
  { slot: 'optical', label: 'Fusion · Optical image', feature: 'fusion', featureLabel: 'Fusion', only: 'optical' },
  { slot: 'sar', label: 'Fusion · SAR image', feature: 'fusion', featureLabel: 'Fusion', only: 'sar' },
];

// The prompt bar on the live map. "Ask" and the scan chips first capture exactly what is on screen
// (via `capture`, supplied by MapView), then send that picture as the single image. The answer is shown
// on the map; scan results come back as an overlay pinned to the captured ground. "Save view as" keeps a
// capture in one of the features' image slots instead (for example the Before and After of a change
// detection). On the SAR layer the capture is tagged SAR.
export default function MapAsk({ ws, layer = 'optical', layerLabel = null, capture, overlay, onOverlay, onClearOverlay, onToggleOverlay, onOpenFeature }) {
  const [text, setText] = useState('');
  const [capturing, setCapturing] = useState(false);
  const [card, setCard] = useState(null);          // { title, answer, open? }
  const [captureError, setCaptureError] = useState(null);
  const [failed, setFailed] = useState(false);      // the last analysis failed; ws.error says why
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const sar = layer === 'sar';
  const modality = sar ? 'sar' : 'optical';      // the Past (historical) layer is optical imagery

  const busy = capturing || ws.isExecuting;
  const status = capturing ? 'Capturing view…' : ws.isExecuting ? 'Analysing…' : null;

  useEffect(() => {
    if (!menuOpen) return undefined;
    const close = (e) => { if (e.type === 'keydown' ? e.key === 'Escape' : !menuRef.current?.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', close);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', close); };
  }, [menuOpen]);

  const begin = () => {
    setCard(null);
    setCaptureError(null);
    setFailed(false);
    onClearOverlay();
  };

  const grab = async () => {
    setCapturing(true);
    try {
      return await capture();
    } catch (err) {
      setCaptureError(err.message);
      return null;
    } finally {
      setCapturing(false);
    }
  };

  const go = async ({ question, adapter = 'general', title }) => {
    if (busy) return;
    begin();
    const shot = await grab();
    if (!shot) return;
    const data = await ws.analyzeView(shot.file, { question, adapter, modality, bounds: shot.bounds, zoom: shot.zoom });
    if (!data) {
      setFailed(true);
      return;
    }
    setCard({ title: sar ? `${title} (SAR view)` : title, answer: data.answer, scan: data.scan || null, link: mapLink({ bounds: shot.bounds }) });
    if (adapter !== 'general' && data.visual_evidence_url) {
      onOverlay({ url: evidenceUrl(data), bounds: shot.bounds });
    }
  };

  const saveAs = async (target) => {
    setMenuOpen(false);
    if (busy) return;
    begin();
    const shot = await grab();
    if (!shot) return;
    const label = layerLabel || (sar ? 'SAR' : 'Optical');
    const file = new File([shot.file], captureFileName(label), { type: 'image/png' });
    ws.setImage(target.slot, file, { reveal: false, modality, meta: { label, bounds: shot.bounds, zoom: shot.zoom } });
    setCard({
      title: 'View saved',
      answer: `Saved this ${layer === 'past' ? label : sar ? 'SAR' : 'optical'} view as ${target.label}.`,
      open: { feature: target.feature, label: `Open ${target.featureLabel}` },
    });
  };

  const submit = (e) => {
    e.preventDefault();
    const question = text.trim();
    if (!question || busy) return;
    setText('');
    go({ question, title: question });
  };

  const error = captureError || (failed ? ws.errorFor('map') || ws.error?.message : null);
  const targets = SAVE_TARGETS.filter((t) => !t.only || t.only === modality);

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
                {card.scan
                  ? <ScanResult scan={card.scan} link={card.link} compact />
                  : <FormattedAnswer text={card.answer} className="text-[13px] leading-relaxed text-slate-100" />}
              </div>
              <div className="flex flex-wrap gap-2">
                {card.open
                  ? <Button variant="subtle" size="sm" icon={ArrowRight} onClick={() => onOpenFeature?.(card.open.feature)}>{card.open.label}</Button>
                  : <Button variant="subtle" size="sm" icon={MessageSquare} onClick={() => onOpenFeature?.('assistant')}>Continue in Assistant</Button>}
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
            placeholder={sar ? "Ask about this SAR view…" : "Ask about what's on the map…"}
            className="min-w-0 flex-1 rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50"
          />
          <Button type="submit" size="sm" icon={SendHorizontal} disabled={busy || !text.trim()} aria-label="Ask about this view" className="h-9 w-9 shrink-0 !p-0" />
        </form>
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[11px] text-slate-400">{status || (sar ? 'Feature scans are optical-only:' : 'Scan this view:')}</span>
          {!status && SCANS.map((scan) => (
            <Button
              key={scan.id}
              variant={scan.tone}
              size="sm"
              disabled={busy || sar}
              title={sar ? 'The feature scans were built for optical imagery' : undefined}
              onClick={() => go({ adapter: scan.id, title: `${scan.title} scan` })}
              aria-label={`Scan this view for ${scan.title}`}
            >
              {scan.title}
            </Button>
          ))}

          <div ref={menuRef} className="relative ml-auto">
            <Button variant="subtle" size="sm" icon={ImageDown} disabled={busy} onClick={() => setMenuOpen((open) => !open)} aria-haspopup="menu" aria-expanded={menuOpen}>
              Save view as <ChevronDown size={12} className={cx('transition', menuOpen && 'rotate-180')} />
            </Button>
            {menuOpen && (
              <div role="menu" aria-label="Save this view as" className="absolute bottom-full right-0 z-20 mb-1.5 w-56 overflow-hidden rounded-lg border border-white/15 bg-black/85 py-1 backdrop-blur-md">
                {targets.map((target) => (
                  <button key={target.slot} type="button" role="menuitem" onClick={() => saveAs(target)}
                    className="block w-full cursor-pointer px-3 py-2 text-left text-[12px] text-slate-200 transition hover:bg-white/10 focus:bg-white/10 focus:outline-none">
                    {target.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
