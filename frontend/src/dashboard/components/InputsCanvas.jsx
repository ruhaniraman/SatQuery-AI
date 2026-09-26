import { useRef, useState } from 'react';
import { ImageIcon, Loader2, RefreshCw, TriangleAlert, Upload } from 'lucide-react';
import { Button, EmptyState, PreviewImage } from './ui';
import { cx } from './uiHelpers';
import ZoomPan from './ZoomPan';
import CompareSlider from './CompareSlider';
import { ACCEPT } from '../constants';
import { evidenceUrl } from '../utils/api';
import { useBackendImage } from '../hooks/useBackendImage';
import { useImageAspect } from '../hooks/useImageAspect';

const Centered = ({ children }) => (
  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-6 text-center text-slate-400">{children}</div>
);

// What one image slot shows: the image, a spinner while a GeoTIFF is being rendered by the backend, or
// a plain message when it cannot be shown (the file can still be analysed).
function SlotBody({ slot }) {
  if (slot.previewState === 'loading') {
    return <Centered><Loader2 size={26} className="animate-spin text-blue-400" /><p className="text-xs">Rendering a preview of this GeoTIFF…</p></Centered>;
  }
  if (slot.previewState === 'failed' || !slot.preview) {
    return (
      <Centered>
        <TriangleAlert size={30} className="text-amber-400" />
        <p className="text-sm font-semibold text-slate-200">Preview unavailable</p>
        <p className="max-w-sm text-xs">{slot.previewError || 'This file cannot be drawn in the browser.'} It can still be analysed.</p>
      </Centered>
    );
  }
  return <PreviewImage src={slot.preview} alt={slot.file?.name || 'Image'} className="absolute inset-0 h-full w-full object-contain" />;
}

// Shows one layer at a time (an input image, the before/after swipe, or the server-generated evidence),
// uncropped, with zoom and pan. `layer` is already resolved by the Viewer. This is also where images are
// added when the viewer is open: an empty layer offers an upload button, files can be dropped anywhere on
// the canvas, and a shown image has a Replace button. `targets` are the slots the open feature owns.
export default function InputsCanvas({ ws, layer, evidenceRun, targets = [], onFile }) {
  const { slots, focus } = ws;
  const evidence = evidenceUrl(evidenceRun?.data);
  const aspect = useImageAspect(useBackendImage(evidence).src);
  const inputRef = useRef(null);
  const pending = useRef(null);
  const [dragging, setDragging] = useState(false);
  const canUpload = targets.length > 0 && typeof onFile === 'function';
  const shownSlot = layer && slots[layer] && targets.some((t) => t.id === layer) ? layer : null;

  const choose = (id) => {
    pending.current = id;
    inputRef.current?.click();
  };

  // Dropped files go to the slot on screen first, then to the feature's other empty slots, in order
  const drop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (!canUpload) return;
    const order = [];
    if (shownSlot) order.push(shownSlot);
    targets.forEach((t) => { if (!order.includes(t.id) && !slots[t.id]?.file) order.push(t.id); });
    Array.from(event.dataTransfer.files || []).slice(0, order.length).forEach((file, i) => onFile(order[i], file));
  };

  let body = null;
  let caption = '';

  if (layer === 'evidence' && evidence) {
    body = (
      <ZoomPan key={evidence} focus={focus && aspect ? { ...focus, aspect } : null}>
        <PreviewImage src={evidence} alt="Analysis evidence" className="absolute inset-0 h-full w-full object-contain" />
      </ZoomPan>
    );
    caption = evidenceRun.title;
  } else if (layer === 'compare' && evidence && slots.a.preview) {
    body = <ZoomPan key={`compare-${evidence}`}><CompareSlider before={slots.a.preview} after={evidence} beforeLabel="Original" afterLabel="Highlights" /></ZoomPan>;
    caption = 'Swipe to compare the original with the highlights';
  } else if (layer === 'swipe') {
    const { before, after } = slots;
    const ready = before.previewState === 'ready' && after.previewState === 'ready';
    body = ready
      ? <ZoomPan key="swipe"><CompareSlider before={before.preview} after={after.preview} /></ZoomPan>
      : <div className="absolute inset-0"><SlotBody slot={before.previewState === 'ready' ? after : before} /></div>;
    caption = 'Swipe to compare before and after';
  } else if (layer && slots[layer]?.file) {
    const slot = slots[layer];
    const zoomable = slot.previewState === 'ready' && slot.preview;
    body = zoomable ? <ZoomPan key={slot.preview}><SlotBody slot={slot} /></ZoomPan> : <div className="absolute inset-0"><SlotBody slot={slot} /></div>;
    caption = slot.file.name;
  }

  // Nothing to draw: offer the upload for the slot on screen, or one button per empty slot of the feature
  const emptyTargets = targets.filter((t) => !slots[t.id]?.file);
  const prompt = shownSlot ? targets.filter((t) => t.id === shownSlot) : emptyTargets.length ? emptyTargets : targets;

  return (
    <div
      className="absolute inset-0"
      onDragOver={(e) => { if (canUpload) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false); }}
      onDrop={drop}
    >
      <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-10" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <pattern id="sq-grid" width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#94a3b8" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#sq-grid)" />
      </svg>

      {canUpload && <input ref={inputRef} type="file" accept={ACCEPT} className="hidden" data-canvas-upload
        onChange={(e) => { const file = e.target.files?.[0]; if (file && pending.current) onFile(pending.current, file); e.target.value = ''; }} />}

      {body ? (
        <>
          {body}
          {caption && (
            <span className="pointer-events-none absolute left-3 top-3 z-10 max-w-[60%] truncate rounded-lg border border-white/10 bg-black/60 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 backdrop-blur-md" title={caption}>
              {caption}
            </span>
          )}
          {canUpload && shownSlot && (
            <Button variant="subtle" size="sm" icon={RefreshCw} onClick={() => choose(shownSlot)} className="absolute right-3 top-3 z-10 !bg-black/60 backdrop-blur-md">Replace</Button>
          )}
        </>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <EmptyState icon={canUpload ? Upload : ImageIcon} title={canUpload ? 'Add an image' : 'Nothing to show yet'}
            action={canUpload && (
              <div className="mt-2 flex flex-wrap justify-center gap-2">
                {prompt.map((t) => <Button key={t.id} size="sm" icon={Upload} onClick={() => choose(t.id)} aria-label={`Add ${t.label}`}>{`Add ${t.label}`}</Button>)}
              </div>
            )}
          >
            {canUpload ? 'Choose a file, or drop it anywhere on this view.' : 'Run an analysis to see its evidence here.'}
          </EmptyState>
        </div>
      )}

      {dragging && canUpload && (
        <div className={cx('pointer-events-none absolute inset-2 z-20 flex items-center justify-center rounded-xl border-2 border-dashed border-blue-500 bg-blue-500/10 text-sm font-semibold text-blue-200 backdrop-blur-[1px]')}>
          Drop to add
        </div>
      )}
    </div>
  );
}
