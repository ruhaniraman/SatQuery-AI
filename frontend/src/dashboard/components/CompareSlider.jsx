import React, { useRef, useState } from 'react';
import { MoveHorizontal } from 'lucide-react';
import { PreviewImage } from './ui';

const clamp = (value) => Math.min(100, Math.max(0, value));

// Before / after by swiping: the Before image on the left of the handle, the After image on the right.
// Both are drawn in the same box, so a change shows up as the picture "flipping" as the handle passes it.
// Drag the handle (or press the arrow keys, Home and End when it is focused). It sits inside ZoomPan,
// so zooming works too; the handle is data-no-pan so dragging it does not pan the picture.
export default function CompareSlider({ before, after, beforeLabel = 'Before', afterLabel = 'After' }) {
  const box = useRef(null);
  const [position, setPosition] = useState(50);
  const [dragging, setDragging] = useState(false);

  const moveTo = (clientX) => {
    const rect = box.current.getBoundingClientRect();
    if (rect.width > 0) setPosition(clamp(((clientX - rect.left) / rect.width) * 100));
  };

  const onKeyDown = (e) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft') setPosition((p) => clamp(p - step));
    else if (e.key === 'ArrowRight') setPosition((p) => clamp(p + step));
    else if (e.key === 'Home') setPosition(0);
    else if (e.key === 'End') setPosition(100);
    else return;
    e.preventDefault();
  };

  return (
    <div ref={box} className="absolute inset-0">
      <PreviewImage src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
        <PreviewImage src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" />
      </div>

      <span className="pointer-events-none absolute left-3 top-12 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] font-semibold text-slate-100 backdrop-blur-md">{beforeLabel}</span>
      <span className="pointer-events-none absolute right-3 top-12 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] font-semibold text-slate-100 backdrop-blur-md">{afterLabel}</span>

      <div data-no-pan className="absolute inset-y-0 z-10 w-8 -translate-x-1/2 cursor-ew-resize touch-none"
        style={{ left: `${position}%` }}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setDragging(true); moveTo(e.clientX); }}
        onPointerMove={(e) => { if (dragging) moveTo(e.clientX); }}
        onPointerUp={(e) => { setDragging(false); e.currentTarget.releasePointerCapture?.(e.pointerId); }}
        onPointerCancel={() => setDragging(false)}
      >
        <div className="pointer-events-none absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]" />
        <button
          type="button"
          role="slider"
          aria-label="Before and after position"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(position)}
          onKeyDown={onKeyDown}
          className="absolute left-1/2 top-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border border-white/70 bg-blue-600 text-white shadow-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300"
        >
          <MoveHorizontal size={16} />
        </button>
      </div>
    </div>
  );
}
