import React, { useEffect, useRef, useState } from 'react';
import { Maximize, ZoomIn, ZoomOut } from 'lucide-react';
import { IDENTITY, focusView, isZoomed, panBy, wheelFactor, zoomAt, zoomLabel } from '../utils/viewMath';
import { cx } from './ui';

// Wheel to zoom about the cursor, drag to pan, double-click to zoom in or back out, and +/-/0 keys. The
// arithmetic (and its limits) is in utils/viewMath.js. Give it a `key` that changes with its content so
// each image starts fitted. `focus` zooms to a region of the image (see focusView). Anything inside marked data-no-pan (a slider handle) keeps its own drags.
export default function ZoomPan({ children, className = '', focus = null }) {
  const box = useRef(null);
  const drag = useRef(null);
  const [view, setView] = useState(IDENTITY);
  const [dragging, setDragging] = useState(false);

  const measure = () => {
    const rect = box.current.getBoundingClientRect();
    return { width: rect.width, height: rect.height, left: rect.left, top: rect.top };
  };
  const zoomBy = (factor, px, py) => setView((v) => zoomAt(v, factor, px, py, measure()));
  const zoomCentre = (factor) => { const { width, height } = measure(); zoomBy(factor, width / 2, height / 2); };

  // Zoom to a region when asked: focus = { box: [ymin, xmin, ymax, xmax], aspect, token }. A new token zooms again.
  useEffect(() => {
    if (!focus?.box || !focus.aspect) return undefined;
    const frame = requestAnimationFrame(() => setView(focusView(focus.box, focus.aspect, measure())));   // after layout
    return () => cancelAnimationFrame(frame);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.token, focus?.aspect]);

  // React registers wheel listeners as passive, which forbids preventDefault; the page must not scroll
  // while the wheel zooms the image.
  useEffect(() => {
    const el = box.current;
    const onWheel = (e) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      setView((v) => zoomAt(v, wheelFactor(e.deltaY), e.clientX - rect.left, e.clientY - rect.top, rect));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerDown = (e) => {
    if (e.button !== 0 || e.target.closest('[data-no-pan]') || !isZoomed(view)) return;
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    setView((v) => panBy(v, dx, dy, measure()));
  };
  const endDrag = (e) => {
    if (!drag.current) return;
    drag.current = null;
    setDragging(false);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
  };
  const onDoubleClick = (e) => {
    if (e.target.closest('[data-no-pan]')) return;
    if (isZoomed(view)) setView(IDENTITY);
    else { const rect = measure(); zoomBy(2.5, e.clientX - rect.left, e.clientY - rect.top); }
  };
  const onKeyDown = (e) => {
    if (e.key === '+' || e.key === '=') zoomCentre(1.4);
    else if (e.key === '-' || e.key === '_') zoomCentre(1 / 1.4);
    else if (e.key === '0') setView(IDENTITY);
  };

  const control = 'inline-flex h-8 w-8 cursor-pointer items-center justify-center text-slate-200 transition enabled:hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-400/70 disabled:cursor-not-allowed disabled:opacity-40';

  return (
    <div
      ref={box}
      tabIndex={0}
      role="group"
      aria-label="Image viewer. Scroll to zoom, drag to pan, double-click to zoom, 0 to reset."
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={onDoubleClick}
      onKeyDown={onKeyDown}
      className={cx('absolute inset-0 touch-none select-none overflow-hidden outline-none', isZoomed(view) ? (dragging ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-zoom-in', className)}
    >
      <div className="absolute inset-0 will-change-transform" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`, transformOrigin: '0 0' }}>
        {children}
      </div>

      <div data-no-pan className="absolute bottom-3 right-3 z-10 flex items-center overflow-hidden rounded-lg border border-white/15 bg-black/60 backdrop-blur-md" onDoubleClick={(e) => e.stopPropagation()}>
        <button type="button" className={control} onClick={() => zoomCentre(1 / 1.4)} disabled={!isZoomed(view)} aria-label="Zoom out" title="Zoom out"><ZoomOut size={15} /></button>
        <span className="w-12 select-none text-center font-mono text-[11px] text-slate-300" aria-live="polite">{zoomLabel(view)}</span>
        <button type="button" className={control} onClick={() => zoomCentre(1.4)} aria-label="Zoom in" title="Zoom in"><ZoomIn size={15} /></button>
        <button type="button" className={cx(control, 'border-l border-white/10')} onClick={() => setView(IDENTITY)} disabled={!isZoomed(view)} aria-label="Fit to view" title="Fit to view (0)"><Maximize size={14} /></button>
      </div>
    </div>
  );
}
