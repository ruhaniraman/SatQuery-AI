import React, { useRef, useState } from 'react';
import { ImageIcon, Loader2, RefreshCw, TriangleAlert, Upload, X } from 'lucide-react';
import { Button, Card, PreviewImage, Segmented, cx } from './ui';
import { formatBytes, isTiffFile } from '../utils/api';
import { describeCapture } from '../utils/views';

export const ACCEPT = 'image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff';
export const SENSORS = [
  { value: 'optical', label: 'Optical', title: 'Colour imagery' },
  { value: 'sar', label: 'SAR', title: 'Radar imagery: despeckled and described by brightness, not colour' },
];
const SENSOR_LABEL = { optical: 'Optical', sar: 'SAR' };

// The small square next to a chosen file: the image itself, or a spinner while a TIFF is being rendered
// by the backend, or an icon when there is nothing to show.
function Thumb({ slot }) {
  const tiff = isTiffFile(slot.file);
  let content = <ImageIcon size={20} />;
  if (slot.previewState === 'loading') content = <Loader2 size={18} className="animate-spin" aria-label="Rendering preview" />;
  else if (slot.previewState === 'failed') content = <TriangleAlert size={18} className="text-amber-400" />;
  else if (slot.preview) content = <PreviewImage src={slot.preview} alt="" className="h-full w-full object-cover" fallback={<ImageIcon size={20} />} />;
  return (
    <div
      className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-black/40 text-blue-400"
      title={tiff && slot.previewState === 'loading' ? 'Rendering a browser preview of this GeoTIFF…' : undefined}
    >
      {content}
    </div>
  );
}

// One image input. `sensor` is 'toggle' (user picks Optical/SAR), 'fixed' (the slot decides, shown as a
// badge) or null. Used by Imagery, Change detection and Fusion, so all three look and behave the same.
export default function ImageSlot({
  id, title, hint, slot, onFile, onRemove, sensor = null, onSensor, disabled = false, disabledNote, className = '',
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const pick = (fileList) => {
    const file = fileList && fileList[0];
    if (file && !disabled) onFile(id, file);
  };

  return (
    <Card className={cx('p-3.5 transition', disabled && 'opacity-60', className)}>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-100">{title}</p>
          {hint && <p className="text-[11px] leading-snug text-slate-400">{hint}</p>}
        </div>
        {sensor === 'toggle' && (
          <Segmented size="sm" label={`${title} sensor type`} value={slot.modality} onChange={(value) => onSensor(id, value)} options={SENSORS} />
        )}
        {sensor === 'fixed' && (
          <span className="shrink-0 rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[11px] font-medium text-slate-300">
            {SENSOR_LABEL[slot.modality]}
          </span>
        )}
      </div>

      <input
        ref={inputRef}
        data-slot={id}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          pick(Array.from(e.target.files || []));
          e.target.value = '';   // so choosing the same file again still fires onChange
        }}
      />

      {slot.file ? (
        <div
          className={cx('rounded-lg border bg-black/30 p-2.5 transition', dragging ? 'border-blue-500 bg-blue-500/10' : 'border-white/10')}
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(Array.from(e.dataTransfer.files || [])); }}
        >
          <div className="flex items-center gap-3">
            <Thumb slot={slot} />
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] text-slate-100" title={slot.file.name}>{slot.file.name}</p>
              <p className="text-[11px] text-slate-400">
                {formatBytes(slot.file.size)}{isTiffFile(slot.file) ? ' · GeoTIFF' : ''}
              </p>
              {slot.meta?.label && <p className="truncate text-[11px] text-blue-300/80" title={describeCapture(slot.meta)}>{describeCapture(slot.meta)}</p>}
            </div>
            <div className="flex shrink-0 gap-1">
              <Button variant="ghost" size="sm" icon={RefreshCw} onClick={() => inputRef.current?.click()} aria-label={`Replace ${title}`} title="Replace" />
              {onRemove && <Button variant="ghost" size="sm" icon={X} onClick={() => onRemove(id)} aria-label={`Remove ${title}`} title="Remove" />}
            </div>
          </div>
          {slot.previewState === 'failed' && (
            <p className="mt-2 text-[11px] leading-snug text-amber-400" role="status">
              Preview unavailable: {slot.previewError || 'the backend could not render this file'}. The file can still be analysed.
            </p>
          )}
        </div>
      ) : (
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(Array.from(e.dataTransfer.files || [])); }}
          className={cx(
            'flex w-full flex-col items-center gap-1.5 rounded-lg border border-dashed px-4 py-6 text-center transition',
            'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
            disabled ? 'cursor-not-allowed border-white/10' : 'cursor-pointer hover:border-blue-500/70 hover:bg-blue-500/5',
            dragging ? 'border-blue-500 bg-blue-500/10' : 'border-white/20 bg-black/20',
          )}
        >
          <Upload size={18} className="text-blue-400" />
          <span className="text-xs font-medium text-slate-200">{disabled ? disabledNote : 'Drop an image or click to browse'}</span>
          {!disabled && <span className="text-[11px] text-slate-500">JPEG, PNG, WebP or GeoTIFF</span>}
        </button>
      )}
    </Card>
  );
}
