import React, { useRef, useState } from 'react';
import { ImageIcon, Layers, RefreshCw, Upload, X } from 'lucide-react';
import { Button, Card, Label, PreviewImage, Segmented, cx } from '../ui';
import { analysisMode, formatBytes, isTiffFile } from '../../utils/api';

const MODALITIES = [
  { value: 'optical', label: 'Optical' },
  { value: 'sar', label: 'SAR' },
];
const ACCEPT = 'image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff';

function ImageSlot({ which, title, hint, slot, onFile, onModality, onRemove, disabled, disabledNote }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const pick = (fileList) => {
    const file = fileList && fileList[0];
    if (file && !disabled) onFile(which, file);
  };

  return (
    <Card className={cx('p-3.5', disabled && 'opacity-60')}>
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-100">{title}</p>
          <p className="text-[11px] leading-snug text-slate-400">{hint}</p>
        </div>
        <Segmented
          size="sm"
          label={`${title} sensor type`}
          value={slot.modality}
          onChange={(value) => onModality(which, value)}
          options={MODALITIES}
        />
      </div>

      <input
        ref={inputRef}
        data-slot={which}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          pick(Array.from(e.target.files || []));
          e.target.value = '';   // so choosing the same file again still fires onChange
        }}
      />

      {slot.file ? (
        <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-black/30 p-2.5">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-md border border-white/10 bg-black/40 text-blue-400">
            {slot.preview && !isTiffFile(slot.file)
              ? <PreviewImage src={slot.preview} alt="" className="h-full w-full object-cover" fallback={<ImageIcon size={20} />} />
              : <ImageIcon size={20} />}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] text-slate-100" title={slot.file.name}>{slot.file.name}</p>
            <p className="text-[11px] text-slate-400">
              {formatBytes(slot.file.size)}{isTiffFile(slot.file) ? ' · GeoTIFF' : ''}
            </p>
          </div>
          <div className="flex shrink-0 gap-1">
            <Button variant="ghost" size="sm" icon={RefreshCw} onClick={() => inputRef.current?.click()} aria-label={`Replace ${title}`} title="Replace" />
            {onRemove && <Button variant="ghost" size="sm" icon={X} onClick={onRemove} aria-label={`Remove ${title}`} title="Remove" />}
          </div>
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

export function ModeCard({ mode }) {
  const active = mode.id !== 'none';
  return (
    <Card className={cx('flex gap-3 p-3.5', active && 'border-blue-700/40 bg-blue-950/20')}>
      <div className={cx('flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', active ? 'bg-blue-500/20 text-blue-300' : 'bg-white/5 text-slate-500')}>
        <Layers size={16} />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Analysis mode</p>
        <p className="text-sm font-semibold text-slate-100">{mode.label}</p>
        <p className="mt-0.5 text-xs leading-relaxed text-slate-400">{mode.hint}</p>
      </div>
    </Card>
  );
}

export default function ImageryPanel({ ws }) {
  const { slots, setImage, setModality, removeImageB, mode } = ws;
  return (
    <div className="space-y-3">
      <Label icon={Upload}>Inputs</Label>
      <ImageSlot
        which="a" title="Image A" hint="Primary image (the BEFORE image when comparing)"
        slot={slots.a} onFile={setImage} onModality={setModality}
      />
      <ImageSlot
        which="b" title="Image B" hint="Optional second image for change detection or fusion"
        slot={slots.b} onFile={setImage} onModality={setModality} onRemove={slots.b.file ? removeImageB : undefined}
        disabled={!slots.a.file} disabledNote="Add Image A first"
      />
      <ModeCard mode={mode} />
    </div>
  );
}
