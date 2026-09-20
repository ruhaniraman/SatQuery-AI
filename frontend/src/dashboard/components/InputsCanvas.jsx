import React from 'react';
import { ImageIcon, Upload } from 'lucide-react';
import { Button, EmptyState, PreviewImage } from './ui';
import { evidenceUrl, isTiffFile } from '../utils/api';

const SENSOR = { optical: 'Optical', sar: 'SAR' };

// Browsers can't draw a GeoTIFF, so it gets a placeholder instead of a broken image
function GeoTiffPlaceholder({ file }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-1 text-blue-300/80">
      <ImageIcon size={44} className="mb-2" />
      <p className="font-semibold tracking-wide">GeoTIFF loaded</p>
      <p className="max-w-xs truncate text-xs text-slate-400">{file.name}</p>
    </div>
  );
}

// Shows one layer at a time (Image A, Image B, or the server-generated evidence), uncropped
export default function InputsCanvas({ ws, onAddImagery }) {
  const { activeLayer, slots, result } = ws;
  const evidence = evidenceUrl(result);

  let body = null;
  let caption = '';

  if (activeLayer === 'evidence' && evidence) {
    body = <PreviewImage src={evidence} alt="Analysis evidence" className="absolute inset-0 h-full w-full object-contain" />;
  } else {
    const slot = activeLayer === 'imageB' ? slots.b : slots.a;
    if (slot.file) {
      const label = `${activeLayer === 'imageB' ? 'Image B' : 'Image A'} · ${SENSOR[slot.modality]}`;
      caption = slot.file.name;
      body = isTiffFile(slot.file)
        ? <GeoTiffPlaceholder file={slot.file} />
        : <PreviewImage src={slot.preview} alt={label} className="absolute inset-0 h-full w-full object-contain" />;
    }
  }

  return (
    <div className="absolute inset-0">
      <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-10" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <defs>
          <pattern id="sq-grid" width="32" height="32" patternUnits="userSpaceOnUse">
            <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#94a3b8" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#sq-grid)" />
      </svg>

      {body ? (
        <>
          {body}
          {caption && (
            <span className="absolute left-3 top-3 z-10 max-w-[60%] truncate rounded-lg border border-white/10 bg-black/60 px-2.5 py-1.5 text-[11px] font-medium text-slate-200 backdrop-blur-md" title={caption}>
              {caption}
            </span>
          )}
        </>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <EmptyState
            icon={ImageIcon}
            title="No imagery loaded"
            action={<Button variant="subtle" size="sm" icon={Upload} className="mt-2" onClick={onAddImagery}>Add imagery</Button>}
          >
            Upload an image to view it here, or switch back to the live map.
          </EmptyState>
        </div>
      )}
    </div>
  );
}
