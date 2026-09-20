import React, { Suspense, lazy } from 'react';
import { Globe, ImageIcon, Loader2 } from 'lucide-react';
import { Segmented } from './ui';
import InputsCanvas from './InputsCanvas';

// The map (and Leaflet with it) is loaded on demand
const MapView = lazy(() => import('./MapView'));

const SENSOR = { optical: 'Optical', sar: 'SAR' };

export default function Viewer({ ws, onAddImagery, onOpenFeature }) {
  const { viewMode, setViewMode, activeLayer, setActiveLayer, slots, result } = ws;

  const layers = [
    { value: 'imageA', label: `Image A · ${SENSOR[slots.a.modality]}`, disabled: !slots.a.file },
    { value: 'imageB', label: `Image B · ${SENSOR[slots.b.modality]}`, disabled: !slots.b.file },
    { value: 'evidence', label: 'Evidence', disabled: !result?.visual_evidence_url },
  ];

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-white/10 bg-black/50 backdrop-blur-md" aria-label="Viewer">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <Segmented
          label="Viewer"
          value={viewMode}
          onChange={setViewMode}
          options={[
            { value: 'map', label: 'Live map', icon: Globe, title: 'Satellite basemap' },
            { value: 'inputs', label: 'Inputs', icon: ImageIcon, title: 'Your images and analysis evidence' },
          ]}
        />
        {viewMode === 'inputs' ? (
          <Segmented label="Layer" value={activeLayer} onChange={setActiveLayer} options={layers} size="sm" />
        ) : (
          <span className="hidden text-[11px] text-slate-500 sm:inline">Satellite basemap &middot; drag to pan, scroll to zoom</span>
        )}
      </div>

      <div className="relative min-h-0 flex-1">
        <Suspense fallback={<div className="absolute inset-0 flex items-center justify-center text-slate-500"><Loader2 className="animate-spin" size={20} /></div>}>
          <MapView active={viewMode === 'map'} ws={ws} onOpenAssistant={() => onOpenFeature?.('assistant')} />
        </Suspense>
        {viewMode === 'inputs' && (
          <InputsCanvas ws={ws} onAddImagery={onAddImagery} />
        )}
      </div>
    </section>
  );
}
