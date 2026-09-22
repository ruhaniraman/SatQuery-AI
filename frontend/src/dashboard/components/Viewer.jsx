import React, { Suspense, lazy, useState } from 'react';
import { Globe, ImageIcon, Loader2 } from 'lucide-react';
import { Segmented } from './ui';
import InputsCanvas from './InputsCanvas';
import { effectiveLayer } from '../features';

// The map (and Leaflet with it) is loaded on demand
const MapView = lazy(() => import('./MapView'));

// The viewer follows the open feature: Change detection offers Before / After / Swipe / Changes, Fusion
// offers Optical / SAR / Composite, the single-image tools offer the image and the scan evidence.
export default function Viewer({ ws, feature, onOpenFeature }) {
  const { viewMode, setViewMode, activeLayer, setActiveLayer } = ws;
  // The map draws its imagery-type switch and the date timeline into these two header slots (portals)
  const [timelineSlot, setTimelineSlot] = useState(null);
  const [layerSlot, setLayerSlot] = useState(null);
  const evidenceRun = feature.evidenceRun(ws);
  const layers = feature.layers(ws, evidenceRun);
  const layer = effectiveLayer(layers, activeLayer);
  const targets = feature.uploadTargets(ws);

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-xl border border-white/10 bg-black/30 backdrop-blur-md" aria-label="Viewer">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2">
        <Segmented
          label="Viewer"
          value={viewMode}
          onChange={setViewMode}
          options={[
            { value: 'map', label: 'Live map', icon: Globe, title: 'Satellite basemap' },
            { value: 'inputs', label: 'Inputs', icon: ImageIcon, title: `Your images and the evidence for ${feature.title}` },
          ]}
        />
        {/* Between the two switches: the date timeline (only while the map shows past imagery) */}
        <div ref={setTimelineSlot} className="min-w-0 flex-1 basis-[22rem]" />
        {viewMode === 'inputs' ? <Segmented label="Layer" value={layer} onChange={setActiveLayer} options={layers} size="sm" /> : <div ref={setLayerSlot} />}
      </div>

      <div className="relative min-h-0 flex-1">
        <Suspense fallback={<div className="absolute inset-0 flex items-center justify-center text-slate-500"><Loader2 className="animate-spin" size={20} /></div>}>
          <MapView active={viewMode === 'map'} ws={ws} onOpenFeature={onOpenFeature} layerSlot={layerSlot} timelineSlot={timelineSlot} />
        </Suspense>
        {viewMode === 'inputs' && (
          <InputsCanvas ws={ws} layer={layer} evidenceRun={evidenceRun} targets={targets} onFile={ws.setImage} />
        )}
      </div>
    </section>
  );
}
