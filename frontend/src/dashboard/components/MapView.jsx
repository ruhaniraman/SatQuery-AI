import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ImageOverlay, MapContainer, Pane, ScaleControl, TileLayer, ZoomControl, useMap, useMapEvents } from 'react-leaflet';
import { Tag } from 'lucide-react';
import { MAP_CONFIG, formatLatLng } from '../utils/mapConfig';
import { captureMapView } from '../utils/mapCapture';
import MapAsk from './MapAsk';
import { cx } from './ui';

// Leaflet measures its container once. When the tool panel collapses, the window resizes, or the
// map is shown again after being hidden, it has to be told to re-measure or tiles render offset.
function KeepSized({ active }) {
  const map = useMap();
  useEffect(() => {
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(map.getContainer());
    return () => observer.disconnect();
  }, [map]);
  useEffect(() => {
    if (active) map.invalidateSize();
  }, [active, map]);
  return null;
}

function CursorTracker({ onMove }) {
  useMapEvents({
    mousemove: (e) => onMove(e.latlng),
    mouseout: () => onMove(null),
  });
  return null;
}

// The live satellite map. It stays mounted while hidden so the user's position and zoom survive a
// trip to the inputs view. The ask bar on top of it captures what is on screen and analyses it.
export default function MapView({ active, ws, onOpenAssistant }) {
  const [labels, setLabels] = useState(true);
  const [overlay, setOverlay] = useState(null);      // { url, bounds, visible }: a scan drawn on the map
  const mapRef = useRef(null);
  const imageryRef = useRef(null);
  const readoutRef = useRef(null);
  const readoutBoxRef = useRef(null);

  // Written straight to the DOM: a state update on every mouse move would re-render the whole map
  const onMove = (latlng) => {
    if (readoutRef.current) readoutRef.current.textContent = latlng ? formatLatLng(latlng) : '';
    if (readoutBoxRef.current) readoutBoxRef.current.hidden = !latlng;
  };

  // Only the imagery layer is captured (not the place labels), exactly as currently framed
  const capture = useCallback(() => {
    if (!mapRef.current || !imageryRef.current) return Promise.reject(new Error('The map is not ready yet.'));
    return captureMapView(mapRef.current, imageryRef.current);
  }, []);

  return (
    <div className={cx('absolute inset-0 sq-map', !active && 'hidden')}>
      {/* isolate: keeps Leaflet's high z-index panes from painting over the dashboard's overlays */}
      <div className="absolute inset-0 isolate">
        <MapContainer
          ref={mapRef}
          center={MAP_CONFIG.center}
          zoom={MAP_CONFIG.zoom}
          minZoom={MAP_CONFIG.minZoom}
          maxZoom={MAP_CONFIG.maxZoom}
          zoomControl={false}
          worldCopyJump
          className="h-full w-full"
        >
          <TileLayer
            ref={imageryRef}
            url={MAP_CONFIG.imageryUrl}
            attribution={MAP_CONFIG.imageryAttribution}
            maxNativeZoom={MAP_CONFIG.maxZoom}
            crossOrigin={MAP_CONFIG.crossOrigin}
          />
          {overlay?.visible && <ImageOverlay key={overlay.url} url={overlay.url} bounds={overlay.bounds} opacity={0.92} />}
          {/* Own pane above the overlay pane (400), so place names stay crisp on top of a scan overlay */}
          <Pane name="labels" style={{ zIndex: 450, pointerEvents: 'none' }}>
            {labels && <TileLayer url={MAP_CONFIG.labelsUrl} attribution={MAP_CONFIG.labelsAttribution} maxNativeZoom={MAP_CONFIG.maxZoom} />}
          </Pane>
          <ZoomControl position="bottomright" />
          <ScaleControl position="bottomleft" imperial={false} />
          <KeepSized active={active} />
          <CursorTracker onMove={onMove} />
        </MapContainer>
      </div>

      <div className="pointer-events-none absolute left-3 top-3 z-10 flex items-center gap-2">
        <button
          type="button"
          aria-pressed={labels}
          onClick={() => setLabels((v) => !v)}
          className={cx(
            'pointer-events-auto inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[11px] font-medium backdrop-blur-md transition cursor-pointer',
            'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
            labels ? 'border-blue-500/50 bg-blue-950/60 text-blue-200' : 'border-white/15 bg-black/60 text-slate-300 hover:bg-black/70',
          )}
        >
          <Tag size={12} /> Place labels
        </button>
      </div>

      <div ref={readoutBoxRef} hidden className="pointer-events-none absolute right-3 top-3 z-10 rounded-lg border border-white/10 bg-black/60 px-2.5 py-1.5 font-mono text-[11px] text-slate-300 backdrop-blur-md">
        <span ref={readoutRef} />
      </div>

      {active && ws && MAP_CONFIG.canExportView && (
        <MapAsk
          ws={ws}
          capture={capture}
          overlay={overlay}
          onOverlay={(o) => setOverlay({ ...o, visible: true })}
          onClearOverlay={() => setOverlay(null)}
          onToggleOverlay={() => setOverlay((o) => (o ? { ...o, visible: !o.visible } : o))}
          onOpenAssistant={onOpenAssistant}
        />
      )}
    </div>
  );
}
