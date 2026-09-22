import { createPortal } from 'react-dom';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CircleMarker, ImageOverlay, MapContainer, Pane, ScaleControl, TileLayer, Tooltip, ZoomControl, useMap, useMapEvents } from 'react-leaflet';
import { Settings2, Tag } from 'lucide-react';
import { MAP_CONFIG, formatLatLng, layerSettings } from '../utils/mapConfig';
import { captureMapView } from '../utils/mapCapture';
import { clearSarSource, loadSarSource, resolveSarConfig, saveSarSource } from '../utils/sarSource';
import { loadWaybackReleases } from '../utils/wayback';
import { captureFileName } from '../utils/views';
import MapAsk from './MapAsk';
import CoordinateBox from './CoordinateBox';
import Timeline from './Timeline';
import SarDialog from './SarDialog';
import { Segmented, cx } from './ui';

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

// Flies to a searched place or typed coordinate. A place with a bounding box (a city, a region) is framed by it;
// a bare point zooms in to at least street-block level, but never zooms OUT from where the user already is.
function FlyToPin({ pin }) {
  const map = useMap();
  useEffect(() => {
    if (!pin) return;
    if (pin.bounds) map.flyToBounds(pin.bounds, { duration: 1.2, maxZoom: 16 });
    else map.flyTo([pin.lat, pin.lng], Math.max(map.getZoom(), 13), { duration: 1.2 });
  }, [map, pin]);
  return null;
}

// How long the date slider must rest before the map loads that date (dragging across 15 years would
// otherwise reload every tile at every step)
const DATE_SETTLE_MS = 350;

// The live satellite map. It stays mounted while hidden so the user's position and zoom survive a
// trip to the inputs view. The ask bar on top of it captures what is on screen and analyses it.
export default function MapView({ active, ws, onOpenFeature, layerSlot = null, timelineSlot = null }) {
  const [labels, setLabels] = useState(true);
  const [layer, setLayer] = useState('optical');       // 'optical' | 'sar' | 'past'
  const [pin, setPin] = useState(null);                // a typed coordinate, { lat, lng }
  const [overlay, setOverlay] = useState(null);        // { url, bounds, visible }: a scan drawn on the map

  // SAR: the provider connected from a personal account (kept in this browser) overrides the environment
  const [sarSource, setSarSource] = useState(() => loadSarSource());
  const [sarDialog, setSarDialog] = useState(false);
  const config = useMemo(() => resolveSarConfig(MAP_CONFIG, sarSource), [sarSource]);

  // Past imagery (Esri Wayback): the list loads the first time it is asked for
  const [past, setPast] = useState({ status: 'idle', releases: [], error: null });
  const [dateIndex, setDateIndex] = useState(0);       // where the slider is
  const [appliedIndex, setAppliedIndex] = useState(0); // the date the map shows (follows the slider after it settles)
  const [capturing, setCapturing] = useState(false);
  const [captureNote, setCaptureNote] = useState(null);
  const settleTimer = useRef(null);

  const mapRef = useRef(null);
  const imageryRef = useRef(null);
  const readoutRef = useRef(null);
  const readoutBoxRef = useRef(null);
  useEffect(() => () => clearTimeout(settleTimer.current), []);

  const release = past.status === 'ready' ? past.releases[appliedIndex] : null;
  const settings = layerSettings(layer, config, release);
  const layerLabel = layer === 'past' && release ? `Wayback ${release.date}` : null;

  const loadPast = useCallback(async () => {
    setPast({ status: 'loading', releases: [], error: null });
    try {
      const releases = await loadWaybackReleases();
      setPast({ status: 'ready', releases, error: null });
      setDateIndex(releases.length - 1);               // start at the newest, then look back in time
      setAppliedIndex(releases.length - 1);
    } catch (err) {
      setPast({ status: 'error', releases: [], error: err.message });
    }
  }, []);

  const chooseLayer = (value) => {
    if (value === 'sar' && !config.hasSar) { setSarDialog(true); return; }   // nothing connected yet: offer to connect
    setLayer(value);
    if (value === 'past' && past.status === 'idle') loadPast();
  };

  const chooseDate = (index) => {
    setDateIndex(index);
    clearTimeout(settleTimer.current);
    settleTimer.current = setTimeout(() => setAppliedIndex(index), DATE_SETTLE_MS);
  };

  const connectSar = (source) => {
    saveSarSource(source);
    setSarSource(source);
    setSarDialog(false);
    setLayer('sar');
  };
  const disconnectSar = () => {
    clearSarSource();
    setSarSource(null);
    setSarDialog(false);
    if (layer === 'sar' && !MAP_CONFIG.hasSar) setLayer('optical');
  };

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

  // "Use as Before / After": capture the view at the chosen date into Change detection, remembering the date
  // and the area so a mismatch between the two captures can be flagged
  const useAs = async (slot) => {
    if (!release || capturing) return;
    setCapturing(true);
    setCaptureNote(null);
    try {
      const shot = await capture();
      const label = `Wayback ${release.date}`;
      const file = new File([shot.file], captureFileName(label), { type: 'image/png' });
      ws.setImage(slot, file, { reveal: false, modality: 'optical', meta: { label, date: release.date, bounds: shot.bounds, zoom: shot.zoom } });
    } catch (err) {
      setCaptureNote(err.message);
    } finally {
      setCapturing(false);
    }
  };

  const layerOptions = [
    { value: 'optical', label: 'Optical', title: 'Optical satellite imagery' },
    { value: 'sar', label: 'SAR', title: config.hasSar ? 'Radar (SAR) imagery' : 'Connect a SAR imagery provider' },
    { value: 'past', label: 'Past', title: 'Historical imagery from Esri Wayback: pick a date' },
  ];
  const tileKey = `${layer}-${release?.id ?? ''}`;

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
          {settings.url && (
            <TileLayer
              key={tileKey}
              ref={imageryRef}
              url={settings.url}
              attribution={settings.attribution}
              maxNativeZoom={settings.maxNativeZoom}
              crossOrigin={settings.crossOrigin}
            />
          )}
          {overlay?.visible && <ImageOverlay key={overlay.url} url={overlay.url} bounds={overlay.bounds} opacity={0.92} />}
          {/* Own pane above the overlay pane (400), so place names stay crisp on top of a scan overlay */}
          <Pane name="labels" style={{ zIndex: 450, pointerEvents: 'none' }}>
            {labels && <TileLayer url={MAP_CONFIG.labelsUrl} attribution={MAP_CONFIG.labelsAttribution} maxNativeZoom={MAP_CONFIG.maxZoom} />}
          </Pane>
          {pin && (
            <CircleMarker center={[pin.lat, pin.lng]} radius={9} pathOptions={{ color: '#3b82f6', weight: 3, fillColor: '#60a5fa', fillOpacity: 0.3 }}>
              <Tooltip permanent direction="top" offset={[0, -8]}>{pin.label ? `${pin.label} · ${formatLatLng(pin)}` : formatLatLng(pin)}</Tooltip>
            </CircleMarker>
          )}
          <ZoomControl position="bottomright" />
          <ScaleControl position="bottomleft" imperial={false} />
          <KeepSized active={active} />
          <CursorTracker onMove={onMove} />
          <FlyToPin pin={pin} />
        </MapContainer>
      </div>

      <div className="pointer-events-none absolute left-3 top-3 z-10 flex flex-wrap items-center gap-2">
        {sarSource || config.hasSar ? (
          <button type="button" onClick={() => setSarDialog(true)} aria-label="SAR provider settings" title="SAR provider settings"
            className="pointer-events-auto inline-flex h-7 w-7 cursor-pointer items-center justify-center rounded-lg border border-white/15 bg-black/60 text-slate-300 backdrop-blur-md transition hover:bg-black/70 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
            <Settings2 size={13} />
          </button>
        ) : null}
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

      {active && layerSlot && createPortal(
        <Segmented label="Imagery type" size="sm" value={layer} onChange={chooseLayer} options={layerOptions} />,
        layerSlot,
      )}

      {sarDialog && <SarDialog source={sarSource} onSave={connectSar} onDisconnect={disconnectSar} onClose={() => setSarDialog(false)} />}

      {active && timelineSlot && layer === 'past' && !sarDialog && createPortal(
        <Timeline
          status={past.status} error={past.error} releases={past.releases} index={dateIndex}
          ready={Boolean(release) && appliedIndex === dateIndex} busy={capturing} note={captureNote}
          before={ws.slots.before} after={ws.slots.after}
          onIndex={chooseDate} onRetry={loadPast} onUse={useAs} onOpenChange={() => onOpenFeature?.('change')}
        />,
        timelineSlot,
      )}

      {/* The pointer's coordinates live under the coordinate box, top right: the bottom of the map belongs to the
          ask bar and the zoom buttons, and a readout down there used to collide with them. */}
      <CoordinateBox
        onGo={(point) => setPin({ ...point })}
        onClear={() => setPin(null)}
        hasPin={Boolean(pin)}
        footer={(
          <div ref={readoutBoxRef} hidden className="pointer-events-none rounded-lg border border-white/10 bg-black/60 px-2.5 py-1 font-mono text-[11px] text-slate-300 backdrop-blur-md">
            <span className="mr-1.5 font-sans text-slate-500">Pointer</span><span ref={readoutRef} />
          </div>
        )}
      />

      {active && ws && settings.canExport && (
        <MapAsk
          ws={ws}
          layer={layer}
          layerLabel={layerLabel}
          capture={capture}
          overlay={overlay}
          onOverlay={(o) => setOverlay({ ...o, visible: true })}
          onClearOverlay={() => setOverlay(null)}
          onToggleOverlay={() => setOverlay((o) => (o ? { ...o, visible: !o.visible } : o))}
          onOpenFeature={onOpenFeature}
        />
      )}
    </div>
  );
}
