// Basemap settings. Defaults use Esri's World Imagery (no API key needed). It is Esri's regularly
// updated basemap, not a real-time feed. Check Esri's terms before production use, or point the
// app at a keyed provider without touching code:
//   VITE_MAP_TILE_URL=https://.../{z}/{y}/{x}   VITE_MAP_ATTRIBUTION="..."
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services';

// "Ask about this view" needs to read the map's pixels, which the browser only allows when the tile
// server sends CORS headers (Esri's does). For a provider without them set VITE_MAP_TILE_CORS=false:
// the map still displays, and the on-map ask bar is hidden instead of failing.
const CORS_ENABLED = import.meta.env?.VITE_MAP_TILE_CORS !== 'false';

// SAR basemap. No free, keyless SAR tile service exists, so this stays off until you point it at one
// you have an account for (for example a Sentinel Hub Sentinel-1 layer exposed as an XYZ/WMTS tile URL):
//   VITE_SAR_TILE_URL=https://.../{z}/{x}/{y}     VITE_SAR_ATTRIBUTION="..."
//   VITE_SAR_MAX_ZOOM=14 (Sentinel-1 is ~10 m per pixel, so tiles stop being sharper around zoom 14)
//   VITE_SAR_TILE_CORS=false  if the provider sends no CORS headers (the view can then be shown, not captured)
const SAR_CORS_ENABLED = import.meta.env?.VITE_SAR_TILE_CORS !== 'false';
const SAR_URL = import.meta.env?.VITE_SAR_TILE_URL || '';

export const MAP_CONFIG = {
  crossOrigin: CORS_ENABLED ? 'anonymous' : false,
  canExportView: CORS_ENABLED,
  imageryUrl: import.meta.env?.VITE_MAP_TILE_URL || `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
  imageryAttribution:
    import.meta.env?.VITE_MAP_ATTRIBUTION || 'Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community',
  labelsUrl: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
  labelsAttribution: 'Labels &copy; Esri',
  sarUrl: SAR_URL,
  hasSar: Boolean(SAR_URL),
  sarAttribution: import.meta.env?.VITE_SAR_ATTRIBUTION || 'SAR imagery',
  sarMaxNativeZoom: Number(import.meta.env?.VITE_SAR_MAX_ZOOM) || 14,
  sarCrossOrigin: SAR_CORS_ENABLED ? 'anonymous' : false,
  sarCanExportView: SAR_CORS_ENABLED,
  center: [20, 0],
  zoom: 3,
  minZoom: 2,
  maxZoom: 18,
};

export function formatLatLng(latlng) {
  if (!latlng) return '';
  const lat = `${Math.abs(latlng.lat).toFixed(4)}°${latlng.lat >= 0 ? 'N' : 'S'}`;
  // Leaflet lets longitude run past +/-180 when the world is panned; wrap it back
  const lngWrapped = ((((latlng.lng + 180) % 360) + 360) % 360) - 180;
  const lng = `${Math.abs(lngWrapped).toFixed(4)}°${lngWrapped >= 0 ? 'E' : 'W'}`;
  return `${lat}  ${lng}`;
}

export const SAR_SETUP_HINT = 'SAR tiles need a provider account. Set VITE_SAR_TILE_URL in frontend/.env (see CLAUDE.md), then restart the dev server.';

// Tile-layer settings for the optical or the SAR basemap, so MapView and the capture code read one place.
export function layerSettings(layer, config = MAP_CONFIG, release = null) {
  if (layer === 'past') {
    // Esri Wayback: the tile URL depends on the release (date) chosen on the timeline. Its tiles send CORS headers.
    return {
      url: release?.template || '', attribution: `Esri World Imagery Wayback${release ? ` ${release.date}` : ''}`,
      maxNativeZoom: config.maxZoom, crossOrigin: 'anonymous', canExport: true, modality: 'optical',
    };
  }
  if (layer === 'sar') {
    return {
      url: config.sarUrl, attribution: config.sarAttribution, maxNativeZoom: config.sarMaxNativeZoom,
      crossOrigin: config.sarCrossOrigin, canExport: config.sarCanExportView, modality: 'sar',
    };
  }
  return {
    url: config.imageryUrl, attribution: config.imageryAttribution, maxNativeZoom: config.maxZoom,
    crossOrigin: config.crossOrigin, canExport: config.canExportView, modality: 'optical',
  };
}
