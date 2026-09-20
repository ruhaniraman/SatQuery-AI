// Basemap settings. Defaults use Esri's World Imagery (no API key needed). It is Esri's regularly
// updated basemap, not a real-time feed. Check Esri's terms before production use, or point the
// app at a keyed provider without touching code:
//   VITE_MAP_TILE_URL=https://.../{z}/{y}/{x}   VITE_MAP_ATTRIBUTION="..."
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services';

// "Ask about this view" needs to read the map's pixels, which the browser only allows when the tile
// server sends CORS headers (Esri's does). For a provider without them set VITE_MAP_TILE_CORS=false:
// the map still displays, and the on-map ask bar is hidden instead of failing.
const CORS_ENABLED = import.meta.env?.VITE_MAP_TILE_CORS !== 'false';

export const MAP_CONFIG = {
  crossOrigin: CORS_ENABLED ? 'anonymous' : false,
  canExportView: CORS_ENABLED,
  imageryUrl: import.meta.env?.VITE_MAP_TILE_URL || `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
  imageryAttribution:
    import.meta.env?.VITE_MAP_ATTRIBUTION || 'Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community',
  labelsUrl: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
  labelsAttribution: 'Labels &copy; Esri',
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
