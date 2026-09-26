import { useEffect, useState } from 'react';
import { needsTunnelHeader, tunnelHeaders } from '../utils/api';

// Images served by the backend (evidence, report thumbnails). Normally the URL is used as-is. Behind an ngrok
// tunnel an <img> cannot send the header that skips ngrok's warning page, so the image is fetched with it and
// shown from a blob: URL instead. Each URL is fetched once per page load (report images never change).
const cache = new Map();

export function loadBackendImage(url) {
  if (!url || !needsTunnelHeader(url)) return Promise.resolve(url);
  if (!cache.has(url)) {
    const pending = fetch(url, { headers: tunnelHeaders(url) })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => URL.createObjectURL(blob));
    pending.catch(() => cache.delete(url));          // a failed load may be retried later
    cache.set(url, pending);
  }
  return cache.get(url);
}

// -> { src, loading, failed }: src is what an <img> can use (null while loading or after a failure).
export function useBackendImage(url) {
  const direct = !url || !needsTunnelHeader(url);
  const [loaded, setLoaded] = useState({ url: null, src: null, failed: false });
  useEffect(() => {
    if (direct) return undefined;
    let live = true;
    loadBackendImage(url).then(
      (src) => { if (live) setLoaded({ url, src, failed: false }); },
      () => { if (live) setLoaded({ url, src: null, failed: true }); },
    );
    return () => { live = false; };
  }, [url, direct]);
  if (direct) return { src: url || null, loading: false, failed: false };
  if (loaded.url !== url) return { src: null, loading: true, failed: false };
  return { src: loaded.src, loading: false, failed: loaded.failed };
}
