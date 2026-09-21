import { useCallback, useEffect, useState } from 'react';
import { checkHealth } from '../utils/api';

// Polls the backend's /health so the header can say whether analyses will work: ready, model still
// loading, model failed to load, or backend not running. Polls faster while something is wrong (so the
// pill turns green soon after the model finishes loading) and pauses while the tab is hidden.
export function useBackendStatus({ okMs = 15000, retryMs = 4000 } = {}) {
  const [status, setStatus] = useState({ state: 'checking', message: 'Checking the backend…' });

  const refresh = useCallback(async () => {
    const next = await checkHealth();
    setStatus((prev) => (prev.state === next.state && prev.message === next.message ? prev : next));
    return next;
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const loop = async () => {
      const next = document.hidden ? { state: 'ready' } : await refresh();
      if (!cancelled) timer = setTimeout(loop, next.state === 'ready' ? okMs : retryMs);
    };
    loop();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [refresh, okMs, retryMs]);

  return { ...status, refresh };
}
