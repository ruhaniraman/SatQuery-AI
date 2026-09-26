import { useEffect, useRef } from 'react';

// Plain helpers shared by the dashboard components (kept out of ui.jsx so that file only exports components).

export const cx = (...parts) => parts.filter(Boolean).join(' ');

// When a run finishes its result card is usually below the inputs, out of sight. Scroll it to the top of
// the panel once per new run (not on every render, so the user can scroll away).
export function useScrollToNew(key) {
  const ref = useRef(null);
  useEffect(() => {
    if (key) ref.current?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
  }, [key]);
  return ref;
}
