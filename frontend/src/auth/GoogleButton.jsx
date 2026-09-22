import React, { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';

const SCRIPT_SRC = 'https://accounts.google.com/gsi/client';
let scriptPromise = null;

// Google's sign-in library, loaded once and only on the pages that need it
function loadGoogleScript() {
  if (globalThis.google?.accounts?.id) return Promise.resolve(globalThis.google);
  if (!scriptPromise) {
    scriptPromise = new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.src = SCRIPT_SRC;
      tag.async = true;
      tag.onload = () => resolve(globalThis.google);
      tag.onerror = () => { scriptPromise = null; tag.remove(); reject(new Error('Could not load Google sign-in. Check your connection or ad blocker.')); };
      document.head.appendChild(tag);
    });
  }
  return scriptPromise;
}

// "Continue with Google". Google draws the button and hands back an ID token, which the backend verifies.
// With no client id configured on the server the button is replaced by a note saying how to set it up.
export default function GoogleButton({ clientId, onCredential, onError, disabled = false }) {
  const holder = useRef(null);
  const [loaded, setState] = useState('loading');
  const state = clientId ? loaded : 'unconfigured';
  const callbacks = useRef({ onCredential, onError });
  useEffect(() => { callbacks.current = { onCredential, onError }; });

  useEffect(() => {
    if (!clientId) return undefined;
    let cancelled = false;
    loadGoogleScript().then((google) => {
      if (cancelled || !holder.current) return;
      google.accounts.id.initialize({
        client_id: clientId,
        callback: (response) => callbacks.current.onCredential(response.credential),
      });
      holder.current.replaceChildren();
      google.accounts.id.renderButton(holder.current, { type: 'standard', theme: 'outline', size: 'large', text: 'continue_with', shape: 'rectangular', width: 320 });
      setState('ready');
    }).catch((err) => {
      if (cancelled) return;
      setState('failed');
      callbacks.current.onError?.(err.message);
    });
    return () => { cancelled = true; };
  }, [clientId]);

  if (state === 'unconfigured') {
    return (
      <p className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-[12px] leading-snug text-slate-400">
        Google sign-in is off: set <code className="text-slate-200">GOOGLE_CLIENT_ID</code> for the backend (a Google Cloud OAuth “Web” client id, with this site's address as an authorised JavaScript origin).
      </p>
    );
  }
  return (
    <div className={disabled ? 'pointer-events-none opacity-50' : undefined} aria-busy={state === 'loading'}>
      {state === 'loading' && <div className="flex h-10 items-center justify-center gap-2 text-xs text-slate-400"><Loader2 size={14} className="animate-spin" /> Loading Google…</div>}
      <div ref={holder} className="flex justify-center" />
    </div>
  );
}
