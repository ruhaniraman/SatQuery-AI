import React from 'react';
import { Link, Navigate, useLocation } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { useAuth } from './authState';
import AuthShell from './AuthShell';

// Guards a page: signed in -> the page; not signed in -> the login page (and back here afterwards).
export default function RequireAuth({ children }) {
  const { status, recheck } = useAuth();
  const location = useLocation();

  if (status === 'authed') return children;
  if (status === 'anon') return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  if (status === 'offline') {
    return (
      <AuthShell title="Cannot reach the server">
        <p className="text-sm text-slate-300">You are signed in on this browser, but the SatQuery-AI backend is not answering. Start it and try again.</p>
        <div className="mt-5 flex gap-2">
          <button type="button" onClick={recheck} className="cursor-pointer rounded-lg bg-blue-600/90 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-blue-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300">Try again</button>
          <Link to="/login" className="rounded-lg border border-white/15 px-4 py-2 text-sm text-slate-200 transition hover:bg-white/5">Go to login</Link>
        </div>
      </AuthShell>
    );
  }
  return (
    <AuthShell title="Checking your session">
      <div className="flex items-center gap-2 text-sm text-slate-300"><Loader2 size={16} className="animate-spin text-blue-400" /> One moment…</div>
    </AuthShell>
  );
}
