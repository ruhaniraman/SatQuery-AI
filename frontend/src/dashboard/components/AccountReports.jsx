import { useCallback, useContext, useEffect, useRef, useState } from 'react';
import { Check, Download, History, Loader2, Pencil, RotateCcw, X } from 'lucide-react';
import { AuthContext } from '../../auth/authState';
import { readToken, renameReport as requestRenameReport, fetchMyReports } from '../utils/auth';
import { evidenceUrl, humanizeIdentifier, reportUrl, savePdf } from '../utils/api';
import { Card, Label, Notice, PreviewImage, cx } from './ui';

const when = (epochSeconds) => new Date(epochSeconds * 1000).toLocaleString(undefined, {
  dateStyle: 'medium', timeStyle: 'short',
});

// The name box while editing: pre-filled with the report's current name (its default is the original
// query, set server-side), Enter/the check saves, Escape/the cross cancels back to that same value.
function RenameForm({ entry, onSave, onCancel }) {
  const [value, setValue] = useState(entry.title);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);
  useEffect(() => { inputRef.current?.focus(); inputRef.current?.select(); }, []);

  const submit = async (e) => {
    e.preventDefault();
    if (!value.trim()) { setError('Enter a name.'); return; }
    setBusy(true);
    setError(null);
    try {
      await onSave(value.trim());
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  return (
    <form onSubmit={submit} className="flex min-w-0 flex-1 items-center gap-1">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => { setValue(e.target.value); if (error) setError(null); }}
        onKeyDown={(e) => { if (e.key === 'Escape') onCancel(); }}
        maxLength={200}
        aria-label="Report name"
        aria-invalid={error ? 'true' : undefined}
        disabled={busy}
        className={cx(
          'min-w-0 flex-1 rounded-md border bg-black/40 px-2 py-1 text-[13px] text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
          error ? 'border-red-700/60' : 'border-white/15',
        )}
      />
      <button type="submit" disabled={busy} aria-label="Save name" title="Save"
        className="inline-flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-emerald-300 transition enabled:hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 disabled:cursor-progress disabled:opacity-60">
        {busy ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />}
      </button>
      <button type="button" onClick={onCancel} disabled={busy} aria-label="Cancel renaming" title="Cancel"
        className="inline-flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition enabled:hover:bg-white/10 enabled:hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 disabled:cursor-progress disabled:opacity-60">
        <X size={13} />
      </button>
      {error && <span role="alert" className="absolute mt-9 text-[11px] text-red-300">{error}</span>}
    </form>
  );
}

function ReportRow({ entry, editing, busy, onDownload, onStartEdit, onCancelEdit, onSaveEdit }) {
  return (
    <li className="relative flex items-center gap-2.5 rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
      <span className="h-9 w-12 shrink-0 overflow-hidden rounded-md border border-white/10 bg-black/40">
        <PreviewImage src={evidenceUrl(entry)} alt="" className="h-full w-full object-cover" />
      </span>

      {editing ? (
        <RenameForm entry={entry} onSave={(title) => onSaveEdit(entry, title)} onCancel={onCancelEdit} />
      ) : (
        <>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13px] text-slate-100" title={entry.title}>{entry.title}</span>
            <span className="block text-[11px] text-slate-500">{when(entry.created_at)} · {humanizeIdentifier(entry.task || '')}</span>
          </span>
          <button
            type="button"
            onClick={() => onStartEdit(entry.session_id)}
            title="Rename this report"
            aria-label={`Rename the report “${entry.title}”`}
            className="inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition hover:bg-white/10 hover:text-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"
          >
            <Pencil size={14} />
          </button>
          <button
            type="button"
            onClick={() => onDownload(entry)}
            disabled={busy === entry.session_id}
            aria-busy={busy === entry.session_id}
            title="Download this report"
            aria-label={`Download the report for “${entry.title}”`}
            className="inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-blue-300 transition enabled:hover:bg-white/10 enabled:hover:text-blue-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 disabled:cursor-progress disabled:opacity-60"
          >
            {busy === entry.session_id ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
          </button>
        </>
      )}
    </li>
  );
}

// Every signed-in /analyze run is saved to the account (backend/auth.py, GET /auth/reports) and stays
// downloadable and renameable here across sessions and devices, independent of this browser tab's own
// run history above. A report's default name is the question it was asked; renaming only changes the
// display name (the original question is kept server-side too).
export default function AccountReports({ className = '' }) {
  // A plain useContext, not the throwing useAuth(): this panel can render standalone (e.g. outside
  // <AuthProvider> in isolation), in which case it simply has no account history to show.
  const status = useContext(AuthContext)?.status;
  const [state, setState] = useState({ loading: true, error: null, reports: [] });
  const [busy, setBusy] = useState(null);
  const [editingId, setEditingId] = useState(null);

  const load = useCallback(async () => {
    const token = readToken();
    if (!token) { setState({ loading: false, error: null, reports: [] }); return; }
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const { reports } = await fetchMyReports(token);
      setState({ loading: false, error: null, reports });
    } catch (err) {
      setState({ loading: false, error: err.message, reports: [] });
    }
  }, []);

  useEffect(() => {
    if (status !== 'authed') return undefined;
    const id = setTimeout(load, 0);   // deferred so this effect never calls setState synchronously itself
    return () => clearTimeout(id);
  }, [status, load]);

  if (status !== 'authed') return null;

  const download = async (entry) => {
    setBusy(entry.session_id);
    await savePdf(reportUrl(entry), `satquery-report-${entry.session_id}.pdf`);
    setBusy(null);
  };

  const saveEdit = async (entry, title) => {
    await requestRenameReport(readToken(), entry.session_id, title);
    setState((s) => ({ ...s, reports: s.reports.map((r) => (r.session_id === entry.session_id ? { ...r, title } : r)) }));
    setEditingId(null);
  };

  return (
    <div className={cx('space-y-2', className)}>
      <Label
        icon={History}
        right={(
          <button type="button" onClick={load} disabled={state.loading} aria-label="Refresh your report history"
            className="inline-flex cursor-pointer items-center gap-1 text-[11px] font-medium text-blue-300 hover:underline disabled:cursor-progress disabled:opacity-60">
            <RotateCcw size={11} className={state.loading ? 'animate-spin' : ''} /> Refresh
          </button>
        )}
      >
        Your saved reports
      </Label>
      {state.error && <Notice tone="error" title="Could not load your saved reports">{state.error}</Notice>}
      {!state.error && state.loading && (
        <Card className="flex items-center gap-2 p-3.5 text-[12px] text-slate-400">
          <Loader2 size={14} className="animate-spin" /> Loading…
        </Card>
      )}
      {!state.error && !state.loading && state.reports.length === 0 && (
        <Card className="p-3.5 text-[12px] leading-relaxed text-slate-400">
          Reports you generate while signed in are saved here automatically, ready to download (and rename) anytime.
        </Card>
      )}
      {!state.error && state.reports.length > 0 && (
        <ul className="space-y-1.5">
          {state.reports.map((entry) => (
            <ReportRow
              key={entry.session_id}
              entry={entry}
              editing={editingId === entry.session_id}
              busy={busy}
              onDownload={download}
              onStartEdit={setEditingId}
              onCancelEdit={() => setEditingId(null)}
              onSaveEdit={saveEdit}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
