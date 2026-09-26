import { useEffect, useRef, useState } from 'react';
import { CircleAlert, ExternalLink, Loader2, Unplug, X } from 'lucide-react';
import { Button, Segmented } from './ui';
import { cx } from './uiHelpers';
import {
  CDSE_HELP, CDSE_SENTINEL_HUB, CDSE_SIGNUP, buildCdseTemplate, describeSarSource, testSarTemplate, validateTemplate,
} from '../utils/sarSource';

const INPUT = 'w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-[13px] text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none disabled:opacity-50';
const TABS = [
  { value: 'cdse', label: 'Copernicus (free account)' },
  { value: 'custom', label: 'Any tile URL' },
];

function Field({ id, label, hint, children }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        {label} {hint && <span className="font-normal normal-case tracking-normal text-slate-500">{hint}</span>}
      </label>
      {children}
    </div>
  );
}

const Link = ({ href, children }) => (
  <a href={href} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-0.5 font-semibold text-blue-300 underline">{children}<ExternalLink size={10} aria-hidden="true" /></a>
);

// Connect a SAR imagery provider from a personal account, without editing any file. The details are checked
// (format, then a real test tile) before they are saved, and are kept only in this browser.
export default function SarDialog({ source, onSave, onDisconnect, onClose }) {
  const [tab, setTab] = useState(source?.kind === 'custom' ? 'custom' : 'cdse');
  const [instanceId, setInstanceId] = useState(source?.instanceId || '');
  const [layer, setLayer] = useState(source?.layer || '');
  const [timeRange, setTimeRange] = useState(source?.timeRange || '');
  const [url, setUrl] = useState(source?.kind === 'custom' ? source.template : '');
  const [attribution, setAttribution] = useState(source?.kind === 'custom' ? source.attribution || '' : '');
  const [testing, setTesting] = useState(false);
  const [problem, setProblem] = useState(null);       // { message, pending? } pending = a source the user may still choose to use
  const box = useRef(null);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    box.current?.querySelector('input')?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const candidate = () => {
    if (tab === 'cdse') {
      const built = buildCdseTemplate({ instanceId, layer, timeRange });
      return built.ok
        ? { ok: true, source: { kind: 'cdse', template: built.template, instanceId: instanceId.trim(), layer: layer.trim(), timeRange: timeRange.trim(), attribution: 'Copernicus Sentinel data (Sentinel Hub)', maxZoom: 14 } }
        : built;
    }
    const checked = validateTemplate(url);
    return checked.ok ? { ok: true, source: { kind: 'custom', template: checked.template, attribution: attribution.trim(), maxZoom: 14 } } : checked;
  };

  const connect = async (e) => {
    e?.preventDefault();
    setProblem(null);
    const next = candidate();
    if (!next.ok) { setProblem({ message: next.error }); return; }
    setTesting(true);
    const result = await testSarTemplate(next.source.template);
    setTesting(false);
    if (result.ok) onSave(next.source);
    else setProblem({ message: result.message, pending: next.source });
  };

  return (
    <div ref={box} role="dialog" aria-label="Connect SAR imagery" className="pointer-events-auto absolute left-3 top-[3.25rem] z-30 w-[min(28rem,calc(100%-1.5rem))] rounded-xl border border-white/15 bg-black/85 p-4 shadow-xl backdrop-blur-md">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-50">Connect SAR imagery</h3>
          {source && <p className="text-[11px] text-slate-400">Connected: {describeSarSource(source)}</p>}
        </div>
        <button type="button" onClick={onClose} aria-label="Close" className="cursor-pointer rounded p-0.5 text-slate-400 hover:text-slate-100"><X size={15} /></button>
      </div>

      <Segmented size="sm" label="Provider" value={tab} onChange={(v) => { setTab(v); setProblem(null); }} options={TABS} className="mb-3" />

      <form onSubmit={connect} className="space-y-3">
        {tab === 'cdse' ? (
          <>
            <ol className="list-decimal space-y-1 pl-4 text-[11px] leading-relaxed text-slate-400">
              <li>Create a free account at <Link href={CDSE_SIGNUP}>dataspace.copernicus.eu</Link>.</li>
              <li>Open <Link href={CDSE_SENTINEL_HUB}>Sentinel Hub</Link> &rarr; Configuration Utility &rarr; New configuration. Add a Sentinel-1 layer and copy the Instance ID.</li>
              <li>Paste the Instance ID and the layer name here. <Link href={CDSE_HELP}>Details</Link></li>
            </ol>
            <Field id="sar-instance" label="Instance ID"><input id="sar-instance" className={INPUT} value={instanceId} onChange={(e) => setInstanceId(e.target.value)} placeholder="1a2b3c4d-…" spellCheck={false} autoComplete="off" /></Field>
            <div className="grid grid-cols-2 gap-2.5">
              <Field id="sar-layer" label="Layer name"><input id="sar-layer" className={INPUT} value={layer} onChange={(e) => setLayer(e.target.value)} placeholder="e.g. S1-VV" spellCheck={false} autoComplete="off" /></Field>
              <Field id="sar-time" label="Dates" hint="(optional)"><input id="sar-time" className={INPUT} value={timeRange} onChange={(e) => setTimeRange(e.target.value)} placeholder="2024-01-01/2024-03-31" spellCheck={false} autoComplete="off" /></Field>
            </div>
          </>
        ) : (
          <>
            <Field id="sar-url" label="Tile URL" hint="with {z} {x} {y}"><input id="sar-url" className={INPUT} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://…/{z}/{x}/{y}.png" spellCheck={false} autoComplete="off" /></Field>
            <Field id="sar-attr" label="Credit" hint="(optional)"><input id="sar-attr" className={INPUT} value={attribution} onChange={(e) => setAttribution(e.target.value)} placeholder="shown on the map" autoComplete="off" /></Field>
          </>
        )}

        {problem && (
          <div role="alert" className="space-y-2 rounded-lg border border-red-800/40 bg-red-950/30 p-2.5 text-[11px] leading-relaxed text-red-200">
            <p className="flex items-start gap-1.5"><CircleAlert size={13} className="mt-0.5 shrink-0" />{problem.message}</p>
            {problem.pending && <Button size="sm" variant="subtle" onClick={() => onSave(problem.pending)}>Use it anyway</Button>}
          </div>
        )}

        <p className="text-[10px] text-slate-500">Kept only in this browser. The Instance ID works like a password for your layers.</p>
        <div className="flex items-center gap-2">
          <Button type="submit" size="sm" loading={testing} disabled={testing}>{testing ? 'Testing' : 'Connect'}</Button>
          <Button size="sm" variant="ghost" onClick={onClose}>Cancel</Button>
          {source && <Button size="sm" variant="ghost" icon={Unplug} onClick={onDisconnect} className={cx('ml-auto', '!text-red-300')}>Disconnect</Button>}
        </div>
      </form>
      {testing && <p className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-400"><Loader2 size={11} className="animate-spin" /> Asking the provider for a test tile…</p>}
    </div>
  );
}
