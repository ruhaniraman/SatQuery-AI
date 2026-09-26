import { useState } from 'react';
import { Check, ChevronDown, Copy, Crosshair, MapPin, TriangleAlert } from 'lucide-react';
import { Button, Notice } from './ui';
import { cx } from './uiHelpers';
import { findingLine, levelOf, pinClass, summaryText, thresholdPercent } from '../utils/scanResult';

function Pin({ scan, n, className = '' }) {
  return (
    <span className={cx('flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2 border-white text-[11px] font-bold text-white', pinClass(scan), className)}>{n}</span>
  );
}

function FindingRow({ scan, finding, onShow }) {
  return (
    <li className="flex items-center gap-3 rounded-lg border border-white/10 bg-black/30 p-2.5">
      <Pin scan={scan} n={finding.id} />
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium capitalize text-slate-100">{finding.where}</p>
        <p className="text-[11px] text-slate-400">{finding.label} &middot; {finding.coverage_pct}% of the view</p>
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/10" role="img" aria-label={`Average score ${finding.confidence} percent`}>
          <div className={cx('h-full rounded-full', pinClass(scan))} style={{ width: `${finding.confidence}%` }} />
        </div>
      </div>
      {onShow && (
        <Button variant="subtle" size="sm" icon={Crosshair} onClick={() => onShow(finding)} aria-label={`Show area ${finding.id} on the image`}>Show</Button>
      )}
    </li>
  );
}

// The scores as they came back, so nothing is hidden behind the summary: 16 cells, brighter = surer
function ScoreGrid({ scan }) {
  return (
    <div className="grid w-40 grid-cols-4 gap-0.5" role="img" aria-label="Score of each of the 16 areas of the image">
      {scan.grid.flat().map((p, i) => (
        <span
          key={i}
          className={cx('flex h-8 items-center justify-center rounded text-[10px] font-medium', p >= scan.threshold ? 'text-white' : 'text-slate-300')}
          style={{ background: `rgba(59,130,246,${(0.1 + 0.8 * p).toFixed(2)})`, outline: p >= scan.threshold ? '1px solid rgba(255,255,255,0.7)' : 'none' }}
        >
          {Math.round(p * 100)}
        </span>
      ))}
    </div>
  );
}

// A feature scan's result, answer first: what was found and how sure the scan is, then one card per
// numbered area (matching the pins on the image), then how it was produced. `compact` is the short form
// for the map card. `onShow(finding)` zooms the viewer to an area; `link` is a map link for map captures.
export default function ScanResult({ scan, link = null, onShow, compact = false }) {
  const [copied, setCopied] = useState(false);
  if (!scan) return null;
  const level = levelOf(scan);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summaryText(scan, link));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  if (compact) {
    return (
      <div className="space-y-2">
        <span className={cx('inline-block rounded-md border px-2 py-0.5 text-[11px] font-semibold', level.chip)}>{level.label}</span>
        <p className="text-[13px] font-medium leading-snug text-slate-100">{scan.headline}</p>
        {scan.note && <Notice tone="warn" icon={TriangleAlert}>{scan.note}</Notice>}
        {scan.findings.length > 0 && (
          <ul className="flex flex-wrap gap-x-4 gap-y-1">
            {scan.findings.map((f) => (
              <li key={f.id} className="flex items-center gap-1.5 text-[12px] capitalize text-slate-300"><Pin scan={scan} n={f.id} className="!h-5 !w-5 !text-[10px]" />{findingLine(f)}</li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <span className={cx('inline-block rounded-md border px-2 py-0.5 text-[11px] font-semibold', level.chip)}>{level.label}</span>
        <p className="text-[15px] font-semibold leading-snug text-slate-50">{scan.headline}</p>
      </div>

      {scan.note && <Notice tone="warn" icon={TriangleAlert} title="Check this one by eye">{scan.note}</Notice>}

      {scan.findings.length > 0 && (
        <ul className="space-y-1.5" aria-label="Areas found">
          {scan.findings.map((f) => <FindingRow key={f.id} scan={scan} finding={f} onShow={onShow} />)}
        </ul>
      )}

      <div className="flex flex-wrap gap-2">
        <Button variant="subtle" size="sm" icon={copied ? Check : Copy} onClick={copy}>{copied ? 'Copied' : 'Copy summary'}</Button>
        {link && (
          <a href={link} target="_blank" rel="noreferrer"
            className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs font-semibold text-slate-200 transition hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
            <MapPin size={14} />Open on map
          </a>
        )}
      </div>

      <details className="group rounded-lg border border-white/10 bg-black/20 text-[12px] text-slate-300">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3 py-2 font-medium text-slate-200">
          How this was produced
          <ChevronDown size={14} className="transition group-open:rotate-180" />
        </summary>
        <div className="space-y-2.5 border-t border-white/10 px-3 py-2.5 leading-relaxed">
          <p>
            The image was split into 16 areas and each was checked for {scan.subject}. The number in each cell below is how sure the
            scan was, from 0 to 100. Areas at {thresholdPercent(scan)} or more are outlined and numbered; fainter ones are only tinted.
          </p>
          <ScoreGrid scan={scan} />
          <p className="text-slate-500">This is an automated screening, not a survey, and it was not measured against ground truth.</p>
        </div>
      </details>
    </div>
  );
}
