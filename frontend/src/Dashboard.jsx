import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Cpu, PanelLeftClose } from 'lucide-react';
import earthBg from './assets/earth.jpeg';
import './dashboard/dashboard.css';
import { FEATURES, featureById } from './dashboard/features';
import { useWorkspace } from './dashboard/hooks/useWorkspace';
import NavRail from './dashboard/components/NavRail';
import Viewer from './dashboard/components/Viewer';

// Layout: header, then [feature nav] [active feature's panel] [viewer]. The viewer shows the live
// satellite map by default and can be switched to the user's inputs and analysis evidence.
export default function Dashboard() {
  const ws = useWorkspace();
  const [featureId, setFeatureId] = useState('imagery');
  const [panelOpen, setPanelOpen] = useState(true);
  const feature = featureById(featureId);
  const Panel = feature.Panel;

  // Clicking the feature that is already open collapses its panel, giving the viewer the full width
  const selectFeature = (id) => {
    if (id === featureId) setPanelOpen((open) => !open);
    else {
      setFeatureId(id);
      setPanelOpen(true);
    }
  };
  const goTo = (id) => {
    setFeatureId(id);
    setPanelOpen(true);
  };

  return (
    <div
      className="sq-root fixed inset-0 flex flex-col overflow-hidden font-sans text-slate-100"
      style={{ backgroundImage: `url(${earthBg})`, backgroundSize: 'cover', backgroundPosition: 'center' }}
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.80) 0%, rgba(0,0,0,0.40) 45%, rgba(0,0,0,0.75) 100%)' }}
      />

      <div className="relative z-10 flex h-full min-h-0 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between border-b border-white/10 bg-black/50 px-5 backdrop-blur-md">
          <Link to="/" className="flex items-center gap-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 rounded-lg" title="Back to home">
            <span className="rounded-lg bg-blue-500/90 p-1.5 text-slate-950"><Cpu size={18} /></span>
            <span className="text-lg font-bold leading-none tracking-wide">SatQuery-AI</span>
          </Link>
          <span className="flex items-center rounded-full border border-blue-700/40 bg-blue-950/40 px-3 py-1 text-xs text-blue-300">
            <span className={`mr-2 h-2 w-2 rounded-full bg-blue-400 ${ws.isExecuting ? 'animate-pulse' : ''}`} />
            {ws.isExecuting ? 'Analysing' : 'Agent ready'}
          </span>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:flex-row">
          <NavRail features={FEATURES} activeId={featureId} panelOpen={panelOpen} onSelect={selectFeature} ws={ws} />

          {panelOpen && (
            <aside
              aria-label={feature.title}
              className="flex max-h-[46vh] w-full shrink-0 flex-col overflow-hidden rounded-xl border border-white/10 bg-black/50 backdrop-blur-md lg:max-h-none lg:w-[400px] xl:w-[430px]"
            >
              <div className="flex shrink-0 items-start justify-between gap-3 border-b border-white/10 px-5 py-4">
                <div className="min-w-0">
                  <h2 className="text-base font-semibold leading-tight text-slate-50">{feature.title}</h2>
                  <p className="mt-0.5 text-xs leading-snug text-slate-400">{feature.summary}</p>
                </div>
                <button
                  type="button"
                  onClick={() => setPanelOpen(false)}
                  title="Collapse panel"
                  aria-label="Collapse panel"
                  className="shrink-0 cursor-pointer rounded-md p-1.5 text-slate-400 transition hover:bg-white/5 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"
                >
                  <PanelLeftClose size={16} />
                </button>
              </div>
              <div className={`min-h-0 flex-1 px-5 py-4 ${feature.id === 'assistant' ? '' : 'overflow-y-auto sq-scroll'}`}>
                <Panel ws={ws} goTo={goTo} />
              </div>
            </aside>
          )}

          <main className="flex min-h-0 min-w-0 flex-1">
            <Viewer ws={ws} onAddImagery={() => goTo('imagery')} onOpenFeature={goTo} />
          </main>
        </div>
      </div>
    </div>
  );
}
