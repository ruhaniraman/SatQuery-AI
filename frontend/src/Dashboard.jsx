import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { PanelLeftClose } from 'lucide-react';
import earthBg from './assets/earth.jpeg';
import logoMark from './assets/logo-mark.png';
import './dashboard/dashboard.css';
import { FEATURES, featureById } from './dashboard/features';
import { useWorkspace } from './dashboard/hooks/useWorkspace';
import { useBackendStatus } from './dashboard/hooks/useBackendStatus';
import NavRail from './dashboard/components/NavRail';
import Viewer from './dashboard/components/Viewer';
import ThemeToggle from './dashboard/components/ThemeToggle';
import StatusPill from './dashboard/components/StatusPill';
import UserMenu from './dashboard/components/UserMenu';
import { readTheme, saveTheme } from './dashboard/utils/theme';

// Layout: header, then [feature nav] [active feature's panel] [viewer]. The viewer shows the live
// satellite map by default and can be switched to the open feature's images and analysis evidence.
export default function Dashboard() {
  const ws = useWorkspace();
  const health = useBackendStatus();
  const [featureId, setFeatureId] = useState('imagery');
  const [panelOpen, setPanelOpen] = useState(true);
  const [theme, setTheme] = useState(readTheme);
  const feature = featureById(featureId);
  const Panel = feature.Panel;

  const changeTheme = (next) => {
    setTheme(next);
    saveTheme(next);
  };

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
      data-theme={theme}
      className="sq-root fixed inset-0 flex flex-col overflow-hidden font-sans text-slate-100"
      style={{ backgroundImage: `url(${earthBg})`, backgroundSize: 'cover', backgroundPosition: 'center' }}
    >
      <div className="sq-overlay pointer-events-none absolute inset-0" />

      <div className="relative z-10 flex h-full min-h-0 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-white/10 bg-black/30 px-3 backdrop-blur-md">
          {/* The logo sits in a column as wide as the nav rail (76px), so it is centred over the nav icons, and
              the title starts where the side panel starts (rail + the 12px gap of the layout below). */}
          <Link to="/" aria-label="SatQuery-AI home" className="flex items-center gap-3 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70" title="Back to home">
            <span className="flex w-[76px] shrink-0 justify-center">
              <span role="img" aria-hidden="true" className="sq-logo-mark h-10 w-10" style={{ '--sq-logo': `url(${logoMark})` }} />
            </span>
            <span className="text-2xl font-bold leading-none tracking-wide">SatQuery-AI</span>
          </Link>
          <div className="flex items-center gap-2.5 pr-2">
            <StatusPill health={health} executing={ws.isExecuting} onRefresh={health.refresh} />
            <ThemeToggle theme={theme} onChange={changeTheme} />
            <UserMenu />
          </div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 lg:flex-row">
          <NavRail features={FEATURES} activeId={featureId} panelOpen={panelOpen} onSelect={selectFeature} ws={ws} />

          {panelOpen && (
            <aside
              aria-label={feature.title}
              className="flex max-h-[46vh] w-full shrink-0 flex-col overflow-hidden rounded-xl border border-white/10 bg-black/30 backdrop-blur-md lg:max-h-none lg:w-[400px] xl:w-[430px]"
            >
              <div className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 px-5 py-3.5">
                <div className="min-w-0">
                  <h2 className="text-base font-semibold leading-tight text-slate-50">{feature.title}</h2>
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
            <Viewer ws={ws} feature={feature} onOpenFeature={goTo} />
          </main>
        </div>
      </div>
    </div>
  );
}
