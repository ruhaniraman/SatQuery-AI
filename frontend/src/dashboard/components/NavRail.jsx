import React from 'react';
import { Loader2 } from 'lucide-react';
import { cx } from './ui';
import { GROUP_LABELS } from '../features';

// Vertical feature navigation (a row on narrow screens). Built entirely from the feature registry; a
// divider and a small caption separate the groups (single-image tools, two-image tools, the report).
export default function NavRail({ features, activeId, panelOpen, onSelect, ws }) {
  return (
    <nav aria-label="Features" className="flex shrink-0 gap-1 rounded-xl border border-white/10 bg-black/50 p-2 backdrop-blur-md max-lg:items-center max-lg:overflow-x-auto lg:w-[76px] lg:flex-col">
      {features.map((feature, index) => {
        const Icon = feature.icon;
        const selected = feature.id === activeId && panelOpen;
        const status = feature.status?.(ws);
        const startsGroup = index === 0 || features[index - 1].group !== feature.group;
        const caption = GROUP_LABELS[feature.group];
        return (
          <React.Fragment key={feature.id}>
            {startsGroup && index > 0 && (
              <div aria-hidden="true" className="mx-2 my-1 border-t border-white/10 max-lg:mx-1 max-lg:my-0 max-lg:h-8 max-lg:border-l max-lg:border-t-0" />
            )}
            {startsGroup && caption && (
              <span className="px-1 pb-0.5 pt-0.5 text-center text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500 max-lg:hidden">{caption}</span>
            )}
            <button
              type="button"
              onClick={() => onSelect(feature.id)}
              aria-current={selected ? 'page' : undefined}
              title={selected ? `Hide ${feature.title}` : `${feature.title}: ${feature.summary}`}
              className={cx(
                'group relative flex min-w-[64px] flex-col items-center gap-1 rounded-lg border px-2 py-2.5 transition cursor-pointer lg:w-full lg:min-w-0',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70',
                selected
                  ? 'border-blue-500/40 bg-blue-500/15 text-blue-200'
                  : 'border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-100',
              )}
            >
              {selected && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-blue-400 max-lg:hidden" aria-hidden="true" />}
              <span className="relative">
                <Icon size={20} />
                {status === 'busy' && <Loader2 size={11} className="absolute -right-2 -top-2 animate-spin text-blue-300" aria-label="Working" />}
                {status === 'ready' && <span className="absolute -right-1.5 -top-1 h-2 w-2 rounded-full bg-blue-400 ring-2 ring-black/70" aria-hidden="true" />}
              </span>
              <span className="text-[10px] font-medium leading-none tracking-wide">{feature.label}</span>
            </button>
          </React.Fragment>
        );
      })}
    </nav>
  );
}
