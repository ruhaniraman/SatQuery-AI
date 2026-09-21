import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { nextTheme } from '../utils/theme';

// Icon shows the theme you will switch TO (sun while dark, moon while light).
export default function ThemeToggle({ theme, onChange }) {
  const target = nextTheme(theme);
  const Icon = theme === 'dark' ? Sun : Moon;
  return (
    <button
      type="button"
      onClick={() => onChange(target)}
      aria-label={`Switch to ${target} theme`}
      title={`Switch to ${target} theme`}
      className="inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"
    >
      <Icon size={15} />
    </button>
  );
}
