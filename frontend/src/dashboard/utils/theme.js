// Dark/light theme: pure helpers (no DOM, no React) so the rules are testable in plain Node.
// The choice is remembered per browser. Dark is the default: it is the look the dashboard was built for.
export const THEME_KEY = 'sq-theme';
export const THEMES = ['dark', 'light'];
export const DEFAULT_THEME = 'dark';

const browserStorage = () => {
  try { return globalThis.localStorage ?? null; } catch { return null; }   // blocked storage can throw on access
};

export function readTheme(storage = browserStorage()) {
  try {
    const value = storage?.getItem(THEME_KEY);
    return THEMES.includes(value) ? value : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

export function saveTheme(theme, storage = browserStorage()) {
  if (!THEMES.includes(theme)) return false;
  try {
    storage?.setItem(THEME_KEY, theme);
    return true;
  } catch {
    return false;                                            // private mode / quota: the theme still applies, just not remembered
  }
}

export const nextTheme = (theme) => (theme === 'light' ? 'dark' : 'light');
