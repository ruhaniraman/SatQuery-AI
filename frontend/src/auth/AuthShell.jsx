import { Link } from 'react-router-dom';
import earthBg from '../assets/earth.jpeg';
import logoMark from '../assets/logo-mark.png';
import '../dashboard/dashboard.css';
import { readTheme } from '../dashboard/utils/theme';

// The frame shared by the login, sign-up and session-check screens: the dashboard's earth backdrop and glass card,
// in whichever theme the person chose there.
export default function AuthShell({ title, subtitle, children }) {
  return (
    <div
      data-theme={readTheme()}
      className="sq-root fixed inset-0 overflow-y-auto font-sans text-slate-100"
      style={{ backgroundImage: `url(${earthBg})`, backgroundSize: 'cover', backgroundPosition: 'center' }}
    >
      <div className="sq-overlay pointer-events-none fixed inset-0" />
      <div className="relative z-10 flex min-h-full flex-col items-center justify-center px-4 py-10">
        <Link to="/" aria-label="SatQuery-AI home" className="mb-6 flex items-center gap-3 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
          <span role="img" aria-hidden="true" className="sq-logo-mark h-11 w-11" style={{ '--sq-logo': `url(${logoMark})` }} />
          <span className="text-2xl font-bold leading-none tracking-wide">SatQuery-AI</span>
        </Link>
        <main className="w-full max-w-sm rounded-xl border border-white/10 bg-black/50 p-6 backdrop-blur-md">
          <h1 className="text-lg font-semibold text-slate-50">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
          <div className="mt-5">{children}</div>
        </main>
      </div>
    </div>
  );
}
