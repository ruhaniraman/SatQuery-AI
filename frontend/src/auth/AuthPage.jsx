import { useEffect, useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { useAuth } from './authState';
import AuthShell from './AuthShell';
import GoogleButton from './GoogleButton';
import {
  MIN_PASSWORD_LENGTH, fetchAuthConfig, requestGoogleLogin, requestLogin, requestSignup, requestVerifySignup,
  requestResendOtp, safeRedirect, validateAuthForm,
} from '../dashboard/utils/auth';

const FIELD = 'w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-400/70 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/40';

function Field({ label, id, ...input }) {
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-[11px] font-semibold uppercase tracking-wider text-slate-400">{label}</label>
      <input id={id} className={FIELD} {...input} />
    </div>
  );
}

// Login and sign-up: the same form, with a name and a confirmation added for sign-up.
export default function AuthPage({ mode = 'login' }) {
  const signup = mode === 'signup';
  const { status, signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const destination = safeRedirect(location.state?.from);

  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState({ googleClientId: null, minLength: MIN_PASSWORD_LENGTH, loaded: false });
  // Signup is two steps: the form, then the emailed code. `pendingEmail` set = show the code step.
  const [pendingEmail, setPendingEmail] = useState(null);
  const [code, setCode] = useState('');
  const [notice, setNotice] = useState(null);
  const [resendCooldown, setResendCooldown] = useState(0);

  useEffect(() => {
    if (resendCooldown <= 0) return undefined;
    const t = setInterval(() => setResendCooldown((s) => Math.max(0, s - 1)), 1000);
    return () => clearInterval(t);
  }, [resendCooldown]);

  useEffect(() => {
    let cancelled = false;
    fetchAuthConfig()
      .then((c) => { if (!cancelled) setConfig({ googleClientId: c.google_client_id, minLength: c.min_password_length || MIN_PASSWORD_LENGTH, loaded: true }); })
      .catch(() => { if (!cancelled) setConfig((c) => ({ ...c, loaded: true })); });
    return () => { cancelled = true; };
  }, []);

  if (status === 'authed') return <Navigate to={destination} replace />;

  const set = (key) => (e) => { setForm((f) => ({ ...f, [key]: e.target.value })); if (error) setError(null); };

  const finish = (session) => { signIn(session); navigate(destination, { replace: true }); };

  const submit = async (e) => {
    e.preventDefault();
    const problem = validateAuthForm({ mode, ...form }, config.minLength);
    if (problem) { setError(problem); return; }
    setBusy(true);
    setError(null);
    try {
      if (signup) {
        const email = form.email.trim();
        const res = await requestSignup({ name: form.name.trim(), email, password: form.password });
        setPendingEmail(email);
        setResendCooldown(30);
        setNotice(`We sent a 6-digit code to ${email}. It expires in ${Math.round((res.expires_in || 600) / 60)} minutes.`);
        setBusy(false);
      } else {
        finish(await requestLogin({ email: form.email.trim(), password: form.password }));
      }
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  const verify = async (e) => {
    e.preventDefault();
    if (!code.trim()) { setError('Enter the 6-digit code from your email.'); return; }
    setBusy(true);
    setError(null);
    try {
      finish(await requestVerifySignup({ email: pendingEmail, code: code.trim() }));
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  const resend = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await requestResendOtp({ email: pendingEmail });
      setResendCooldown(30);
      setNotice(`We sent a new code to ${pendingEmail}. It expires in ${Math.round((res.expires_in || 600) / 60)} minutes.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const backToForm = () => { setPendingEmail(null); setCode(''); setError(null); setNotice(null); };

  const google = async (credential) => {
    setBusy(true);
    setError(null);
    try {
      finish(await requestGoogleLogin(credential));
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  if (signup && pendingEmail) {
    return (
      <AuthShell title="Check your email" subtitle={`Enter the 6-digit code we sent to ${pendingEmail}.`}>
        <form onSubmit={verify} className="space-y-3.5" noValidate>
          <Field
            label="Verification code" id="auth-otp" type="text" inputMode="numeric" autoComplete="one-time-code"
            maxLength={6} value={code}
            onChange={(e) => { setCode(e.target.value.replace(/\D/g, '')); if (error) setError(null); }}
            placeholder="000000"
          />
          {notice && !error && (
            <p role="status" className="rounded-lg border border-blue-800/40 bg-blue-950/40 px-3 py-2 text-[12px] leading-snug text-blue-200">{notice}</p>
          )}
          {error && (
            <p role="alert" className="rounded-lg border border-red-800/40 bg-red-950/60 px-3 py-2 text-[12px] leading-snug text-red-200">{error}</p>
          )}
          <button type="submit" disabled={busy} aria-busy={busy}
            className="inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-blue-600/90 px-4 py-2.5 text-sm font-semibold text-slate-950 transition enabled:hover:bg-blue-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 disabled:cursor-progress disabled:opacity-80">
            {busy && <Loader2 size={15} className="animate-spin" />}
            Verify and create account
          </button>
        </form>
        <p className="mt-5 flex items-center justify-between text-[13px] text-slate-400">
          <button type="button" onClick={backToForm} className="font-semibold text-slate-300 hover:text-slate-100 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
            Use a different email
          </button>
          <button type="button" onClick={resend} disabled={busy || resendCooldown > 0}
            className="font-semibold text-blue-300 enabled:hover:text-blue-200 enabled:hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70 disabled:text-slate-500 disabled:cursor-not-allowed">
            {resendCooldown > 0 ? `Resend code (${resendCooldown}s)` : 'Resend code'}
          </button>
        </p>
      </AuthShell>
    );
  }

  return (
    <AuthShell
      title={signup ? 'Create your account' : 'Welcome back'}
      subtitle={signup ? 'Sign up to use the SatQuery-AI dashboard.' : 'Log in to open the SatQuery-AI dashboard.'}
    >
      <GoogleButton clientId={config.googleClientId} onCredential={google} onError={setError} disabled={busy} />

      <div className="my-4 flex items-center gap-3 text-[11px] uppercase tracking-wider text-slate-500" aria-hidden="true">
        <span className="h-px flex-1 bg-white/10" /> or with email <span className="h-px flex-1 bg-white/10" />
      </div>

      <form onSubmit={submit} className="space-y-3.5" noValidate>
        {signup && <Field label="Name" id="auth-name" type="text" autoComplete="name" value={form.name} onChange={set('name')} placeholder="Your name" />}
        <Field label="Email" id="auth-email" type="email" autoComplete="email" value={form.email} onChange={set('email')} placeholder="you@example.com" />
        <Field label="Password" id="auth-password" type="password" autoComplete={signup ? 'new-password' : 'current-password'} value={form.password} onChange={set('password')} placeholder={signup ? `At least ${config.minLength} characters` : 'Your password'} />
        {signup && <Field label="Confirm password" id="auth-confirm" type="password" autoComplete="new-password" value={form.confirm} onChange={set('confirm')} placeholder="Repeat the password" />}

        {error && (
          <p role="alert" className="rounded-lg border border-red-800/40 bg-red-950/60 px-3 py-2 text-[12px] leading-snug text-red-200">{error}</p>
        )}

        <button type="submit" disabled={busy} aria-busy={busy}
          className="inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-lg bg-blue-600/90 px-4 py-2.5 text-sm font-semibold text-slate-950 transition enabled:hover:bg-blue-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-300 disabled:cursor-progress disabled:opacity-80">
          {busy && <Loader2 size={15} className="animate-spin" />}
          {signup ? 'Create account' : 'Log in'}
        </button>
      </form>

      <p className="mt-5 text-center text-[13px] text-slate-400">
        {signup ? 'Already have an account?' : 'New to SatQuery-AI?'}{' '}
        <Link to={signup ? '/login' : '/signup'} state={location.state} className="font-semibold text-blue-300 hover:text-blue-200 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70">
          {signup ? 'Log in' : 'Sign up'}
        </Link>
      </p>
    </AuthShell>
  );
}
