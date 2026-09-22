// Talking to the backend's /auth endpoints, plus the client-side form checks. No JSX, no DOM: tested in plain Node.
// The session token is kept in this browser (localStorage) and sent as a Bearer header.
import { BACKEND_URL, readErrorMessage } from './api.js';

export const TOKEN_KEY = 'sq-token';
export const MIN_PASSWORD_LENGTH = 8;

const browserStorage = () => {
  try { return globalThis.localStorage ?? null; } catch { return null; }   // blocked storage can throw on access
};

export function readToken(storage = browserStorage()) {
  try { return storage?.getItem(TOKEN_KEY) || null; } catch { return null; }
}

export function saveToken(token, storage = browserStorage()) {
  try { storage?.setItem(TOKEN_KEY, token); return true; } catch { return false; }
}

export function clearToken(storage = browserStorage()) {
  try { storage?.removeItem(TOKEN_KEY); } catch { /* nothing to clear */ }
}

export class AuthError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'AuthError';
    this.status = status;                 // 0 = the server could not be reached
  }
}

async function call(path, { method = 'GET', body, token, fetchImpl = fetch, baseUrl = BACKEND_URL } = {}) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetchImpl(`${baseUrl}/auth${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new AuthError('Cannot reach the SatQuery-AI server. Check that the backend is running and try again.');
  }
  if (!response.ok) throw new AuthError(await readErrorMessage(response, 'Sign-in'), response.status);
  return response.json();
}

// -> { google_client_id: string | null, min_password_length }
export const fetchAuthConfig = (opts) => call('/config', opts);
// Signup is two calls: /signup validates the form and emails a 6-digit code ({ pending, email, expires_in }, no
// token yet); /verify-signup takes that code and only then creates the account and returns { token, user }.
export const requestSignup = ({ name, email, password }, opts) => call('/signup', { ...opts, method: 'POST', body: { name, email, password } });
export const requestVerifySignup = ({ email, code }, opts) => call('/verify-signup', { ...opts, method: 'POST', body: { email, code } });
export const requestResendOtp = ({ email }, opts) => call('/resend-otp', { ...opts, method: 'POST', body: { email } });
export const requestLogin = ({ email, password }, opts) => call('/login', { ...opts, method: 'POST', body: { email, password } });
export const requestGoogleLogin = (credential, opts) => call('/google', { ...opts, method: 'POST', body: { credential } });
export const requestLogout = (token, opts) => call('/logout', { ...opts, method: 'POST', token });

// This account's past /analyze runs (newest first), each with a ready-to-download report PDF and
// evidence image URL (relative to BACKEND_URL, like requestAnalysis's own visual_evidence_url /
// report_download_url). Throws AuthError(401) if the token is not (or no longer) valid.
export const fetchMyReports = (token, opts) => call('/reports', { ...opts, token });

// Renames a saved report (its default name is the original question). Throws AuthError: 401 not signed
// in, 404 no such report in this account, 422 an empty or too-long name.
export const renameReport = (token, sessionId, title, opts) =>
  call(`/reports/${encodeURIComponent(sessionId)}`, { ...opts, method: 'PATCH', token, body: { title } });

// The signed-in user for a stored token, or null when the server says the token is no longer valid.
// A server that cannot be reached throws instead, so being offline never signs anybody out.
export async function requestMe(token, opts) {
  try {
    return (await call('/me', { ...opts, token })).user;
  } catch (err) {
    if (err instanceof AuthError && err.status === 401) return null;
    throw err;
  }
}

const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

// The first problem with the form, or null. The server checks again; this only saves a round trip.
export function validateAuthForm({ mode, name = '', email = '', password = '', confirm = '' }, minLength = MIN_PASSWORD_LENGTH) {
  if (mode === 'signup' && !name.trim()) return 'Enter your name.';
  if (!EMAIL.test(email.trim())) return 'Enter a valid email address.';
  if (!password) return 'Enter your password.';
  if (mode === 'signup') {
    if (password.length < minLength) return `Use at least ${minLength} characters for the password.`;
    if (password !== confirm) return 'The two passwords do not match.';
  }
  return null;
}

// Only follow a "go back to where you were" path that stays inside this app
export const safeRedirect = (from) => (typeof from === 'string' && /^\/(?!\/)/.test(from) ? from : '/dashboard');
