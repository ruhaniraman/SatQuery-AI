import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AuthError, TOKEN_KEY, clearToken, fetchAuthConfig, fetchMyReports, readToken, renameReport, requestLogin,
  requestMe, requestResendOtp, requestSignup, requestVerifySignup, safeRedirect, saveToken, validateAuthForm,
} from './auth.js';

const reply = (status, body) => async () => ({ ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) });
const memory = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };

test('signup and login post JSON to /auth and return the body', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return reply(200, { token: 't', user: { email: 'a@b.co' } })(); };
  const out = await requestSignup({ name: 'A', email: 'a@b.co', password: 'longenough' }, { fetchImpl, baseUrl: 'http://x' });
  assert.equal(out.token, 't');
  assert.equal(seen.url, 'http://x/auth/signup');
  assert.equal(seen.init.method, 'POST');
  assert.deepEqual(JSON.parse(seen.init.body), { name: 'A', email: 'a@b.co', password: 'longenough' });
  await requestLogin({ email: 'a@b.co', password: 'p' }, { fetchImpl, baseUrl: 'http://x' });
  assert.equal(seen.url, 'http://x/auth/login');
});

test('verify-signup and resend-otp post JSON to /auth and return the body', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return reply(200, { pending: true })(); };
  await requestVerifySignup({ email: 'a@b.co', code: '123456' }, { fetchImpl, baseUrl: 'http://x' });
  assert.equal(seen.url, 'http://x/auth/verify-signup');
  assert.deepEqual(JSON.parse(seen.init.body), { email: 'a@b.co', code: '123456' });
  await requestResendOtp({ email: 'a@b.co' }, { fetchImpl, baseUrl: 'http://x' });
  assert.equal(seen.url, 'http://x/auth/resend-otp');
  assert.deepEqual(JSON.parse(seen.init.body), { email: 'a@b.co' });
});

test('fetchMyReports sends the bearer token and returns the account\'s saved reports', async () => {
  let seen;
  const reports = [{ session_id: 'abc', query: 'what changed?', task: 'CHANGE_DETECTION', created_at: 1700000000 }];
  const fetchImpl = async (url, init) => { seen = { url, init }; return reply(200, { reports })(); };
  const out = await fetchMyReports('tok', { fetchImpl, baseUrl: 'http://x' });
  assert.equal(seen.url, 'http://x/auth/reports');
  assert.equal(seen.init.headers.Authorization, 'Bearer tok');
  assert.deepEqual(out.reports, reports);
});

test('renameReport PATCHes /auth/reports/<session_id> with the bearer token and the new title', async () => {
  let seen;
  const fetchImpl = async (url, init) => { seen = { url, init }; return reply(200, { ok: true })(); };
  await renameReport('tok', 'abc def', 'Mining scan, north pit', { fetchImpl, baseUrl: 'http://x' });
  assert.equal(seen.url, 'http://x/auth/reports/abc%20def');   // the session id is URL-encoded
  assert.equal(seen.init.method, 'PATCH');
  assert.equal(seen.init.headers.Authorization, 'Bearer tok');
  assert.deepEqual(JSON.parse(seen.init.body), { title: 'Mining scan, north pit' });
});

test('a server error becomes an AuthError carrying the server message and status', async () => {
  await assert.rejects(
    requestLogin({ email: 'a@b.co', password: 'p' }, { fetchImpl: reply(401, { detail: 'Wrong email or password.' }) }),
    (err) => err instanceof AuthError && err.status === 401 && err.message === 'Wrong email or password.',
  );
});

test('an unreachable server says so, with status 0', async () => {
  await assert.rejects(fetchAuthConfig({ fetchImpl: async () => { throw new TypeError('Failed to fetch'); } }), (err) => err.status === 0 && /backend is running/.test(err.message));
});

test('requestMe: valid token -> user, rejected token -> null, offline -> throws (never signs out)', async () => {
  const ok = await requestMe('tok', { fetchImpl: async (url, init) => { assert.equal(init.headers.Authorization, 'Bearer tok'); return reply(200, { user: { name: 'A' } })(); } });
  assert.deepEqual(ok, { name: 'A' });
  assert.equal(await requestMe('tok', { fetchImpl: reply(401, { detail: 'Not signed in.' }) }), null);
  await assert.rejects(requestMe('tok', { fetchImpl: async () => { throw new Error('offline'); } }), (err) => err.status === 0);
  await assert.rejects(requestMe('tok', { fetchImpl: reply(500, {}) }), (err) => err.status === 500);
});

test('the token is stored, read and cleared, and blocked storage never throws', () => {
  const s = memory();
  assert.equal(readToken(s), null);
  assert.equal(saveToken('abc', s), true);
  assert.equal(s.getItem(TOKEN_KEY), 'abc');
  assert.equal(readToken(s), 'abc');
  clearToken(s);
  assert.equal(readToken(s), null);
  const blocked = { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); }, removeItem() { throw new Error('denied'); } };
  assert.equal(readToken(blocked), null);
  assert.equal(saveToken('x', blocked), false);
  assert.doesNotThrow(() => clearToken(blocked));
});

test('form validation reports the first problem', () => {
  const good = { mode: 'signup', name: 'Ada', email: 'ada@example.com', password: 'longenough', confirm: 'longenough' };
  assert.equal(validateAuthForm(good), null);
  assert.match(validateAuthForm({ ...good, name: ' ' }), /name/);
  assert.match(validateAuthForm({ ...good, email: 'nope' }), /email/);
  assert.match(validateAuthForm({ ...good, password: 'short', confirm: 'short' }), /at least 8/);
  assert.match(validateAuthForm({ ...good, confirm: 'different1' }), /do not match/);
  assert.equal(validateAuthForm({ mode: 'login', email: 'a@b.co', password: 'x' }), null);   // login does not judge the length
  assert.match(validateAuthForm({ mode: 'login', email: 'a@b.co', password: '' }), /password/);
});

test('redirects stay inside the app', () => {
  assert.equal(safeRedirect('/dashboard'), '/dashboard');
  assert.equal(safeRedirect('/about'), '/about');
  assert.equal(safeRedirect('//evil.example'), '/dashboard');
  assert.equal(safeRedirect('https://evil.example'), '/dashboard');
  assert.equal(safeRedirect(undefined), '/dashboard');
});
