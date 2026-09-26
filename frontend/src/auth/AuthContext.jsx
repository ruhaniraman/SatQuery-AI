import { useCallback, useEffect, useMemo, useState } from 'react';
import { clearToken, readToken, requestLogout, requestMe, saveToken } from '../dashboard/utils/auth';
import { AuthContext } from './authState';

// `status`: 'loading' (checking a stored token), 'anon', 'authed', or 'offline' (a token is stored but the server
// could not be reached, so we neither sign the person out nor let them in).
async function resolveSession() {
  const token = readToken();
  if (!token) return { status: 'anon', user: null };
  try {
    const user = await requestMe(token);
    if (user) return { status: 'authed', user };
    clearToken();
    return { status: 'anon', user: null };
  } catch {
    return { status: 'offline', user: null };
  }
}

export function AuthProvider({ children }) {
  const [state, setState] = useState(() => (readToken() ? { status: 'loading', user: null } : { status: 'anon', user: null }));

  useEffect(() => {
    if (!readToken()) return undefined;
    let cancelled = false;
    resolveSession().then((next) => { if (!cancelled) setState(next); });
    return () => { cancelled = true; };
  }, []);

  const recheck = useCallback(async () => {
    setState({ status: 'loading', user: null });
    setState(await resolveSession());
  }, []);

  const signIn = useCallback(({ token, user }) => {
    saveToken(token);
    setState({ status: 'authed', user });
  }, []);

  const signOut = useCallback(async () => {
    const token = readToken();
    clearToken();
    setState({ status: 'anon', user: null });
    if (token) { try { await requestLogout(token); } catch { /* the token is already forgotten here; it expires on the server */ } }
  }, []);

  const value = useMemo(() => ({ ...state, signIn, signOut, recheck }), [state, signIn, signOut, recheck]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
