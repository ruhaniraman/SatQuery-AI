import { createContext, useContext } from 'react';

// Who is signed in. Kept apart from the provider component so that file only exports a component.
export const AuthContext = createContext(null);

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>.');
  return ctx;
}
