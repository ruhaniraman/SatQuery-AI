import { LogOut } from 'lucide-react';
import { useAuth } from '../../auth/authState';

// Who is signed in, with a sign-out button, for the dashboard header. Draws nothing when nobody is signed in.
export default function UserMenu() {
  const { user, signOut } = useAuth();
  if (!user) return null;
  const initial = (user.name || user.email || '?').trim().charAt(0).toUpperCase();
  return (
    <div className="flex items-center gap-2">
      <span className="hidden items-center gap-2 sm:inline-flex" title={user.email}>
        <span aria-hidden="true" className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-blue-600/90 text-xs font-bold text-slate-950">{initial}</span>
        <span className="max-w-[10rem] truncate text-xs text-slate-300">{user.name || user.email}</span>
      </span>
      <button
        type="button"
        onClick={signOut}
        aria-label="Sign out"
        title="Sign out"
        className="inline-flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/70"
      >
        <LogOut size={15} />
      </button>
    </div>
  );
}
