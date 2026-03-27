import React, { useState, useEffect } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';

export const GeminiProfile: React.FC = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const token = localStorage.getItem('eudora_token');
      const response = await fetch('/api/auth/profile', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setUser(data);
        setName(data.name);
        setEmail(data.email);
      }
    } catch (err) {
      console.error("Failed to fetch profile", err);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    setUpdating(true);
    setMessage(null);
    setError(null);

    try {
      const token = localStorage.getItem('eudora_token');
      const response = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ name, email, password })
      });
      const data = await response.json();
      if (response.ok) {
        setMessage("Profile updated successfully");
        localStorage.setItem('eudora_user', JSON.stringify(data.user));
        setPassword('');
      } else {
        setError(data.error);
      }
    } catch (_err) {
      setError("Failed to update profile");
    } finally {
      setUpdating(false);
    }
  };

  const handleVerify = async () => {
    try {
      const token = localStorage.getItem('eudora_token');
      const response = await fetch('/api/auth/verify-email', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      if (response.ok) {
        setMessage(data.message);
        fetchProfile();
      }
    } catch (_err) {
      setError("Verification failed");
    }
  };

  if (loading) return <div className="flex items-center justify-center h-full"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div></div>;

  return (
    <div className="max-w-2xl mx-auto py-12 animate-fade-in space-y-12">
      <div className="flex items-center gap-6">
        <div className="w-24 h-24 rounded-[2.5rem] bg-blue-600/20 border border-blue-500/20 flex items-center justify-center text-blue-400 text-4xl font-black">
          {user?.name?.charAt(0)}
        </div>
        <div className="space-y-1">
          <h1 className="text-4xl font-black text-white tracking-tighter uppercase">{user?.name}</h1>
          <div className="flex items-center gap-2">
            <span className="text-zinc-500 text-sm">{user?.email}</span>
            {user?.isVerified ? (
              <span className="px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[9px] font-black text-emerald-400 uppercase tracking-widest">Verified</span>
            ) : (
              <button 
                onClick={handleVerify}
                className="px-2 py-0.5 bg-amber-500/10 border border-amber-500/20 rounded-full text-[9px] font-black text-amber-400 uppercase tracking-widest hover:bg-amber-500/20 transition-colors"
              >
                Verify Email
              </button>
            )}
          </div>
        </div>
      </div>

      <form onSubmit={handleUpdate} className="space-y-8 bg-black/40 p-10 rounded-[3rem] border border-white/5 shadow-2xl">
        <h2 className="text-xs font-black text-zinc-500 uppercase tracking-[0.3em]">Profile Settings</h2>
        
        {message && <p className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider text-center bg-emerald-500/10 py-2 rounded-lg border border-emerald-500/20">{message}</p>}
        {error && <p className="text-[10px] text-red-400 font-bold uppercase tracking-wider text-center bg-red-500/10 py-2 rounded-lg border border-red-500/20">{error}</p>}

        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-1">
            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Display Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Email Address</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">New Access Key (Optional)</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
            placeholder="Leave blank to keep current"
          />
        </div>

        <Button 
          type="submit" 
          loading={updating} 
          className="w-full bg-blue-600 shadow-xl" 
          icon={Icons.Check}
        >
          Save Changes
        </Button>
      </form>
    </div>
  );
};
