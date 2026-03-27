
import React, { useState } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';

export const GeminiAuth: React.FC<{ onLinked: (user: any) => void }> = ({ onLinked }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
    const body = isLogin ? { email, password } : { email, password, name };

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Authentication failed');

      localStorage.setItem('eudora_token', data.token);
      localStorage.setItem('eudora_user', JSON.stringify(data.user));
      onLinked(data.user);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto py-12 animate-fade-in space-y-8">
      <div className="w-20 h-20 bg-blue-600/10 border border-blue-500/20 rounded-3xl flex items-center justify-center mx-auto text-blue-400">
        <Icons.Key className="w-10 h-10" />
      </div>
      
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-black text-white tracking-tighter uppercase">
          {isLogin ? 'Neural Identity' : 'Create Core ID'}
        </h1>
        <p className="text-zinc-500 text-xs">
          {isLogin ? 'Authenticate to access the Synthesis Forge.' : 'Register your neural signature in the EudoraX network.'}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 bg-black/40 p-8 rounded-[2rem] border border-white/5 shadow-2xl">
        {!isLogin && (
          <div className="space-y-1">
            <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Full Name</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
              placeholder="John Doe"
            />
          </div>
        )}
        
        <div className="space-y-1">
          <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Email Address</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
            placeholder="pilot@eudorax.io"
          />
        </div>

        <div className="space-y-1">
          <label className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Access Key</label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-dark-card border border-dark-border rounded-xl px-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
            placeholder="••••••••"
          />
        </div>

        {error && <p className="text-[10px] text-red-400 font-bold uppercase tracking-wider text-center">{error}</p>}

        <Button 
          type="submit" 
          loading={loading} 
          className="w-full bg-blue-600 shadow-xl mt-4" 
          icon={isLogin ? Icons.Logout : Icons.Check}
        >
          {isLogin ? 'Sync Identity' : 'Initialize ID'}
        </Button>

        <div className="pt-4 text-center">
          <button 
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-[10px] font-black text-zinc-500 uppercase tracking-widest hover:text-blue-400 transition-colors"
          >
            {isLogin ? "Don't have an ID? Register" : "Already registered? Login"}
          </button>
        </div>
      </form>
    </div>
  );
};
