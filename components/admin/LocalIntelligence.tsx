
import React, { useEffect, useState } from 'react';
import { Icons } from '../Icons';
import { checkSystemCapabilities, SystemCapability } from '../../services/chromeAiService';

export const LocalIntelligence: React.FC = () => {
  const [capabilities, setCapabilities] = useState<SystemCapability[]>([]);
  const [nodeStatus] = useState<{version: string, v8: string, status: string}>({
      version: 'Node.js 25.0.0-nightly',
      v8: 'V8 14.2 (Turbofan)',
      status: 'CONNECTED'
  });

  useEffect(() => {
    const fetchCapabilities = async () => {
      const caps = await checkSystemCapabilities();
      setCapabilities(caps);
    };
    fetchCapabilities();
  }, []);

  return (
    <div className="bg-dark-card border border-dark-border rounded-xl p-6 animate-fade-in">
      <div className="flex justify-between items-start mb-6">
          <div>
            <h3 className="text-lg font-medium text-white flex items-center gap-2">
                <Icons.Chip className="w-5 h-5 text-orange-500" /> 
                Local Intelligence & Runtime
            </h3>
            <p className="text-xs text-zinc-400">
                Monitor Chrome Canary (Client) and Node.js 25 (BFF) capabilities.
            </p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-green-900/10 border border-green-500/20 rounded-lg">
             <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
             <span className="text-[10px] font-mono font-bold text-green-400">NODE_25_ACTIVE</span>
          </div>
      </div>
      
      {/* Node.js 25 Runtime Block */}
      <div className="mb-6 bg-black/30 rounded-xl p-4 border border-green-500/10 relative overflow-hidden group">
         <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Icons.Server className="w-16 h-16 text-green-500" />
         </div>
         <h4 className="text-xs font-bold text-zinc-300 uppercase tracking-widest mb-3">Middleware Runtime (BFF)</h4>
         <div className="grid grid-cols-3 gap-4">
             <div>
                 <p className="text-[10px] text-zinc-500">Engine Version</p>
                 <p className="text-sm font-mono text-green-400">{nodeStatus.version}</p>
             </div>
             <div>
                 <p className="text-[10px] text-zinc-500">JS VM</p>
                 <p className="text-sm font-mono text-zinc-300">{nodeStatus.v8}</p>
             </div>
             <div>
                 <p className="text-[10px] text-zinc-500">Feature Flags</p>
                 <p className="text-[10px] font-mono text-zinc-400 mt-0.5">--experimental-strip-types</p>
             </div>
         </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {capabilities.map(cap => (
          <div key={cap.id} className="flex items-center justify-between p-3 bg-black/20 rounded-lg border border-white/5 group hover:border-white/10 transition-colors">
            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${cap.available ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-zinc-600'}`}></div>
              <div className="overflow-hidden">
                <p className="text-sm font-medium text-zinc-200 truncate">{cap.name}</p>
                <p className="text-[10px] text-zinc-500 truncate">{cap.description}</p>
              </div>
            </div>
            <div className="text-right flex-shrink-0 ml-2">
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${cap.available ? 'bg-green-900/20 text-green-400 border-green-500/20' : 'bg-red-900/10 text-red-400 border-red-500/10'}`}>
                {cap.available ? 'ACTIVE' : 'OFFLINE'}
              </span>
              {cap.details && <p className="text-[9px] text-zinc-600 mt-1">{cap.details}</p>}
            </div>
          </div>
        ))}
      </div>

      <div className="pt-4 mt-4 border-t border-white/5">
        <p className="text-[10px] text-zinc-500 bg-orange-900/10 p-2 rounded border border-orange-500/10 flex items-start gap-2">
          <Icons.Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <span>
            System running in hybrid mode: <strong>Python 3.14 (Backend)</strong> + <strong>Node.js 25 (Middleware)</strong>. Ensure `node:ai` bindings are active.
          </span>
        </p>
      </div>
    </div>
  );
};
