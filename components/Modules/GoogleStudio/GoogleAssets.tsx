
import React from 'react';
import { Icons } from '../../../Icons';

export const GoogleAssets: React.FC = () => {
  return (
    <div className="flex h-full gap-8">
      <div className="flex-1 flex flex-col space-y-6">
        <div className="flex items-center justify-between px-4">
           <div className="flex items-center gap-2">
              <Icons.Layers className="w-4 h-4 text-blue-500" />
              <h3 className="text-xs font-black text-white uppercase tracking-widest">Global Asset Vault</h3>
           </div>
        </div>
        <div className="flex-1 grid grid-cols-4 gap-6 overflow-y-auto pr-4">
          {[1, 2, 3, 4, 5, 6, 7, 8].map(i => (
            <div key={i} className="group relative aspect-[4/5] bg-dark-card border border-white/5 rounded-3xl overflow-hidden hover:border-blue-500/30 transition-all cursor-pointer">
               <div className="absolute inset-0 bg-zinc-900 flex items-center justify-center">
                  <Icons.Image className="w-8 h-8 text-zinc-800" />
               </div>
               <div className="absolute inset-x-0 bottom-0 p-4 bg-gradient-to-t from-black to-transparent">
                  <div className="text-[8px] font-black text-blue-400 uppercase">ASSET_{i}</div>
               </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
