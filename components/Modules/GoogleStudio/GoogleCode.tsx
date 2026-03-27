
import React from 'react';
import { Icons } from '../../../Icons';
import { Button } from '../../../Button';

export const GoogleCode: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-6 font-mono">
      <div className="flex-1 bg-[#1e1e1e] border border-white/5 rounded-[1rem] overflow-hidden flex flex-col">
        <div className="h-8 bg-[#252526] border-b border-black flex items-center px-4 text-xs text-zinc-400">
          <Icons.Code className="w-3 h-3 mr-2" />
          <span>forge_v1.ts</span>
        </div>
        <div className="flex-1 p-10 flex items-center justify-center opacity-20">
          <div className="text-center">
            <Icons.Terminal className="w-24 h-24 mx-auto mb-4" />
            <p className="text-xs font-black uppercase tracking-widest">Native Code Forge Environment</p>
          </div>
        </div>
        <div className="h-8 bg-blue-600 text-white flex items-center px-4 text-xs font-bold">
           <span>-- NORMAL --</span>
        </div>
      </div>
      <div className="flex gap-4 p-4 bg-dark-sidebar/20 border border-white/5 rounded-2xl items-center">
         <Button size="sm" variant="primary" icon={Icons.Play}>Execute</Button>
      </div>
    </div>
  );
};
