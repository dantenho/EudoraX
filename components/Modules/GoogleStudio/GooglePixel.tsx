
import React, { useState } from 'react';
import { Icons } from '../../../Icons';
import { Button } from '../../../Button';

export const GooglePixel: React.FC = () => {
  const [prompt, setPrompt] = useState('');

  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col items-center justify-center p-10">
         <div className="w-64 h-64 border border-zinc-800 rounded-lg flex items-center justify-center relative opacity-20">
             <Icons.Gamepad className="w-16 h-16" />
         </div>
         <p className="mt-8 text-xs font-black text-zinc-600 uppercase tracking-widest">Pixel Forge Standby</p>
      </div>
      <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A retro sprite..."
              className="flex-1 h-20 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none"
          />
          <Button className="h-14 w-40 bg-blue-600" icon={Icons.Zap}>Forge</Button>
      </div>
    </div>
  );
};
