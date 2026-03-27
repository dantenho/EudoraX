
/**
 * @file GeminiPixel.tsx
 * @description Optimized pixel art synthesis using Gemini 2.5 Image kernels.
 * @backend Python 3.14 (Numba / OpenCV Pixelation Kernels)
 * @jules_hint Apply post-process pixel-snapping in Python to ensure 8-bit grid alignment.
 */
import React, { useState } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';

export const GeminiPixel: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [gridSize, setGridSize] = useState<'32x32' | '64x64' | '128x128'>('64x64');

  return (
    <div className="flex flex-col h-full space-y-6 animate-fade-in">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl relative">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-10 relative">
          <div className="w-64 h-64 border border-zinc-800 rounded-lg flex items-center justify-center relative group">
             <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'linear-gradient(to right, #444 1px, transparent 1px), linear-gradient(to bottom, #444 1px, transparent 1px)', backgroundSize: '16px 16px' }}></div>
             <Icons.Gamepad className="w-16 h-16 text-zinc-800 group-hover:text-blue-500/20 transition-colors" />
             <div className="absolute -bottom-6 left-0 right-0 text-center">
                <span className="text-[8px] font-mono text-zinc-600 uppercase tracking-widest">Renderer: PixelForge-v1</span>
             </div>
          </div>
        </div>

        <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
             <div className="flex justify-between items-center">
                <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Sprite Command</label>
                <div className="flex gap-2">
                   {['32x32', '64x64', '128x128'].map(g => (
                     <button 
                        key={g} 
                        onClick={() => setGridSize(g as any)}
                        className={`text-[8px] font-bold px-2 py-1 rounded border transition-all ${gridSize === g ? 'bg-blue-500/20 border-blue-500/40 text-blue-400' : 'bg-transparent border-white/5 text-zinc-600'}`}
                      >
                        {g}
                      </button>
                   ))}
                </div>
             </div>
             <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. A cybernetic owl sprite, 16-bit color palette, top-down view..."
              className="w-full h-20 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none"
            />
          </div>
          <Button className="h-14 w-40 bg-blue-600" icon={Icons.Zap}>Forge Pixel</Button>
        </div>
      </div>
    </div>
  );
};
