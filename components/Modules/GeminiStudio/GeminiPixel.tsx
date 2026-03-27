
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';

export const GeminiPixel: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [gridSize, setGridSize] = useState<'32x32' | '64x64' | '128x128'>('64x64');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleExecute = async () => {
    if (!prompt) return;
    setLoading(true);
    try {
        const response = await GeminiPlugin.executeSynthesis(prompt, 'PIXEL_FORGE', {
            // Passing grid size as prompt context or config
            prompt: `${prompt}, ${gridSize} pixel art style` 
        });
        if (response.url) setResult(response.url);
    } catch (e) {
        console.error(e);
    } finally {
        setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6 animate-fade-in">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl relative">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-10 relative">
          
          {loading ? (
             <div className="flex flex-col items-center gap-4">
                 <div className="w-12 h-12 border-4 border-dashed border-purple-500 rounded animate-spin" />
                 <span className="text-xs font-mono text-purple-400">PIXELATING_KERNELS...</span>
             </div>
          ) : result ? (
             <img src={result} className="h-64 w-64 object-contain image-pixelated shadow-[0_0_30px_rgba(168,85,247,0.4)]" style={{ imageRendering: 'pixelated' }} alt="Pixel Art" />
          ) : (
            <div className="w-64 h-64 border border-zinc-800 rounded-lg flex items-center justify-center relative group">
                <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'linear-gradient(to right, #444 1px, transparent 1px), linear-gradient(to bottom, #444 1px, transparent 1px)', backgroundSize: '16px 16px' }}></div>
                <Icons.Gamepad className="w-16 h-16 text-zinc-800 group-hover:text-blue-500/20 transition-colors" />
                <div className="absolute -bottom-6 left-0 right-0 text-center">
                    <span className="text-[8px] font-mono text-zinc-600 uppercase tracking-widest">Renderer: PixelForge-v1</span>
                </div>
            </div>
          )}
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
                        className={`text-[8px] font-bold px-2 py-1 rounded border transition-all ${gridSize === g ? 'bg-purple-500/20 border-purple-500/40 text-purple-400' : 'bg-transparent border-white/5 text-zinc-600'}`}
                      >
                        {g}
                      </button>
                   ))}
                </div>
             </div>
             <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. A cybernetic owl sprite..."
              className="w-full h-20 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none focus:ring-1 focus:ring-purple-500/50"
            />
          </div>
          <Button onClick={handleExecute} loading={loading} className="h-14 w-40 bg-purple-600 hover:bg-purple-700" icon={Icons.Zap}>Forge Pixel</Button>
        </div>
      </div>
    </div>
  );
};
