
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { synthesizeImage } from '../../../services/geminiService.ts';
import { LoRAStyle } from '../../../types.ts';
import { LORA_STYLES } from './imagegeneratorfilterfunction.tsx';

export const TxtToImageExtension: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [useThinking, setUseThinking] = useState(false);
  const [useSearch, setUseSearch] = useState(false);
  const [refImage, setRefImage] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<LoRAStyle>(LORA_STYLES[0]);

  const handleSynthesize = async () => {
    if (!prompt && !refImage) return;
    setLoading(true);
    try {
      const finalPrompt = selectedStyle.id !== 'none' 
        ? `${prompt}. Style influence: ${selectedStyle.prompts?.positive}` 
        : prompt;

      const { imageUrl } = await synthesizeImage({
        prompt: finalPrompt,
        aspectRatio: '1:1',
        useThinking,
        useSearch,
        referenceImage: refImage || undefined
      });
      if (imageUrl) setResult(imageUrl);
    } catch (e) {
      console.error("[SYNTHESIS_ERROR]", e);
    } finally {
      setLoading(false);
    }
  };

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setRefImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4 animate-fade-in">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col gap-8 overflow-hidden">
        {/* Workspace Display */}
        <div className="flex-1 bg-[#050507] rounded-3xl flex items-center justify-center relative overflow-hidden group border border-white/5 shadow-inner">
          {loading ? (
            <div className="flex flex-col items-center gap-6">
              <div className="w-14 h-14 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div className="text-center space-y-2">
                <p className="text-[10px] font-black uppercase tracking-[0.3em] text-blue-400 animate-pulse">
                  {useThinking ? 'Deep Neural Reasoning...' : 'Synthesizing Latents...'}
                </p>
                {useSearch && <p className="text-[8px] font-mono text-zinc-600 uppercase">Search Grounding: Active</p>}
              </div>
            </div>
          ) : result ? (
            <img src={result} className="max-h-full max-w-full object-contain rounded-xl shadow-2xl animate-fade-in transition-transform duration-700 hover:scale-[1.02]" alt="Synthesis Output" />
          ) : (
            <div className="text-zinc-800 flex flex-col items-center gap-6 opacity-[0.05]">
              <Icons.Sparkles className="w-32 h-32" />
              <p className="text-sm font-black uppercase tracking-[0.6em]">Neural Forge Ready</p>
            </div>
          )}
          
          {/* Reference Overlay */}
          {refImage && (
            <div className="absolute top-6 left-6 w-24 h-24 rounded-2xl border border-blue-500/40 overflow-hidden shadow-2xl bg-black transition-all duration-500 hover:w-48 hover:h-48">
              <img src={refImage} className="w-full h-full object-cover" alt="Source Reference" />
              <button 
                className="absolute top-2 right-2 p-1.5 bg-black/60 rounded-lg hover:bg-red-600/40 transition-colors"
                onClick={() => setRefImage(null)}
              >
                <Icons.Plus className="w-3 h-3 rotate-45 text-white" />
              </button>
            </div>
          )}
        </div>

        {/* Inputs */}
        <div className="flex gap-6 items-end">
          <div className="flex-1 space-y-6">
            <div className="flex gap-4">
              <label className={`flex-1 flex items-center justify-center gap-3 p-4 rounded-2xl border transition-all cursor-pointer ${useThinking ? 'bg-blue-600/10 border-blue-500/40' : 'bg-white/5 border-transparent opacity-40 hover:opacity-100'}`}>
                <input type="checkbox" className="hidden" checked={useThinking} onChange={() => setUseThinking(!useThinking)} />
                <Icons.Brain className={`w-4 h-4 ${useThinking ? 'text-blue-400' : 'text-zinc-500'}`} />
                <span className="text-[10px] font-black text-zinc-300 uppercase tracking-widest">Thinking</span>
              </label>
              <label className={`flex-1 flex items-center justify-center gap-3 p-4 rounded-2xl border transition-all cursor-pointer ${useSearch ? 'bg-emerald-600/10 border-emerald-500/40' : 'bg-white/5 border-transparent opacity-40 hover:opacity-100'}`}>
                <input type="checkbox" className="hidden" checked={useSearch} onChange={() => setUseSearch(!useSearch)} />
                <Icons.Search className={`w-4 h-4 ${useSearch ? 'text-emerald-400' : 'text-zinc-500'}`} />
                <span className="text-[10px] font-black text-zinc-300 uppercase tracking-widest">Grounding</span>
              </label>
              <label className={`flex-1 flex items-center justify-center gap-3 p-4 rounded-2xl border bg-white/5 border-transparent opacity-40 hover:opacity-100 transition-all cursor-pointer`}>
                <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
                <Icons.Upload className="w-4 h-4 text-zinc-500" />
                <span className="text-[10px] font-black text-zinc-300 uppercase tracking-widest">Ref Asset</span>
              </label>
            </div>
            
            <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
              {LORA_STYLES.map(style => (
                <button
                  key={style.id}
                  onClick={() => setSelectedStyle(style)}
                  className={`px-3 py-1.5 rounded-xl text-[9px] font-black uppercase whitespace-nowrap transition-all border ${selectedStyle.id === style.id ? 'bg-blue-600 border-blue-500 text-white' : 'bg-white/5 border-white/5 text-zinc-500 hover:bg-white/10'}`}
                >
                  {style.name}
                </button>
              ))}
            </div>
            
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-28 bg-black/60 border border-white/5 rounded-3xl p-6 text-sm text-white focus:ring-1 focus:ring-blue-500/50 transition-all resize-none shadow-inner custom-scrollbar"
              placeholder="Input neural synthesis command (e.g., 'A cyberpunk skyline at dusk, 4k, hyper-detailed')..."
            />
          </div>
          <Button 
            loading={loading} 
            onClick={handleSynthesize} 
            className="h-28 w-48 bg-blue-600 rounded-3xl shadow-[0_0_30px_rgba(37,99,235,0.2)] hover:shadow-[0_0_40px_rgba(37,99,235,0.4)]" 
            icon={Icons.Zap}
          >
            Forge
          </Button>
        </div>
      </div>
    </div>
  );
};
