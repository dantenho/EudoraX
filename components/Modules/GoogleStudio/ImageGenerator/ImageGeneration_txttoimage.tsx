
import React, { useState } from 'react';
import { Icons } from '../Icons';
import { Button } from '../Button';
import { synthesizeImage } from '../../services/geminiService';
import { LoRAStyle } from '../../types';
import { LORA_STYLES } from './imagegeneratorfilterfunction';

export const TxtToImageExtension: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [useThinking, setUseThinking] = useState(false);
  const [useSearch, setUseSearch] = useState(false);
  const [refImage, setRefImage] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<LoRAStyle>(LORA_STYLES[0]);

  const handleSynthesize = async () => {
    setLoading(true);
    try {
      const finalPrompt = selectedStyle.id !== 'none' 
        ? `${prompt}. Follow style: ${selectedStyle.prompts?.positive}` 
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
      console.error(e);
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
    <div className="flex flex-col h-full space-y-4">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-3xl p-6 flex flex-col gap-6 overflow-hidden">
        {/* Workspace Display */}
        <div className="flex-1 bg-[#050507] rounded-2xl flex items-center justify-center relative overflow-hidden group">
          {loading ? (
            <div className="flex flex-col items-center gap-4">
              <div className="w-12 h-12 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-[10px] font-black uppercase tracking-widest text-blue-400 animate-pulse">
                {useThinking ? 'Deep Reasoning in Progress...' : 'Neural Forge Active'}
              </p>
            </div>
          ) : result ? (
            <img src={result} className="max-h-full max-w-full object-contain rounded-lg shadow-2xl animate-fade-in" alt="Synthesis" />
          ) : (
            <div className="text-zinc-800 flex flex-col items-center gap-4 opacity-20">
              <Icons.Image className="w-24 h-24" />
              <p className="text-xs font-black uppercase tracking-[0.5em]">Forge Standby</p>
            </div>
          )}
          
          {/* Reference Overlay */}
          {refImage && (
            <div className="absolute top-4 left-4 w-20 h-20 rounded-lg border border-blue-500/50 overflow-hidden shadow-xl bg-black group-hover:w-40 group-hover:h-40 transition-all duration-500">
              <img src={refImage} className="w-full h-full object-cover" alt="Ref" />
              <div className="absolute top-0 right-0 p-1 bg-black/60 cursor-pointer" onClick={() => setRefImage(null)}>
                <Icons.Plus className="w-3 h-3 rotate-45" />
              </div>
            </div>
          )}
        </div>

        {/* Controls Overlay */}
        <div className="flex gap-4 items-end">
          <div className="flex-1 space-y-4">
            <div className="flex gap-4">
              <label className={`flex-1 flex items-center gap-3 p-3 rounded-xl border transition-all cursor-pointer ${useThinking ? 'bg-purple-500/10 border-purple-500/30' : 'bg-white/5 border-transparent opacity-60'}`}>
                <input type="checkbox" className="hidden" checked={useThinking} onChange={() => setUseThinking(!useThinking)} />
                <Icons.Brain className={`w-4 h-4 ${useThinking ? 'text-purple-400' : 'text-zinc-500'}`} />
                <span className="text-[10px] font-bold text-zinc-300 uppercase">Think</span>
              </label>
              <label className={`flex-1 flex items-center gap-3 p-3 rounded-xl border transition-all cursor-pointer ${useSearch ? 'bg-blue-500/10 border-blue-500/30' : 'bg-white/5 border-transparent opacity-60'}`}>
                <input type="checkbox" className="hidden" checked={useSearch} onChange={() => setUseSearch(!useSearch)} />
                <Icons.Search className={`w-4 h-4 ${useSearch ? 'text-blue-400' : 'text-zinc-500'}`} />
                <span className="text-[10px] font-bold text-zinc-300 uppercase">Web Search</span>
              </label>
              <label className={`flex-1 flex items-center gap-3 p-3 rounded-xl border bg-white/5 border-transparent opacity-60 hover:opacity-100 transition-all cursor-pointer`}>
                <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
                <Icons.Upload className="w-4 h-4 text-zinc-500" />
                <span className="text-[10px] font-bold text-zinc-300 uppercase">AssetFollow</span>
              </label>
            </div>
            
            <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
              {LORA_STYLES.map(style => (
                <button
                  key={style.id}
                  onClick={() => setSelectedStyle(style)}
                  className={`px-3 py-1.5 rounded-lg text-[9px] font-bold uppercase whitespace-nowrap transition-all border ${selectedStyle.id === style.id ? 'bg-blue-600 border-blue-500 text-white' : 'bg-white/5 border-white/5 text-zinc-500 hover:bg-white/10'}`}
                >
                  {style.name}
                </button>
              ))}
            </div>
            
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-24 bg-black/40 border border-white/5 rounded-2xl p-4 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all resize-none"
              placeholder="Enter your neural command..."
            />
          </div>
          <Button loading={loading} onClick={handleSynthesize} className="h-24 w-40 bg-blue-600 shadow-[0_0_20px_rgba(37,99,235,0.3)]" icon={Icons.Zap}>Synthesize</Button>
        </div>
      </div>
    </div>
  );
};
