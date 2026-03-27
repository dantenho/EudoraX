
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';

export const GeminiImg2Img: React.FC = () => {
  const [sourceImage, setSourceImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [denoise, setDenoise] = useState(0.75);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setSourceImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleExecute = async () => {
    if (!sourceImage || !prompt) return;
    setLoading(true);
    try {
        const response = await GeminiPlugin.executeSynthesis(prompt, 'IMG_TO_IMG', {
            inputImage: sourceImage,
            // Passing denoise strength as a control parameter
            controlMode: 'canny' 
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
      <div className="flex-1 grid grid-cols-2 gap-6">
        {/* Source Panel */}
        <div className="bg-black/40 border border-white/5 rounded-[2rem] p-6 flex flex-col items-center justify-center relative overflow-hidden group">
          {sourceImage ? (
             <div className="relative w-full h-full">
                <img src={sourceImage} className="w-full h-full object-contain rounded-xl" alt="Source" />
                <button 
                    onClick={() => setSourceImage(null)}
                    className="absolute top-2 right-2 p-2 bg-black/60 rounded-full hover:bg-red-500/80 transition-colors"
                >
                    <Icons.Erase className="w-4 h-4 text-white" />
                </button>
             </div>
          ) : (
             <label className="cursor-pointer flex flex-col items-center gap-4 text-zinc-600 hover:text-blue-400 transition-colors">
                <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center border border-white/10 group-hover:scale-110 transition-transform">
                    <Icons.Upload className="w-8 h-8" />
                </div>
                <span className="text-xs font-black uppercase tracking-widest">Upload Source Image</span>
                <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
             </label>
          )}
        </div>

        {/* Result Panel */}
        <div className="bg-black/40 border border-white/5 rounded-[2rem] p-6 flex flex-col items-center justify-center relative overflow-hidden">
          {loading ? (
              <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs font-black uppercase tracking-widest text-blue-400 animate-pulse">Re-imagining...</span>
              </div>
          ) : result ? (
              <img src={result} className="w-full h-full object-contain rounded-xl animate-scale-in" alt="Result" />
          ) : (
              <div className="text-center opacity-10">
                <Icons.Image className="w-16 h-16 mx-auto mb-4" />
                <p className="text-xs font-black uppercase tracking-widest">Variation Output</p>
              </div>
          )}
        </div>
      </div>

      <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
            <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Variation Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe changes (e.g. 'make it cybernetic, sunset lighting')..."
              className="w-full h-24 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none focus:ring-1 focus:ring-blue-500/50 transition-all"
            />
          </div>
          <div className="w-64 space-y-4">
             <div className="space-y-2">
                <div className="flex items-center justify-between text-[10px] text-zinc-500 font-bold px-1">
                    <span>DENOISING STRENGTH</span>
                    <span className="text-blue-400">{denoise.toFixed(2)}</span>
                </div>
                <input 
                    type="range" 
                    min="0.1" 
                    max="1.0" 
                    step="0.05" 
                    value={denoise}
                    onChange={(e) => setDenoise(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
             </div>
             <Button 
                onClick={handleExecute} 
                loading={loading}
                disabled={!sourceImage}
                className="w-full h-14 bg-blue-600 rounded-xl" 
                icon={Icons.Zap}
             >
                Generate Variation
             </Button>
          </div>
      </div>
    </div>
  );
};
