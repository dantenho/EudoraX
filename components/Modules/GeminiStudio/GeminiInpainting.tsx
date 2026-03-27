
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';

export const GeminiInpainting: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleExecute = async () => {
      if (!image || !prompt) return;
      setLoading(true);
      try {
          // In a real app, you would draw on a canvas to generate a mask
          // Here we assume full image context for semantic inpainting
          const response = await GeminiPlugin.executeSynthesis(prompt, 'INPAINTING', {
              inputImage: image,
              maskImage: image, // Simulating a mask
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
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] flex flex-col items-center justify-center p-10 relative overflow-hidden group">
          {image && !result && (
              <div className="absolute top-6 left-6 flex gap-2 z-10">
                <button className="p-3 bg-blue-600 rounded-xl text-white shadow-lg"><Icons.Erase className="w-5 h-5" /></button>
                <button className="p-3 bg-zinc-800 rounded-xl text-zinc-400 hover:text-white"><Icons.MousePointer2 className="w-5 h-5" /></button>
              </div>
          )}
          
          {loading ? (
             <div className="text-center">
                 <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4 mx-auto" />
                 <span className="text-xs font-black uppercase tracking-widest text-emerald-500">Inpainting Latents...</span>
             </div>
          ) : result ? (
             <img src={result} className="w-full h-full object-contain rounded-xl shadow-2xl animate-fade-in" alt="Inpainted" />
          ) : image ? (
             <div className="relative w-full h-full">
                <img src={image} className="w-full h-full object-contain rounded-xl opacity-60" alt="Canvas" />
                <div className="absolute inset-0 flex items-center justify-center">
                    <p className="text-[10px] bg-black/80 px-4 py-2 rounded-full text-white uppercase tracking-widest">Mock Masking Active</p>
                </div>
             </div>
          ) : (
             <label className="text-center opacity-40 hover:opacity-100 transition-opacity cursor-pointer">
                 <Icons.Erase className="w-24 h-24 mx-auto mb-6 text-zinc-600" />
                 <p className="text-sm font-black uppercase tracking-[0.5em] text-zinc-500">Inpaint Canvas Standby</p>
                 <p className="text-[10px] mt-2 font-mono text-zinc-600">Click to Upload Base Image</p>
                 <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
             </label>
          )}
      </div>

      <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
             <div className="flex justify-between">
               <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Inpaint Prompt</label>
               <span className="text-[10px] text-blue-400 font-mono font-bold">MODE: FILL_GENERATIVE</span>
             </div>
             <input 
               type="text" 
               value={prompt}
               onChange={(e) => setPrompt(e.target.value)}
               placeholder="What should fill the masked area?" 
               className="w-full bg-dark-bg/60 border border-white/5 rounded-2xl h-14 px-6 text-sm text-white focus:ring-1 focus:ring-blue-500/50 outline-none"
             />
          </div>
          <Button 
            onClick={handleExecute} 
            loading={loading}
            disabled={!image}
            className="w-48 h-14 bg-blue-600" 
            icon={Icons.Wand2}
          >
            Smart Fill
          </Button>
      </div>
    </div>
  );
};
