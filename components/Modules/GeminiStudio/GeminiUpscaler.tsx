
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';

export const GeminiUpscaler: React.FC = () => {
  const [scale, setScale] = useState(4);
  const [image, setImage] = useState<string | null>(null);
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
      if (!image) return;
      setLoading(true);
      try {
          const response = await GeminiPlugin.executeSynthesis("Upscale this image", 'UPSCALER', {
              inputImage: image,
              upscaleFactor: scale
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
         <div className="bg-black/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col justify-center items-center relative overflow-hidden group">
            {image ? (
                <div className="relative w-full h-full">
                    <img src={image} className="w-full h-full object-contain rounded-xl" alt="Original" />
                    <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded-md text-[9px] font-bold text-zinc-400">ORIGINAL</div>
                </div>
            ) : (
                <label className="w-full h-full border-2 border-dashed border-zinc-800 rounded-2xl flex items-center justify-center text-zinc-600 hover:text-blue-500 hover:border-blue-500/30 transition-all cursor-pointer">
                    <div className="text-center">
                        <Icons.Upload className="w-10 h-10 mx-auto mb-2" />
                        <span className="text-xs font-bold uppercase">Drop Image</span>
                    </div>
                    <input type="file" className="hidden" onChange={handleFile} accept="image/*" />
                </label>
            )}
         </div>

         <div className="bg-black/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col justify-center items-center relative overflow-hidden">
             <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5" />
             {loading ? (
                 <div className="text-center z-10">
                     <div className="w-12 h-12 border-4 border-white border-t-transparent rounded-full animate-spin mb-4 mx-auto" />
                     <p className="text-xs font-black uppercase tracking-widest text-white animate-pulse">Running Real-ESRGAN...</p>
                 </div>
             ) : result ? (
                 <div className="relative w-full h-full z-10 animate-fade-in">
                     <img src={result} className="w-full h-full object-contain rounded-xl" alt="Upscaled" />
                     <div className="absolute top-2 right-2 px-2 py-1 bg-blue-600/80 rounded-md text-[9px] font-bold text-white shadow-lg">{scale}X ENHANCED</div>
                 </div>
             ) : (
                 <>
                    <Icons.Maximize2 className="w-20 h-20 text-blue-500/20 mb-4" />
                    <p className="text-xs font-black uppercase tracking-widest text-zinc-600">Enhanced Output</p>
                 </>
             )}
         </div>
      </div>

      <div className="h-32 bg-dark-sidebar/20 border-t border-white/5 flex items-center px-10 gap-10">
         <div className="flex items-center gap-4">
            <span className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Upscale Factor</span>
            <div className="flex bg-black rounded-xl p-1 border border-white/10">
               {[2, 4, 8].map(s => (
                  <button 
                    key={s} 
                    onClick={() => setScale(s)}
                    className={`px-6 py-2 rounded-lg text-xs font-bold transition-all ${scale === s ? 'bg-blue-600 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
                  >
                    {s}x
                  </button>
               ))}
            </div>
         </div>
         <div className="h-10 w-px bg-white/5" />
         <div className="flex-1">
             <div className="flex justify-between text-[10px] font-bold text-zinc-500 mb-2">
                <span>MODEL: REAL_ESRGAN_v3</span>
                <span>DENOISE: ACTIVE</span>
             </div>
             <div className="h-1 bg-zinc-800 rounded-full w-48 overflow-hidden">
                <div className="h-full bg-blue-500 w-full opacity-50"></div>
             </div>
         </div>
         <Button 
            onClick={handleExecute} 
            loading={loading}
            disabled={!image}
            className="w-48 h-14 bg-blue-600 shadow-xl shadow-blue-500/10" 
            icon={Icons.Zap}
         >
            Enhance
         </Button>
      </div>
    </div>
  );
};
