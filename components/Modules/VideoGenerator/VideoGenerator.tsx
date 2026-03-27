
import React, { useState } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';
import { generateVideo } from '../../../services/geminiService';

export const VideoGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [aspectRatio, setAspectRatio] = useState('16:9');

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    try {
      const url = await generateVideo(prompt);
      if (url) setVideoUrl(url);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-12 space-y-12 animate-fade-in max-w-6xl mx-auto">
      <div className="flex flex-col lg:flex-row gap-12">
        {/* Workspace */}
        <div className="flex-1 space-y-8">
          <div className="aspect-video bg-dark-card/40 rounded-[3rem] border border-white/5 overflow-hidden relative flex items-center justify-center group shadow-2xl">
            {loading ? (
              <div className="flex flex-col items-center gap-6">
                <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <div className="space-y-2 text-center">
                  <p className="text-xs font-black uppercase tracking-[0.3em] text-blue-500 animate-pulse">Temporal Synthesis Node Active</p>
                  <p className="text-[10px] text-zinc-600 font-mono">Frame Inception: Phase 4/12</p>
                </div>
              </div>
            ) : videoUrl ? (
              <video src={videoUrl} controls autoPlay loop className="w-full h-full object-cover" />
            ) : (
              <div className="flex flex-col items-center gap-8 opacity-20 group-hover:opacity-30 transition-opacity">
                <Icons.Clapperboard className="w-32 h-32 text-zinc-400" />
                <p className="text-xs font-black uppercase tracking-[0.5em]">Veo Motion Studio Standby</p>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h3 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Motion Sequence Specification</h3>
              <div className="flex gap-2">
                {['16:9', '9:16'].map(ratio => (
                  <button 
                    key={ratio}
                    onClick={() => setAspectRatio(ratio)}
                    className={`px-4 py-1.5 text-[10px] font-bold rounded-lg border transition-all ${aspectRatio === ratio ? 'bg-blue-600/10 border-blue-500/30 text-blue-400' : 'bg-transparent border-white/5 text-zinc-600'}`}
                  >
                    {ratio}
                  </button>
                ))}
              </div>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe temporal dynamics, camera motion, and object physics..."
              className="w-full h-32 bg-dark-card/60 border border-white/5 rounded-[2rem] p-6 text-sm text-white focus:ring-2 focus:ring-blue-500/20 transition-all resize-none shadow-inner"
            />
            <Button 
              loading={loading} 
              onClick={handleGenerate} 
              className="w-full h-16 bg-blue-600 shadow-2xl shadow-blue-500/20 rounded-2xl" 
              icon={Icons.Video}
            >
              Initiate Temporal Forge
            </Button>
          </div>
        </div>

        {/* Info Panel */}
        <div className="w-full lg:w-80 space-y-8">
          <div className="p-8 bg-dark-card/40 border border-white/5 rounded-[2.5rem] space-y-6">
            <div className="flex items-center gap-3">
              <Icons.Info className="w-4 h-4 text-blue-500" />
              <h4 className="text-xs font-black uppercase tracking-widest text-zinc-300">Veo v3.1 Logic</h4>
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Temporal synthesis utilizes Whisk Veo 3.1 kernels for cinematic motion stability and physical consistency. 
              Render times may vary based on sequence complexity.
            </p>
            <div className="pt-4 space-y-3">
               <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-zinc-600 uppercase">Resolution</span>
                  <span className="text-zinc-300">1080p_NATIVE</span>
               </div>
               <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-zinc-600 uppercase">FPS_Target</span>
                  <span className="text-zinc-300">24_CINEMATIC</span>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
