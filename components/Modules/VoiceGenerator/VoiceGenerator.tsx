
import React, { useState } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';
import { generateSpeech } from '../../../services/geminiService';
// Import VoiceName enum from shared types
import { VoiceName } from '../../../types';

export const VoiceGenerator: React.FC = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [selectedVoice, setSelectedVoice] = useState<VoiceName>(VoiceName.Kore);

  const handleSynthesize = async () => {
    if (!text) return;
    setLoading(true);
    try {
      const buffer = await generateSpeech(text, selectedVoice);
      if (buffer) setAudioBuffer(buffer);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const playAudio = () => {
    if (!audioBuffer) return;
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start(0);
  };

  return (
    <div className="p-12 space-y-12 animate-fade-in max-w-5xl mx-auto">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight uppercase">Neural Voice Forge</h2>
          <p className="text-zinc-500 text-sm mt-1">Multi-persona audio synthesis powered by Gemini 2.5 Native Audio.</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/5 border border-emerald-500/10 rounded-xl">
           <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
           <span className="text-[10px] font-black text-emerald-500 uppercase tracking-widest">WAV_READY</span>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
        <div className="lg:col-span-2 space-y-8">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter narration text or dialogue script..."
            className="w-full h-48 bg-dark-card/40 border border-white/5 rounded-[2.5rem] p-8 text-lg text-zinc-200 focus:ring-2 focus:ring-blue-500/20 transition-all resize-none shadow-2xl"
          />
          
          <div className="flex gap-4">
             <Button 
                loading={loading} 
                onClick={handleSynthesize} 
                className="flex-1 h-16 bg-blue-600 rounded-2xl" 
                icon={Icons.Voice}
              >
                Synthesize Audio
             </Button>
             {audioBuffer && (
               <Button onClick={playAudio} variant="secondary" className="h-16 w-16 rounded-2xl" icon={Icons.Play}> </Button>
             )}
          </div>
        </div>

        <div className="space-y-8">
           <div className="p-8 bg-dark-card/40 border border-white/5 rounded-[2.5rem] space-y-6">
              <h3 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Persona Library</h3>
              <div className="grid gap-3">
                 {/* Fixed: Cast enum values to string array to resolve unknown type errors in React key and children */}
                 {(Object.values(VoiceName) as string[]).map(voice => (
                   <button
                    key={voice}
                    onClick={() => setSelectedVoice(voice as VoiceName)}
                    className={`flex items-center justify-between p-4 rounded-2xl border transition-all ${selectedVoice === voice ? 'bg-blue-600 text-white border-blue-500' : 'bg-white/5 border-white/5 text-zinc-500 hover:text-zinc-300'}`}
                   >
                     <span className="text-xs font-bold uppercase">{voice}</span>
                     {selectedVoice === voice && <Icons.Check className="w-3.5 h-3.5" />}
                   </button>
                 ))}
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};
