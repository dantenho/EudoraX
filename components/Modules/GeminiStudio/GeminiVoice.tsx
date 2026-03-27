
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { generateSpeech } from '../../../services/geminiService.ts';
import { VoiceName } from '../../../types.ts';

export const GeminiVoice: React.FC = () => {
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
    <div className="flex flex-col h-full space-y-6 animate-fade-in">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl relative">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-10 relative">
          {audioBuffer ? (
             <div className="flex gap-1 h-32 items-center justify-center w-full">
                {[...Array(20)].map((_, i) => (
                  <div key={i} className="w-2 bg-blue-500 rounded-full animate-pulse" style={{ height: `${Math.random() * 100}%`, animationDuration: `${0.5 + Math.random()}s` }} />
                ))}
             </div>
          ) : (
            <div className="text-center opacity-10">
              <Icons.Voice className="w-32 h-32 mx-auto mb-4 text-zinc-500" />
              <p className="text-2xl font-black uppercase tracking-[0.5em] text-zinc-700">Voice Forge Standby</p>
            </div>
          )}
          
          <div className="absolute top-6 right-6 flex gap-2">
            <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
               <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
               <span className="text-[9px] font-black text-emerald-500 uppercase tracking-widest">Gemini 2.5 TTS</span>
            </div>
          </div>
        </div>

        <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-start">
          <div className="flex-1 space-y-4">
             <div className="flex items-center gap-4 overflow-x-auto pb-2 custom-scrollbar">
                {(Object.values(VoiceName) as string[]).map(voice => (
                   <button
                    key={voice}
                    onClick={() => setSelectedVoice(voice as VoiceName)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-[10px] font-bold uppercase transition-all whitespace-nowrap ${selectedVoice === voice ? 'bg-blue-600 border-blue-500 text-white' : 'bg-white/5 border-white/5 text-zinc-500 hover:text-zinc-300'}`}
                   >
                     {voice}
                     {selectedVoice === voice && <Icons.Check className="w-3 h-3" />}
                   </button>
                 ))}
             </div>
             <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter narration text or dialogue script..."
              className="w-full h-24 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none focus:ring-1 focus:ring-blue-500/50"
            />
          </div>
          <div className="flex flex-col gap-3 w-40">
             <Button 
                loading={loading} 
                onClick={handleSynthesize} 
                className="h-14 w-full bg-blue-600 rounded-xl" 
                icon={Icons.Voice}
              >
                Synthesize
             </Button>
             {audioBuffer && (
               <Button onClick={playAudio} variant="secondary" className="h-10 w-full rounded-xl" icon={Icons.Play}>Play Audio</Button>
             )}
          </div>
        </div>
      </div>
    </div>
  );
};
