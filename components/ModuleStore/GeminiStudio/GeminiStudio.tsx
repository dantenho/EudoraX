
/**
 * @file GeminiStudio.tsx
 * @description Central hub for the Unified LoRa Forge (v4.8).
 * Consolidates Image, Voice, Video, Pixel, and LoRA Training modules.
 */
import React, { useState, useEffect, ReactElement } from 'react';
import { Icons } from '../../Icons.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';
import { GeminiOperationMode, GEMINI_STUDIO_PLUGINS } from '../../../plugins/geminiStudioActions.ts';
import { ToolType } from '../../../types.ts';
import { GeminiErrorBoundary } from './GeminiErrorHandler.tsx';

// Synthesis & Service Modules
import { GeminiAuth } from './GeminiAuth.tsx';
import { GeminiNano } from './GeminiNano.tsx';
import { GeminiSensor } from './GeminiSensor.tsx';
import { GeminiAssets } from './GeminiAssets.tsx';
import { GeminiVeo } from './GeminiVeo.tsx';
import { GeminiTxt2Img } from './GeminiTxt2Img.tsx';
import { GeminiImg2Img } from './GeminiImg2Img.tsx';
import { GeminiInpainting } from './GeminiInpainting.tsx';
import { GeminiUpscaler } from './GeminiUpscaler.tsx';
import { GeminiNodeStudio } from './GeminiNodeStudio.tsx';
import { GeminiPixel } from './GeminiPixel.tsx';
import { GeminiLoRA } from './GeminiLoRA.tsx';
import { GeminiCode } from './GeminiCode.tsx';
import { GeminiAgent } from './GeminiAgent.tsx';
import { GeminiVoice } from './GeminiVoice.tsx';

interface GeminiStudioProps {
  activeSubMode?: ToolType;
}

export const GeminiStudio: React.FC<GeminiStudioProps> = ({ activeSubMode }): ReactElement => {
  const [opMode, setOpMode] = useState<GeminiOperationMode>(GeminiOperationMode.LORA_SERVICE);
  const [loading, setLoading] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>('');
  const [result, setResult] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState({ vram: '18.4GB', latency: '0ms', engine: 'v4.8-Ultra' });
  const [user, setUser] = useState<any>(() => {
    const saved = localStorage.getItem('eudora_user');
    return saved ? JSON.parse(saved) : null;
  });

  const handleLogout = () => {
    localStorage.removeItem('eudora_token');
    localStorage.removeItem('eudora_user');
    setUser(null);
    setOpMode(GeminiOperationMode.AUTH);
  };

  useEffect(() => {
    if (activeSubMode) {
      const mapping: Record<string, GeminiOperationMode> = {
        [ToolType.IMAGE_GENERATOR]: GeminiOperationMode.TXT_TO_IMAGE,
        [ToolType.IMAGE_TO_IMAGE]: GeminiOperationMode.IMG_TO_IMG,
        [ToolType.INPAINTING]: GeminiOperationMode.INPAINTING,
        [ToolType.UPSCALER]: GeminiOperationMode.UPSCALER,
        [ToolType.STUDIO_NODES]: GeminiOperationMode.STUDIO_NODES,
        [ToolType.PIXEL_ART_GENERATOR]: GeminiOperationMode.PIXEL_FORGE,
        [ToolType.LORA_TRAINER]: GeminiOperationMode.LORA_SERVICE,
        [ToolType.IMAGE_ASSETS]: GeminiOperationMode.ASSET_LIBRARY,
        [ToolType.GEMINI_NANO]: GeminiOperationMode.NANO_BANANA,
        [ToolType.GEMINI_VEO]: GeminiOperationMode.WHISK_VEO,
        [ToolType.GEMINI_VOICE]: GeminiOperationMode.VOICE_SYNTH,
        [ToolType.GEMINI_SENSOR]: GeminiOperationMode.SENSOR_FUSION,
        [ToolType.GEMINI_AUTH]: GeminiOperationMode.AUTH,
        [ToolType.GEMINI_CODE]: GeminiOperationMode.CODE_FORGE,
      };
      
      if (mapping[activeSubMode]) {
        setOpMode(mapping[activeSubMode]);
      }
    }
  }, [activeSubMode]);

  const handleExecute = async (): Promise<void> => {
    setLoading(true);
    const start = performance.now();
    try {
      const synthResponse = await GeminiPlugin.executeSynthesis(prompt, opMode);
      setResult(synthResponse.url);
      setTelemetry(prev => ({ 
        ...prev, 
        latency: `${Math.round(performance.now() - start)}ms` 
      }));
    } catch (e) {
      console.error("[GEMINI_STUDIO] Synthesis Fault:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex bg-[#030305] text-zinc-300 overflow-hidden font-sans">
      {/* LoRA Forge Sidebar (Modality Switcher) */}
      <div className="w-20 border-r border-white/5 bg-black/40 flex flex-col items-center py-6 gap-4 overflow-y-auto custom-scrollbar">
        {[
          { id: GeminiOperationMode.LORA_SERVICE, icon: Icons.Dna, label: 'LoRA Trainer' },
          { id: GeminiOperationMode.TXT_TO_IMAGE, icon: Icons.Sparkles, label: 'Images' },
          { id: GeminiOperationMode.VOICE_SYNTH, icon: Icons.Voice, label: 'Voices' },
          { id: GeminiOperationMode.WHISK_VEO, icon: Icons.Clapperboard, label: 'Video' },
          { id: GeminiOperationMode.PIXEL_FORGE, icon: Icons.Gamepad, label: 'Pixel Art' },
          { id: GeminiOperationMode.ASSET_LIBRARY, icon: Icons.Layers, label: 'Assets' },
          { id: GeminiOperationMode.NANO_BANANA, icon: Icons.Zap, label: 'Nano' },
          { id: GeminiOperationMode.CODE_FORGE, icon: Icons.Code, label: 'Code' },
          { id: GeminiOperationMode.SENSOR_FUSION, icon: Icons.Globe, label: 'Sensors' },
        ].map(item => (
          <button 
            key={item.id} 
            title={item.label}
            onClick={() => setOpMode(item.id)}
            className={`p-4 rounded-2xl transition-all duration-500 relative group ${
              opMode === item.id 
              ? 'bg-blue-600 text-white shadow-xl scale-110' 
              : 'text-zinc-600 hover:text-zinc-300 hover:bg-white/5'
            }`}
          >
            <item.icon className="w-5 h-5" />
            {opMode === item.id && <div className="absolute -right-1 top-1 w-2 h-2 bg-emerald-400 rounded-full border border-black animate-pulse" />}
          </button>
        ))}
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-20 border-b border-white/5 bg-black/20 flex items-center justify-between px-10 backdrop-blur-3xl z-10">
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
               <span className="text-[9px] font-black tracking-[0.3em] text-zinc-600 uppercase">LoRA Forge Engine</span>
               <span className="text-base font-bold text-white tracking-widest uppercase">
                 {GEMINI_STUDIO_PLUGINS.find(p => p.id === opMode)?.label || opMode.replace(/_/g, ' ')}
               </span>
            </div>
            <div className="h-6 w-px bg-white/10" />
            <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 rounded-full border border-emerald-500/20">
               <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
               <span className="text-[9px] font-mono text-emerald-400 font-bold uppercase">System_Active</span>
            </div>
          </div>
          <div className="flex items-center gap-8">
             {user ? (
               <div className="flex items-center gap-3">
                 <div className="flex flex-col text-right">
                   <span className="text-[10px] font-black text-white uppercase tracking-widest">{user.name}</span>
                   <button 
                     onClick={handleLogout}
                     className="text-[9px] font-bold text-zinc-500 hover:text-red-400 uppercase tracking-tighter transition-colors"
                   >
                     Disconnect
                   </button>
                 </div>
                 <div className="w-10 h-10 rounded-xl bg-blue-600/20 border border-blue-500/20 flex items-center justify-center text-blue-400 font-black">
                   {user.name.charAt(0)}
                 </div>
               </div>
             ) : (
               <>
                 <div className="flex flex-col items-end">
                    <span className="text-[8px] font-black text-zinc-500 uppercase">Engine Build</span>
                    <span className="text-[10px] font-mono text-blue-400">{telemetry.engine}</span>
                 </div>
                 <div className="h-8 w-px bg-white/5" />
                 <div className="flex flex-col items-end text-blue-400 font-black italic tracking-tighter text-sm uppercase">
                    Unified creativity
                 </div>
               </>
             )}
          </div>
        </header>

        <div className="flex-1 p-10 overflow-y-auto custom-scrollbar bg-[#050507]">
          <GeminiErrorBoundary moduleName={opMode}>
             {opMode === GeminiOperationMode.LORA_SERVICE && <GeminiLoRA />}
             {opMode === GeminiOperationMode.TXT_TO_IMAGE && <GeminiTxt2Img />}
             {opMode === GeminiOperationMode.VOICE_SYNTH && <GeminiVoice />}
             {opMode === GeminiOperationMode.WHISK_VEO && <GeminiVeo loading={loading} result={result} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.PIXEL_FORGE && <GeminiPixel />}
             {opMode === GeminiOperationMode.ASSET_LIBRARY && <GeminiAssets />}
             
             {opMode === GeminiOperationMode.IMG_TO_IMG && <GeminiImg2Img />}
             {opMode === GeminiOperationMode.INPAINTING && <GeminiInpainting />}
             {opMode === GeminiOperationMode.UPSCALER && <GeminiUpscaler />}
             {opMode === GeminiOperationMode.STUDIO_NODES && <GeminiNodeStudio />}
             {opMode === GeminiOperationMode.NANO_BANANA && <GeminiNano loading={loading} result={result} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.AGENT_TRAINING && <GeminiAgent loading={loading} agentLog={[]} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.CODE_FORGE && <GeminiCode />}
             {opMode === GeminiOperationMode.SENSOR_FUSION && <GeminiSensor />}
             {opMode === GeminiOperationMode.AUTH && <GeminiAuth onLinked={(u) => { setUser(u); setOpMode(GeminiOperationMode.LORA_SERVICE); }} />}
          </GeminiErrorBoundary>
        </div>
      </div>

      {/* Pipeline Telemetry Sidebar */}
      <aside className="w-80 border-l border-white/5 bg-black/40 p-8 flex flex-col gap-10">
         <section className="space-y-4">
            <h3 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Hardware Telemetry</h3>
            <div className="bg-white/5 rounded-[2rem] p-6 space-y-6 border border-white/5">
                <div className="space-y-2">
                   <div className="flex justify-between text-[10px] font-bold">
                      <span className="text-zinc-500 uppercase tracking-tighter">VRAM Utilization</span>
                      <span className="text-blue-400">{telemetry.vram}</span>
                   </div>
                   <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 w-[74%] transition-all duration-700 shadow-[0_0_8px_rgba(59,130,246,0.5)]"></div>
                   </div>
                </div>
                <div className="space-y-2">
                   <div className="flex justify-between text-[10px] font-bold">
                      <span className="text-zinc-500 uppercase tracking-tighter">Dispatch Latency</span>
                      <span className="text-emerald-400">{telemetry.latency}</span>
                   </div>
                   <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 w-[32%] transition-all duration-700"></div>
                   </div>
                </div>
            </div>
         </section>

         <section className="flex-1 space-y-4 overflow-hidden">
            <h3 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Active Kernels</h3>
            <div className="space-y-2 overflow-y-auto custom-scrollbar pr-2 h-full">
               {[
                 { label: 'LORA_TRAINING_PASS', status: 'Idle', id: '0x01' },
                 { label: 'NATIVE_AUDIO_TTS', status: 'Idle', id: '0x02' },
                 { label: 'VEO_TEMPORAL_ENC', status: 'Standby', id: '0x03' },
                 { label: 'PIXEL_SNAP_GRID', status: 'Standby', id: '0x04' },
                 { label: 'ASSET_R2_SYNC', status: 'Active', id: '0x05' }
               ].map((kernel) => (
                 <div key={kernel.id} className="flex items-center gap-4 p-4 rounded-2xl bg-white/5 border border-white/5 group hover:border-blue-500/30 transition-all cursor-pointer">
                    <div className={`w-1.5 h-1.5 rounded-full ${kernel.status === 'Active' ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-zinc-800'}`} />
                    <div className="flex-1 text-[10px] font-mono">
                       <div className="text-zinc-300 font-black">{kernel.label}</div>
                       <div className="text-zinc-600 uppercase text-[8px] mt-1 tracking-widest">{kernel.status}</div>
                    </div>
                 </div>
               ))}
            </div>
         </section>
      </aside>
    </div>
  );
};
