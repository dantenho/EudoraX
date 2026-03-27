
/**
 * @file GeminiStudio.tsx
 * @description Central hub for the Unified Creativity Engine. Consolidates Intelligence and Synthesis pillars.
 */
import React, { useState, useEffect, ReactElement } from 'react';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';
import { GeminiOperationMode, GEMINI_STUDIO_PLUGINS } from '../../../plugins/geminiStudioActions.ts';
import { ToolType } from '../../../types.ts';
import { GeminiErrorBoundary } from './GeminiErrorHandler.tsx';

// Unified Imports (Flattened Structure)
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
import { GeminiProfile } from './GeminiProfile.tsx';

interface GeminiStudioProps {
  activeSubMode?: ToolType;
}

export const GeminiStudio: React.FC<GeminiStudioProps> = ({ activeSubMode }): ReactElement => {
  const [opMode, setOpMode] = useState<GeminiOperationMode>(GeminiOperationMode.TXT_TO_IMAGE);
  const [loading, setLoading] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>('');
  const [result, setResult] = useState<string | null>(null);
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
      // Map external ToolTypes to internal OperationModes
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
    try {
      const synthResponse = await GeminiPlugin.executeSynthesis(prompt, opMode);
      setResult(synthResponse.url);
      
      // Save to assets
      const token = localStorage.getItem('eudora_token');
      if (token && synthResponse.url) {
        await fetch('/api/assets', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            name: `Synth_${Date.now()}`,
            url: synthResponse.url,
            tags: ['generated', opMode.toLowerCase()],
          })
        });
      }
    } catch (e) {
      console.error("[GEMINI_STUDIO] Synthesis Fault:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex bg-[#030305] text-zinc-300 overflow-hidden font-sans">
      {/* Unified Module Selector */}
      <div className="w-20 border-r border-white/5 bg-black/40 flex flex-col items-center py-6 gap-4 overflow-y-auto custom-scrollbar">
        {GEMINI_STUDIO_PLUGINS.map(item => (
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
        <header className="h-20 border-b border-white/5 bg-black/20 flex items-center justify-between px-10">
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
               <span className="text-[9px] font-black tracking-[0.3em] text-zinc-600 uppercase">Unified Hub</span>
               <span className="text-base font-bold text-white tracking-widest uppercase">
                 {GEMINI_STUDIO_PLUGINS.find(p => p.id === opMode)?.label || opMode.replace(/_/g, ' ')}
               </span>
            </div>
            <div className="h-6 w-px bg-white/10" />
            <div className="flex items-center gap-2 px-3 py-1 bg-blue-500/10 rounded-full border border-blue-500/20">
               <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
               <span className="text-[9px] font-mono text-blue-400 font-bold uppercase">Kernel: Online</span>
            </div>
          </div>
          <div className="flex items-center gap-6">
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
              <div className="text-zinc-700 font-black italic text-2xl tracking-tighter opacity-20">EUDORAX</div>
            )}
          </div>
        </header>

        <div className="flex-1 p-10 overflow-y-auto custom-scrollbar">
          <GeminiErrorBoundary moduleName={opMode}>
             {opMode === GeminiOperationMode.TXT_TO_IMAGE && <GeminiTxt2Img />}
             {opMode === GeminiOperationMode.IMG_TO_IMG && <GeminiImg2Img />}
             {opMode === GeminiOperationMode.INPAINTING && <GeminiInpainting />}
             {opMode === GeminiOperationMode.UPSCALER && <GeminiUpscaler />}
             {opMode === GeminiOperationMode.STUDIO_NODES && <GeminiNodeStudio />}
             {opMode === GeminiOperationMode.NANO_BANANA && <GeminiNano loading={loading} result={result} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.WHISK_VEO && <GeminiVeo loading={loading} result={result} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.VOICE_SYNTH && <GeminiVoice />}
             {opMode === GeminiOperationMode.PIXEL_FORGE && <GeminiPixel />}
             {opMode === GeminiOperationMode.LORA_SERVICE && <GeminiLoRA />}
             {opMode === GeminiOperationMode.AGENT_TRAINING && <GeminiAgent loading={loading} agentLog={[]} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.CODE_FORGE && <GeminiCode />}
             {opMode === GeminiOperationMode.SENSOR_FUSION && <GeminiSensor />}
             {opMode === GeminiOperationMode.AUTH && <GeminiAuth onLinked={(u) => { setUser(u); setOpMode(GeminiOperationMode.TXT_TO_IMAGE); }} />}
             {opMode === GeminiOperationMode.ASSET_LIBRARY && <GeminiAssets />}
             {opMode === GeminiOperationMode.PROFILE && <GeminiProfile />}
          </GeminiErrorBoundary>
        </div>
      </div>
    </div>
  );
};
