
/**
 * @file GoogleStudio.tsx
 * @description Simplified hub for intelligence pillars.
 */
import React, { useState, useEffect, ReactElement } from 'react';
import { Icons } from '../../Icons.tsx';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';
import { GeminiOperationMode } from '../../../plugins/geminiStudioActions.ts';
import { ToolType } from '../../../types.ts';
import { GoogleErrorBoundary } from './GoogleErrorHandler.tsx';

import { GoogleAuth } from './GoogleAuth.tsx';
import { GoogleNano } from './GoogleNano.tsx';
import { GoogleSensor } from './GoogleSensor.tsx';

interface GoogleStudioProps {
  activeSubMode?: ToolType;
}

export const GoogleStudio: React.FC<GoogleStudioProps> = ({ activeSubMode }): ReactElement => {
  const [opMode, setOpMode] = useState<GeminiOperationMode>(GeminiOperationMode.NANO_BANANA);
  const [loading, setLoading] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>('');
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    if (activeSubMode) {
      if (activeSubMode === ToolType.GEMINI_AUTH) setOpMode(GeminiOperationMode.AUTH);
      if (activeSubMode === ToolType.GEMINI_NANO) setOpMode(GeminiOperationMode.NANO_BANANA);
      if (activeSubMode === ToolType.GEMINI_SENSOR) setOpMode(GeminiOperationMode.SENSOR_FUSION);
    }
  }, [activeSubMode]);

  const handleExecute = async (): Promise<void> => {
    setLoading(true);
    try {
      const synthResponse = await GeminiPlugin.executeSynthesis(prompt, opMode);
      setResult(synthResponse.url);
    } catch (e) {
      console.error("[GOOGLE_STUDIO] Synthesis Fault:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex bg-[#030305] text-zinc-300 overflow-hidden">
      <div className="w-16 border-r border-white/5 bg-black/40 flex flex-col items-center py-8 gap-8">
        {[
           { id: GeminiOperationMode.NANO_BANANA, icon: Icons.Image, label: 'Nano' },
           { id: GeminiOperationMode.SENSOR_FUSION, icon: Icons.Globe, label: 'Sensor' },
           { id: GeminiOperationMode.AUTH, icon: Icons.Key, label: 'Auth' }
        ].map(item => (
          <button 
            key={item.id} 
            title={item.label}
            onClick={() => setOpMode(item.id as any)}
            className={`p-3 rounded-2xl transition-all duration-300 ${
              opMode === item.id 
              ? 'bg-blue-600 text-white shadow-xl scale-110' 
              : 'text-zinc-600 hover:text-zinc-300 hover:bg-white/5'
            }`}
          >
            <item.icon className="w-5 h-5" />
          </button>
        ))}
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b border-white/5 bg-black/20 flex items-center justify-between px-10">
          <div className="text-[10px] font-bold text-blue-400 tracking-widest uppercase">{opMode.replace('_', ' ')} Node</div>
          <div className="text-blue-400 font-black italic text-sm">EUDORAX</div>
        </header>

        <div className="flex-1 p-10 overflow-y-auto">
          <GoogleErrorBoundary moduleName={opMode}>
             {opMode === GeminiOperationMode.NANO_BANANA && <GoogleNano loading={loading} result={result} prompt={prompt} onPromptChange={setPrompt} onExecute={handleExecute} />}
             {opMode === GeminiOperationMode.SENSOR_FUSION && <GoogleSensor />}
             {opMode === GeminiOperationMode.AUTH && <GoogleAuth onLinked={() => {}} />}
          </GoogleErrorBoundary>
        </div>
      </div>
    </div>
  );
};
