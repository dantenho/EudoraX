
import React, { useState } from 'react';
import { Icons } from '../Icons';
import { ImageGenExtension } from './types';
import { TxtToImageExtension } from './ImageGeneration_txttoimage';
import { Button } from '../Button';
import { GeminiPixel } from '../GeminiStudio/modules/GeminiPixel';
import { GeminiLoRA } from '../GeminiStudio/modules/GeminiLoRA';
import { GeminiVeo } from '../GeminiStudio/modules/GeminiVeo';
import { VoiceGenerator } from '../voicegeneration/VoiceGenerator';
import { GeminiAssets } from '../GeminiStudio/modules/GeminiAssets';

const EXTENSIONS: ImageGenExtension[] = [
  {
    id: 'txt2img',
    name: 'Neural Txt2Img',
    description: 'Bmm3-optimized image synthesis.',
    longDescription: 'High-fidelity image generation using Gemini 3 Pro with Thinking Mode and Search Grounding.',
    version: '2.4.0',
    author: 'EudoraX Core',
    icon: 'Palette',
    category: 'Synthesis',
    tags: ['Realism', 'Search', 'Thinking'],
    component: TxtToImageExtension
  },
  {
    id: 'lora_service',
    name: 'LoRA Orchestrator',
    description: 'Fine-tune & layer style adapters.',
    longDescription: 'Train and manage custom LoRA weights for cross-module style consistency.',
    version: '1.1.0',
    author: 'EudoraX Logic',
    icon: 'Dna',
    category: 'Utility',
    tags: ['Training', 'Styles', 'Adapters'],
    component: GeminiLoRA
  },
  {
    id: 'pixel_forge',
    name: 'Pixel Art Forge',
    description: 'Retro 8/16-bit asset creator.',
    longDescription: 'Specialized kernel for generating grid-aligned game sprites and tiles.',
    version: '1.0.5',
    author: 'EudoraX Games',
    icon: 'Gamepad',
    category: 'Synthesis',
    tags: ['Sprites', 'Retro', 'Gaming'],
    component: GeminiPixel
  },
  {
    id: 'motion_forge',
    name: 'Veo Motion Forge',
    description: 'Temporal video synthesis.',
    longDescription: 'Generate cinematic motion sequences using the Veo 3.1 temporal encoder.',
    version: '3.1.2',
    author: 'Whisk AI',
    icon: 'Clapperboard',
    category: 'Motion',
    tags: ['Video', 'Cinematic', 'Veo'],
    component: () => <GeminiVeo loading={false} result={null} prompt="" onPromptChange={() => {}} onExecute={() => {}} />
  },
  {
    id: 'voice_forge',
    name: 'Neural Voice Forge',
    description: 'Multi-persona audio synthesis.',
    longDescription: 'Generate lifelike narration and dialogue with Gemini 2.5 Native Audio.',
    version: '2.1.0',
    author: 'EudoraX Audio',
    icon: 'Voice',
    category: 'Audio',
    tags: ['TTS', 'Multi-Speaker', 'Emotion'],
    component: VoiceGenerator
  },
  {
    id: 'asset_vault',
    name: 'Unified Asset Vault',
    description: 'Cross-module resource manager.',
    longDescription: 'View and manage generated images, videos, audio, and 3D meshes.',
    version: '4.5.0',
    author: 'System',
    icon: 'Layers',
    category: 'Utility',
    tags: ['Management', 'Cloud', 'PBR'],
    component: GeminiAssets
  }
];

export const ImageGenerator: React.FC = () => {
  const [activeExtId, setActiveExtId] = useState('txt2img');
  const [showInspector, setShowInspector] = useState(false);
  
  const activeExt = EXTENSIONS.find(e => e.id === activeExtId) || EXTENSIONS[0];

  return (
    <div className="h-full flex bg-[#030305] text-zinc-300 overflow-hidden font-sans">
      {/* Extension Sidebar HUD */}
      <div className="w-80 border-r border-white/5 bg-black/40 p-6 flex flex-col gap-6 overflow-y-auto">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Icons.Grid className="w-4 h-4 text-blue-500" />
            <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">LoRA Forge Suite</h3>
          </div>
          <button onClick={() => setShowInspector(!showInspector)} className="p-2 hover:bg-white/5 rounded-lg transition-colors">
            <Icons.Info className="w-4 h-4 text-zinc-600" />
          </button>
        </div>

        <div className="space-y-3">
          {EXTENSIONS.map(ext => {
            const Icon = Icons[ext.icon as keyof typeof Icons];
            const isActive = activeExtId === ext.id;
            return (
              <button
                key={ext.id}
                onClick={() => setActiveExtId(ext.id)}
                className={`w-full group flex flex-col gap-3 p-4 rounded-2xl border transition-all duration-300 ${
                  isActive 
                    ? 'bg-blue-600/10 border-blue-500/30 ring-1 ring-blue-500/10' 
                    : 'bg-transparent border-white/5 hover:border-white/10'
                }`}
              >
                <div className="flex items-center gap-4">
                  <div className={`p-2.5 rounded-xl transition-all ${isActive ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-500 group-hover:text-zinc-300'}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 text-left overflow-hidden">
                    <div className="text-xs font-black uppercase tracking-tight truncate">{ext.name}</div>
                    <div className="text-[9px] text-zinc-500 uppercase font-mono mt-0.5">v{ext.version}</div>
                  </div>
                </div>
                {isActive && (
                  <p className="text-[10px] text-zinc-400 leading-relaxed animate-fade-in px-1">
                    {ext.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>

        {/* Create Extension Call-to-Action */}
        <div className="mt-6 p-4 rounded-2xl border border-dashed border-white/10 bg-white/5 hover:bg-white/10 transition-all cursor-pointer group">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-500 group-hover:text-blue-400">
              <Icons.Plus className="w-4 h-4" />
            </div>
            <div className="flex-1 text-left">
              <div className="text-[10px] font-black uppercase text-zinc-400">Add Extension</div>
              <div className="text-[8px] text-zinc-600 uppercase">Plug in custom logic</div>
            </div>
          </div>
        </div>

        {/* Runtime HUD */}
        <div className="mt-auto pt-6 border-t border-white/5 space-y-4">
           <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-zinc-600 uppercase">Engine Status</span>
              <span className="text-emerald-400 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                V4.8-VIM
              </span>
           </div>
           <div className="bg-black/60 rounded-xl p-3 border border-white/5">
              <div className="flex justify-between items-center text-[9px] text-zinc-500 mb-2">
                <span>CUDA_DISPATCH</span>
                <span className="font-mono">18.4 GB</span>
              </div>
              <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 w-[68%] transition-all duration-1000"></div>
              </div>
           </div>
        </div>
      </div>

      {/* Main Extension Workspace */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#050507]">
        <header className="h-16 border-b border-white/5 bg-black/20 flex items-center justify-between px-8 backdrop-blur-3xl z-10">
          <div className="flex items-center gap-4">
            <div className="flex flex-col">
              <span className="text-[9px] font-black text-zinc-600 uppercase tracking-[0.2em]">Active Forge</span>
              <span className="text-xs font-bold text-blue-400 tracking-tight uppercase">{activeExt.name}</span>
            </div>
            <div className="h-4 w-px bg-white/5" />
            <div className="flex gap-1.5">
              {activeExt.tags.map(tag => (
                <span key={tag} className="px-2 py-0.5 bg-zinc-800/50 border border-white/5 rounded text-[8px] font-bold text-zinc-500 uppercase">
                  {tag}
                </span>
              ))}
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" icon={Icons.Terminal}>Kernel Log</Button>
            <Button variant="secondary" size="sm" icon={Icons.Share}>Deploy Node</Button>
          </div>
        </header>

        <div className="flex-1 p-8 overflow-y-auto custom-scrollbar">
          <activeExt.component />
        </div>
      </div>

      {/* Extension Inspector (Right Sidebar) */}
      {showInspector && (
        <div className="w-80 border-l border-white/5 bg-black/40 p-8 flex flex-col gap-8 animate-slide-in-right">
          <div className="space-y-2">
            <h4 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Metadata Inspector</h4>
            <div className="p-6 bg-white/5 border border-white/5 rounded-3xl space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-600/10 rounded-xl flex items-center justify-center text-blue-400">
                  {Icons[activeExt.icon as keyof typeof Icons] ? React.createElement(Icons[activeExt.icon as keyof typeof Icons] as any, { className: "w-5 h-5" }) : null}
                </div>
                <div>
                  <div className="text-xs font-black text-white">{activeExt.name}</div>
                  <div className="text-[9px] text-zinc-500">By {activeExt.author}</div>
                </div>
              </div>
              <p className="text-[10px] text-zinc-400 leading-relaxed">
                {activeExt.longDescription}
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Extension Requirements</h4>
            <div className="space-y-2">
              {[
                { label: 'Runtime', value: 'NodeJS 25' },
                { label: 'Memory', value: '8GB VRAM Min' },
                { label: 'Modality', value: activeExt.category }
              ].map(req => (
                <div key={req.label} className="flex justify-between items-center p-3 bg-black/20 rounded-xl border border-white/5">
                  <span className="text-[9px] font-bold text-zinc-500 uppercase">{req.label}</span>
                  <span className="text-[9px] font-mono text-zinc-300">{req.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
