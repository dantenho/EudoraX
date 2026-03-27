
/**
 * @file ImageGenerator.tsx
 * @description Central hub for the Synthesis Forge (v4.8).
 */
import React, { useState, useMemo } from 'react';
import { Icons } from '../../Icons.tsx';
import { ImageGenExtension } from './types.ts';
import { TxtToImageExtension } from './ImageGeneration_txttoimage.tsx';
// Strict relative paths
import { GeminiPixel } from '../GeminiStudio/GeminiPixel.tsx';
import { GeminiLoRA } from '../GeminiStudio/GeminiLoRA.tsx';
import { GeminiAssets } from '../GeminiStudio/GeminiAssets.tsx';
import { ToolType } from '../../../types.ts';

interface Props {
  activeSubMode?: ToolType;
}

const EXTENSIONS: ImageGenExtension[] = [
  {
    id: 'txt2img',
    name: 'Neural Synthesis',
    description: 'Bmm3-optimized image generation using Gemini 3 Pro kernels.',
    longDescription: 'High-fidelity text-to-image synthesis with thinking-mode reasoning and search grounding.',
    version: '3.0.0',
    author: 'EudoraX Core',
    icon: 'Palette',
    category: 'Synthesis',
    tags: ['4K', 'Neural'],
    component: TxtToImageExtension
  },
  {
    id: 'lora_service',
    name: 'LoRA Orchestrator',
    description: 'Fine-tune & layer custom style adapters for consistent results.',
    longDescription: 'Manage custom weights and cross-module style conditioning across all synthesis pillars.',
    version: '1.5.0',
    author: 'EudoraX Logic',
    icon: 'Dna',
    category: 'Utility',
    tags: ['Styles', 'Adapters'],
    component: GeminiLoRA
  },
  {
    id: 'pixel_forge',
    name: 'Pixel Art Forge',
    description: 'Specialized Retro 8/16-bit asset creator for game development.',
    longDescription: 'Grid-aligned sprite and tile synthesis kernel with retro hardware palette emulation.',
    version: '1.2.0',
    author: 'EudoraX Games',
    icon: 'Gamepad',
    category: 'Synthesis',
    tags: ['Retro', 'Sprites'],
    component: GeminiPixel
  },
  {
    id: 'asset_vault',
    name: 'Unified Asset Vault',
    description: 'Cross-module manager for all generated creative assets.',
    longDescription: 'Secure cloud storage with Three.js-powered inspection for generated 3D assets and textures.',
    version: '4.8.0',
    author: 'System',
    icon: 'Layers',
    category: 'Utility',
    tags: ['Vault', 'PBR'],
    component: GeminiAssets
  }
];

export const ImageGenerator: React.FC<Props> = ({ activeSubMode }) => {
  const [overrideExtId, setOverrideExtId] = useState<string | null>(null);

  const activeExtId = useMemo(() => {
    if (overrideExtId) return overrideExtId;
    if (activeSubMode === ToolType.LORA_TRAINER) return 'lora_service';
    if (activeSubMode === ToolType.PIXEL_ART_GENERATOR) return 'pixel_forge';
    if (activeSubMode === ToolType.IMAGE_GENERATOR) return 'txt2img';
    if (activeSubMode === ToolType.IMAGE_ASSETS) return 'asset_vault';
    return 'txt2img';
  }, [activeSubMode, overrideExtId]);

  const activeExt = EXTENSIONS.find(e => e.id === activeExtId) || EXTENSIONS[0];

  return (
    <div className="h-full flex bg-[#030305] overflow-hidden font-sans">
      {/* Extension Sidebar HUD */}
      <div className="w-80 border-r border-white/5 bg-black/40 p-10 flex flex-col gap-12 overflow-y-auto">
        <div className="flex items-center gap-4 mb-2">
          <Icons.Grid className="w-5 h-5 text-blue-500" />
          <h3 className="text-[10px] font-black uppercase tracking-[0.5em] text-zinc-600">Forge Extensions</h3>
        </div>

        <div className="space-y-6">
          {EXTENSIONS.map(ext => {
            const Icon = Icons[ext.icon as keyof typeof Icons];
            const isActive = activeExtId === ext.id;
            return (
              <button
                key={ext.id}
                onClick={() => setOverrideExtId(ext.id)}
                className={`w-full group flex flex-col gap-6 p-8 rounded-[2.5rem] border transition-all duration-700 ${
                  isActive 
                    ? 'bg-blue-600/10 border-blue-500/40 shadow-2xl' 
                    : 'bg-transparent border-white/5 hover:border-white/10'
                }`}
              >
                <div className="flex items-center gap-6">
                  <div className={`p-4 rounded-2xl transition-all duration-500 ${isActive ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-500 group-hover:text-zinc-300'}`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex-1 text-left">
                    <span className="text-xs font-black text-zinc-200 block uppercase tracking-tight">{ext.name}</span>
                    <span className="text-[9px] text-zinc-600 uppercase font-mono mt-1 font-bold">Node v{ext.version}</span>
                  </div>
                </div>
                {isActive && (
                  <p className="text-[11px] text-zinc-500 leading-relaxed text-left animate-fade-in font-medium">
                    {ext.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>

        {/* System Monitoring Stats */}
        <div className="mt-auto pt-10 border-t border-white/5 space-y-8">
           <div className="bg-black/40 rounded-[2rem] p-8 border border-white/5 space-y-6">
              <div className="flex justify-between items-center text-[10px] font-black text-zinc-600 uppercase tracking-widest">
                <span>CUDA_CORE_LOAD</span>
                <span className="font-mono text-blue-400 font-bold">READY</span>
              </div>
              <div className="h-1.5 bg-zinc-900 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 w-[68%] transition-all duration-1000 shadow-[0_0_15px_rgba(59,130,246,0.3)]" />
              </div>
           </div>
        </div>
      </div>

      {/* Main Execution Workspace */}
      <div className="flex-1 p-16 overflow-y-auto bg-[#050507] custom-scrollbar">
        <activeExt.component />
      </div>
    </div>
  );
};
