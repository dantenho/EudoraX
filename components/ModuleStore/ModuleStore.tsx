
import React, { useState } from 'react';
import { Icons } from '../Icons';
import { Button } from '../Button';

interface Module {
  id: string;
  name: string;
  description: string;
  icon: React.ElementType;
  isInstalled: boolean;
  version: string;
  category: 'Creation' | 'Logic' | 'Marketing' | 'System';
}

const MODULES: Module[] = [
  { id: 'img_gen', name: 'Image Generator', description: 'Advanced SDXL/Flux generation with LoRA.', icon: Icons.Palette, isInstalled: true, version: '2.4.0', category: 'Creation' },
  { id: 'vid_gen', name: 'Video Forge', description: 'Motion synthesis via Google Veo 3.1.', icon: Icons.Clapperboard, isInstalled: true, version: '1.2.0', category: 'Creation' },
  { id: 'voice_gen', name: 'Voice Synth', description: 'Native audio multi-speaker synthesis.', icon: Icons.Voice, isInstalled: true, version: '1.0.5', category: 'Creation' },
  { id: 'comfy_ui', name: 'ComfyUI Bridge', description: 'Direct orchestration of local ComfyUI nodes.', icon: Icons.Network, isInstalled: false, version: '0.8.1-beta', category: 'Logic' },
  { id: 'lora_trainer', name: 'LoRA Trainer', description: 'Train custom styles using PyTorch and vLLM.', icon: Icons.Dna, isInstalled: false, version: '1.1.0', category: 'Creation' },
  { id: 'n8n_bridge', name: 'n8n Automation', description: 'Trigger complex webhooks and automation.', icon: Icons.Workflow, isInstalled: true, version: '2.0.0', category: 'Logic' },
  { id: 'tiktok_tracker', name: 'TikTok Analyzer', description: 'Market intelligence for short-form trends.', icon: Icons.Activity, isInstalled: false, version: '0.5.0', category: 'Marketing' },
];

export const ModuleStore: React.FC = () => {
  const [filter, setFilter] = useState<'All' | 'Installed' | 'New'>('All');

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-fade-in">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center gap-3">
            <Icons.Box className="w-8 h-8 text-eudora-500" />
            Module Store
          </h2>
          <p className="text-zinc-500 mt-2">Extend EudoraX with specialized AI capabilities.</p>
        </div>
        <div className="flex bg-dark-card border border-dark-border rounded-lg p-1">
          {['All', 'Installed', 'New'].map(t => (
            <button 
              key={t}
              onClick={() => setFilter(t as any)}
              className={`px-4 py-1.5 text-xs font-medium rounded-md transition-all ${filter === t ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {MODULES.filter(m => filter === 'All' || (filter === 'Installed' && m.isInstalled) || (filter === 'New' && !m.isInstalled)).map(mod => (
          <div key={mod.id} className="bg-dark-card border border-dark-border rounded-2xl p-6 flex flex-col hover:border-eudora-500/30 transition-all group">
            <div className="flex justify-between items-start mb-4">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${mod.isInstalled ? 'bg-eudora-500/10 text-eudora-400' : 'bg-zinc-800 text-zinc-500'}`}>
                <mod.icon className="w-6 h-6" />
              </div>
              <span className="text-[10px] font-mono text-zinc-600 bg-black/40 px-2 py-0.5 rounded uppercase tracking-widest">{mod.category}</span>
            </div>
            
            <h3 className="text-lg font-bold text-white mb-1">{mod.name}</h3>
            <p className="text-sm text-zinc-500 mb-6 flex-1">{mod.description}</p>
            
            <div className="flex items-center justify-between mt-auto pt-4 border-t border-white/5">
              <div className="text-[10px] text-zinc-600 font-mono">
                v{mod.version}
              </div>
              {mod.isInstalled ? (
                <div className="flex gap-2">
                  <Button variant="secondary" size="sm" icon={Icons.Settings}>Config</Button>
                  <Button variant="ghost" size="sm" className="text-red-900 hover:bg-red-900/10">Uninstall</Button>
                </div>
              ) : (
                <Button variant="primary" size="sm" icon={Icons.Download}>Install</Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
