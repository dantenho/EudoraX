
/**
 * @file GeminiLoRA.tsx
 * @description Advanced Fine-tuning & Style Orchestration Service.
 * @backend Python 3.14 (vLLM / PyTorch / PEFT)
 */
import React, { useState } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';
import { LORA_STYLES } from '../../ImageGenerator/imagegeneratorfilterfunction';

export const GeminiLoRA: React.FC = () => {
  const [training, setTraining] = useState(false);
  const [activeTab, setActiveTab] = useState<'manage' | 'train'>('manage');
  const [selectedStyle, setSelectedStyle] = useState(LORA_STYLES[0].id);

  return (
    <div className="flex flex-col h-full space-y-6 animate-fade-in font-sans">
      <div className="flex gap-4 p-1 bg-white/5 rounded-2xl w-fit border border-white/5 mb-2">
        <button 
          onClick={() => setActiveTab('manage')}
          className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all ${activeTab === 'manage' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
        >
          Manage Adapters
        </button>
        <button 
          onClick={() => setActiveTab('train')}
          className={`px-6 py-2 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all ${activeTab === 'train' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
        >
          Train New Forge
        </button>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Adapter Library */}
        <div className="lg:col-span-2 bg-black/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col space-y-6 shadow-2xl relative overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-[10px] font-black text-white uppercase tracking-[0.2em]">Neural Style Library</h3>
            <span className="text-[10px] font-mono text-blue-500">{LORA_STYLES.length} Adapters Loaded</span>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 overflow-y-auto custom-scrollbar pr-2">
            {LORA_STYLES.map(style => (
              <div 
                key={style.id}
                onClick={() => setSelectedStyle(style.id)}
                className={`group relative aspect-[3/4] rounded-3xl overflow-hidden border transition-all duration-500 cursor-pointer ${selectedStyle === style.id ? 'border-blue-500 ring-2 ring-blue-500/20' : 'border-white/5 hover:border-white/20'}`}
              >
                <img src={style.image} className="w-full h-full object-cover grayscale opacity-40 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-700" alt={style.name} />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent" />
                <div className="absolute inset-x-0 bottom-0 p-4">
                  <div className="text-[8px] font-black text-blue-400 uppercase mb-0.5">{style.category}</div>
                  <div className="text-xs font-bold text-white">{style.name}</div>
                </div>
                {selectedStyle === style.id && (
                  <div className="absolute top-3 right-3 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center shadow-lg animate-scale-in">
                    <Icons.Check className="w-3.5 h-3.5 text-white" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Configuration / Training Console */}
        <div className="bg-dark-sidebar/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col space-y-8 shadow-2xl">
          {activeTab === 'manage' ? (
            <div className="space-y-8">
              <div className="space-y-4">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Adapter Conditioning</h4>
                <div className="space-y-6">
                  <div className="space-y-3">
                    <div className="flex justify-between text-[9px] font-bold text-zinc-400 uppercase">
                      <span>LoRA Intensity</span>
                      <span className="text-blue-400">0.85</span>
                    </div>
                    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="w-[85%] h-full bg-gradient-to-r from-blue-500 to-purple-500" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-[9px] font-bold text-zinc-400 uppercase">
                      <span>Stylization Delta</span>
                      <span className="text-purple-400">0.42</span>
                    </div>
                    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                      <div className="w-[42%] h-full bg-gradient-to-r from-purple-500 to-pink-500" />
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4 pt-6 border-t border-white/5">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Cross-Module Sync</h4>
                <div className="grid grid-cols-2 gap-3">
                   {['Image', 'Video', '3D', 'Voice'].map(mod => (
                     <label key={mod} className="flex items-center gap-2 p-3 bg-black/20 rounded-xl border border-white/5 cursor-pointer hover:bg-white/5 transition-all">
                       <input type="checkbox" defaultChecked className="hidden" />
                       <div className="w-3.5 h-3.5 rounded-md border border-zinc-600 flex items-center justify-center">
                          <div className="w-2 h-2 bg-blue-500 rounded-sm" />
                       </div>
                       <span className="text-[10px] font-bold text-zinc-400">{mod}</span>
                     </label>
                   ))}
                </div>
              </div>
              
              <Button className="w-full h-14 bg-blue-600 shadow-xl shadow-blue-500/20" icon={Icons.Zap}>Apply Style Adapter</Button>
            </div>
          ) : (
            <div className="space-y-8 flex-1 flex flex-col">
              <div className="space-y-4">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Dataset Ingestion</h4>
                <div className="border-2 border-dashed border-white/5 rounded-3xl p-8 flex flex-col items-center justify-center gap-4 hover:bg-white/5 transition-all cursor-pointer group">
                  <div className="w-12 h-12 bg-zinc-900 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform">
                    <Icons.Upload className="w-6 h-6 text-zinc-600 group-hover:text-blue-500" />
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] font-black text-zinc-400 uppercase">Drop Zip / JSONL</div>
                    <div className="text-[8px] text-zinc-600 mt-1 uppercase">Supports CLIP-Interrogated sets</div>
                  </div>
                </div>
              </div>

              <div className="space-y-4 pt-6 border-t border-white/5">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Forge Config</h4>
                <div className="space-y-2">
                  <input type="text" placeholder="LoRA Trigger Word" className="w-full bg-black/40 border border-white/5 rounded-xl p-3 text-xs text-blue-400 font-mono" />
                  <select className="w-full bg-black/40 border border-white/5 rounded-xl p-3 text-xs text-zinc-400 focus:outline-none">
                    <option>Rank 16 (Standard)</option>
                    <option>Rank 32 (High Fidelity)</option>
                    <option>Rank 64 (Experimental)</option>
                  </select>
                </div>
              </div>

              <Button 
                onClick={() => { setTraining(true); setTimeout(() => setTraining(false), 3000); }}
                loading={training}
                className="w-full h-14 bg-gradient-to-r from-blue-700 to-indigo-700 mt-auto" 
                icon={Icons.Brain}
              >
                Initiate Forge Training
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
