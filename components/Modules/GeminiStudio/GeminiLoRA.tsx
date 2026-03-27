
/**
 * @file GeminiLoRA.tsx
 * @description Advanced LoRA Training & Style Orchestration Service.
 * Centralizing the management of style adapters for Image, Video, and Voice.
 */
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';
import { LORA_STYLES } from './GeminiFilterFunctions.ts';
import { GeminiPlugin } from '../../../plugins/gemini/geminiPlugin.ts';

export const GeminiLoRA: React.FC = () => {
  const [training, setTraining] = useState(false);
  const [activeTab, setActiveTab] = useState<'manage' | 'train'>('manage');
  const [selectedStyle, setSelectedStyle] = useState(LORA_STYLES[0].id);
  const [triggerWord, setTriggerWord] = useState('');

  const handleTrain = async () => {
    setTraining(true);
    try {
        await GeminiPlugin.trainLoRA({
            datasetPath: "s3://eudorax/uploads/training_set.zip",
            triggerWord: triggerWord || "style_adapter_alpha",
            rank: 16
        });
        alert("LoRA Training Task successfully dispatched to EudoraX Backend.");
    } catch (e) {
        console.error("[LORA_TRAIN_ERROR]", e);
    } finally {
        setTraining(false);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6 animate-fade-in font-sans">
      <div className="flex gap-4 p-1 bg-white/5 rounded-2xl w-fit border border-white/5 mb-2">
        <button 
          onClick={() => setActiveTab('manage')}
          className={`px-8 py-3 text-[10px] font-black uppercase tracking-[0.2em] rounded-xl transition-all ${activeTab === 'manage' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
        >
          Adapter Vault
        </button>
        <button 
          onClick={() => setActiveTab('train')}
          className={`px-8 py-3 text-[10px] font-black uppercase tracking-[0.2em] rounded-xl transition-all ${activeTab === 'train' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
        >
          Train Logic
        </button>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Style Selection Grid */}
        <div className="lg:col-span-2 bg-black/40 border border-white/5 rounded-[2.5rem] p-10 flex flex-col space-y-8 shadow-2xl relative overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-black text-white uppercase tracking-widest">Available Adapters</h3>
            <div className="flex items-center gap-4 text-[10px] font-mono text-blue-500 uppercase">
              <Icons.History className="w-3.5 h-3.5" />
              <span>Loaded: {LORA_STYLES.length} adapters</span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-6 overflow-y-auto custom-scrollbar pr-4">
            {LORA_STYLES.map(style => (
              <div 
                key={style.id}
                onClick={() => setSelectedStyle(style.id)}
                className={`group relative aspect-[3/4] rounded-[2rem] overflow-hidden border transition-all duration-700 cursor-pointer ${selectedStyle === style.id ? 'border-blue-500 ring-2 ring-blue-500/30' : 'border-white/5 hover:border-white/20'}`}
              >
                <img src={style.image} className="w-full h-full object-cover grayscale opacity-40 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-1000" alt={style.name} />
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/10 to-transparent" />
                <div className="absolute inset-x-0 bottom-0 p-6">
                  <div className="text-[8px] font-black text-blue-400 uppercase tracking-widest mb-1">{style.category}</div>
                  <div className="text-sm font-black text-white">{style.name}</div>
                </div>
                {selectedStyle === style.id && (
                  <div className="absolute top-4 right-4 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center shadow-2xl animate-scale-in">
                    <Icons.Check className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Configuration Cockpit */}
        <div className="bg-dark-sidebar/40 border border-white/5 rounded-[2.5rem] p-10 flex flex-col space-y-10 shadow-2xl">
          {activeTab === 'manage' ? (
            <div className="space-y-10">
              <section className="space-y-6">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Global Conditioning</h4>
                <div className="space-y-8">
                  <div className="space-y-3">
                    <div className="flex justify-between text-[10px] font-black text-zinc-400 uppercase tracking-widest">
                      <span>LoRA Intensity</span>
                      <span className="text-blue-400">0.85</span>
                    </div>
                    <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-white/5">
                      <div className="w-[85%] h-full bg-gradient-to-r from-blue-500 to-indigo-600 shadow-[0_0_10px_rgba(59,130,246,0.3)]" />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-[10px] font-black text-zinc-400 uppercase tracking-widest">
                      <span>Style Variance</span>
                      <span className="text-purple-400">0.42</span>
                    </div>
                    <div className="h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-white/5">
                      <div className="w-[42%] h-full bg-gradient-to-r from-purple-500 to-pink-500" />
                    </div>
                  </div>
                </div>
              </section>

              <section className="space-y-6 pt-10 border-t border-white/5">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Pillar Synchronicity</h4>
                <div className="grid grid-cols-2 gap-4">
                   {['Image Forge', 'Motion Forge', 'Voice Forge', 'Pixel Forge'].map(pillar => (
                     <label key={pillar} className="flex items-center gap-3 p-4 bg-white/5 rounded-2xl border border-white/5 cursor-pointer hover:bg-white/10 transition-all group">
                       <input type="checkbox" defaultChecked className="hidden" />
                       <div className="w-4 h-4 rounded-lg border border-zinc-700 flex items-center justify-center transition-colors group-hover:border-blue-500">
                          <div className="w-2.5 h-2.5 bg-blue-500 rounded-md" />
                       </div>
                       <span className="text-[10px] font-bold text-zinc-400 uppercase">{pillar.split(' ')[0]}</span>
                     </label>
                   ))}
                </div>
              </section>
              
              <Button className="w-full h-16 bg-blue-600 shadow-2xl shadow-blue-500/20 rounded-2xl" icon={Icons.Zap}>Apply Adaptive Weights</Button>
            </div>
          ) : (
            <div className="space-y-10 flex-1 flex flex-col">
              <section className="space-y-6">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Dataset Ingestion</h4>
                <div className="border-2 border-dashed border-white/5 rounded-[2rem] p-10 flex flex-col items-center justify-center gap-6 hover:bg-white/5 hover:border-blue-500/30 transition-all cursor-pointer group">
                  <div className="w-16 h-16 bg-zinc-900 rounded-3xl flex items-center justify-center border border-white/5 group-hover:scale-110 transition-transform duration-500">
                    <Icons.Upload className="w-8 h-8 text-zinc-600 group-hover:text-blue-500" />
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">Upload Dataset</div>
                    <div className="text-[9px] text-zinc-600 mt-2 font-mono uppercase">.zip / .jsonl • 512px Base</div>
                  </div>
                </div>
              </section>

              <section className="space-y-6 pt-10 border-t border-white/5 flex-1">
                <h4 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Training Parameters</h4>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-[9px] font-black text-zinc-600 uppercase ml-1">Trigger Word</label>
                    <input 
                      type="text" 
                      value={triggerWord}
                      onChange={(e) => setTriggerWord(e.target.value)}
                      placeholder="e.g. synth_neon" 
                      className="w-full bg-black/60 border border-white/5 rounded-2xl p-4 text-xs text-blue-400 font-mono focus:ring-1 focus:ring-blue-500/50 transition-all" 
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[9px] font-black text-zinc-600 uppercase ml-1">Network Rank</label>
                    <select className="w-full bg-black/60 border border-white/5 rounded-2xl p-4 text-xs text-zinc-400 focus:outline-none transition-all focus:border-white/20">
                      <option>Rank 16 (Standard)</option>
                      <option>Rank 32 (Fidelity)</option>
                      <option>Rank 64 (Extreme)</option>
                    </select>
                  </div>
                </div>
              </section>

              <Button 
                onClick={handleTrain}
                loading={training}
                className="w-full h-16 bg-gradient-to-r from-blue-700 to-indigo-700 rounded-2xl shadow-2xl shadow-blue-500/20" 
                icon={Icons.Brain}
              >
                Initiate Neural Forge
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
