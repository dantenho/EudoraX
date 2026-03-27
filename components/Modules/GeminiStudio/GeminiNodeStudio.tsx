
import React from 'react';
import { Icons } from '../../Icons.tsx';

export const GeminiNodeStudio: React.FC = () => {
  return (
    <div className="flex flex-col h-full bg-[#111113] relative overflow-hidden animate-fade-in rounded-[2rem] border border-white/5 shadow-2xl">
       <div className="absolute inset-0" style={{ backgroundImage: 'radial-gradient(circle, #333 1px, transparent 1px)', backgroundSize: '24px 24px', opacity: 0.1 }}></div>
       
       <div className="absolute top-4 left-4 z-10 flex gap-2">
           <div className="px-3 py-1 bg-black/60 border border-white/10 rounded-lg text-[10px] font-mono text-zinc-400 uppercase">Workflow: SDXL_Turbo_v4</div>
           <div className="px-3 py-1 bg-green-900/20 border border-green-500/20 rounded-lg text-[10px] font-mono text-green-400 uppercase">Status: Idle</div>
       </div>

       {/* Mock Nodes */}
       <div className="absolute top-20 left-20 w-64 bg-black/80 border border-zinc-700 rounded-xl p-4 shadow-xl backdrop-blur-md">
          <div className="flex justify-between items-center mb-4 border-b border-white/10 pb-2">
             <span className="text-xs font-bold text-blue-400 uppercase">Input: Prompt</span>
             <Icons.MoreHorizontal className="w-4 h-4 text-zinc-500" />
          </div>
          <div className="space-y-2">
             <div className="h-8 bg-zinc-900 rounded border border-white/5 text-[10px] flex items-center px-3 text-zinc-400 font-mono truncate">"A futuristic city..."</div>
             <div className="h-8 bg-zinc-900 rounded border border-white/5 text-[10px] flex items-center px-3 text-zinc-400 font-mono">Seed: 4294967295</div>
          </div>
          <div className="mt-4 flex justify-end relative">
             <div className="w-3 h-3 bg-blue-500 rounded-full border-2 border-white translate-x-1.5 cursor-crosshair hover:scale-125 transition-transform" />
          </div>
       </div>

       {/* Connection Line */}
       <svg className="absolute inset-0 pointer-events-none overflow-visible">
          <path d="M 330 200 C 450 200, 400 250, 500 250" stroke="#3b82f6" strokeWidth="2" fill="none" className="opacity-50" />
       </svg>

       <div className="absolute top-40 left-96 w-64 bg-black/80 border border-emerald-500/30 rounded-xl p-4 shadow-xl shadow-emerald-500/5 backdrop-blur-md">
          <div className="flex justify-between items-center mb-4 border-b border-white/10 pb-2">
             <span className="text-xs font-bold text-emerald-400 uppercase">Kernel: Diffusion</span>
             <Icons.Cpu className="w-4 h-4 text-zinc-500" />
          </div>
          <div className="space-y-2">
             <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full border-2 border-white -translate-x-1.5" />
                <span className="text-[10px] text-zinc-400 font-mono">Latent_Input</span>
             </div>
             <div className="flex items-center gap-2 pt-2">
                 <div className="w-3 h-3 bg-purple-500 rounded-full border-2 border-white -translate-x-1.5" />
                 <span className="text-[10px] text-zinc-400 font-mono">VAE_Decode</span>
             </div>
          </div>
          <div className="mt-8 flex justify-end">
             <div className="w-3 h-3 bg-emerald-500 rounded-full border-2 border-white translate-x-1.5 cursor-pointer" />
          </div>
       </div>

       <div className="absolute top-10 right-10 flex gap-2">
          <button className="px-4 py-2 bg-blue-600 rounded-lg text-xs font-bold text-white shadow-lg flex items-center gap-2 hover:bg-blue-500 transition-colors">
             <Icons.Play className="w-3 h-3" /> Execute Workflow
          </button>
       </div>
    </div>
  );
};
