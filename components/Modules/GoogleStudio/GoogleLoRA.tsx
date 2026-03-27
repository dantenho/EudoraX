
import React, { useState } from 'react';
import { Icons } from '../../../Icons';

export const GoogleLoRA: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'manage' | 'train'>('manage');

  return (
    <div className="flex flex-col h-full space-y-6 font-sans">
      <div className="flex gap-4 p-1 bg-white/5 rounded-2xl w-fit border border-white/5 mb-2">
        <button onClick={() => setActiveTab('manage')} className={`px-6 py-2 text-[10px] font-black uppercase rounded-xl ${activeTab === 'manage' ? 'bg-blue-600 text-white' : 'text-zinc-500'}`}>Manage</button>
        <button onClick={() => setActiveTab('train')} className={`px-6 py-2 text-[10px] font-black uppercase rounded-xl ${activeTab === 'train' ? 'bg-blue-600 text-white' : 'text-zinc-500'}`}>Train</button>
      </div>
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2.5rem] p-8 flex items-center justify-center">
         <div className="text-center opacity-20">
            <Icons.Dna className="w-24 h-24 mx-auto mb-4" />
            <p className="text-xs font-black uppercase tracking-widest">LoRA Service Interface</p>
         </div>
      </div>
    </div>
  );
};
