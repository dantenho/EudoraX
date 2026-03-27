
import React from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';

interface Props {
  loading: boolean;
  agentLog: string[];
  prompt: string;
  onPromptChange: (val: string) => void;
  onExecute: () => void;
}

export const GeminiAgent: React.FC<Props> = ({ loading, agentLog, prompt, onPromptChange, onExecute }) => {
  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-10 relative overflow-hidden">
          <div className="w-full h-full font-mono text-[10px] text-blue-400 overflow-y-auto custom-scrollbar">
            <div className="text-zinc-600 mb-4 font-bold uppercase tracking-widest border-b border-white/5 pb-2">
              WebNN Engine Monitor • [MIMALLOC: ACTIVE] [HUGEPAGES: 2MB]
            </div>
            {agentLog.length > 0 ? agentLog.map((log, i) => (
              <div key={i} className="mb-1 opacity-80 animate-fade-in">&gt; {log}</div>
            ) ) : (
              <div className="flex flex-col items-center justify-center h-full opacity-20 text-center">
                <Icons.Brain className="w-24 h-24 mb-4" />
                <p className="uppercase tracking-widest font-black">NUMA Node Standby</p>
                <p className="text-[8px] mt-2">Ready for hardware-accelerated training (ThinLTO Kernel)</p>
              </div>
            )}
            {loading && <div className="mt-4 text-white font-bold animate-pulse">&gt; _OPTIMIZING_KERNELS_WITH_PGO_...</div>}
          </div>
        </div>
        <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
            <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Neural Configuration</label>
            <textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              placeholder="Define agent behavior and tool set..."
              className="w-full h-24 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none focus:ring-1 focus:ring-blue-500/50"
            />
          </div>
          <div className="w-48">
            <Button onClick={onExecute} loading={loading} className="w-full h-14 bg-blue-600 border-0" icon={Icons.Brain}>Train Cycle</Button>
          </div>
        </div>
      </div>
    </div>
  );
};
