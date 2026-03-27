
import React from 'react';
import { Icons } from '../../../Icons';
import { Button } from '../../../Button';

interface Props {
  loading: boolean;
  agentLog: string[];
  prompt: string;
  onPromptChange: (val: string) => void;
  onExecute: () => void;
}

export const GoogleAgent: React.FC<Props> = ({ loading, agentLog, prompt, onPromptChange, onExecute }) => {
  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-10 relative overflow-hidden font-mono text-[10px] text-blue-400">
           {agentLog.length > 0 ? agentLog.map((log, i) => (
              <div key={i} className="mb-1 opacity-80">&gt; {log}</div>
            ) ) : (
              <div className="flex flex-col items-center justify-center h-full opacity-20 text-center">
                <Icons.Brain className="w-24 h-24 mb-4" />
                <p className="uppercase tracking-widest font-black">NUMA Node Standby</p>
              </div>
            )}
        </div>
        <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
            <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Agent Spec</label>
            <textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              className="w-full h-24 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none"
            />
          </div>
          <Button onClick={onExecute} loading={loading} className="w-48 h-14 bg-blue-600" icon={Icons.Brain}>Train Cycle</Button>
        </div>
      </div>
    </div>
  );
};
