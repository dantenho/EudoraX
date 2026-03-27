
import React from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';

interface Props {
  loading: boolean;
  result: string | null;
  prompt: string;
  onPromptChange: (val: string) => void;
  onExecute: () => void;
}

export const GeminiVeo: React.FC<Props> = ({ loading, result, prompt, onPromptChange, onExecute }) => {
  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="flex-1 bg-black/40 border border-white/5 rounded-[2rem] overflow-hidden flex flex-col shadow-2xl">
        <div className="flex-1 flex items-center justify-center bg-[#030305] p-4 relative">
          {loading ? (
            <div className="text-center">
              <div className="w-16 h-16 border-t-2 border-blue-500 rounded-full animate-spin mb-4 mx-auto" />
              <p className="text-[10px] font-black text-zinc-500 uppercase tracking-widest animate-pulse">Temporal Motion Encoding</p>
            </div>
          ) : result ? (
            <video src={result} controls autoPlay loop className="max-h-full max-w-full rounded-xl border border-white/10" />
          ) : (
            <div className="text-center opacity-[0.03] scale-150">
              <Icons.Clapperboard className="w-64 h-64 mx-auto mb-4" />
              <p className="text-4xl font-black uppercase tracking-[0.5em]">Veo Motion Studio</p>
            </div>
          )}
        </div>
        <div className="p-8 bg-dark-sidebar/20 border-t border-white/5 flex gap-8 items-end">
          <div className="flex-1 space-y-4">
            <label className="text-[10px] font-black uppercase tracking-[0.3em] text-zinc-500">Motion Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              placeholder="Describe motion dynamics for Whisk Veo..."
              className="w-full h-24 bg-dark-bg/60 border border-white/5 rounded-2xl p-4 text-sm text-white resize-none focus:ring-1 focus:ring-blue-500/50"
            />
          </div>
          <div className="w-48">
            <Button onClick={onExecute} loading={loading} className="w-full h-14 bg-blue-600 border-0" icon={Icons.Video}>Render</Button>
          </div>
        </div>
      </div>
    </div>
  );
};
