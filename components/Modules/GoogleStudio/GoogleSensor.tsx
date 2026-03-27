
import React from 'react';
import { Icons } from '../../../Icons';

export const GoogleSensor: React.FC = () => {
  return (
    <div className="flex flex-col h-full space-y-8 font-sans">
      <div className="bg-black/40 border border-white/5 rounded-[2rem] p-10 flex flex-col items-center justify-center flex-1">
          <Icons.Globe className="w-24 h-24 text-zinc-800 mb-6" />
          <p className="text-xl font-black text-white tracking-widest uppercase">Sensor Fusion Active</p>
          <div className="mt-4 text-[10px] font-mono text-emerald-400">TELEMETRY_LINK: 18ms</div>
      </div>
    </div>
  );
};
