
import React, { useState } from 'react';
import { Icons } from '../../Icons.tsx';

interface SensorState {
  npuTemp: number;
  batteryCurrent: string;
  imuStatus: 'CALIBRATED' | 'DRIFTING' | 'OFFLINE';
  gpsLock: '3D_FIX' | 'NO_FIX';
  networkLatency: string;
}

export const GeminiSensor: React.FC = () => {
  const [sensors] = useState<SensorState>({
    npuTemp: 42.5,
    batteryCurrent: '-240mA',
    imuStatus: 'CALIBRATED',
    gpsLock: '3D_FIX',
    networkLatency: '18ms'
  });

  return (
    <div className="flex flex-col h-full space-y-8 animate-fade-in font-sans">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1">
        
        {/* Android Device HUD: Main Telemetrics Visualization */}
        <div className="lg:col-span-2 bg-black/40 border border-white/5 rounded-[2rem] p-10 flex flex-col space-y-10 shadow-2xl relative overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-black text-white uppercase tracking-widest">
              Android Device Hub (Pixel 9 Pro)
            </h3>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
              <span className="text-[10px] font-mono text-emerald-400">TELEMETRY_LINK_ESTABLISHED</span>
            </div>
          </div>

          <div className="flex-1 grid grid-cols-2 gap-10">
            {/* Thermal & Power Metrics Column */}
            <div className="space-y-6">
               <div className="p-6 bg-white/5 rounded-3xl border border-white/5">
                  <div className="text-[10px] font-black text-zinc-500 uppercase mb-4">NPU Thermal Envelope</div>
                  <div className="text-4xl font-mono text-blue-400 font-black">{sensors.npuTemp}°C</div>
                  <div className="h-1 w-full bg-zinc-800 rounded-full mt-4 overflow-hidden">
                     <div className="h-full bg-blue-500 w-[65%] shadow-[0_0_10px_rgba(59,130,246,0.5)] transition-all duration-500"></div>
                  </div>
               </div>
               <div className="p-6 bg-white/5 rounded-3xl border border-white/5">
                  <div className="text-[10px] font-black text-zinc-500 uppercase mb-4">Power Discharge</div>
                  <div className="text-2xl font-mono text-emerald-400 font-black">{sensors.batteryCurrent}</div>
                  <div className="text-[10px] font-bold text-zinc-600 uppercase mt-2">Mode: High Efficiency Synthesis</div>
               </div>
            </div>

            {/* Spatial & Network Metrics Column */}
            <div className="space-y-6">
                <div className="p-6 bg-white/5 rounded-3xl border border-white/5 relative group overflow-hidden">
                    <Icons.Globe className="absolute -right-6 -bottom-6 w-32 h-32 opacity-[0.03] group-hover:opacity-10 transition-opacity" />
                    <div className="text-[10px] font-black text-zinc-500 uppercase mb-4">Spatial Fusion Lock</div>
                    <div className="space-y-3">
                       <div className="flex justify-between text-xs">
                          <span className="text-zinc-400 uppercase font-bold tracking-tighter">IMU Precision</span>
                          <span className="text-emerald-400 font-bold">{sensors.imuStatus}</span>
                       </div>
                       <div className="flex justify-between text-xs">
                          <span className="text-zinc-400 uppercase font-bold tracking-tighter">GPS Constellation</span>
                          <span className="text-blue-400 font-bold">{sensors.gpsLock}</span>
                       </div>
                       <div className="flex justify-between text-xs">
                          <span className="text-zinc-400 uppercase font-bold tracking-tighter">Signal Latency</span>
                          <span className="text-zinc-300 font-bold">{sensors.networkLatency}</span>
                       </div>
                    </div>
                </div>
                <div className="p-6 bg-white/5 rounded-3xl border border-white/5">
                   <div className="text-[10px] font-black text-zinc-500 uppercase mb-4">Kernel Tuning Status</div>
                   <div className="text-[10px] font-mono text-blue-400 space-y-1">
                      <div>&gt; TCP_AUTO_TUNING: STABLE</div>
                      <div>&gt; EBPF_TASK_AFFINITY: NODE_0</div>
                      <div>&gt; HUGETLB_PAGING: ACTIVE (1GB)</div>
                   </div>
                </div>
            </div>
          </div>
        </div>

        {/* Sentiment Analysis Hub */}
        <div className="bg-dark-sidebar/40 border border-white/5 rounded-[2rem] p-10 space-y-10 shadow-2xl">
          <h3 className="text-sm font-black text-white uppercase tracking-widest">Sentiment Kernel</h3>
          <div className="space-y-6">
             <div className="p-6 bg-emerald-500/10 border border-emerald-500/20 rounded-3xl">
                <div className="flex items-center gap-3 mb-2">
                   <Icons.Sparkles className="w-4 h-4 text-emerald-400" />
                   <span className="text-[10px] font-black text-emerald-400 uppercase">Analysis Engine: POSITIVE</span>
                </div>
                <p className="text-sm text-emerald-200/80 italic leading-relaxed">
                  "Prompt intent classified as high-creative. Adjusting synthesis temperature to 1.1."
                </p>
             </div>
             
             <div className="space-y-2">
                <div className="text-[10px] font-black text-zinc-600 uppercase tracking-widest">Confidence Score</div>
                <div className="flex gap-2">
                   {[1, 2, 3, 4, 5].map(i => (
                     <div key={i} className={`h-1.5 flex-1 rounded-full ${i <= 4 ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.3)]' : 'bg-zinc-800'}`}></div>
                   ))}
                </div>
             </div>
          </div>

          <div className="pt-8 border-t border-white/5 space-y-4">
             <p className="text-[10px] text-zinc-500 leading-relaxed font-medium">
                Our Efficient Sentiment Analysis kernel utilizes **AVX-512** vectorized instructions for sub-millisecond string classification.
             </p>
             <button className="w-full py-4 bg-zinc-800 text-zinc-300 font-black text-[10px] uppercase rounded-2xl hover:bg-zinc-700 hover:text-white transition-all tracking-[0.2em]">
                Export Fusion Logs
             </button>
          </div>
        </div>
      </div>
    </div>
  );
};
