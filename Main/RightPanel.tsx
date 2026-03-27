
import React from 'react';
import { MoreHorizontal, User, MessageSquare } from 'lucide-react';

export const RightPanel: React.FC = () => {
  return (
    <aside className="w-80 bg-zinc-900/50 border-l border-zinc-800 flex flex-col animate-fade-in backdrop-blur-md">
      
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-6 border-b border-zinc-800">
        <span className="text-xs font-bold uppercase tracking-widest text-zinc-500">Inspector</span>
        <button className="text-zinc-500 hover:text-white transition-colors">
          <MoreHorizontal className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scrollbar">
        
        {/* Profile / User */}
        <div className="text-center space-y-4">
          <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-full p-[2px]">
            <div className="w-full h-full bg-zinc-900 rounded-full flex items-center justify-center overflow-hidden">
               <User className="w-8 h-8 text-white/50" />
            </div>
          </div>
          <div>
            <h3 className="text-lg font-medium text-white">Admin User</h3>
            <p className="text-xs text-zinc-500">Level 4 Access • Root</p>
          </div>
          <div className="flex justify-center gap-2">
            <button className="px-4 py-1.5 rounded-lg bg-zinc-800 text-xs font-medium text-zinc-300 hover:text-white border border-zinc-700 hover:bg-zinc-700 transition-colors">Profile</button>
            <button className="px-4 py-1.5 rounded-lg bg-zinc-800 text-xs font-medium text-zinc-300 hover:text-white border border-zinc-700 hover:bg-zinc-700 transition-colors">Log Out</button>
          </div>
        </div>

        {/* System Health */}
        <div className="space-y-4">
          <h4 className="text-[10px] font-black uppercase text-zinc-500 tracking-widest">System Telemetry</h4>
          
          <div className="p-4 bg-zinc-950 rounded-xl border border-zinc-800 space-y-4">
             <div className="space-y-2">
                <div className="flex justify-between text-xs text-zinc-500">
                   <span>Memory</span>
                   <span className="text-white">12.4 GB</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                   <div className="h-full w-2/3 bg-purple-500" />
                </div>
             </div>
             
             <div className="space-y-2">
                <div className="flex justify-between text-xs text-zinc-500">
                   <span>NPU Temp</span>
                   <span className="text-white">42°C</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                   <div className="h-full w-1/3 bg-emerald-500" />
                </div>
             </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="space-y-4">
           <h4 className="text-[10px] font-black uppercase text-zinc-500 tracking-widest">Feed</h4>
           <div className="space-y-3">
              {[1, 2].map(i => (
                <div key={i} className="flex gap-3 items-start">
                   <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                   <div>
                      <p className="text-xs text-zinc-300 leading-relaxed">
                         Model <span className="text-blue-500">Gemini-Nano</span> has finished quantization pass.
                      </p>
                      <span className="text-[10px] text-zinc-600 font-mono">10:4{i} AM</span>
                   </div>
                </div>
              ))}
           </div>
        </div>

      </div>

      <div className="p-6 border-t border-zinc-800">
         <button className="w-full flex items-center justify-center gap-2 h-10 rounded-lg bg-blue-600 text-white font-medium text-sm hover:bg-blue-500 transition-colors shadow-lg shadow-blue-500/20">
            <MessageSquare className="w-4 h-4" />
            Support Chat
         </button>
      </div>
    </aside>
  );
};
