
import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar.tsx';
import { GeminiStudio } from './components/Modules/GeminiStudio/GeminiStudio.tsx';
import { AdminPanel } from './components/AdminPanel.tsx';
import { ToolType } from './types.ts';
import { Icons } from './components/Icons.tsx';

/**
 * @component DashboardHome
 * @description Dual-Pillar focal point for EudoraX.
 */
const DashboardHome: React.FC<{ onNavigate: (t: ToolType) => void }> = ({ onNavigate }) => {
  return (
    <div className="p-16 space-y-20 animate-fade-in max-w-7xl mx-auto">
      <header className="space-y-8">
        <div className="flex items-center gap-4 text-blue-500 mb-2">
          <Icons.Logo className="w-10 h-10" />
          <span className="text-xs font-black uppercase tracking-[0.5em] text-zinc-600">EudoraX Structural v4.8</span>
        </div>
        <h1 className="text-9xl font-black text-white tracking-tighter leading-[0.85] max-w-5xl">
          Unified <br /> <span className="text-blue-500 italic">Creativity.</span>
        </h1>
        <p className="text-zinc-500 text-2xl max-w-3xl leading-relaxed font-medium">
          A high-fidelity orchestration environment for distributed intelligence and neural synthesis. 
          Consolidated architecture for maximum efficiency.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        {[
          { 
            id: ToolType.IMAGE_GENERATOR, 
            name: 'Image Generation', 
            desc: 'Neural Txt2Img, LoRA style orchestration, and Retro Pixel kernels.', 
            icon: Icons.Palette, 
            color: 'text-blue-400', 
            bg: 'bg-blue-500/10',
            action: 'Neural Imaging'
          },
          { 
            id: ToolType.GEMINI_STUDIO, 
            name: 'Intelligence Node', 
            desc: 'Nano synthesis, Veo motion, Voice forge, and Sensor telemetry.', 
            icon: Icons.Box, 
            color: 'text-emerald-400', 
            bg: 'bg-emerald-500/10',
            action: 'Gemini Studio'
          },
        ].map(card => (
          <button
            key={card.id}
            onClick={() => onNavigate(card.id)}
            className="group relative p-16 rounded-[5rem] bg-dark-card border border-white/5 hover:border-blue-500/30 transition-all text-left space-y-12 overflow-hidden h-[520px] flex flex-col justify-between"
          >
            <div className="absolute -right-24 -top-24 w-80 h-80 bg-blue-500/5 blur-[120px] group-hover:bg-blue-500/10 transition-colors duration-1000" />
            <div className={`w-28 h-28 ${card.bg} rounded-[3rem] flex items-center justify-center ${card.color} group-hover:scale-110 transition-transform duration-700 shadow-2xl`}>
              <card.icon className="w-14 h-14" />
            </div>
            <div className="space-y-6 relative z-10">
              <h3 className="text-6xl font-black text-white tracking-tighter">{card.name}</h3>
              <p className="text-xl text-zinc-500 font-medium leading-relaxed">{card.desc}</p>
            </div>
            <div className="flex items-center gap-4 text-sm font-black text-blue-500 uppercase tracking-widest opacity-40 group-hover:opacity-100 transition-all duration-500">
              Launch {card.action} <Icons.ArrowRight className="w-5 h-5" />
            </div>
          </button>
        ))}
      </div>

      <div className="pt-16 border-t border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-16">
           <div className="flex items-center gap-4">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
              <div className="flex flex-col">
                <span className="text-[10px] font-black text-zinc-600 uppercase tracking-[0.3em]">NPU Engine</span>
                <span className="text-[10px] font-mono text-blue-400 uppercase font-bold">Structural_Active</span>
              </div>
           </div>
        </div>
        <div className="text-[10px] font-black text-white/20 uppercase tracking-[0.8em] italic">EudoraX v4.8</div>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  const [activeTool, setActiveTool] = useState<ToolType>(ToolType.DASHBOARD);

  const renderContent = () => {
    switch (activeTool) {
      // Unified Routing: All creative tools now route to GeminiStudio
      case ToolType.IMAGE_GENERATOR:
      case ToolType.IMAGE_TO_IMAGE:
      case ToolType.INPAINTING:
      case ToolType.UPSCALER:
      case ToolType.STUDIO_NODES:
      case ToolType.LORA_TRAINER:
      case ToolType.PIXEL_ART_GENERATOR:
      case ToolType.IMAGE_ASSETS:
      case ToolType.GEMINI_STUDIO:
      case ToolType.GEMINI_NANO:
      case ToolType.GEMINI_VEO:
      case ToolType.GEMINI_VOICE:
      case ToolType.GEMINI_SENSOR:
      case ToolType.GEMINI_AUTH:
      case ToolType.GEMINI_CODE:
        return <GeminiStudio activeSubMode={activeTool} />;
      case ToolType.ADMIN_PANEL:
        return <AdminPanel />;
      case ToolType.DASHBOARD:
      default:
        return <DashboardHome onNavigate={setActiveTool} />;
    }
  };

  return (
    <div className="flex h-screen bg-dark-bg text-zinc-100 overflow-hidden font-sans">
      <Sidebar activeTool={activeTool} onSelectTool={setActiveTool} />
      <main className="flex-1 ml-64 h-full overflow-y-auto relative custom-scrollbar">
        <div className="sticky top-0 z-40 bg-dark-bg/40 backdrop-blur-3xl border-b border-white/5 px-16 py-6 flex justify-between items-center">
             <div className="flex items-center gap-8">
                <span className="text-[10px] font-black tracking-[0.5em] text-zinc-600 uppercase">System Hub</span>
                <div className="h-4 w-px bg-white/10" />
                <span className="text-xs font-bold text-blue-400 uppercase tracking-tighter">
                  {activeTool === ToolType.DASHBOARD ? 'Command Center' : activeTool.replace(/_/g, ' ')}
                </span>
             </div>
             <span className="text-[10px] font-black text-white/40 tracking-[0.4em] uppercase italic">EudoraX Core</span>
        </div>
        <div className="min-h-[calc(100vh-80px)]">{renderContent()}</div>
      </main>
    </div>
  );
};

export default App;
