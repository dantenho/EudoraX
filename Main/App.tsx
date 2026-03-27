
import React, { useState } from 'react';
import { LayoutDashboard, Layers, Cpu, Settings, Hexagon } from 'lucide-react';
import { Dashboard } from './Dashboard';
import { RightPanel } from './RightPanel';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const navItems = [
    { id: 'overview', icon: LayoutDashboard },
    { id: 'layers', icon: Layers },
    { id: 'cpu', icon: Cpu },
    { id: 'settings', icon: Settings },
  ];

  return (
    <div className="flex h-screen w-full overflow-hidden bg-zinc-950 text-zinc-100 font-sans selection:bg-blue-500/30">
      {/* Sidebar Navigation */}
      <aside className="w-16 border-r border-zinc-800 flex flex-col items-center py-6 gap-6 bg-zinc-900/50 backdrop-blur-md">
        <div className="p-3 bg-blue-600/10 rounded-xl text-blue-500 mb-4 border border-blue-500/20">
          <Hexagon className="w-6 h-6" />
        </div>
        {navItems.map((item) => (
          <button 
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`p-3 rounded-xl transition-all duration-300 ${
              activeTab === item.id 
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25' 
              : 'text-zinc-500 hover:text-white hover:bg-white/5'
            }`}
          >
            <item.icon className="w-5 h-5" />
          </button>
        ))}
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex min-w-0">
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-16 border-b border-zinc-800 flex items-center justify-between px-8 bg-zinc-900/30 backdrop-blur-sm z-10">
            <h1 className="text-sm font-bold uppercase tracking-widest text-zinc-400 flex items-center gap-3">
              <span className="text-white">EudoraX</span> 
              <span className="w-1 h-1 bg-zinc-700 rounded-full" />
              <span>Atomic Workspace</span>
            </h1>
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-xs font-mono text-emerald-500">SYSTEM_OPTIMAL</span>
            </div>
          </header>
          
          <div className="flex-1 overflow-y-auto p-8 relative custom-scrollbar">
            <Dashboard />
          </div>
        </div>

        {/* Right Context Panel */}
        <RightPanel />
      </main>
    </div>
  );
};
