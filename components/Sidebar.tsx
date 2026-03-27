
import React, { useState } from 'react';
import { TOOLS } from '../constants';
import { ToolType, ToolItem } from '../types';
import { Icons } from './Icons';

interface SidebarProps {
  activeTool: ToolType;
  onSelectTool: (tool: ToolType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTool, onSelectTool }) => {
  const [openCategories, setOpenCategories] = useState<Set<string>>(new Set(['image-forge-parent', 'google-studio-parent']));

  const toggleCategory = (id: string) => {
    setOpenCategories(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const renderToolItem = (tool: ToolItem, level: number) => {
    if (tool.isCategoryHeader) {
      return (
        <div key={tool.id} className="mt-10 mb-3 px-6 flex items-center gap-3">
          <span className="text-[10px] font-black text-zinc-700 uppercase tracking-[0.3em] whitespace-nowrap">{tool.label}</span>
          <div className="h-px flex-1 bg-white/[0.03]" />
        </div>
      );
    }

    const isActive = activeTool === tool.id;
    const isParent = !!tool.children?.length;
    const isOpen = openCategories.has(tool.id as string);
    const Icon = tool.icon;

    return (
      <div key={tool.id} className="px-3">
        <button
          onClick={() => isParent ? toggleCategory(tool.id as string) : onSelectTool(tool.id as ToolType)}
          className={`w-full group flex items-center justify-between p-3 rounded-2xl transition-all duration-300 ${
            isActive 
              ? 'bg-blue-600/10 border border-blue-500/20 text-white shadow-xl shadow-blue-500/5' 
              : 'text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-300'
          }`}
        >
          <div className="flex items-center gap-4">
            <div className={`p-2 rounded-xl transition-all ${isActive ? 'bg-blue-600 text-white' : 'bg-zinc-900 text-zinc-600 group-hover:text-zinc-400'}`}>
              <Icon className="w-4 h-4" />
            </div>
            <div className="flex flex-col items-start">
              <span className="text-xs font-bold tracking-tight">{tool.label}</span>
              {tool.subLabel && <span className="text-[8px] text-zinc-600 uppercase font-mono tracking-tighter mt-0.5">{tool.subLabel}</span>}
            </div>
          </div>
          {isParent && (
            <Icons.ChevronRight className={`w-3 h-3 transition-transform duration-300 ${isOpen ? 'rotate-90' : 'text-zinc-700'}`} />
          )}
        </button>
        {isParent && isOpen && (
          <div className="mt-2 ml-6 pl-4 border-l border-white/5 space-y-1.5 animate-fade-in">
            {tool.children!.map(child => renderToolItem(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <aside className="w-64 h-screen bg-dark-sidebar border-r border-white/5 flex flex-col fixed left-0 top-0 z-50">
      <div className="p-8">
        <div className="flex items-center gap-4 group cursor-pointer" onClick={() => onSelectTool(ToolType.DASHBOARD)}>
          <div className="w-12 h-12 bg-white text-black rounded-2xl flex items-center justify-center shadow-2xl group-hover:scale-105 transition-transform duration-500">
            <Icons.Logo className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-sm font-black text-white uppercase tracking-tighter leading-tight">EudoraX</h1>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="w-1 h-1 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[8px] font-mono text-zinc-600 uppercase">Node_v4.8</span>
            </div>
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto pb-10 custom-scrollbar">
        {TOOLS.map(tool => renderToolItem(tool, 0))}
      </nav>

      {/* Sidebar Footer Telemetry */}
      <div className="p-6 bg-black/20 border-t border-white/5">
        <div className="p-5 rounded-[1.5rem] bg-white/[0.02] border border-white/5 space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-[9px] font-black text-zinc-600 uppercase tracking-widest">Neural Load</span>
            <span className="text-[10px] font-mono text-emerald-500">Normal</span>
          </div>
          <div className="h-1 bg-zinc-900 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 w-[34%] animate-pulse" />
          </div>
          <div className="flex items-center justify-between text-[9px] text-zinc-600">
            <span>MEM: 12.4GB</span>
            <span>GPU: 42%</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
