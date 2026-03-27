
import React, { useState, useEffect } from 'react';
import { Icons } from './Icons';
import { Button } from './Button';
import { checkSystemCapabilities } from '../services/chromeAiService';
import { LocalIntelligence } from './admin/LocalIntelligence';
import { ApiSettings } from './admin/ApiSettings';
import { McpSettings } from './admin/McpSettings';
import { ConfigService } from '../services/configService';

// Toggle switch component
const ToggleSwitch = ({ checked, onChange }: { checked?: boolean, onChange?: () => void }) => (
  <div 
      onClick={onChange}
      className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${checked ? 'bg-eudora-500' : 'bg-zinc-700'}`}
  >
      <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-transform ${checked ? 'left-6' : 'left-1'}`}></div>
  </div>
);

/**
 * @component AdminPanel
 * @description Administration interface for managing users, system status, and global settings.
 * Features a tabbed layout with real-time style analytics visualization.
 * Integrated with Chrome AI capabilities for local hardware monitoring.
 */
export const AdminPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'settings'>('overview');
  // Sub-tabs for Settings
  const [settingsSubTab, setSettingsSubTab] = useState<'general' | 'api' | 'mcp'>('general');
  
  const [npuReady, setNpuReady] = useState(false);

  useEffect(() => {
    const fetchCapabilities = async () => {
        const caps = await checkSystemCapabilities();
        const nano = caps.find(c => c.id === 'npu_nano');
        if (nano && nano.available) {
            setNpuReady(true);
        }
    };
    fetchCapabilities();
  }, []);

  return (
    <div className="h-full flex flex-col bg-dark-bg text-zinc-200 overflow-hidden">
      {/* Header */}
      <div className="px-8 py-6 border-b border-dark-border bg-dark-sidebar/50 flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            <Icons.Shield className="w-6 h-6 text-eudora-500" />
            Administration
          </h2>
          <p className="text-sm text-zinc-500 mt-1">System Control & Analytics Center</p>
        </div>
        <div className="flex items-center gap-3">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${npuReady ? 'bg-green-900/20 border-green-500/20' : 'bg-zinc-800 border-zinc-700'}`}>
                <div className={`w-2 h-2 rounded-full ${npuReady ? 'bg-green-500 animate-pulse' : 'bg-zinc-500'}`}></div>
                <span className={`text-xs font-medium ${npuReady ? 'text-green-400' : 'text-zinc-400'}`}>
                    {npuReady ? 'NPU Online' : 'System Healthy'}
                </span>
            </div>
            <Button variant="secondary" size="sm" icon={Icons.Logout}>Log Out</Button>
        </div>
      </div>

      {/* Main Navigation Tabs */}
      <div className="px-8 pt-4 border-b border-dark-border flex gap-6">
        {[
            { id: 'overview', label: 'Overview', icon: Icons.Activity },
            { id: 'users', label: 'User Management', icon: Icons.Users },
            { id: 'settings', label: 'System Settings', icon: Icons.Settings }
        ].map(tab => (
            <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 pb-4 px-2 text-sm font-medium border-b-2 transition-all ${
                    activeTab === tab.id 
                    ? 'border-eudora-500 text-eudora-400' 
                    : 'border-transparent text-zinc-500 hover:text-zinc-300 hover:border-zinc-700'
                }`}
            >
                <tab.icon className="w-4 h-4" />
                {tab.label}
            </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
            <div className="space-y-8 animate-fade-in p-8 overflow-y-auto h-full">
                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <StatCard icon={Icons.Users} label="Total Users" value="12,450" trend="+5.2%" color="blue" />
                    <StatCard icon={Icons.Zap} label="Credits Used (24h)" value="840,200" trend="+12%" color="yellow" />
                    <StatCard icon={Icons.Server} label="API Latency" value="124ms" trend="-8ms" color="green" />
                    <StatCard icon={Icons.Chip} label="Local Ops (NPU)" value={npuReady ? "Active" : "Idle"} trend={npuReady ? "Ready" : "N/A"} color="purple" />
                </div>
            </div>
        )}

        {/* USERS TAB */}
        {activeTab === 'users' && (
             <div className="space-y-6 animate-fade-in p-8">
                 <p className="text-zinc-500 text-sm italic">User management module loaded.</p>
             </div>
        )}

        {/* SETTINGS TAB */}
        {activeTab === 'settings' && (
            <div className="max-w-4xl mx-auto p-8 space-y-6 animate-fade-in overflow-y-auto h-full">
                
                {/* Sub-Navigation for Settings */}
                <div className="flex gap-2 pb-2 border-b border-white/5 mb-4 overflow-x-auto">
                    <button 
                        onClick={() => setSettingsSubTab('general')}
                        className={`px-4 py-2 text-xs font-medium rounded-lg transition-colors whitespace-nowrap ${settingsSubTab === 'general' ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                        General
                    </button>
                    <button 
                        onClick={() => setSettingsSubTab('api')}
                        className={`px-4 py-2 text-xs font-medium rounded-lg transition-colors flex items-center gap-2 whitespace-nowrap ${settingsSubTab === 'api' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                        <Icons.Server className="w-3 h-3" /> API
                    </button>
                    <button 
                        onClick={() => setSettingsSubTab('mcp')}
                        className={`px-4 py-2 text-xs font-medium rounded-lg transition-colors flex items-center gap-2 whitespace-nowrap ${settingsSubTab === 'mcp' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' : 'text-zinc-500 hover:text-zinc-300'}`}
                    >
                        <Icons.Sliders className="w-3 h-3" /> MCP
                    </button>
                </div>

                {/* General Settings Sub-Tab */}
                {settingsSubTab === 'general' && (
                    <>
                        <LocalIntelligence />

                        <div className="bg-dark-card border border-dark-border rounded-xl p-6">
                            <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
                                <Icons.Lock className="w-5 h-5 text-eudora-500" /> Global Configuration
                            </h3>
                            <div className="space-y-6">
                                <div className="flex items-center justify-between pb-4 border-b border-white/5">
                                    <div>
                                        <p className="text-sm font-medium text-zinc-200">Maintenance Mode</p>
                                        <p className="text-xs text-zinc-500">Disable all tool access for non-admins</p>
                                    </div>
                                    <ToggleSwitch checked={ConfigService.get().system.maintenanceMode} onChange={() => ConfigService.updateSection('system', { maintenanceMode: !ConfigService.get().system.maintenanceMode })} />
                                </div>
                                <div className="flex items-center justify-between pb-4 border-b border-white/5">
                                    <div>
                                        <p className="text-sm font-medium text-zinc-200">Public Registration</p>
                                        <p className="text-xs text-zinc-500">Allow new users to sign up</p>
                                    </div>
                                    <ToggleSwitch checked={ConfigService.get().system.publicRegistration} onChange={() => ConfigService.updateSection('system', { publicRegistration: !ConfigService.get().system.publicRegistration })} />
                                </div>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-sm font-medium text-zinc-200">Debug Mode (NPU Log)</p>
                                        <p className="text-xs text-zinc-500">Enable console logging for Gemini Nano inference</p>
                                    </div>
                                    <ToggleSwitch checked={ConfigService.get().system.debugMode} onChange={() => ConfigService.updateSection('system', { debugMode: !ConfigService.get().system.debugMode })} />
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* API Settings Sub-Tab */}
                {settingsSubTab === 'api' && <ApiSettings />}

                {/* MCP Settings Sub-Tab */}
                {settingsSubTab === 'mcp' && <McpSettings />}

            </div>
        )}

      </div>
    </div>
  );
};

// Helper for Stat Cards (preserved)
const StatCard = ({ icon: Icon, label, value, trend, color }: any) => {
    const colorStyles: any = {
        blue: 'text-blue-500 bg-blue-500/10 border-blue-500/20',
        yellow: 'text-yellow-500 bg-yellow-500/10 border-yellow-500/20',
        green: 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20',
        purple: 'text-purple-500 bg-purple-500/10 border-purple-500/20'
    };
    return (
        <div className="bg-dark-card border border-dark-border p-6 rounded-xl flex flex-col gap-4 hover:border-white/10 transition-colors group">
            <div className="flex justify-between items-start">
                <div className={`p-3 rounded-lg ${colorStyles[color]} bg-opacity-10`}>
                    <Icon className="w-6 h-6" />
                </div>
                <span className={`text-xs font-medium px-2 py-1 rounded ${trend.startsWith('+') || trend === 'Ready' ? 'bg-green-900/30 text-green-400' : trend === 'N/A' ? 'bg-zinc-800 text-zinc-500' : 'bg-red-900/30 text-red-400'}`}>
                    {trend}
                </span>
            </div>
            <div>
                <h4 className="text-2xl font-bold text-white">{value}</h4>
                <p className="text-xs text-zinc-500 uppercase tracking-wider font-medium mt-1">{label}</p>
            </div>
        </div>
    )
}
