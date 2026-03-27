

import React, { useState } from 'react';
import { Icons } from '../Icons';
import { Button } from '../Button';
import { ConfigService, AppConfig } from '../../services/configService';

export const ApiSettings: React.FC = () => {
  const [extConfig, setExtConfig] = useState<AppConfig['externalApis']>(ConfigService.get().externalApis);
  const [isDirty, setIsDirty] = useState(false);
  const [lastSave, setLastSave] = useState<number | null>(null);

  const handleExtChange = (section: keyof AppConfig['externalApis'], key: string, value: any) => {
    setExtConfig(prev => ({
        ...prev,
        [section]: {
            ...prev[section],
            [key]: value
        }
    }));
    setIsDirty(true);
  };

  // Helper for nested config updates (e.g. imageGeneration.replicate.apiKey)
  const handleNestedExtChange = (
    section: keyof AppConfig['externalApis'], 
    subSection: string, 
    key: string, 
    value: any
  ) => {
      setExtConfig(prev => ({
          ...prev,
          [section]: {
              ...prev[section],
              [subSection]: {
                  ...(prev[section] as any)[subSection],
                  [key]: value
              }
          }
      }));
      setIsDirty(true);
  }

  const handleSave = () => {
    ConfigService.updateSection('externalApis', extConfig);
    setIsDirty(false);
    setLastSave(Date.now());
  };

  return (
    <div className="bg-dark-card border border-dark-border rounded-xl p-6 animate-fade-in space-y-8">
      
      {/* --- Connectivity Header --- */}
      <div className="flex justify-between items-start">
        <div>
           <h3 className="text-lg font-medium text-white flex items-center gap-2">
            <Icons.Server className="w-5 h-5 text-blue-500" /> 
            API Connectivity
          </h3>
          <p className="text-xs text-zinc-400 mt-1">
            Manage 3rd party integrations for models, storage, and blockchain services.
          </p>
        </div>
        {isDirty && (
            <span className="text-xs text-yellow-400 animate-pulse font-bold uppercase tracking-wider">Unsaved Changes</span>
        )}
      </div>

      {/* --- AI Models Integration --- */}
      <div className="space-y-6 border-b border-white/5 pb-6">
          <h4 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
            <Icons.Chip className="w-4 h-4" /> Generative Models
          </h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Prompt Gen API */}
            <div className="bg-black/20 border border-white/5 rounded-lg p-4 space-y-4">
                <div className="flex items-center gap-2 mb-2">
                    <Icons.Code className="w-4 h-4 text-purple-400" />
                    <span className="text-sm font-medium text-zinc-200">LLM & Prompting</span>
                </div>
                <div className="space-y-3">
                    <div className="space-y-1">
                        <label className="text-[10px] text-zinc-500">Active Provider</label>
                        <select 
                            value={extConfig.promptGeneration.provider} 
                            onChange={(e) => handleExtChange('promptGeneration', 'provider', e.target.value)}
                            className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300"
                        >
                            <option value="gemini">Google Gemini (Recommended)</option>
                            <option value="anthropic">Anthropic Claude</option>
                            <option value="openai">OpenAI GPT-4</option>
                        </select>
                    </div>
                    <div className="space-y-1">
                        <label className="text-[10px] text-zinc-500">API Key</label>
                        <input 
                            type="password" 
                            value={extConfig.promptGeneration.apiKey}
                            onChange={(e) => handleExtChange('promptGeneration', 'apiKey', e.target.value)}
                            placeholder="sk-..."
                            className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                        />
                    </div>
                </div>
            </div>

            {/* Image Gen API (Multi-Provider) */}
            <div className="bg-black/20 border border-white/5 rounded-lg p-4 space-y-4">
                <div className="flex items-center gap-2 mb-2">
                    <Icons.Image className="w-4 h-4 text-pink-400" />
                    <span className="text-sm font-medium text-zinc-200">Image Generation</span>
                </div>
                
                <div className="space-y-3">
                    <div className="space-y-1">
                         <label className="text-[10px] text-zinc-500">Active Provider</label>
                         <select 
                            value={extConfig.imageGeneration.activeProvider} 
                            onChange={(e) => handleExtChange('imageGeneration', 'activeProvider', e.target.value)}
                            className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300"
                         >
                            <option value="gemini">Gemini 2.5 Flash (Default)</option>
                            <option value="replicate">Replicate (Flux/SDXL)</option>
                            <option value="stability">Stability AI (SD3)</option>
                            <option value="fal">Fal.AI (Realtime)</option>
                            <option value="midjourney">MidJourney (Proxy)</option>
                            <option value="custom">Custom Endpoint</option>
                         </select>
                    </div>

                    {/* Dynamic Settings based on Provider */}
                    {extConfig.imageGeneration.activeProvider === 'replicate' && (
                        <>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500">Replicate API Token</label>
                                <input 
                                    type="password" 
                                    value={extConfig.imageGeneration.replicate.apiKey}
                                    onChange={(e) => handleNestedExtChange('imageGeneration', 'replicate', 'apiKey', e.target.value)}
                                    className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                                />
                            </div>
                            <div className="space-y-1">
                                <label className="text-[10px] text-zinc-500">Model String</label>
                                <input 
                                    type="text" 
                                    value={extConfig.imageGeneration.replicate.modelString}
                                    onChange={(e) => handleNestedExtChange('imageGeneration', 'replicate', 'modelString', e.target.value)}
                                    className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                                />
                            </div>
                        </>
                    )}

                    {extConfig.imageGeneration.activeProvider === 'stability' && (
                         <div className="space-y-1">
                            <label className="text-[10px] text-zinc-500">Stability API Key</label>
                            <input 
                                type="password" 
                                value={extConfig.imageGeneration.stability.apiKey}
                                onChange={(e) => handleNestedExtChange('imageGeneration', 'stability', 'apiKey', e.target.value)}
                                className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                            />
                        </div>
                    )}
                    
                     {extConfig.imageGeneration.activeProvider === 'fal' && (
                         <div className="space-y-1">
                            <label className="text-[10px] text-zinc-500">Fal.AI Key</label>
                            <input 
                                type="password" 
                                value={extConfig.imageGeneration.fal.apiKey}
                                onChange={(e) => handleNestedExtChange('imageGeneration', 'fal', 'apiKey', e.target.value)}
                                className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                            />
                        </div>
                    )}
                </div>
            </div>
          </div>
      </div>

      {/* --- DevOps & Infrastructure --- */}
      <div className="space-y-6 border-b border-white/5 pb-6">
          <h4 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider flex items-center gap-2">
            <Icons.Cloud className="w-4 h-4" /> DevOps & Web3 Infrastructure
          </h4>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
               {/* Code & Data */}
               <div className="bg-black/20 border border-white/5 rounded-lg p-4 space-y-4">
                   <div className="flex items-center gap-2 mb-2">
                       <Icons.Github className="w-4 h-4 text-white" />
                       <span className="text-sm font-medium text-zinc-200">Code & Data Repositories</span>
                   </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-1">
                           <label className="text-[10px] text-zinc-500">GitHub Token</label>
                           <input 
                               type="password" 
                               value={extConfig.infrastructure.github.token}
                               onChange={(e) => handleNestedExtChange('infrastructure', 'github', 'token', e.target.value)}
                               placeholder="ghp_..."
                               className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                           />
                        </div>
                         <div className="space-y-1">
                           <label className="text-[10px] text-zinc-500">HuggingFace Token</label>
                           <input 
                               type="password" 
                               value={extConfig.infrastructure.huggingFace.token}
                               onChange={(e) => handleNestedExtChange('infrastructure', 'huggingFace', 'token', e.target.value)}
                               placeholder="hf_..."
                               className="w-full bg-dark-card border border-dark-border rounded px-2 py-1.5 text-xs text-zinc-300" 
                           />
                        </div>
                   </div>
               </div>
          </div>
      </div>

      <div className="pt-4 border-t border-white/5 flex justify-end gap-3">
            <Button variant="ghost" onClick={() => {
                setExtConfig(ConfigService.get().externalApis);
            }}>Discard</Button>
            <Button onClick={handleSave} icon={Icons.Check} disabled={!isDirty}>Save Integrations</Button>
      </div>
      {lastSave && <p className="text-[10px] text-zinc-500 text-right">Saved at {new Date(lastSave).toLocaleTimeString()}</p>}
    </div>
  );
};