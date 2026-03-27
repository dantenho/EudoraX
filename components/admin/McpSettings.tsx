
import React, { useState } from 'react';
import { Icons } from '../Icons';
import { Button } from '../Button';
import { ConfigService, AppConfig } from '../../services/configService';

export const McpSettings: React.FC = () => {
  const [config, setConfig] = useState<AppConfig['mcp']>(ConfigService.get().mcp);
  const [isDirty, setIsDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (val: string) => {
    setConfig({ customModels: val });
    setIsDirty(true);
    try {
        JSON.parse(val);
        setError(null);
    } catch (_e) {
        setError("Invalid JSON format");
    }
  };

  const handleSave = () => {
    if (error) return;
    ConfigService.updateSection('mcp', config);
    setIsDirty(false);
  };

  return (
    <div className="bg-dark-card border border-dark-border rounded-xl p-6 animate-fade-in space-y-6">
       <div className="flex justify-between items-start">
        <div>
           <h3 className="text-lg font-medium text-white flex items-center gap-2">
            <Icons.Sliders className="w-5 h-5 text-purple-500" /> 
            Model Context Protocol (MCP)
          </h3>
          <p className="text-xs text-zinc-400 mt-1">
            Define custom model definitions and context parameters.
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <label className="text-sm font-medium text-zinc-300">Custom Models JSON</label>
        <div className="relative">
            <textarea
                value={config.customModels}
                onChange={(e) => handleChange(e.target.value)}
                className="w-full h-64 bg-black/40 border border-white/10 rounded-xl p-4 font-mono text-xs text-zinc-300 focus:ring-1 focus:ring-purple-500/50 resize-none"
            />
            {error && <div className="absolute bottom-4 right-4 text-xs text-red-400 bg-black/60 px-2 py-1 rounded">{error}</div>}
        </div>
        <p className="text-[10px] text-zinc-500">
            Define an array of model objects with `id`, `name`, and `type`.
        </p>
      </div>

      <div className="flex justify-end gap-3 pt-4 border-t border-white/5">
        <Button variant="ghost" onClick={() => setConfig(ConfigService.get().mcp)}>Discard</Button>
        <Button onClick={handleSave} disabled={!isDirty || !!error} icon={Icons.Check}>Save MCP Config</Button>
      </div>
    </div>
  );
};
