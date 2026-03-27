
import { ToolType, ToolItem } from './types.ts';
import { Icons } from './components/Icons.tsx';

export const APP_NAME = "EudoraX";

export const TOOLS: ToolItem[] = [
  { id: ToolType.DASHBOARD, label: "Command Center", icon: Icons.Dashboard },

  { isCategoryHeader: true, id: 'creative-header', label: 'LoRA Forge Suite', icon: Icons.Zap },
  {
    id: 'image-forge-parent',
    label: "Creative Synthesis",
    icon: Icons.Palette,
    defaultOpen: true,
    children: [
      { id: ToolType.LORA_TRAINER, label: "LoRA Forge", icon: Icons.Dna, subLabel: "Training" },
      { id: ToolType.IMAGE_GENERATOR, label: "Neural Images", icon: Icons.Sparkles, subLabel: "Synthesis" },
      { id: ToolType.GEMINI_VOICE, label: "Voice Forge", icon: Icons.Voice, subLabel: "TTS" },
      { id: ToolType.GEMINI_VEO, label: "Motion Forge", icon: Icons.Clapperboard, subLabel: "Video" },
      { id: ToolType.PIXEL_ART_GENERATOR, label: "Pixel Forge", icon: Icons.Gamepad, subLabel: "Grid Art" },
      { id: ToolType.IMAGE_ASSETS, label: "Asset Vault", icon: Icons.Layers, subLabel: "Storage" },
    ]
  },
  
  { isCategoryHeader: true, id: 'google-header', label: 'Intelligence Pillar', icon: Icons.Globe },
  {
    id: 'google-studio-parent',
    label: "Intelligence Hub",
    icon: Icons.Box,
    children: [
      { id: ToolType.GEMINI_NANO, label: "Nano Fast", icon: Icons.Zap, subLabel: "Local Gen" },
      { id: ToolType.GEMINI_SENSOR, label: "Sensor Fusion", icon: Icons.Globe, subLabel: "Telemetry" },
      { id: ToolType.GEMINI_CODE, label: "Code Forge", icon: Icons.Code, subLabel: "Vim Mode" },
    ]
  },

  { isCategoryHeader: true, id: 'system-header', label: 'Root Config', icon: Icons.Settings },
  { id: ToolType.ADMIN_PANEL, label: "Kernel Admin", icon: Icons.Shield },
];
