
import { Icons } from '../components/Icons.tsx';

export enum GeminiOperationMode {
  AUTH = 'AUTH',
  TXT_TO_IMAGE = 'TXT_TO_IMAGE',
  IMG_TO_IMG = 'IMG_TO_IMG',
  INPAINTING = 'INPAINTING',
  UPSCALER = 'UPSCALER',
  STUDIO_NODES = 'STUDIO_NODES',
  NANO_BANANA = 'NANO_BANANA',
  WHISK_VEO = 'WHISK_VEO',
  AGENT_TRAINING = 'AGENT_TRAINING',
  LORA_SERVICE = 'LORA_SERVICE',
  PIXEL_FORGE = 'PIXEL_FORGE',
  VOICE_SYNTH = 'VOICE_SYNTH',
  ASSET_LIBRARY = 'ASSET_LIBRARY',
  SENSOR_FUSION = 'SENSOR_FUSION',
  CODE_FORGE = 'CODE_FORGE',
  PROFILE = 'PROFILE',
}

export interface GeminiStudioTool {
  id: GeminiOperationMode;
  label: string;
  icon: any;
  description: string;
  beta?: boolean;
}

export const GEMINI_STUDIO_PLUGINS: GeminiStudioTool[] = [
  { 
    id: GeminiOperationMode.TXT_TO_IMAGE, 
    label: 'Text to Image', 
    icon: Icons.Sparkles, 
    description: 'Neural Text-to-Image Generation.' 
  },
  { 
    id: GeminiOperationMode.IMG_TO_IMG, 
    label: 'Image to Image', 
    icon: Icons.Image, 
    description: 'Variation and Style Transfer.' 
  },
  { 
    id: GeminiOperationMode.LORA_SERVICE, 
    label: 'LoRA Training', 
    icon: Icons.Dna, 
    description: 'Fine-tune styles.' 
  },
  { 
    id: GeminiOperationMode.INPAINTING, 
    label: 'Inpainting', 
    icon: Icons.Erase, 
    description: 'Smart Fill and Edit.' 
  },
  { 
    id: GeminiOperationMode.UPSCALER, 
    label: 'Upscaler', 
    icon: Icons.Maximize2, 
    description: 'Super Resolution.' 
  },
  { 
    id: GeminiOperationMode.STUDIO_NODES, 
    label: 'Studio Nodes', 
    icon: Icons.Network, 
    description: 'Workflow Graph.',
    beta: true
  },
  { 
    id: GeminiOperationMode.PIXEL_FORGE, 
    label: 'Pixel Art', 
    icon: Icons.Gamepad, 
    description: 'Retro Sprite Synthesis.' 
  },
  { 
    id: GeminiOperationMode.WHISK_VEO, 
    label: 'Veo Motion', 
    icon: Icons.Clapperboard, 
    description: 'Temporal Video Synthesis.' 
  },
  { 
    id: GeminiOperationMode.VOICE_SYNTH, 
    label: 'Voice Forge', 
    icon: Icons.Voice, 
    description: 'Native Audio Generation.' 
  },
  { 
    id: GeminiOperationMode.NANO_BANANA, 
    label: 'Nano Fast', 
    icon: Icons.Zap, 
    description: 'High-speed local inference.' 
  },
  { 
    id: GeminiOperationMode.SENSOR_FUSION, 
    label: 'Sensor Fusion', 
    icon: Icons.Globe, 
    description: 'Telemetry dashboard.' 
  },
  { 
    id: GeminiOperationMode.ASSET_LIBRARY, 
    label: 'Asset Vault', 
    icon: Icons.Layers, 
    description: 'Resource management.' 
  },
  { 
    id: GeminiOperationMode.PROFILE, 
    label: 'Profile', 
    icon: Icons.User, 
    description: 'Neural Identity settings.' 
  }
];
