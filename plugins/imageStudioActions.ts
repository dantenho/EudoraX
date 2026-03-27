
import { Icons } from '../components/Icons';

export enum OperationMode {
  TXT2IMG = 'TXT2IMG',
  IMG2IMG = 'IMG2IMG',
  LORA_TRAINING = 'LORA_TRAINING',
  GIF_CREATOR = 'GIF_CREATOR',
  INPAINT = 'INPAINT',
  POSTPROCESSING = 'POSTPROCESSING',
  AI_WORKFLOW = 'AI_WORKFLOW',
}

export interface StudioTool {
  id: OperationMode;
  label: string;
  icon: any;
  description: string;
  beta?: boolean;
}

export const IMAGE_STUDIO_PLUGINS: StudioTool[] = [
  { 
    id: OperationMode.TXT2IMG, 
    label: '1. Txt to Image', 
    icon: Icons.File, 
    description: 'Direct synthesis from text prompts.' 
  },
  { 
    id: OperationMode.IMG2IMG, 
    label: '2. Image to Image', 
    icon: Icons.Image, 
    description: 'Transform existing visuals.' 
  },
  { 
    id: OperationMode.LORA_TRAINING, 
    label: '3. LoRA Training', 
    icon: Icons.Dna, 
    description: 'Fine-tune styles and concepts.' 
  },
  { 
    id: OperationMode.GIF_CREATOR, 
    label: '4. Gif Creator', 
    icon: Icons.Clapperboard, 
    description: 'Animate static frames.' 
  },
  { 
    id: OperationMode.INPAINT, 
    label: '5. Inpaint', 
    icon: Icons.Selection, 
    description: 'Edit specific image regions.' 
  },
  { 
    id: OperationMode.POSTPROCESSING, 
    label: '6. Postprocessing', 
    icon: Icons.Sliders, 
    description: 'Upscale and color correction.' 
  },
  { 
    id: OperationMode.AI_WORKFLOW, 
    label: '7. AI Workflow', 
    icon: Icons.Network, 
    description: 'Gemini Function Call orchestration.',
    beta: true
  }
];
