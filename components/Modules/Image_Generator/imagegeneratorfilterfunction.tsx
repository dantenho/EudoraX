
/**
 * @file imagegeneratorfilterfunction.tsx
 * @description Style definitions and filtering logic for LoRA adapters.
 */
import { LoRAStyle } from '../../../types.ts';

export const LORA_STYLES: LoRAStyle[] = [
  { 
    id: 'none', 
    name: 'Standard Engine', 
    image: 'https://images.unsplash.com/photo-1618193139062-2c5bf4f935b7?q=80&w=300&auto=format&fit=crop', 
    category: 'Realistic',
    prompts: { positive: "", negative: "" }
  },
  { 
    id: 's_pixel_01', 
    name: '8-Bit Retro', 
    image: 'https://images.unsplash.com/photo-1550745165-9bc0b252726f?q=80&w=300&auto=format&fit=crop', 
    category: 'Pixel Art',
    prompts: {
        positive: "pixel art style, 8-bit, retro gaming, limited color palette, blocky, sprite, video game asset, nostalgic",
        negative: "photorealistic, smooth, blurred, high resolution, 3d, realistic lighting"
    }
  },
  { 
    id: 's_pixel_02', 
    name: 'Isometric Sprite', 
    image: 'https://images.unsplash.com/photo-1627398242454-45a1465c2479?q=80&w=300&auto=format&fit=crop', 
    category: 'Pixel Art',
    prompts: {
        positive: "isometric pixel art, video game map, diorama, 16-bit, detailed sprites, sharp edges, game environment",
        negative: "perspective, vanishing point, realism, painting, sketch"
    }
  },
  { 
    id: 's_port_01', 
    name: 'Studio Portrait', 
    image: 'https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?q=80&w=300&auto=format&fit=crop', 
    category: 'Realistic',
    prompts: {
        positive: "studio portrait, cinematic headshot, bokeh background, 85mm lens, sharp focus on eyes, dramatic lighting",
        negative: "outdoor, sun flare, messy, snapshot, low quality"
    }
  }
];

export const getFilteredStyles = (styles: LoRAStyle[], filter: string): LoRAStyle[] => {
  if (filter === 'Most Used') return styles.slice(0, 3);
  if (filter === 'Newest') return styles.slice(-3);
  return styles.filter(s => s.category === filter || filter === 'Art Style');
};
