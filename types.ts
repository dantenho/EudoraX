
/**
 * @file types.ts
 * @description Core TypeScript definitions for EudoraX focused studio.
 */
import React from 'react';

export enum ToolType {
  DASHBOARD = 'DASHBOARD',
  
  // Pillar 1: Image Generation (formerly Synthesis Forge)
  IMAGE_GENERATOR = 'IMAGE_GENERATOR', // Txt2Img
  IMAGE_TO_IMAGE = 'IMAGE_TO_IMAGE',
  INPAINTING = 'INPAINTING',
  UPSCALER = 'UPSCALER',
  STUDIO_NODES = 'STUDIO_NODES',
  LORA_TRAINER = 'LORA_TRAINER',
  PIXEL_ART_GENERATOR = 'PIXEL_ART_GENERATOR',
  IMAGE_ASSETS = 'IMAGE_ASSETS',
  
  // Pillar 2: Intelligence Node (Gemini Studio)
  GEMINI_STUDIO = 'GEMINI_STUDIO',
  GEMINI_NANO = 'GEMINI_NANO',
  GEMINI_VEO = 'GEMINI_VEO',
  GEMINI_VOICE = 'GEMINI_VOICE',
  GEMINI_AUTH = 'GEMINI_AUTH',
  GEMINI_SENSOR = 'GEMINI_SENSOR',
  GEMINI_CODE = 'GEMINI_CODE', // Added based on usage in GeminiStudio
  
  ADMIN_PANEL = 'ADMIN_PANEL',
}

export interface ToolItem {
  id: ToolType | string;
  label: string;
  subLabel?: string;
  icon: React.ElementType;
  children?: ToolItem[];
  isCategoryHeader?: boolean;
  defaultOpen?: boolean;
}

export interface LoRAStyle {
  id: string;
  name: string;
  image: string;
  category: 'Most Used' | 'Art Style' | 'Newest' | 'Pixel Art' | 'Realistic';
  prompts?: {
    positive: string;
    negative: string;
  };
}

export enum VoiceName {
  Kore = 'Kore',
  Puck = 'Puck',
  Charon = 'Charon',
  Fenrir = 'Fenrir',
  Zephyr = 'Zephyr',
}
