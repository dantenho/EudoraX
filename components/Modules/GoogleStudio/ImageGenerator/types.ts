
import React from 'react';
import { Icons } from '../Icons';

export interface ExtensionMetadata {
  id: string;
  name: string;
  description: string;
  longDescription?: string;
  version: string;
  author: string;
  icon: keyof typeof Icons;
  category: 'Synthesis' | 'Motion' | 'Audio' | 'Utility';
  tags: string[];
}

export interface ImageGenExtension extends ExtensionMetadata {
  component: React.FC<any>;
  beta?: boolean;
}

export interface AssetFollowConfig {
  assetUri?: string;
  intensity: number;
  instruction: 'style' | 'structure' | 'composition' | 'timbre';
}
