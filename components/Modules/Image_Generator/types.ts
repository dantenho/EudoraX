
/**
 * @file types.ts
 * @description Type definitions for the Image Generator Module.
 * Provides the structure for extensions, metadata, and asset follow configurations.
 */
import React from 'react';
import { Icons } from '../../Icons';

/**
 * Metadata for a generative extension.
 */
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

/**
 * Full extension object including the React component.
 */
export interface ImageGenExtension extends ExtensionMetadata {
  component: React.FC<any>;
  beta?: boolean;
}

/**
 * Configuration for the AssetFollow feature.
 */
export interface AssetFollowConfig {
  assetUri?: string;
  intensity: number;
  instruction: 'style' | 'structure' | 'composition' | 'timbre';
}
