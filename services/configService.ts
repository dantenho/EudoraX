

/**
 * @file configService.ts
 * @description Service for managing application-wide configuration settings.
 * Persists data to localStorage to simulate backend config management.
 */

export interface AppConfig {
  api: {
    timeout: number; // milliseconds
    logRequests: boolean;
    mockLatency: number; // Simulate network conditions
  };
  externalApis: {
    promptGeneration: {
      provider: 'gemini' | 'anthropic' | 'openai';
      apiKey: string;
      modelId: string;
    };
    imageGeneration: {
      activeProvider: 'gemini' | 'replicate' | 'stability' | 'fal' | 'midjourney' | 'custom';
      gemini: { apiKey?: string }; // Optional override
      replicate: { apiKey: string; modelString: string };
      stability: { apiKey: string; coreModel: string };
      fal: { apiKey: string };
      midjourney: { apiKey: string; endpoint: string }; // Proxy configuration
      custom: { endpoint: string; apiKey: string };
    };
    webScraping: {
      provider: 'browser-npu' | 'external-service';
      apiKey?: string;
    };
    infrastructure: {
        github: { token: string };
        huggingFace: { token: string };
    };
  };
  mcp: {
      customModels: string; // JSON string defining custom models
  };
  imageGeneration: {
    defaultModel: string;
    defaultAspectRatio: string;
    globalNegativePrompt: string;
    safetyFilterLevel: 'block_none' | 'block_few' | 'block_some' | 'block_most';
    enableHDUpscale: boolean;
  };
  system: {
    maintenanceMode: boolean;
    publicRegistration: boolean;
    debugMode: boolean;
  };
}

const CONFIG_KEY = 'eudorax_app_config_v3';

const DEFAULT_CONFIG: AppConfig = {
  api: {
    timeout: 30000,
    logRequests: true,
    mockLatency: 0,
  },
  externalApis: {
    promptGeneration: {
      provider: 'gemini',
      apiKey: '',
      modelId: 'gemini-2.5-pro',
    },
    imageGeneration: {
      activeProvider: 'gemini',
      gemini: {},
      replicate: { apiKey: '', modelString: 'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b' },
      stability: { apiKey: '', coreModel: 'stable-diffusion-xl-1024-v1-0' },
      fal: { apiKey: '' },
      midjourney: { apiKey: '', endpoint: '' },
      custom: { endpoint: '', apiKey: '' },
    },
    webScraping: {
      provider: 'browser-npu',
    },
    infrastructure: {
        github: { token: '' },
        huggingFace: { token: '' }
    }
  },
  mcp: {
    customModels: JSON.stringify([
        { id: "custom-flux-lora-v1", name: "Flux LoRA Custom", type: "image" },
        { id: "llama-3-70b-groq", name: "Llama 3 (Groq)", type: "text" }
    ], null, 2)
  },
  imageGeneration: {
    defaultModel: 'gemini-2.5-flash-image',
    defaultAspectRatio: '1:1',
    globalNegativePrompt: 'blurry, bad anatomy, watermark, text, low quality, distorted, ugly, pixelated',
    safetyFilterLevel: 'block_some',
    enableHDUpscale: true,
  },
  system: {
    maintenanceMode: false,
    publicRegistration: true,
    debugMode: false,
  }
};

export const ConfigService = {
  /**
   * Retrieves the current configuration from storage or returns defaults.
   */
  get: (): AppConfig => {
    try {
      const stored = localStorage.getItem(CONFIG_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        // Deep merge to prevent undefined nested objects
        return {
            ...DEFAULT_CONFIG,
            ...parsed,
            api: { ...DEFAULT_CONFIG.api, ...(parsed.api || {}) },
            externalApis: { 
                ...DEFAULT_CONFIG.externalApis, 
                ...(parsed.externalApis || {}),
                imageGeneration: { ...DEFAULT_CONFIG.externalApis.imageGeneration, ...(parsed.externalApis?.imageGeneration || {}) },
                infrastructure: { ...DEFAULT_CONFIG.externalApis.infrastructure, ...(parsed.externalApis?.infrastructure || {}) }
            },
            mcp: { ...DEFAULT_CONFIG.mcp, ...(parsed.mcp || {}) },
            imageGeneration: { ...DEFAULT_CONFIG.imageGeneration, ...(parsed.imageGeneration || {}) },
            system: { ...DEFAULT_CONFIG.system, ...(parsed.system || {}) }
        };
      }
    } catch (e) {
      console.error("Failed to load config", e);
    }
    return DEFAULT_CONFIG;
  },

  /**
   * Saves the entire configuration object.
   */
  save: (config: AppConfig) => {
    try {
      localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
      window.dispatchEvent(new Event('config-updated'));
    } catch (e) {
      console.error("Failed to save config", e);
    }
  },

  /**
   * Updates a specific section of the configuration.
   */
  updateSection: <K extends keyof AppConfig>(section: K, data: Partial<AppConfig[K]>) => {
    const current = ConfigService.get();
    current[section] = { ...current[section], ...data } as any;
    ConfigService.save(current);
  },
  
  /**
   * Resets configuration to defaults.
   */
  reset: () => {
    ConfigService.save(DEFAULT_CONFIG);
  }
};