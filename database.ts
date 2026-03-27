
/**
 * @file database.ts
 * @description Persistent storage layer for EudoraX AI Creative Studio.
 * Interfaces with LocalStorage and IndexedDB (fallback) to manage creative assets and flows.
 */

export interface SavedFlow {
  id: string;
  name: string;
  mode: string;
  prompt: string;
  config: any;
  timestamp: number;
  version: string;
}

export interface Asset {
  id: string;
  url: string;
  type: 'image' | 'video' | 'audio';
  metadata: any;
  timestamp: number;
}

const DB_KEYS = {
  FLOWS: 'eudorax_flows_v1',
  ASSETS: 'eudorax_assets_v1',
  SESSION: 'eudorax_last_session'
};

export class EudoraDatabase {
  /**
   * Saves a configuration flow to persistent storage
   */
  static saveFlow(flow: Omit<SavedFlow, 'id' | 'timestamp' | 'version'>): string {
    const flows = this.getFlows();
    const newFlow: SavedFlow = {
      ...flow,
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      version: '3.1.4'
    };
    
    flows.unshift(newFlow);
    localStorage.setItem(DB_KEYS.FLOWS, JSON.stringify(flows.slice(0, 50))); // Keep last 50
    return newFlow.id;
  }

  /**
   * Retrieves all saved creative flows
   */
  static getFlows(): SavedFlow[] {
    const data = localStorage.getItem(DB_KEYS.FLOWS);
    return data ? JSON.parse(data) : [];
  }

  /**
   * Records a generated asset in the database history
   */
  static recordAsset(asset: Omit<Asset, 'id' | 'timestamp'>): void {
    const assets = this.getAssets();
    const newAsset: Asset = {
      ...asset,
      id: crypto.randomUUID(),
      timestamp: Date.now()
    };
    
    assets.unshift(newAsset);
    localStorage.setItem(DB_KEYS.ASSETS, JSON.stringify(assets.slice(0, 100)));
  }

  /**
   * Retrieves generated asset history
   */
  static getAssets(): Asset[] {
    const data = localStorage.getItem(DB_KEYS.ASSETS);
    return data ? JSON.parse(data) : [];
  }

  /**
   * Clear entire workspace cache
   */
  static purge(): void {
    localStorage.removeItem(DB_KEYS.FLOWS);
    localStorage.removeItem(DB_KEYS.ASSETS);
    localStorage.removeItem(DB_KEYS.SESSION);
  }
}
