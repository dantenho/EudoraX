
/**
 * @file GeminiAssets.tsx
 * @description Central vault for the Unified Forge. 
 * Manages Images, Video, Audio, and LoRA Safetensors.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Icons } from '../../Icons.tsx';

interface Folder {
  id: string;
  name: string;
}

interface Asset {
  id: string;
  name: string;
  url: string;
  tags: string[];
  folderId: string | null;
  createdAt: string;
}

export const GeminiAssets: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<'generations' | 'weights'>('generations');
  const [folders, setFolders] = useState<Folder[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [search, setSearch] = useState('');
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [newFolderName, setNewFolderName] = useState('');
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [newTag, setNewTag] = useState('');

  const fetchData = useCallback(async () => {
    const token = localStorage.getItem('eudora_token');
    try {
      const [fRes, aRes] = await Promise.all([
        fetch('/api/folders', { headers: { 'Authorization': `Bearer ${token}` } }),
        fetch('/api/assets', { headers: { 'Authorization': `Bearer ${token}` } })
      ]);
      if (fRes.ok) setFolders(await fRes.json());
      if (aRes.ok) setAssets(await aRes.json());
    } catch (err) {
      console.error("Failed to fetch assets", err);
    }
  }, []);

  useEffect(() => {
    const loadData = async () => {
      await fetchData();
    };
    loadData();
  }, [fetchData]);

  const handleCreateFolder = async () => {
    if (!newFolderName) return;
    const token = localStorage.getItem('eudora_token');
    try {
      const res = await fetch('/api/folders', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ name: newFolderName })
      });
      if (res.ok) {
        setNewFolderName('');
        setShowNewFolder(false);
        fetchData();
      }
    } catch (err) {
      console.error("Failed to create folder", err);
    }
  };

  const handleUpdateAsset = async (id: string, updates: Partial<Asset>) => {
    const token = localStorage.getItem('eudora_token');
    try {
      const res = await fetch(`/api/assets/${id}`, {
        method: 'PUT',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(updates)
      });
      if (res.ok) {
        fetchData();
        if (selectedAsset?.id === id) {
          const updated = await res.json();
          setSelectedAsset(updated);
        }
      }
    } catch (err) {
      console.error("Failed to update asset", err);
    }
  };

  const handleDeleteAsset = async (id: string) => {
    const token = localStorage.getItem('eudora_token');
    try {
      const res = await fetch(`/api/assets/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        fetchData();
        setSelectedAsset(null);
      }
    } catch (err) {
      console.error("Failed to delete asset", err);
    }
  };

  useEffect(() => {
    if (!mountRef.current) return;
    
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    
    const width = 300;
    const height = 300;
    
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);

    const geometry = new THREE.IcosahedronGeometry(1, 12);
    const material = new THREE.MeshNormalMaterial({ wireframe: true, opacity: 0.8, transparent: true });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    camera.position.z = 2.5;

    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      sphere.rotation.x += 0.005;
      sphere.rotation.y += 0.008;
      renderer.render(scene, camera);
    };
    animate();

    const currentMount = mountRef.current;
    return () => {
      cancelAnimationFrame(animationId);
      if (currentMount && renderer.domElement) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [selectedAsset]); // Re-init on selection for flair

  const filteredAssets = assets.filter(a => {
    const matchesSearch = a.name.toLowerCase().includes(search.toLowerCase()) || a.tags.some(t => t.toLowerCase().includes(search.toLowerCase()));
    const matchesFolder = selectedFolderId ? a.folderId === selectedFolderId : true;
    return matchesSearch && matchesFolder;
  });

  return (
    <div className="flex h-full gap-8 animate-fade-in font-sans">
      <div className="w-64 flex-shrink-0 flex flex-col space-y-6">
        <div className="flex items-center justify-between px-2">
          <h3 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Collections</h3>
          <button onClick={() => setShowNewFolder(true)} className="text-blue-400 hover:text-blue-300 transition-colors">
            <Icons.Plus className="w-4 h-4" />
          </button>
        </div>

        {showNewFolder && (
          <div className="p-4 bg-white/5 rounded-2xl border border-white/10 space-y-3">
            <input 
              type="text" 
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="Folder name..."
              className="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-2 text-xs text-white"
            />
            <div className="flex gap-2">
              <button onClick={handleCreateFolder} className="flex-1 py-1.5 bg-blue-600 text-white text-[10px] font-black uppercase rounded-lg">Create</button>
              <button onClick={() => setShowNewFolder(false)} className="flex-1 py-1.5 bg-zinc-800 text-zinc-400 text-[10px] font-black uppercase rounded-lg">Cancel</button>
            </div>
          </div>
        )}

        <div className="space-y-1">
          <button 
            onClick={() => setSelectedFolderId(null)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-xs font-bold transition-all ${selectedFolderId === null ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20' : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'}`}
          >
            <Icons.Layers className="w-4 h-4" />
            All Assets
          </button>
          {folders.map(f => (
            <button 
              key={f.id}
              onClick={() => setSelectedFolderId(f.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-2xl text-xs font-bold transition-all ${selectedFolderId === f.id ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20' : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'}`}
            >
              <Icons.Folder className="w-4 h-4" />
              {f.name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 flex flex-col space-y-8 min-w-0">
        <header className="flex items-center justify-between px-6 gap-6">
           <div className="flex-1 relative">
              <Icons.Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
              <input 
                type="text" 
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by name or tags..."
                className="w-full bg-white/5 border border-white/5 rounded-2xl pl-12 pr-4 py-3 text-sm text-white focus:ring-1 focus:ring-blue-500 transition-all"
              />
           </div>
           <div className="flex bg-white/5 p-1 rounded-xl border border-white/5">
              <button onClick={() => setActiveTab('generations')} className={`px-4 py-1.5 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${activeTab === 'generations' ? 'bg-blue-600 text-white' : 'text-zinc-500'}`}>Generations</button>
              <button onClick={() => setActiveTab('weights')} className={`px-4 py-1.5 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all ${activeTab === 'weights' ? 'bg-blue-600 text-white' : 'text-zinc-500'}`}>LoRA Weights</button>
           </div>
        </header>

        <div className="flex-1 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 overflow-y-auto custom-scrollbar pr-4 content-start">
          {activeTab === 'generations' ? (
            filteredAssets.length > 0 ? (
              filteredAssets.map(asset => (
                <div 
                  key={asset.id} 
                  onClick={() => setSelectedAsset(asset)}
                  className={`group relative aspect-[4/5] bg-dark-card border rounded-3xl overflow-hidden transition-all duration-500 cursor-pointer shadow-2xl ${selectedAsset?.id === asset.id ? 'border-blue-500 ring-2 ring-blue-500/20' : 'border-white/5 hover:border-blue-500/30'}`}
                >
                   <img src={asset.url} className={`w-full h-full object-cover transition-all duration-1000 ${selectedAsset?.id === asset.id ? 'grayscale-0 opacity-100' : 'grayscale opacity-40 group-hover:grayscale-0 group-hover:opacity-100'}`} alt={asset.name} referrerPolicy="no-referrer" />
                   <div className="absolute inset-x-0 bottom-0 p-6 bg-gradient-to-t from-black via-black/40 to-transparent">
                      <div className="text-[8px] font-black text-blue-400 uppercase tracking-widest mb-1">IMAGE_FORGE</div>
                      <div className="text-xs font-bold text-white truncate">{asset.name}</div>
                   </div>
                </div>
              ))
            ) : (
              <div className="col-span-full flex flex-col items-center justify-center py-20 space-y-4 opacity-40">
                <Icons.Layers className="w-12 h-12" />
                <p className="text-sm font-black uppercase tracking-widest">No assets found</p>
              </div>
            )
          ) : (
            <div className="col-span-full flex flex-col items-center justify-center py-20 space-y-4 opacity-40">
              <Icons.Dna className="w-12 h-12" />
              <p className="text-sm font-black uppercase tracking-widest">LoRA Weights module offline</p>
            </div>
          )}
        </div>
      </div>

      {/* Neural Inspector Sidecar */}
      <div className="w-80 flex-shrink-0 bg-dark-sidebar/40 border border-white/5 rounded-[2.5rem] p-8 flex flex-col space-y-8 shadow-2xl overflow-y-auto custom-scrollbar">
         <h3 className="text-[10px] font-black text-zinc-500 uppercase tracking-widest text-center">Neural Inspection</h3>
         
         <div ref={mountRef} className="aspect-square w-full flex items-center justify-center bg-black/40 rounded-[2.5rem] border border-white/5 overflow-hidden">
            {selectedAsset ? (
              <img src={selectedAsset.url} className="w-full h-full object-cover opacity-20 blur-sm" alt="Preview" referrerPolicy="no-referrer" />
            ) : (
              <div className="text-zinc-700 font-black italic text-2xl tracking-tighter opacity-20">EUDORAX</div>
            )}
         </div>

         {selectedAsset ? (
           <div className="space-y-6">
              <div className="space-y-4">
                <div className="flex flex-col">
                  <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest">Asset Name</span>
                  <input 
                    type="text" 
                    value={selectedAsset.name}
                    onChange={(e) => handleUpdateAsset(selectedAsset.id, { name: e.target.value })}
                    className="bg-transparent border-none p-0 text-white font-bold focus:ring-0"
                  />
                </div>

                <div className="space-y-2">
                  <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest">Tags</span>
                  <div className="flex flex-wrap gap-2">
                    {selectedAsset.tags.map(tag => (
                      <span key={tag} className="px-2 py-1 bg-blue-500/10 border border-blue-500/20 rounded-lg text-[9px] font-black text-blue-400 uppercase flex items-center gap-1">
                        {tag}
                        <button onClick={() => handleUpdateAsset(selectedAsset.id, { tags: selectedAsset.tags.filter(t => t !== tag) })}>
                          <Icons.X className="w-2 h-2" />
                        </button>
                      </span>
                    ))}
                    <div className="flex items-center gap-1">
                      <input 
                        type="text" 
                        value={newTag}
                        onChange={(e) => setNewTag(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && newTag) {
                            handleUpdateAsset(selectedAsset.id, { tags: [...selectedAsset.tags, newTag] });
                            setNewTag('');
                          }
                        }}
                        placeholder="Add tag..."
                        className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-[9px] text-white w-20"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <span className="text-[8px] font-black text-zinc-500 uppercase tracking-widest">Move to Collection</span>
                  <select 
                    value={selectedAsset.folderId || ''}
                    onChange={(e) => handleUpdateAsset(selectedAsset.id, { folderId: e.target.value || null })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-[10px] text-white font-bold"
                  >
                    <option value="">No Collection</option>
                    {folders.map(f => (
                      <option key={f.id} value={f.id}>{f.name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="bg-black/40 p-6 rounded-3xl border border-white/5 space-y-4">
                 <div className="flex justify-between text-[10px] font-bold text-zinc-500 uppercase tracking-tighter">
                    <span>Created</span>
                    <span className="text-white font-mono">{new Date(selectedAsset.createdAt).toLocaleDateString()}</span>
                 </div>
                 <div className="flex justify-between text-[10px] font-bold text-zinc-500 uppercase tracking-tighter">
                    <span>Format</span>
                    <span className="text-white font-mono">PNG_SYNTH</span>
                 </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                 <button 
                  onClick={() => handleDeleteAsset(selectedAsset.id)}
                  className="py-3 bg-red-500/10 text-red-400 font-black text-[9px] uppercase rounded-xl hover:bg-red-500/20 transition-all tracking-widest border border-red-500/20"
                 >
                    Delete
                 </button>
                 <a 
                  href={selectedAsset.url} 
                  download 
                  target="_blank" 
                  rel="noreferrer"
                  className="py-3 bg-blue-600 text-white font-black text-[9px] uppercase rounded-xl hover:bg-blue-500 transition-all tracking-widest shadow-lg shadow-blue-500/20 text-center"
                 >
                    Download
                 </a>
              </div>
           </div>
         ) : (
           <div className="flex-1 flex flex-col items-center justify-center text-center space-y-4 opacity-20">
              <Icons.Layers className="w-12 h-12" />
              <p className="text-[10px] font-black uppercase tracking-[0.2em]">Select an asset to inspect</p>
           </div>
         )}
      </div>
    </div>
  );
};

