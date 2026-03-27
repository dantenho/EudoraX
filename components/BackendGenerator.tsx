
import React, { useState } from 'react';
import { GoogleGenAI } from "@google/genai";
import { Button } from './Button.tsx';
import { Icons } from './Icons.tsx';

/**
 * @component BackendBuilder
 * @description AI-driven backend generator that creates Python FastAPI services.
 */
export const BackendBuilder: React.FC = () => {
  const [status, setStatus] = useState<string>('Idle');
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (msg: string) => setLogs(prev => [...prev, `> ${msg}`]);

  // Mock deployment function
  const deployToFirebaseFunctions = async (_code: string) => {
    addLog("Packaging Python runtime...");
    await new Promise(resolve => setTimeout(resolve, 800));
    addLog("Pushing to Firebase Cloud Functions (us-central1)...");
    await new Promise(resolve => setTimeout(resolve, 1500));
    addLog("Deployment successful: https://api.eudorax.io/v1/users");
  };

  const createPythonBackend = async () => {
    setStatus('Generating');
    addLog("Initializing Gemini 3 Pro for Python code generation...");
    
    try {
      // Initialize Gemini with the API Key
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      const prompt = `Create a production-ready FastAPI (Python 3.14) application file content with:
      - Firestore client initialization using firebase-admin
      - /api/users CRUD endpoints (GET, POST)
      - CORS enabled for localhost:3000
      - Use Pydantic v2 models for User schema
      - Include comments for uvloop optimization
      Return only the raw Python code. Do not include markdown backticks.`;

      const response = await ai.models.generateContent({
        model: "gemini-3-pro-preview",
        contents: prompt,
        config: {
            temperature: 0.2,
            systemInstruction: "You are a Senior Python Backend Engineer. Write clean, typed, high-performance FastAPI code.",
        }
      });

      const generatedCode = response.text || "";
      addLog("Code generation complete.");
      addLog(`Generated ${generatedCode.length} bytes of Python.`);
      
      setStatus('Deploying');
      await deployToFirebaseFunctions(generatedCode);
      setStatus('Idle');
      
    } catch (error) {
      addLog(`Error: ${error instanceof Error ? error.message : String(error)}`);
      setStatus('Error');
    }
  };

  return (
    <div className="p-8 bg-dark-card border border-white/5 rounded-3xl max-w-2xl mx-auto space-y-6 font-sans animate-fade-in">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-blue-600/10 rounded-xl text-blue-400 border border-blue-500/20">
            <Icons.Server className="w-6 h-6" />
        </div>
        <div>
            <h3 className="text-lg font-bold text-white">Backend Generator</h3>
            <p className="text-sm text-zinc-500">Auto-provision FastAPI services via Gemini</p>
        </div>
      </div>

      <div className="bg-black/40 rounded-xl p-4 font-mono text-xs text-zinc-400 h-64 overflow-y-auto custom-scrollbar border border-white/5 shadow-inner">
        {logs.length === 0 ? <span className="opacity-30">// System Logs... Waiting for command.</span> : logs.map((l, i) => <div key={i} className="mb-1">{l}</div>)}
        {status === 'Deploying' && <div className="animate-pulse text-blue-400">&gt; Uploading artifacts...</div>}
      </div>

      <Button 
        onClick={createPythonBackend} 
        loading={status !== 'Idle' && status !== 'Error'} 
        icon={Icons.Zap}
        className="w-full h-12"
      >
        {status === 'Idle' || status === 'Error' ? 'Generate & Deploy Backend' : status}
      </Button>
    </div>
  );
};
