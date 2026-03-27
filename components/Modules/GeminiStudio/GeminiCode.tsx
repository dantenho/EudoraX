
import React, { useState, useEffect, useRef } from 'react';
import { Icons } from '../../Icons.tsx';
import { Button } from '../../Button.tsx';

// Mock Prism for type safety in this snippet
declare const Prism: any;

export const GeminiCode: React.FC = () => {
  const [mode, setMode] = useState<'NORMAL' | 'INSERT' | 'VISUAL'>('NORMAL');
  const [code, setCode] = useState<string>(`// EudoraX React 2026 Component
import { useAI } from 'react';

export default function NeuralComponent() {
  const { prediction } = useAI('gemini-nano-v2', {
    temperature: 0.7,
    stream: true
  });

  return (
    <div className="neural-container">
      <h1>{prediction || "Thinking..."}</h1>
      <Suspense fallback={<Skeleton />}>
        <AsyncDataStream source="node:ai" />
      </Suspense>
    </div>
  );
}`);
  
  const [cursor, setCursor] = useState({ line: 0, col: 0 });
  const [command, setCommand] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (mode === 'NORMAL') {
        if (e.key === 'i') {
          e.preventDefault();
          setMode('INSERT');
        } else if (e.key === ':') {
          e.preventDefault();
          setCommand(':');
        } else if (e.key === 'Escape') {
          setCommand('');
        }
        else if (e.key === 'h') setCursor(prev => ({ ...prev, col: Math.max(0, prev.col - 1) }));
        else if (e.key === 'l') setCursor(prev => ({ ...prev, col: prev.col + 1 }));
        else if (e.key === 'j') setCursor(prev => ({ ...prev, line: prev.line + 1 }));
        else if (e.key === 'k') setCursor(prev => ({ ...prev, line: Math.max(0, prev.line - 1) }));
      } else if (mode === 'INSERT') {
        if (e.key === 'Escape') {
          setMode('NORMAL');
          textareaRef.current?.blur();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mode]);

  useEffect(() => {
    if (typeof Prism !== 'undefined' && codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [code]);

  const handleCommandSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (command === ':w') {
      alert("Simulated Save: Written to EudoraVFS (Node 25)");
      setCommand('');
    } else if (command === ':q') {
      setCommand('');
    } else {
      setCommand('E492: Not an editor command: ' + command.substring(1));
      setTimeout(() => setCommand(''), 2000);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-6 animate-fade-in font-mono">
      <div className="flex-1 bg-[#1e1e1e] border border-white/5 rounded-[1rem] overflow-hidden flex flex-col shadow-2xl relative">
        <div className="h-8 bg-[#252526] border-b border-black flex items-center justify-between px-4 text-xs text-zinc-400 select-none">
          <div className="flex items-center gap-2">
            <Icons.Code className="w-3 h-3 text-blue-400" />
            <span>module.tsx</span>
            <span className="text-zinc-600">[RO]</span>
          </div>
          <div>typescript • react-2026</div>
        </div>

        <div className="flex-1 relative overflow-auto custom-scrollbar flex">
          <div className="w-10 bg-[#1e1e1e] text-zinc-600 text-right pr-3 pt-4 text-xs select-none border-r border-white/5">
            {code.split('\n').map((_, i) => (
              <div key={i} className="leading-6">{i + 1}</div>
            ))}
          </div>

          <div className="flex-1 relative pt-4 pl-4 text-sm leading-6">
            <pre className="m-0 bg-transparent! p-0!">
              <code ref={codeRef} className="language-tsx bg-transparent! p-0! block">
                {code}
              </code>
            </pre>
            
            {mode === 'INSERT' && (
              <textarea
                ref={textareaRef}
                className="absolute inset-0 w-full h-full bg-transparent text-transparent caret-white resize-none outline-none pl-4 pt-4 leading-6 font-mono z-10"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                autoFocus
                spellCheck={false}
              />
            )}
          </div>
        </div>

        <div className="h-8 bg-eudora-600 text-white flex items-center px-4 text-xs font-bold justify-between select-none">
          <div className="flex items-center gap-4">
            <span className="uppercase">{mode}</span>
            <span className="text-white/60">master</span>
            {command && <span className="text-yellow-300 ml-4">{command}</span>}
          </div>
          <div className="flex items-center gap-4">
            <span>utf-8</span>
            <span>unix</span>
            <span>ts</span>
            <span>{cursor.line + 1}:{cursor.col + 1}</span>
            <span>Top</span>
          </div>
        </div>
        
        {command.startsWith(':') && mode === 'NORMAL' && (
           <form onSubmit={handleCommandSubmit} className="absolute bottom-0 left-0 right-0 h-8 bg-[#2d2d2d] flex items-center px-2 z-20">
              <input 
                autoFocus
                className="w-full bg-transparent text-white text-xs outline-none font-mono"
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                onBlur={() => { if(command === ':') setCommand(''); }}
              />
           </form>
        )}
      </div>

      <div className="flex gap-4 p-4 bg-dark-sidebar/20 border border-white/5 rounded-2xl items-center">
         <div className="p-3 bg-blue-600/10 rounded-xl border border-blue-500/20">
            <Icons.Zap className="w-5 h-5 text-blue-400" />
         </div>
         <div className="flex-1">
            <h4 className="text-xs font-bold text-zinc-300 uppercase">React 2026 Compiler Status</h4>
            <p className="text-[10px] text-zinc-500">Zero-Bundle-Size Compilation Active. Using `node:ai` imports.</p>
         </div>
         <div className="flex gap-2">
            <Button size="sm" variant="secondary" icon={Icons.Play}>Run Component</Button>
            <Button size="sm" variant="primary" icon={Icons.Share}>Deploy Edge</Button>
         </div>
      </div>
    </div>
  );
};
