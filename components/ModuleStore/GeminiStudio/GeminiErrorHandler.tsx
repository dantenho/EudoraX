
import React, { ErrorInfo, ReactNode } from 'react';
import { Icons } from '../../Icons';
import { Button } from '../../Button';

interface Props {
  children?: ReactNode;
  moduleName: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * @class GeminiErrorBoundary
 * @description Provides error boundaries for Gemini Studio modules.
 * Ensures the app stays functional even if a specific local AI module fails.
 */
// Fix: Renamed to GeminiErrorBoundary to match the import in GeminiStudio.tsx 
// and explicitly used React.Component to resolve property access errors (state, props, setState).
export class GeminiErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  /**
   * Component Error Lifecycle.
   * Note: Extensive logging removed to prevent main thread lag during cascading failures.
   */
  public componentDidCatch(error: Error, _errorInfo: ErrorInfo) {
    // Lightweight console logging only.
    console.error(`[${this.props.moduleName}] Kernel Panic:`, error);
  }

  /**
   * Resets the error state and reloads the interface.
   * Using arrow function to maintain 'this' context.
   */
  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    // Force a reload to clear any tainted WASM memory or GL contexts
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full p-8 text-center space-y-4 bg-red-950/10 border border-red-500/20 rounded-3xl animate-fade-in">
          <Icons.Alert className="w-12 h-12 text-red-500 mb-2" />
          <h2 className="text-xl font-bold text-white uppercase tracking-tighter">Module Kernel Panic</h2>
          <p className="text-sm text-zinc-500 max-w-md">
            The {this.props.moduleName} service encountered a critical execution error.
          </p>
          <div className="bg-black/40 p-3 rounded-lg font-mono text-[10px] text-red-400 text-left w-full max-w-sm overflow-hidden truncate">
            {this.state.error?.message || "Unknown Runtime Error"}
          </div>
          <div className="flex gap-3">
            <Button variant="danger" size="sm" onClick={this.handleReset}>Hot Reload Engine</Button>
            <Button variant="secondary" size="sm" onClick={() => window.open('https://chrome.google.com/flags', '_blank')}>Check Chrome Flags</Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
