
/**
 * @file GoogleErrorHandler.tsx
 * @description specialized error boundary for the Google Studio environment.
 */
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

// Fix: Explicitly using React.Component to resolve property access errors (props, state, setState) within the class component.
export class GoogleErrorBoundary extends React.Component<Props, State> {
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

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[GoogleStudio:${this.props.moduleName}] Panic:`, error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full p-8 text-center space-y-4 bg-red-950/10 border border-red-500/20 rounded-3xl">
          <Icons.Alert className="w-12 h-12 text-red-500 mb-2" />
          <h2 className="text-xl font-bold text-white uppercase">Module Kernel Panic</h2>
          <p className="text-sm text-zinc-500 max-w-md">
            The {this.props.moduleName} service encountered a critical error.
          </p>
          <Button variant="danger" size="sm" onClick={this.handleReset}>Reload Engine</Button>
        </div>
      );
    }
    return this.props.children;
  }
}
