/**
 * @file Button.tsx
 * @description Standardized action component for the EudoraX design system.
 */
import React, { ReactNode, ReactElement } from 'react';

/**
 * @interface ButtonProps
 * @description Configuration for the high-performance action button.
 */
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** The content to be rendered inside the button */
  children: ReactNode;
  /** Visual variant affecting background and borders */
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  /** Preset size definitions */
  size?: 'sm' | 'md' | 'lg';
  /** Optional icon component from Lucide */
  icon?: React.ElementType;
  /** Triggers the internal loading spinner */
  loading?: boolean;
}

/**
 * @component Button
 * @description A multi-variant button with support for hardware-accelerated transitions.
 */
export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  icon: Icon,
  loading,
  className = '',
  ...props
}): ReactElement => {
  /** Base hardware-accelerated utility classes */
  const baseStyles: string = "inline-flex items-center justify-center rounded-lg font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0f0f12] disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98]";
  
  /** Design system color mappings */
  const variants = {
    primary: "bg-eudora-600 text-white hover:bg-eudora-700 focus:ring-eudora-500 shadow-[0_0_15px_rgba(124,58,237,0.3)]",
    secondary: "bg-dark-card border border-dark-border text-zinc-200 hover:bg-zinc-800 focus:ring-zinc-500",
    ghost: "text-zinc-400 hover:text-white hover:bg-white/5",
    danger: "bg-red-900/20 text-red-400 border border-red-900/50 hover:bg-red-900/40"
  };

  /** Size mappings based on design tokens */
  const sizes = {
    sm: "h-8 px-3 text-xs",
    md: "h-10 px-4 text-sm",
    lg: "h-12 px-6 text-base",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      ) : Icon ? (
        <Icon className={`mr-2 ${size === 'sm' ? 'h-3.5 w-3.5' : 'h-4 w-4'}`} />
      ) : null}
      {children}
    </button>
  );
};
