
import React, { ReactNode } from 'react';
import { Icon, IconName } from './Icon';

interface CardProps {
  children: ReactNode;
  title?: string;
  icon?: IconName;
  variant?: 'default' | 'glass';
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, title, icon, variant = 'default', className = '' }) => {
  const base = "rounded-2xl border p-6 transition-all duration-300";
  const styles = {
    default: "bg-surface border-border hover:border-zinc-700",
    glass: "bg-white/5 border-white/10 backdrop-blur-lg hover:bg-white/10"
  };

  return (
    <div className={`${base} ${styles[variant]} ${className}`}>
      {(title || icon) && (
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-muted">
            {icon && <Icon name={icon} className="w-5 h-5 text-primary" />}
            {title && <h3 className="text-xs font-bold uppercase tracking-widest">{title}</h3>}
          </div>
          <Icon name="MoreHorizontal" className="w-4 h-4 text-zinc-700 cursor-pointer hover:text-white" />
        </div>
      )}
      {children}
    </div>
  );
};
