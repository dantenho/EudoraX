
import React from 'react';
import { 
  Cpu, Network, Database, Activity, Zap, Code, Terminal, ArrowRight, MoreHorizontal 
} from 'lucide-react';

// Internal Helper Components for cleanliness within the single file
const DashboardCard: React.FC<{ 
  title?: string; 
  icon?: React.ElementType; 
  children: React.ReactNode; 
  variant?: 'default' | 'glass';
}> = ({ title, icon: Icon, children, variant = 'default' }) => (
  <div className={`rounded-2xl border p-6 transition-all duration-300 ${
    variant === 'glass' 
      ? 'bg-white/5 border-white/10 backdrop-blur-lg hover:bg-white/10' 
      : 'bg-zinc-900 border-zinc-800 hover:border-zinc-700'
  }`}>
    {(title || Icon) && (
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-zinc-400">
          {Icon && <Icon className="w-5 h-5 text-blue-500" />}
          {title && <h3 className="text-xs font-bold uppercase tracking-widest">{title}</h3>}
        </div>
        <MoreHorizontal className="w-4 h-4 text-zinc-700 cursor-pointer hover:text-white" />
      </div>
    )}
    {children}
  </div>
);

const ActionButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & { icon?: React.ElementType, variant?: 'primary' | 'secondary' }> = ({ 
  children, icon: Icon, variant = 'primary', className = '', ...props 
}) => {
  const baseClass = "inline-flex items-center justify-center rounded-lg font-medium transition-all focus:outline-none active:scale-95 h-10 px-4 text-sm gap-2";
  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-500/20",
    secondary: "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 hover:text-white border border-zinc-700"
  };
  return (
    <button className={`${baseClass} ${variants[variant]} ${className}`} {...props}>
      {Icon && <Icon className="w-4 h-4" />}
      {children}
    </button>
  );
};

export const Dashboard: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in max-w-7xl mx-auto">
      
      {/* Welcome Hero */}
      <div className="col-span-full mb-8">
        <h2 className="text-3xl font-light text-white mb-2">Hello, <span className="font-bold text-blue-500">Architect</span></h2>
        <p className="text-zinc-400 text-sm max-w-xl">
          Welcome to the new atomic interface. Your neural engines are primed and ready for synthesis tasks.
        </p>
      </div>

      {/* Stat Cards */}
      <DashboardCard title="Compute Load" icon={Cpu} variant="glass">
        <div className="flex items-end gap-2 mt-4">
          <span className="text-4xl font-mono font-light text-white">42%</span>
          <span className="text-xs text-emerald-400 mb-1">▲ Stable</span>
        </div>
        <div className="h-1 w-full bg-zinc-800 mt-4 rounded-full overflow-hidden">
          <div className="h-full w-[42%] bg-blue-500" />
        </div>
      </DashboardCard>

      <DashboardCard title="Active Nodes" icon={Network}>
        <div className="flex items-end gap-2 mt-4">
          <span className="text-4xl font-mono font-light text-white">08</span>
          <span className="text-xs text-zinc-500 mb-1">/ 12 Clusters</span>
        </div>
        <div className="flex gap-1 mt-4">
           {[...Array(8)].map((_, i) => <div key={i} className="h-2 w-2 rounded-full bg-emerald-500" />)}
           {[...Array(4)].map((_, i) => <div key={i} className="h-2 w-2 rounded-full bg-zinc-800" />)}
        </div>
      </DashboardCard>

      <DashboardCard title="Storage Vault" icon={Database}>
        <div className="flex items-end gap-2 mt-4">
          <span className="text-4xl font-mono font-light text-white">1.2</span>
          <span className="text-xs text-zinc-500 mb-1">PB Available</span>
        </div>
        <div className="mt-4 text-xs text-zinc-500 font-mono">
          &gt; S3_BUCKET_LINKED<br/>
          &gt; FIREBASE_SYNC_ACTIVE
        </div>
      </DashboardCard>

      {/* Main Action Area */}
      <div className="col-span-full lg:col-span-2 mt-4">
        <div className="flex items-center justify-between mb-4">
           <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-500">Recent Operations</h3>
           <button className="text-zinc-400 hover:text-white text-xs flex items-center gap-1">
             View All <ArrowRight className="w-3 h-3" />
           </button>
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="group p-4 bg-zinc-900 border border-zinc-800 rounded-xl flex items-center justify-between hover:border-blue-500/50 transition-colors cursor-pointer">
              <div className="flex items-center gap-4">
                <div className="p-2 bg-zinc-950 rounded-lg text-zinc-500 group-hover:text-blue-500 transition-colors">
                  <Activity className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-sm font-medium text-white">Neural Synthesis Batch #{2040 + i}</div>
                  <div className="text-xs text-zinc-500">Completed 2m ago • Vector Quantization</div>
                </div>
              </div>
              <div className="px-3 py-1 rounded-full bg-zinc-950 border border-zinc-800 text-[10px] font-mono text-zinc-500">
                340ms
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="mt-4 space-y-4">
        <h3 className="text-sm font-bold uppercase tracking-wider text-zinc-500 mb-4">Quick Deploy</h3>
        <DashboardCard>
           <div className="space-y-3">
             <ActionButton className="w-full justify-start" icon={Zap}>New Instance</ActionButton>
             <ActionButton className="w-full justify-start" variant="secondary" icon={Code}>Open Code Editor</ActionButton>
             <ActionButton className="w-full justify-start" variant="secondary" icon={Terminal}>System Logs</ActionButton>
           </div>
        </DashboardCard>
      </div>

    </div>
  );
};
