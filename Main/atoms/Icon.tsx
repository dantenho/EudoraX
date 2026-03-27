
import React from 'react';
import { 
  LayoutDashboard, Layers, Cpu, Settings, Hexagon, 
  ArrowRight, Activity, Zap, Code, Terminal, 
  Database, Network, MoreHorizontal, User, 
  MessageSquare, Search, Command
} from 'lucide-react';

const icons = {
  LayoutDashboard, Layers, Cpu, Settings, Hexagon,
  ArrowRight, Activity, Zap, Code, Terminal,
  Database, Network, MoreHorizontal, User,
  MessageSquare, Search, Command
};

export type IconName = keyof typeof icons;

interface IconProps extends React.SVGProps<SVGSVGElement> {
  name: IconName;
}

export const Icon: React.FC<IconProps> = ({ name, ...props }) => {
  const IconComponent = icons[name];
  if (!IconComponent) return null;
  return <IconComponent {...props} />;
};
