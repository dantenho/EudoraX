
import React from 'react';
import { Icons } from '../../../Icons';
import { Button } from '../../../Button';

export const GoogleAuth: React.FC<{ onLinked: () => void }> = ({ onLinked }) => {
  return (
    <div className="max-w-3xl mx-auto py-12 text-center space-y-12">
      <div className="w-20 h-20 bg-blue-600/10 border border-blue-500/20 rounded-3xl flex items-center justify-center mx-auto text-blue-400">
        <Icons.Key className="w-10 h-10" />
      </div>
      <div className="space-y-4">
        <h1 className="text-4xl font-black text-white tracking-tighter uppercase">Identity Forge</h1>
        <p className="text-zinc-500 text-sm max-w-md mx-auto">Authorize Google Studio services via Cloud IAM or provision manual hardware keys.</p>
      </div>
      <div className="flex gap-4 justify-center pt-6">
        <Button className="bg-white text-black px-8" icon={Icons.Link} onClick={onLinked}>Sync Cloud ID</Button>
        <Button variant="secondary" className="px-8">Manual Provision</Button>
      </div>
    </div>
  );
};
