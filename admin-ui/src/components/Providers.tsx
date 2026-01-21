'use client';

import { ToastProvider } from '@/components/Toast';
import { ConnectionStatus } from '@/components/ConnectionStatus';

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ToastProvider>
      <ConnectionStatus />
      {children}
    </ToastProvider>
  );
}
