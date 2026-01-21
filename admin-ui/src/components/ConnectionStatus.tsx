'use client';

import { useEffect, useState } from 'react';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { onConnectionChange, getConnectionStatus, checkHealth } from '@/lib/api';
import { cn } from '@/lib/utils';

export function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState(true);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Get initial status
    setIsConnected(getConnectionStatus());

    // Subscribe to connection changes
    const unsubscribe = onConnectionChange((connected) => {
      setIsConnected(connected);
      if (!connected) {
        setShowBanner(true);
      } else {
        // Hide banner after a short delay when reconnected
        setTimeout(() => setShowBanner(false), 2000);
      }
    });

    return unsubscribe;
  }, []);

  const handleReconnect = async () => {
    setIsReconnecting(true);
    try {
      await checkHealth();
      setIsConnected(true);
      setTimeout(() => setShowBanner(false), 1000);
    } catch {
      setIsConnected(false);
    } finally {
      setIsReconnecting(false);
    }
  };

  if (!showBanner) return null;

  return (
    <div
      className={cn(
        'fixed top-0 left-0 right-0 z-50 px-4 py-2 flex items-center justify-center gap-3 text-sm transition-all',
        isConnected
          ? 'bg-white/10 text-white'
          : 'bg-white/5 text-white/80'
      )}
    >
      {isConnected ? (
        <>
          <Wifi className="w-4 h-4" />
          <span>Connected to backend</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4" />
          <span>Cannot connect to backend server</span>
          <button
            onClick={handleReconnect}
            disabled={isReconnecting}
            className="ml-2 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 transition-colors disabled:opacity-50"
          >
            {isReconnecting ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              'Retry'
            )}
          </button>
        </>
      )}
    </div>
  );
}
