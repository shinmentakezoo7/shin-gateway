'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { onConnectionChange, getConnectionStatus, checkHealth, HealthStatus } from '@/lib/api';

export function useConnection() {
  const [isConnected, setIsConnected] = useState(true);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  const checkConnection = useCallback(async () => {
    setIsChecking(true);
    try {
      const healthData = await checkHealth();
      setHealth(healthData);
      setIsConnected(true);
      setLastCheck(new Date());
    } catch {
      setIsConnected(false);
      setHealth(null);
    } finally {
      setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    // Subscribe to connection changes
    const unsubscribe = onConnectionChange((connected) => {
      setIsConnected(connected);
    });

    // Initial check
    checkConnection();

    // Periodic health check
    const interval = setInterval(checkConnection, 30000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [checkConnection]);

  return {
    isConnected,
    isChecking,
    lastCheck,
    health,
    checkConnection,
  };
}

export function useAutoRefresh<T>(
  fetchFn: () => Promise<T>,
  intervalMs: number = 5000,
  enabled: boolean = true
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Store fetchFn in a ref to avoid re-creating the callback
  const fetchFnRef = useRef(fetchFn);
  fetchFnRef.current = fetchFn;

  const refresh = useCallback(async () => {
    try {
      const result = await fetchFnRef.current();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    refresh();

    const interval = setInterval(refresh, intervalMs);
    return () => clearInterval(interval);
  }, [refresh, intervalMs, enabled]);

  return { data, loading, error, refresh, lastUpdated };
}

export function usePolling<T>(
  fetchFn: () => Promise<T>,
  options: {
    interval?: number;
    enabled?: boolean;
    onSuccess?: (data: T) => void;
    onError?: (error: Error) => void;
  } = {}
) {
  const { interval = 5000, enabled = true, onSuccess, onError } = options;
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Store callbacks in refs to avoid re-creating the poll function
  const fetchFnRef = useRef(fetchFn);
  const onSuccessRef = useRef(onSuccess);
  const onErrorRef = useRef(onError);

  fetchFnRef.current = fetchFn;
  onSuccessRef.current = onSuccess;
  onErrorRef.current = onError;

  const poll = useCallback(async () => {
    try {
      const result = await fetchFnRef.current();
      setData(result);
      setError(null);
      onSuccessRef.current?.(result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      onErrorRef.current?.(error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    poll();
    const intervalId = setInterval(poll, interval);
    return () => clearInterval(intervalId);
  }, [poll, interval, enabled]);

  return { data, loading, error, refresh: poll };
}
