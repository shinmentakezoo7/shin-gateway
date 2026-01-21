'use client';

import { useEffect, useState } from 'react';
import {
  Settings,
  RefreshCw,
  Trash2,
  Database,
  Activity,
  Shield,
  Server,
  AlertTriangle,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  PageHeader,
  LoadingSpinner,
  Badge,
  Separator,
} from '@/components/ui';
import { useToast } from '@/components/Toast';
import { useConnection } from '@/hooks/useConnection';
import {
  getOverview,
  getProviders,
  getModels,
  getApiKeys,
  resetStats,
  OverviewStats,
  formatDuration,
  formatDate,
} from '@/lib/api';
import { cn } from '@/lib/utils';

export default function SettingsPage() {
  const { success, error } = useToast();
  const { isConnected, health, checkConnection, isChecking } = useConnection();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [counts, setCounts] = useState({ providers: 0, models: 0, apiKeys: 0 });
  const [resetting, setResetting] = useState(false);

  const loadData = async () => {
    try {
      const [overview, providersData, modelsData, keysData] = await Promise.all([
        getOverview(),
        getProviders(),
        getModels(),
        getApiKeys(),
      ]);
      setStats(overview);
      setCounts({
        providers: providersData.total || 0,
        models: modelsData.total || 0,
        apiKeys: keysData.total || 0,
      });
    } catch (err) {
      console.error('Failed to load settings data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleResetStats = async () => {
    if (!confirm('Are you sure you want to reset all statistics? This action cannot be undone.')) {
      return;
    }

    setResetting(true);
    try {
      await resetStats();
      success('Statistics Reset', 'All real-time statistics have been cleared');
      loadData();
    } catch (err) {
      error('Failed to reset statistics', err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setResetting(false);
    }
  };

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="p-8 min-h-screen bg-background">
      <PageHeader
        title="Settings"
        description="Gateway configuration and system information"
      />

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Connection Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Connection Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-xl bg-secondary border border-border">
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    'w-3 h-3 rounded-full',
                    isConnected ? 'status-online' : 'bg-white/30'
                  )}
                />
                <div>
                  <p className="font-medium text-foreground">Backend Server</p>
                  <p className="text-sm text-muted-foreground">
                    {isConnected ? 'Connected to gateway API' : 'Connection lost'}
                  </p>
                </div>
              </div>
              <Badge variant={isConnected ? 'default' : 'muted'}>
                {isConnected ? 'Online' : 'Offline'}
              </Badge>
            </div>

            {health && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-2 border-b border-border">
                  <span className="text-muted-foreground">Status</span>
                  <Badge variant={health.status === 'healthy' ? 'default' : 'muted'}>
                    {health.status}
                  </Badge>
                </div>
                {health.version && (
                  <div className="flex justify-between py-2 border-b border-border">
                    <span className="text-muted-foreground">Version</span>
                    <span className="text-foreground font-mono">{health.version}</span>
                  </div>
                )}
                {health.components && (
                  <>
                    <div className="flex justify-between py-2 border-b border-border">
                      <span className="text-muted-foreground">Database</span>
                      <Badge variant={health.components.database ? 'default' : 'muted'}>
                        {health.components.database ? 'OK' : 'Error'}
                      </Badge>
                    </div>
                    <div className="flex justify-between py-2 border-b border-border">
                      <span className="text-muted-foreground">Providers</span>
                      <Badge variant={health.components.providers ? 'default' : 'muted'}>
                        {health.components.providers ? 'OK' : 'Error'}
                      </Badge>
                    </div>
                  </>
                )}
              </div>
            )}

            <Button
              variant="secondary"
              onClick={checkConnection}
              disabled={isChecking}
              className="w-full"
            >
              {isChecking ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Check Connection
            </Button>
          </CardContent>
        </Card>

        {/* System Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="w-5 h-5" />
              System Overview
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 rounded-xl bg-secondary border border-border">
                <p className="text-2xl font-bold text-foreground">{counts.providers}</p>
                <p className="text-sm text-muted-foreground">Providers</p>
              </div>
              <div className="text-center p-4 rounded-xl bg-secondary border border-border">
                <p className="text-2xl font-bold text-foreground">{counts.models}</p>
                <p className="text-sm text-muted-foreground">Models</p>
              </div>
              <div className="text-center p-4 rounded-xl bg-secondary border border-border">
                <p className="text-2xl font-bold text-foreground">{counts.apiKeys}</p>
                <p className="text-sm text-muted-foreground">API Keys</p>
              </div>
            </div>

            {stats && (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-2 border-b border-border">
                  <span className="text-muted-foreground">Uptime</span>
                  <span className="text-foreground font-medium">
                    {stats.uptime_formatted || formatDuration(stats.uptime_seconds)}
                  </span>
                </div>
                <div className="flex justify-between py-2 border-b border-border">
                  <span className="text-muted-foreground">Total Requests</span>
                  <span className="text-foreground font-medium">{stats.requests_total.toLocaleString()}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border">
                  <span className="text-muted-foreground">Total Tokens</span>
                  <span className="text-foreground font-medium">{stats.tokens_total.toLocaleString()}</span>
                </div>
                <div className="flex justify-between py-2 border-b border-border">
                  <span className="text-muted-foreground">Current RPS</span>
                  <span className="text-foreground font-medium">{stats.current_rps}</span>
                </div>
                <div className="flex justify-between py-2">
                  <span className="text-muted-foreground">Avg Latency</span>
                  <span className="text-foreground font-medium">{stats.avg_latency_ms}ms</span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Data Management */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Data Management
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-xl bg-secondary border border-border">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-white/70 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Reset Statistics</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Clear all real-time metrics and usage statistics. This will reset RPS,
                    latency measurements, and all provider statistics. Historical usage records
                    in the database will not be affected.
                  </p>
                </div>
              </div>
            </div>

            <Button
              variant="destructive"
              onClick={handleResetStats}
              disabled={resetting}
              className="w-full"
            >
              {resetting ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Trash2 className="w-4 h-4" />
              )}
              Reset All Statistics
            </Button>
          </CardContent>
        </Card>

        {/* API Information */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              API Information
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">API Endpoint</span>
                <span className="text-foreground font-mono text-xs">/v1/messages</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Protocol</span>
                <span className="text-foreground">Anthropic Messages API</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Models Endpoint</span>
                <span className="text-foreground font-mono text-xs">/v1/models</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Health Check</span>
                <span className="text-foreground font-mono text-xs">/health</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Documentation</span>
                <span className="text-foreground font-mono text-xs">/docs</span>
              </div>
            </div>

            <Separator />

            <div className="p-4 rounded-xl bg-secondary border border-border">
              <p className="text-sm text-muted-foreground mb-2">Example Request Header:</p>
              <code className="text-xs text-foreground font-mono block p-2 bg-background rounded-lg border border-border">
                x-api-key: your-api-key-here
              </code>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
