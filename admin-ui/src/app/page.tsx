'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Activity,
  Zap,
  Clock,
  TrendingUp,
  Server,
  Wifi,
  RefreshCw,
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { StatCard, Card, CardHeader, CardTitle, CardContent, PageHeader, LoadingSpinner, Badge, Button } from '@/components/ui';
import { useToast } from '@/components/Toast';
import {
  getOverview,
  getLiveMetrics,
  getTimeseries,
  OverviewStats,
  LiveMetrics,
  formatNumber,
  ApiError,
} from '@/lib/api';
import { cn } from '@/lib/utils';

export default function OverviewPage() {
  const { error: showError } = useToast();
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
  const [timeseries, setTimeseries] = useState<{ time: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    try {
      const [overviewData, liveData, timeseriesData] = await Promise.all([
        getOverview(),
        getLiveMetrics(),
        getTimeseries(60),
      ]);
      setOverview(overviewData);
      setLiveMetrics(liveData);
      setTimeseries(timeseriesData.timeseries || []);
    } catch (err) {
      // Log error without causing re-renders
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []); // Remove showError from dependencies to prevent infinite loop

  useEffect(() => {
    loadData();
    const interval = setInterval(() => loadData(), 5000);
    return () => clearInterval(interval);
  }, [loadData]);

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
        title="Dashboard Overview"
        description="Real-time monitoring and metrics for your gateway"
        action={
          <Button variant="secondary" onClick={() => loadData(true)} disabled={refreshing}>
            <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
          </Button>
        }
      />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <StatCard
          title="Total Requests"
          value={formatNumber(overview?.requests_total || 0)}
          icon={Activity}
        />
        <StatCard
          title="Total Tokens"
          value={formatNumber(overview?.tokens_total || 0)}
          icon={Zap}
        />
        <StatCard
          title="Current RPS"
          value={overview?.current_rps || 0}
          icon={TrendingUp}
        />
        <StatCard
          title="Avg Latency"
          value={overview?.avg_latency_ms || 0}
          suffix="ms"
          icon={Clock}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Requests Over Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeseries}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(0 0% 8%)',
                      border: '1px solid hsl(0 0% 15%)',
                      borderRadius: '12px',
                      boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
                    }}
                    labelStyle={{ color: 'hsl(0 0% 60%)' }}
                    itemStyle={{ color: '#ffffff' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke="#ffffff"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorValue)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <Card>
          <CardHeader>
            <CardTitle>System Health</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-xl bg-secondary border border-border">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                  <Wifi className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">Uptime</p>
                  <p className="text-xs text-muted-foreground">System availability</p>
                </div>
              </div>
              <span className="text-lg font-semibold text-foreground">
                {overview?.uptime_formatted || '-'}
              </span>
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-secondary border border-border">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                  <Server className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">Providers</p>
                  <p className="text-xs text-muted-foreground">Active backends</p>
                </div>
              </div>
              <span className="text-lg font-semibold text-foreground">
                {liveMetrics?.providers?.length || 0}
              </span>
            </div>

            <div className="flex items-center justify-between p-4 rounded-xl bg-secondary border border-border">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">Throughput</p>
                  <p className="text-xs text-muted-foreground">Tokens per second</p>
                </div>
              </div>
              <span className="text-lg font-semibold text-foreground">
                {formatNumber(
                  liveMetrics?.providers?.reduce((sum, p) => sum + p.tps, 0) || 0
                )}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Provider Status */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Provider Status</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {liveMetrics?.providers?.map((provider, index) => (
            <div
              key={provider.id}
              className={cn(
                'flex items-center justify-between p-4 rounded-xl',
                'bg-secondary border border-border',
                'transition-all duration-200 animate-fade-in card-interactive'
              )}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center gap-4">
                <div className="relative">
                  <div
                    className={cn(
                      'w-3 h-3 rounded-full',
                      provider.status === 'healthy' && 'status-online',
                      provider.status === 'degraded' && 'bg-white/50',
                      provider.status === 'unhealthy' && 'bg-white/20'
                    )}
                  />
                  {provider.status === 'healthy' && (
                    <div
                      className="absolute inset-0 w-3 h-3 rounded-full animate-ping status-online"
                      style={{ opacity: 0.3 }}
                    />
                  )}
                </div>
                <div>
                  <p className="font-medium text-foreground">{provider.id}</p>
                  <Badge variant={provider.status === 'healthy' ? 'default' : 'muted'} dot online={provider.status === 'healthy'}>
                    {provider.status}
                  </Badge>
                </div>
              </div>

              <div className="flex items-center gap-8">
                <div className="text-center">
                  <p className="text-lg font-semibold text-foreground">{provider.rps}</p>
                  <p className="text-xs text-muted-foreground">RPS</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-semibold text-foreground">
                    {formatNumber(provider.tps)}
                  </p>
                  <p className="text-xs text-muted-foreground">TPS</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-semibold text-foreground">{provider.latency_p50}</p>
                  <p className="text-xs text-muted-foreground">P50 ms</p>
                </div>
                <div className="text-center">
                  <p className="text-lg font-semibold text-foreground">{provider.latency_p95}</p>
                  <p className="text-xs text-muted-foreground">P95 ms</p>
                </div>
                <div className="text-center">
                  <p
                    className={cn(
                      'text-lg font-semibold',
                      provider.error_rate > 5 ? 'text-white/50' : 'text-foreground'
                    )}
                  >
                    {provider.error_rate}%
                  </p>
                  <p className="text-xs text-muted-foreground">Errors</p>
                </div>
              </div>
            </div>
          ))}
          {(!liveMetrics?.providers || liveMetrics.providers.length === 0) && (
            <div className="text-center py-12 text-muted-foreground">
              <Server className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No active providers</p>
              <p className="text-sm mt-1">Add providers to start proxying requests</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
