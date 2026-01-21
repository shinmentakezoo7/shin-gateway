'use client';

import { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import {
  BarChart3,
  Activity,
  Zap,
  CheckCircle,
  Clock,
  TrendingUp,
  RefreshCw,
  DollarSign,
  Gauge,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Timer,
  Server,
  Cpu,
  Radio,
  Hash,
  Layers,
  ArrowRightLeft,
  CircleDollarSign,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area,
  ComposedChart,
  Legend,
  CartesianGrid,
} from 'recharts';
import {
  MetricCard,
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  PageHeader,
  LoadingSpinner,
  Button,
  Badge,
  Progress,
} from '@/components/ui';
import { useToast } from '@/components/Toast';
import {
  getUsageStats,
  getUsageByProvider,
  getUsageByModel,
  getLiveMetrics,
  getTimeseries,
  getOverview,
  UsageStats,
  ProviderUsage,
  ModelUsage,
  LiveMetrics,
  OverviewStats,
  TimeseriesPoint,
  formatNumber,
  ApiError,
} from '@/lib/api';
import { cn } from '@/lib/utils';

const timeRanges = [
  { value: 1, label: '1h' },
  { value: 24, label: '24h' },
  { value: 168, label: '7d' },
  { value: 720, label: '30d' },
];

// Monochrome color palette for providers
const COLORS = ['#ffffff', '#e5e5e5', '#b3b3b3', '#808080', '#4d4d4d'];

// Distinct color palette for models (vibrant colors for better differentiation)
const MODEL_COLORS = [
  '#3b82f6', // blue
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#6366f1', // indigo
];

const CHART_COLORS = {
  primary: '#ffffff',
  secondary: '#999999',
  tertiary: '#666666',
  accent: '#cccccc',
};

// Custom tooltip style
const tooltipStyle = {
  backgroundColor: 'hsl(0 0% 8%)',
  border: '1px solid hsl(0 0% 15%)',
  borderRadius: '12px',
  boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
};

// Simulated cost calculation (approximate based on tokens)
const calculateCost = (inputTokens: number, outputTokens: number): number => {
  // Approximate pricing: $0.01 per 1K input tokens, $0.03 per 1K output tokens
  return (inputTokens / 1000) * 0.01 + (outputTokens / 1000) * 0.03;
};

// Get status based on usage percentage
const getUsageStatus = (current: number, limit: number): 'healthy' | 'warning' | 'danger' => {
  const percentage = (current / limit) * 100;
  if (percentage >= 90) return 'danger';
  if (percentage >= 70) return 'warning';
  return 'healthy';
};

interface TimeseriesData {
  time: string;
  requests: number;
  tokens: number;
  latency: number;
  rpm: number;
  tpm: number;
  cost: number;
}

export default function StatisticsPage() {
  const { error: showError } = useToast();
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [providerUsage, setProviderUsage] = useState<ProviderUsage[]>([]);
  const [modelUsage, setModelUsage] = useState<ModelUsage[]>([]);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesData[]>([]);
  const [loading, setLoading] = useState(true);
  const [initialLoadDone, setInitialLoadDone] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedRange, setSelectedRange] = useState(24);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isLive, setIsLive] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const showErrorRef = useRef(showError);
  const selectedRangeRef = useRef(selectedRange);

  // Keep refs in sync
  useEffect(() => {
    showErrorRef.current = showError;
  }, [showError]);

  useEffect(() => {
    selectedRangeRef.current = selectedRange;
  }, [selectedRange]);

  // Calculate cumulative data for area charts - memoized
  const cumulativeData = useMemo(
    () =>
      timeseries.reduce((acc: TimeseriesData[], point, index) => {
        const prev = acc[index - 1];
        acc.push({
          ...point,
          tokens: (prev?.tokens || 0) + point.tokens,
          cost: (prev?.cost || 0) + point.cost,
        });
        return acc;
      }, []),
    [timeseries]
  );

  const loadStats = useCallback(
    async (hours: number, isManualRefresh = false) => {
      // Only show full loading on initial load, not on time range changes
      if (!initialLoadDone) {
        setLoading(true);
      } else {
        setRefreshing(true);
      }
      setSelectedRange(hours);

      try {
        const [usage, byProvider, byModel, live, overviewData, timeseriesData] = await Promise.all([
          getUsageStats(hours),
          getUsageByProvider(hours),
          getUsageByModel(hours),
          getLiveMetrics(),
          getOverview(),
          getTimeseries(Math.min(hours * 60, 1440)), // Max 24 hours of minute-level data
        ]);
        setUsageStats(usage);
        setProviderUsage(byProvider.providers || []);
        setModelUsage(byModel.models || []);
        setLiveMetrics(live);
        setOverview(overviewData);

        // Transform timeseries data with additional calculated fields
        const transformedTimeseries: TimeseriesData[] = (timeseriesData.timeseries || []).map(
          (point: TimeseriesPoint) => {
            const requests = point.requests || point.value || 0;
            const tokens = point.tokens || 0;
            // Calculate RPM/TPM based on minute intervals
            const rpm = requests;
            const tpm = tokens;
            // Approximate cost calculation
            const inputTokens = Math.floor(tokens * 0.3);
            const outputTokens = tokens - inputTokens;
            const cost = calculateCost(inputTokens, outputTokens);

            return {
              time: point.time,
              requests,
              tokens,
              latency: 0,
              rpm,
              tpm,
              cost,
            };
          }
        );
        setTimeseries(transformedTimeseries);
        setLastUpdated(new Date());
        setInitialLoadDone(true);
      } catch (err) {
        if (err instanceof ApiError) {
          showErrorRef.current('Failed to load statistics', err.detail);
        }
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [initialLoadDone] // Depend on initialLoadDone to update behavior
  );

  // Load live metrics only (for real-time updates) - updates all data without loading state
  const loadLiveMetrics = useCallback(async () => {
    try {
      // Get the current selected range from ref
      const hours = selectedRangeRef.current;

      const [usage, byProvider, byModel, live, overviewData, timeseriesData] = await Promise.all([
        getUsageStats(hours),
        getUsageByProvider(hours),
        getUsageByModel(hours),
        getLiveMetrics(),
        getOverview(),
        getTimeseries(60), // Always get last 60 minutes for live view
      ]);

      // Update all state without triggering loading
      setUsageStats(usage);
      setProviderUsage(byProvider.providers || []);
      setModelUsage(byModel.models || []);
      setLiveMetrics(live);
      setOverview(overviewData);

      // Transform timeseries data
      const transformedTimeseries: TimeseriesData[] = (timeseriesData.timeseries || []).map(
        (point: TimeseriesPoint) => {
          const requests = point.requests || point.value || 0;
          const tokens = point.tokens || 0;
          const rpm = requests;
          const tpm = tokens;
          const inputTokens = Math.floor(tokens * 0.3);
          const outputTokens = tokens - inputTokens;
          const cost = calculateCost(inputTokens, outputTokens);

          return {
            time: point.time,
            requests,
            tokens,
            latency: 0,
            rpm,
            tpm,
            cost,
          };
        }
      );
      setTimeseries(transformedTimeseries);
      setLastUpdated(new Date());
    } catch {
      // Silent fail for live updates
    }
  }, []); // No dependencies - uses ref for selectedRange

  // Initial load only
  useEffect(() => {
    loadStats(24);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Set up real-time polling - separate from initial load
  useEffect(() => {
    if (isLive) {
      // Start polling
      intervalRef.current = setInterval(loadLiveMetrics, 5000);
    } else {
      // Stop polling
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isLive, loadLiveMetrics]);

  // Memoize computed data to prevent unnecessary recalculations
  const providerChartData = useMemo(
    () =>
      providerUsage.map((p) => ({
        name: p.provider_id,
        requests: p.total_requests,
        tokens: p.total_input_tokens + p.total_output_tokens,
        inputTokens: p.total_input_tokens,
        outputTokens: p.total_output_tokens,
      })),
    [providerUsage]
  );

  const modelChartData = useMemo(
    () =>
      modelUsage.map((m) => ({
        name: m.model_alias,
        requests: m.total_requests,
        latency: m.avg_latency_ms,
      })),
    [modelUsage]
  );

  // Memoize calculated totals and per-request metrics
  const {
    totalInputTokens,
    totalOutputTokens,
    totalCost,
    currentRPM,
    currentTPM,
    tokensPerRequest,
    costPerRequest,
    inputTokensPerRequest,
    outputTokensPerRequest,
    avgTPS,
  } = useMemo(() => {
    const inputTokens = usageStats?.total_input_tokens || 0;
    const outputTokens = usageStats?.total_output_tokens || 0;
    const totalRequests = usageStats?.total_requests || 0;
    const totalTokens = usageStats?.total_tokens || 0;
    const cost = calculateCost(inputTokens, outputTokens);

    return {
      totalInputTokens: inputTokens,
      totalOutputTokens: outputTokens,
      totalCost: cost,
      currentRPM: overview?.current_rps ? overview.current_rps * 60 : 0,
      currentTPM: liveMetrics?.providers?.reduce((sum, p) => sum + p.tps * 60, 0) || 0,
      // Per-request metrics
      tokensPerRequest: totalRequests > 0 ? Math.round(totalTokens / totalRequests) : 0,
      costPerRequest: totalRequests > 0 ? cost / totalRequests : 0,
      inputTokensPerRequest: totalRequests > 0 ? Math.round(inputTokens / totalRequests) : 0,
      outputTokensPerRequest: totalRequests > 0 ? Math.round(outputTokens / totalRequests) : 0,
      avgTPS: liveMetrics?.providers?.reduce((sum, p) => sum + p.tps, 0) || 0,
    };
  }, [usageStats, overview, liveMetrics]);

  // Rate limit defaults (would come from API in real implementation)
  const rpmLimit = 10000;
  const tpmLimit = 1000000;

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
        title="Usage Statistics"
        description="Real-time analytics and performance metrics"
        action={
          <div className="flex items-center gap-3">
            {/* Live indicator */}
            <button
              onClick={() => setIsLive(!isLive)}
              className={cn(
                'flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all',
                isLive
                  ? 'bg-white text-black'
                  : 'bg-secondary text-muted-foreground border border-border hover:bg-accent'
              )}
            >
              <Radio className={cn('w-3 h-3', isLive && 'animate-pulse')} />
              {isLive ? 'Live' : 'Paused'}
            </button>

            <Button variant="secondary" onClick={() => loadStats(selectedRange, true)} disabled={refreshing}>
              <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
            </Button>
          </div>
        }
      />

      {/* Last updated indicator */}
      {lastUpdated && (
        <div className="flex items-center gap-2 mb-6 text-xs text-muted-foreground">
          <div className={cn('w-1.5 h-1.5 rounded-full', isLive ? 'status-online' : 'bg-muted-foreground')} />
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      )}

      {/* Time Range Selector */}
      <div className="flex gap-2 mb-8">
        {timeRanges.map((range) => (
          <button
            key={range.value}
            onClick={() => loadStats(range.value)}
            disabled={refreshing}
            className={cn(
              'px-4 py-2 rounded-xl font-medium text-sm transition-all duration-200',
              selectedRange === range.value
                ? 'bg-white text-black'
                : 'bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground border border-border',
              refreshing && 'opacity-50 cursor-not-allowed'
            )}
          >
            {range.label}
          </button>
        ))}
      </div>

      {/* Real-time Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <MetricCard
          title="Requests per Minute"
          value={formatNumber(Math.round(currentRPM))}
          suffix="RPM"
          icon={Gauge}
          status={getUsageStatus(currentRPM, rpmLimit)}
          progress={{ value: currentRPM, max: rpmLimit }}
          subValue={`${formatNumber(usageStats?.total_requests || 0)} total`}
        />
        <MetricCard
          title="Tokens per Minute"
          value={formatNumber(Math.round(currentTPM))}
          suffix="TPM"
          icon={Zap}
          status={getUsageStatus(currentTPM, tpmLimit)}
          progress={{ value: currentTPM, max: tpmLimit }}
          subValue={`${formatNumber(usageStats?.total_tokens || 0)} total`}
        />
        <MetricCard
          title="Success Rate"
          value={`${usageStats?.success_rate || 0}`}
          suffix="%"
          icon={CheckCircle}
          status={
            (usageStats?.success_rate || 0) >= 99 ? 'healthy' : (usageStats?.success_rate || 0) >= 95 ? 'warning' : 'danger'
          }
          subValue={`${formatNumber(usageStats?.error_count || 0)} errors`}
        />
        <MetricCard
          title="Estimated Cost"
          value={`$${totalCost.toFixed(2)}`}
          icon={DollarSign}
          subValue={`In/Out: ${formatNumber(totalInputTokens)} / ${formatNumber(totalOutputTokens)}`}
        />
      </div>

      {/* Secondary Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <MetricCard title="Total Requests" value={formatNumber(usageStats?.total_requests || 0)} icon={Activity} />
        <MetricCard title="Total Tokens" value={formatNumber(usageStats?.total_tokens || 0)} icon={Zap} />
        <MetricCard
          title="Avg Latency"
          value={usageStats?.avg_latency_ms || 0}
          suffix="ms"
          icon={Clock}
          status={(usageStats?.avg_latency_ms || 0) < 500 ? 'healthy' : (usageStats?.avg_latency_ms || 0) < 1000 ? 'warning' : 'danger'}
        />
        <MetricCard
          title="Active Providers"
          value={liveMetrics?.providers?.filter((p) => p.status === 'healthy').length || 0}
          suffix={`/ ${liveMetrics?.providers?.length || 0}`}
          icon={Server}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* RPM/TPM Over Time - Line Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Request & Token Rate</CardTitle>
              <Badge variant="secondary">Real-time</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeseries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <YAxis
                    yAxisId="left"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: 'hsl(0 0% 60%)' }}
                    itemStyle={{ color: '#ffffff' }}
                  />
                  <Legend
                    wrapperStyle={{ paddingTop: '10px' }}
                    formatter={(value) => <span style={{ color: 'hsl(0 0% 60%)' }}>{value}</span>}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="rpm"
                    name="RPM"
                    stroke={CHART_COLORS.primary}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: CHART_COLORS.primary }}
                    isAnimationActive={!initialLoadDone}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="tpm"
                    name="TPM"
                    stroke={CHART_COLORS.secondary}
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4, fill: CHART_COLORS.secondary }}
                    isAnimationActive={!initialLoadDone}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Cumulative Token Usage - Area Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Cumulative Token Usage</CardTitle>
              <Badge variant="secondary">Over time</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={cumulativeData}>
                  <defs>
                    <linearGradient id="colorTokens" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ffffff" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: 'hsl(0 0% 60%)' }}
                    formatter={(value: number) => [formatNumber(value), 'Tokens']}
                  />
                  <Area
                    type="monotone"
                    dataKey="tokens"
                    stroke="#ffffff"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorTokens)"
                    isAnimationActive={!initialLoadDone}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Response Time Trends - Line Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Response Time Trend</CardTitle>
              <Badge variant="secondary">P50/P95</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeseries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: 'hsl(0 0% 60%)' }}
                    formatter={(value: number) => [`${value}ms`, 'Latency']}
                  />
                  <Line
                    type="monotone"
                    dataKey="requests"
                    name="Requests"
                    stroke={CHART_COLORS.primary}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={!initialLoadDone}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Cost Over Time - Area Chart */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Cumulative Cost</CardTitle>
              <Badge variant="secondary">Estimated</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={cumulativeData}>
                  <defs>
                    <linearGradient id="colorCost" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#999999" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#999999" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
                  <XAxis
                    dataKey="time"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                  />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelStyle={{ color: 'hsl(0 0% 60%)' }}
                    formatter={(value: number) => [`$${value.toFixed(4)}`, 'Cost']}
                  />
                  <Area
                    type="monotone"
                    dataKey="cost"
                    stroke="#999999"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#colorCost)"
                    isAnimationActive={!initialLoadDone}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Composed Chart - Multiple Metrics */}
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Combined Metrics Overview</CardTitle>
            <Badge variant="secondary">RPM + TPM + Cost</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timeseries}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                <YAxis yAxisId="left" axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }}
                />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'hsl(0 0% 60%)' }} />
                <Legend
                  wrapperStyle={{ paddingTop: '10px' }}
                  formatter={(value) => <span style={{ color: 'hsl(0 0% 60%)' }}>{value}</span>}
                />
                <Bar yAxisId="left" dataKey="requests" name="Requests" fill={CHART_COLORS.tertiary} opacity={0.6} isAnimationActive={!initialLoadDone} />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="tpm"
                  name="TPM"
                  stroke={CHART_COLORS.primary}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={!initialLoadDone}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cost"
                  name="Cost ($)"
                  stroke={CHART_COLORS.secondary}
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="5 5"
                  isAnimationActive={!initialLoadDone}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Provider and Model Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Provider Usage Comparison - Bar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Provider Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            {providerChartData.length > 0 ? (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={providerChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(0 0% 15%)" horizontal={false} />
                    <XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: 'hsl(0 0% 60%)', fontSize: 11 }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: 'hsl(0 0% 80%)', fontSize: 12 }}
                      width={100}
                    />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'hsl(0 0% 60%)' }} />
                    <Legend
                      wrapperStyle={{ paddingTop: '10px' }}
                      formatter={(value) => <span style={{ color: 'hsl(0 0% 60%)' }}>{value}</span>}
                    />
                    <Bar dataKey="requests" name="Requests" fill={CHART_COLORS.primary} radius={[0, 4, 4, 0]} isAnimationActive={!initialLoadDone} />
                    <Bar dataKey="tokens" name="Tokens" fill={CHART_COLORS.secondary} radius={[0, 4, 4, 0]} isAnimationActive={!initialLoadDone} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-72 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No provider data available</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Distribution - Pie Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Model Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {modelChartData.length > 0 ? (
              <div className="h-72 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={modelChartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="requests"
                      label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                      labelLine={false}
                      isAnimationActive={!initialLoadDone}
                    >
                      {modelChartData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={MODEL_COLORS[index % MODEL_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: 'hsl(0 0% 60%)' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-72 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No model data available</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Provider Status with Health Indicators */}
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Provider Health & Rate Limits</CardTitle>
            <Badge variant="secondary" dot online={liveMetrics?.providers?.every((p) => p.status === 'healthy')}>
              {liveMetrics?.providers?.every((p) => p.status === 'healthy') ? 'All Healthy' : 'Issues Detected'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {liveMetrics?.providers?.map((provider, index) => {
            const rpmUsage = provider.rps * 60;
            const tpmUsage = provider.tps * 60;
            const providerRpmLimit = 5000;
            const providerTpmLimit = 500000;

            return (
              <div
                key={provider.id}
                className={cn(
                  'p-4 rounded-xl border transition-all duration-200 animate-fade-in',
                  provider.status === 'healthy'
                    ? 'bg-secondary border-white/20'
                    : provider.status === 'degraded'
                    ? 'bg-secondary border-white/10'
                    : 'bg-secondary border-white/5'
                )}
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div
                        className={cn(
                          'w-3 h-3 rounded-full',
                          provider.status === 'healthy' && 'status-online',
                          provider.status === 'degraded' && 'bg-white/50',
                          provider.status === 'unhealthy' && 'bg-white/20 animate-pulse'
                        )}
                      />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">{provider.id}</p>
                      <Badge
                        variant={provider.status === 'healthy' ? 'default' : provider.status === 'degraded' ? 'secondary' : 'muted'}
                      >
                        {provider.status}
                      </Badge>
                    </div>
                  </div>

                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-center">
                      <p className="text-lg font-semibold text-foreground">{provider.rps}</p>
                      <p className="text-xs text-muted-foreground">RPS</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-foreground">{formatNumber(provider.tps)}</p>
                      <p className="text-xs text-muted-foreground">TPS</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-foreground">{provider.latency_p50}ms</p>
                      <p className="text-xs text-muted-foreground">P50</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-foreground">{provider.latency_p95}ms</p>
                      <p className="text-xs text-muted-foreground">P95</p>
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

                {/* Rate limit progress bars */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>RPM Usage</span>
                      <span>{Math.round(rpmUsage).toLocaleString()} / {providerRpmLimit.toLocaleString()}</span>
                    </div>
                    <Progress
                      value={rpmUsage}
                      max={providerRpmLimit}
                      variant={getUsageStatus(rpmUsage, providerRpmLimit) === 'danger' ? 'danger' : getUsageStatus(rpmUsage, providerRpmLimit) === 'warning' ? 'warning' : 'default'}
                      size="sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>TPM Usage</span>
                      <span>{formatNumber(tpmUsage)} / {formatNumber(providerTpmLimit)}</span>
                    </div>
                    <Progress
                      value={tpmUsage}
                      max={providerTpmLimit}
                      variant={getUsageStatus(tpmUsage, providerTpmLimit) === 'danger' ? 'danger' : getUsageStatus(tpmUsage, providerTpmLimit) === 'warning' ? 'warning' : 'default'}
                      size="sm"
                    />
                  </div>
                </div>
              </div>
            );
          })}
          {(!liveMetrics?.providers || liveMetrics.providers.length === 0) && (
            <div className="text-center py-12 text-muted-foreground">
              <Server className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No active providers</p>
              <p className="text-sm mt-1">Add providers to start seeing metrics</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Provider Details */}
        <Card>
          <CardHeader>
            <CardTitle>Provider Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {providerUsage.length > 0 ? (
              providerUsage.map((p, index) => (
                <div
                  key={p.provider_id}
                  className={cn(
                    'flex items-center justify-between p-4 rounded-xl',
                    'bg-secondary border border-border',
                    'animate-fade-in card-interactive'
                  )}
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    />
                    <span className="font-medium text-foreground">{p.provider_id}</span>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                      <p className="text-foreground font-medium">{formatNumber(p.total_requests)}</p>
                      <p className="text-xs text-muted-foreground">requests</p>
                    </div>
                    <div className="text-right">
                      <p className="text-foreground font-medium">
                        {formatNumber(p.total_input_tokens + p.total_output_tokens)}
                      </p>
                      <p className="text-xs text-muted-foreground">tokens</p>
                    </div>
                    <div className="text-right">
                      <p className="text-foreground font-medium">
                        ${calculateCost(p.total_input_tokens, p.total_output_tokens).toFixed(2)}
                      </p>
                      <p className="text-xs text-muted-foreground">cost</p>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <TrendingUp className="w-10 h-10 mx-auto mb-2 opacity-50" />
                <p>No usage data</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Details */}
        <Card>
          <CardHeader>
            <CardTitle>Model Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {modelUsage.length > 0 ? (
              modelUsage.map((m, index) => (
                <div
                  key={m.model_alias}
                  className={cn(
                    'flex items-center justify-between p-4 rounded-xl',
                    'bg-secondary border border-border',
                    'animate-fade-in card-interactive'
                  )}
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: MODEL_COLORS[index % MODEL_COLORS.length] }}
                    />
                    <span className="font-mono text-sm text-foreground">{m.model_alias}</span>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                      <p className="text-foreground font-medium">{formatNumber(m.total_requests)}</p>
                      <p className="text-xs text-muted-foreground">requests</p>
                    </div>
                    <div className="text-right">
                      <p className="text-foreground font-medium">{m.avg_latency_ms}ms</p>
                      <p className="text-xs text-muted-foreground">avg latency</p>
                    </div>
                    <div className="text-right">
                      <Badge variant={m.avg_latency_ms < 500 ? 'default' : m.avg_latency_ms < 1000 ? 'secondary' : 'muted'}>
                        {m.avg_latency_ms < 500 ? 'Fast' : m.avg_latency_ms < 1000 ? 'Normal' : 'Slow'}
                      </Badge>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Cpu className="w-10 h-10 mx-auto mb-2 opacity-50" />
                <p>No model data</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
