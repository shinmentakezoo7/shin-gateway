'use client';

import { useEffect, useState, useCallback, useMemo } from 'react';
import { Plus, Pencil, Trash2, Cpu, ArrowRight, RefreshCw, Search, ChevronLeft, ChevronRight, Filter } from 'lucide-react';
import { Card, CardContent, Button, Badge, Input, Select, PageHeader, LoadingSpinner, EmptyState } from '@/components/ui';
import { Modal } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import {
  getModels,
  getProviders,
  createModel,
  updateModel,
  deleteModel,
  ModelAlias,
  Provider,
  ApiError,
} from '@/lib/api';
import { cn } from '@/lib/utils';

type ModelForm = {
  alias: string;
  provider_id: string;
  target_model: string;
  default_temperature: string;
  default_max_tokens: string;
};

const emptyForm: ModelForm = {
  alias: '',
  provider_id: '',
  target_model: '',
  default_temperature: '',
  default_max_tokens: '',
};

const PAGE_SIZE_OPTIONS = [10, 25, 50];

export default function ModelsPage() {
  const { success, error: showError } = useToast();
  const [models, setModels] = useState<ModelAlias[]>([]);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<ModelAlias | null>(null);
  const [form, setForm] = useState<ModelForm>(emptyForm);
  const [saving, setSaving] = useState(false);

  // Search and filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [filterProvider, setFilterProvider] = useState<string>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const loadData = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    try {
      const [modelsData, providersData] = await Promise.all([
        getModels(),
        getProviders(),
      ]);
      setModels(modelsData.models || []);
      setProviders(providersData.providers || []);
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to load data', err.detail);
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [showError]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Filter and search models
  const filteredModels = useMemo(() => {
    let result = models;

    // Filter by provider
    if (filterProvider !== 'all') {
      result = result.filter(m => m.provider_id === filterProvider);
    }

    // Search by alias or target model
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(m =>
        m.alias.toLowerCase().includes(query) ||
        m.target_model.toLowerCase().includes(query)
      );
    }

    return result;
  }, [models, filterProvider, searchQuery]);

  // Pagination
  const totalPages = Math.ceil(filteredModels.length / pageSize);
  const paginatedModels = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredModels.slice(start, start + pageSize);
  }, [filteredModels, currentPage, pageSize]);

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, filterProvider, pageSize]);

  // Group models by provider for display
  const modelsByProvider = useMemo(() => {
    const grouped: Record<string, ModelAlias[]> = {};
    paginatedModels.forEach(model => {
      const providerName = providers.find(p => p.id === model.provider_id)?.name || model.provider_id;
      if (!grouped[providerName]) {
        grouped[providerName] = [];
      }
      grouped[providerName].push(model);
    });
    return grouped;
  }, [paginatedModels, providers]);

  const openAddModal = () => {
    setEditingModel(null);
    setForm({
      ...emptyForm,
      provider_id: providers[0]?.id || '',
    });
    setModalOpen(true);
  };

  const openEditModal = (model: ModelAlias) => {
    setEditingModel(model);
    setForm({
      alias: model.alias,
      provider_id: model.provider_id,
      target_model: model.target_model,
      default_temperature: model.default_temperature?.toString() || '',
      default_max_tokens: model.default_max_tokens?.toString() || '',
    });
    setModalOpen(true);
  };

  const handleSave = async () => {
    if (!form.alias || !form.provider_id || !form.target_model) {
      showError('Validation Error', 'Alias, Provider, and Target Model are required');
      return;
    }

    setSaving(true);
    try {
      const data = {
        alias: form.alias,
        provider_id: form.provider_id,
        target_model: form.target_model,
        enabled: true,
        default_temperature: form.default_temperature
          ? parseFloat(form.default_temperature)
          : undefined,
        default_max_tokens: form.default_max_tokens
          ? parseInt(form.default_max_tokens)
          : undefined,
      };

      if (editingModel) {
        await updateModel(editingModel.id, data);
        success('Model Updated', `${form.alias} has been updated`);
      } else {
        await createModel(data);
        success('Model Created', `${form.alias} has been added`);
      }

      setModalOpen(false);
      loadData();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to save model', err.detail);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (model: ModelAlias) => {
    if (!confirm(`Delete model alias "${model.alias}"? This action cannot be undone.`)) return;
    try {
      await deleteModel(model.id);
      success('Model Deleted', `${model.alias} has been removed`);
      loadData();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to delete model', err.detail);
      }
    }
  };

  const getProviderName = (providerId: string) => {
    return providers.find(p => p.id === providerId)?.name || providerId;
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
        title="Model Aliases"
        description={`${models.length} models mapped to ${providers.length} providers`}
        action={
          <div className="flex items-center gap-2">
            <Button variant="secondary" onClick={() => loadData(true)} disabled={refreshing}>
              <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
            </Button>
            <Button onClick={openAddModal} icon={Plus} disabled={providers.length === 0}>
              Add Model
            </Button>
          </div>
        }
      />

      {providers.length === 0 && (
        <Card className="mb-6">
          <CardContent className="py-4">
            <p className="text-sm text-muted-foreground text-center">
              You need to add a provider before creating model aliases.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Search and Filter Bar */}
      {models.length > 0 && (
        <div className="mb-6 flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by alias or target model..."
              className={cn(
                'w-full pl-10 pr-4 py-2.5 rounded-xl',
                'bg-secondary border border-border',
                'text-foreground placeholder:text-muted-foreground',
                'focus:outline-none focus:ring-2 focus:ring-white/20 focus:border-white/20',
                'transition-all'
              )}
            />
          </div>

          {/* Provider Filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-muted-foreground" />
            <select
              value={filterProvider}
              onChange={(e) => setFilterProvider(e.target.value)}
              className={cn(
                'px-4 py-2.5 rounded-xl',
                'bg-secondary border border-border',
                'text-foreground',
                'focus:outline-none focus:ring-2 focus:ring-white/20',
                'transition-all min-w-[150px]'
              )}
            >
              <option value="all">All Providers</option>
              {providers.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          {/* Page Size */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Show:</span>
            <select
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value))}
              className={cn(
                'px-3 py-2.5 rounded-xl',
                'bg-secondary border border-border',
                'text-foreground',
                'focus:outline-none focus:ring-2 focus:ring-white/20',
                'transition-all'
              )}
            >
              {PAGE_SIZE_OPTIONS.map(size => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Results Summary */}
      {models.length > 0 && (
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {paginatedModels.length} of {filteredModels.length} models
            {searchQuery && ` matching "${searchQuery}"`}
            {filterProvider !== 'all' && ` in ${getProviderName(filterProvider)}`}
          </p>
        </div>
      )}

      {models.length === 0 ? (
        <Card>
          <CardContent>
            <EmptyState
              icon={Cpu}
              title="No model aliases configured"
              description="Create model aliases to map client requests to specific provider models."
              action={
                <Button onClick={openAddModal} icon={Plus} disabled={providers.length === 0}>
                  Add Model
                </Button>
              }
            />
          </CardContent>
        </Card>
      ) : filteredModels.length === 0 ? (
        <Card>
          <CardContent className="py-12">
            <div className="text-center">
              <Search className="w-12 h-12 mx-auto mb-3 text-muted-foreground opacity-50" />
              <p className="text-muted-foreground">No models found matching your search</p>
              <Button
                variant="secondary"
                className="mt-4"
                onClick={() => {
                  setSearchQuery('');
                  setFilterProvider('all');
                }}
              >
                Clear Filters
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Models grouped by provider */}
          <div className="space-y-6">
            {Object.entries(modelsByProvider).map(([providerName, providerModels]) => (
              <div key={providerName}>
                {/* Provider Header */}
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                    <Cpu className="w-4 h-4 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">{providerName}</h3>
                  <Badge variant="secondary">{providerModels.length} models</Badge>
                </div>

                {/* Models Table */}
                <div className="rounded-xl border border-border overflow-hidden">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-secondary/50 border-b border-border">
                        <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Alias</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Target Model</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Settings</th>
                        <th className="text-left px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Status</th>
                        <th className="text-right px-4 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {providerModels.map((model) => (
                        <tr
                          key={model.id}
                          className="bg-card hover:bg-secondary/30 transition-colors group"
                        >
                          <td className="px-4 py-3">
                            <span className="font-mono text-sm text-foreground font-medium">
                              {model.alias}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <ArrowRight className="w-3 h-3" />
                              <span className="font-mono text-foreground">{model.target_model}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              {model.default_temperature !== undefined && model.default_temperature !== null && (
                                <Badge variant="secondary" className="text-xs">temp: {model.default_temperature}</Badge>
                              )}
                              {model.default_max_tokens !== undefined && model.default_max_tokens !== null && (
                                <Badge variant="secondary" className="text-xs">max: {model.default_max_tokens}</Badge>
                              )}
                              {model.default_temperature === undefined && model.default_max_tokens === undefined && (
                                <span className="text-xs text-muted-foreground">Default</span>
                              )}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <Badge variant={model.enabled ? 'default' : 'muted'} dot online={model.enabled}>
                              {model.enabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                              <button
                                onClick={() => openEditModal(model)}
                                className="w-8 h-8 rounded-lg bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground flex items-center justify-center transition-all"
                                title="Edit"
                              >
                                <Pencil className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleDelete(model)}
                                className="w-8 h-8 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 flex items-center justify-center transition-all"
                                title="Delete"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-6 flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Page {currentPage} of {totalPages}
              </p>
              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                  Previous
                </Button>

                {/* Page numbers */}
                <div className="flex items-center gap-1">
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    let pageNum: number;
                    if (totalPages <= 5) {
                      pageNum = i + 1;
                    } else if (currentPage <= 3) {
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i;
                    } else {
                      pageNum = currentPage - 2 + i;
                    }

                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={cn(
                          'w-8 h-8 rounded-lg text-sm font-medium transition-all',
                          currentPage === pageNum
                            ? 'bg-white text-black'
                            : 'bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground'
                        )}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                </div>

                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Model Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editingModel ? 'Edit Model' : 'Add Model Alias'}
      >
        <div className="space-y-4">
          <Input
            label="Alias (client-facing name)"
            value={form.alias}
            onChange={(e) => setForm({ ...form, alias: e.target.value })}
            placeholder="claude-3-5-sonnet-20241022"
            required
          />
          <Select
            label="Provider"
            value={form.provider_id}
            onChange={(v) => setForm({ ...form, provider_id: v })}
            options={providers.map((p) => ({ value: p.id, label: p.name }))}
          />
          <Input
            label="Target Model"
            value={form.target_model}
            onChange={(e) => setForm({ ...form, target_model: e.target.value })}
            placeholder="llama-3.3-70b-versatile"
            required
          />
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Default Temperature"
              type="number"
              value={form.default_temperature}
              onChange={(e) => setForm({ ...form, default_temperature: e.target.value })}
              placeholder="0.7"
            />
            <Input
              label="Default Max Tokens"
              type="number"
              value={form.default_max_tokens}
              onChange={(e) => setForm({ ...form, default_max_tokens: e.target.value })}
              placeholder="8192"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-border">
          <Button variant="secondary" onClick={() => setModalOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : null}
            {editingModel ? 'Update' : 'Create'} Model
          </Button>
        </div>
      </Modal>
    </div>
  );
}
