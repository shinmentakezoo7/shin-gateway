'use client';

import { useEffect, useState, useCallback } from 'react';
import { Plus, Pencil, Trash2, Power, ExternalLink, Server, RefreshCw, Download, Check } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, Button, Badge, Input, Select, PageHeader, LoadingSpinner, EmptyState } from '@/components/ui';
import { Modal } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import {
  getProviders,
  createProvider,
  updateProvider,
  deleteProvider,
  toggleProvider,
  fetchProviderModels,
  importProviderModels,
  Provider,
  ProviderModel,
  formatNumber,
  ApiError,
} from '@/lib/api';
import { cn } from '@/lib/utils';

type ProviderForm = {
  name: string;
  type: 'openai' | 'anthropic';
  base_url: string;
  api_key: string;
  api_key_env: string;
  timeout: string;
  rate_limit_rpm: string;
  rate_limit_tpm: string;
};

const emptyForm: ProviderForm = {
  name: '',
  type: 'openai',
  base_url: '',
  api_key: '',
  api_key_env: '',
  timeout: '120',
  rate_limit_rpm: '',
  rate_limit_tpm: '',
};

export default function ProvidersPage() {
  const { success, error: showError } = useToast();
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [editingProvider, setEditingProvider] = useState<Provider | null>(null);
  const [form, setForm] = useState<ProviderForm>(emptyForm);
  const [saving, setSaving] = useState(false);

  // Fetch models modal state
  const [modelsModalOpen, setModelsModalOpen] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [importingModels, setImportingModels] = useState(false);
  const [availableModels, setAvailableModels] = useState<ProviderModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [modelsProvider, setModelsProvider] = useState<Provider | null>(null);

  const loadProviders = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    try {
      const data = await getProviders();
      setProviders(data.providers || []);
    } catch (err) {
      console.error('Failed to load providers:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadProviders();
  }, [loadProviders]);

  const openAddModal = () => {
    setEditingProvider(null);
    setForm(emptyForm);
    setModalOpen(true);
  };

  const openEditModal = (provider: Provider) => {
    setEditingProvider(provider);
    setForm({
      name: provider.name,
      type: provider.type,
      base_url: provider.base_url,
      api_key: '',
      api_key_env: provider.api_key_env || '',
      timeout: provider.timeout?.toString() || '120',
      rate_limit_rpm: provider.rate_limit_rpm?.toString() || '',
      rate_limit_tpm: provider.rate_limit_tpm?.toString() || '',
    });
    setModalOpen(true);
  };

  const handleSave = async (fetchModelsAfter: boolean = false) => {
    if (!form.name || !form.base_url) {
      showError('Validation Error', 'Name and Base URL are required');
      return;
    }

    setSaving(true);
    try {
      const data = {
        name: form.name,
        type: form.type,
        base_url: form.base_url,
        api_key: form.api_key || undefined,
        api_key_env: form.api_key_env || undefined,
        timeout: form.timeout ? parseInt(form.timeout) : 120,
        rate_limit_rpm: form.rate_limit_rpm ? parseInt(form.rate_limit_rpm) : undefined,
        rate_limit_tpm: form.rate_limit_tpm ? parseInt(form.rate_limit_tpm) : undefined,
      };

      let savedProvider: Provider;

      if (editingProvider) {
        const result = await updateProvider(editingProvider.id, data);
        savedProvider = result.provider;
        success('Provider Updated', `${form.name} has been updated`);
      } else {
        const result = await createProvider(data);
        savedProvider = result.provider;
        success('Provider Created', `${form.name} has been added`);
      }

      setModalOpen(false);
      loadProviders();

      // Open fetch models modal if requested
      if (fetchModelsAfter && savedProvider) {
        // Small delay to let the provider modal close
        setTimeout(() => {
          openFetchModelsModal(savedProvider);
        }, 300);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to save provider', err.detail);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (provider: Provider) => {
    if (!confirm(`Delete provider "${provider.name}"? This action cannot be undone.`)) return;
    try {
      await deleteProvider(provider.id);
      success('Provider Deleted', `${provider.name} has been removed`);
      loadProviders();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to delete provider', err.detail);
      }
    }
  };

  const handleToggle = async (provider: Provider) => {
    try {
      await toggleProvider(provider.id);
      success(
        provider.enabled ? 'Provider Disabled' : 'Provider Enabled',
        `${provider.name} has been ${provider.enabled ? 'disabled' : 'enabled'}`
      );
      loadProviders();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to toggle provider', err.detail);
      }
    }
  };

  const openFetchModelsModal = async (provider: Provider) => {
    setModelsProvider(provider);
    setAvailableModels([]);
    setSelectedModels([]);
    setModelsModalOpen(true);
    setFetchingModels(true);

    try {
      const result = await fetchProviderModels(provider.id);
      setAvailableModels(result.models);
      // Select all models by default
      setSelectedModels(result.models.map(m => m.id));
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to fetch models', err.detail);
      }
      setModelsModalOpen(false);
    } finally {
      setFetchingModels(false);
    }
  };

  const toggleModelSelection = (modelId: string) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(m => m !== modelId)
        : [...prev, modelId]
    );
  };

  const selectAllModels = () => {
    setSelectedModels(availableModels.map(m => m.id));
  };

  const deselectAllModels = () => {
    setSelectedModels([]);
  };

  const handleImportModels = async () => {
    if (!modelsProvider || selectedModels.length === 0) return;

    setImportingModels(true);
    try {
      const result = await importProviderModels(modelsProvider.id, {
        models: selectedModels,
      });

      if (result.total_created > 0) {
        success(
          'Models Imported',
          `${result.total_created} model(s) imported successfully`
        );
      }
      if (result.total_skipped > 0) {
        showError(
          'Some models skipped',
          `${result.total_skipped} model(s) already exist`
        );
      }

      setModelsModalOpen(false);
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to import models', err.detail);
      }
    } finally {
      setImportingModels(false);
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
        title="Providers"
        description="Manage your backend API providers"
        action={
          <div className="flex items-center gap-2">
            <Button variant="secondary" onClick={() => loadProviders(true)} disabled={refreshing}>
              <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
            </Button>
            <Button onClick={openAddModal} icon={Plus}>
              Add Provider
            </Button>
          </div>
        }
      />

      {providers.length === 0 ? (
        <Card>
          <CardContent>
            <EmptyState
              icon={Server}
              title="No providers configured"
              description="Add your first provider to start routing requests to different backends."
              action={
                <Button onClick={openAddModal} icon={Plus}>
                  Add Provider
                </Button>
              }
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {providers.map((provider, index) => (
            <div
              key={provider.id}
              className={cn(
                'group rounded-xl border border-border bg-card',
                'overflow-hidden transition-all duration-200',
                'hover:border-white/20 card-interactive',
                'animate-fade-in'
              )}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div className="p-5">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div
                      className={cn(
                        'w-12 h-12 rounded-xl flex items-center justify-center',
                        'bg-white/10 text-white'
                      )}
                    >
                      <Server className="w-6 h-6" />
                    </div>
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-lg font-semibold text-foreground">{provider.name}</h3>
                        <Badge variant="secondary">
                          {provider.type}
                        </Badge>
                        <Badge variant={provider.enabled ? 'default' : 'muted'} dot online={provider.enabled}>
                          {provider.enabled ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <span className="font-mono">{provider.base_url}</span>
                        <ExternalLink className="w-3 h-3" />
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => openFetchModelsModal(provider)}
                      className="w-9 h-9 rounded-lg bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground flex items-center justify-center transition-all"
                      title="Fetch & Import Models"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleToggle(provider)}
                      className={cn(
                        'w-9 h-9 rounded-lg flex items-center justify-center transition-all',
                        provider.enabled
                          ? 'bg-white/20 text-white hover:bg-white/30'
                          : 'bg-secondary text-muted-foreground hover:bg-accent'
                      )}
                      title={provider.enabled ? 'Disable' : 'Enable'}
                    >
                      <Power className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => openEditModal(provider)}
                      className="w-9 h-9 rounded-lg bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground flex items-center justify-center transition-all"
                      title="Edit"
                    >
                      <Pencil className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(provider)}
                      className="w-9 h-9 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 flex items-center justify-center transition-all"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Provider Details */}
                <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-border text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">Timeout:</span>
                    <span className="text-foreground">{provider.timeout}s</span>
                  </div>
                  {provider.rate_limit_rpm && (
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">RPM:</span>
                      <span className="text-foreground">{provider.rate_limit_rpm}</span>
                    </div>
                  )}
                  {provider.rate_limit_tpm && (
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">TPM:</span>
                      <span className="text-foreground">{formatNumber(provider.rate_limit_tpm)}</span>
                    </div>
                  )}
                  {provider.api_key_env && (
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground">API Key:</span>
                      <span className="text-foreground font-mono text-xs">${provider.api_key_env}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Provider Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editingProvider ? 'Edit Provider' : 'Add Provider'}
      >
        <div className="space-y-4">
          <Input
            label="Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="my-provider"
            required
          />
          <Select
            label="Type"
            value={form.type}
            onChange={(v) => setForm({ ...form, type: v as 'openai' | 'anthropic' })}
            options={[
              { value: 'openai', label: 'OpenAI Compatible' },
              { value: 'anthropic', label: 'Anthropic' },
            ]}
          />
          <Input
            label="Base URL"
            value={form.base_url}
            onChange={(e) => setForm({ ...form, base_url: e.target.value })}
            placeholder="https://api.example.com/v1"
            required
          />
          <Input
            label="Timeout (seconds)"
            type="number"
            value={form.timeout}
            onChange={(e) => setForm({ ...form, timeout: e.target.value })}
            placeholder="120"
          />
          <Input
            label="API Key (leave empty to use environment variable)"
            type="password"
            value={form.api_key}
            onChange={(e) => setForm({ ...form, api_key: e.target.value })}
            placeholder="sk-..."
          />
          <Input
            label="API Key Environment Variable"
            value={form.api_key_env}
            onChange={(e) => setForm({ ...form, api_key_env: e.target.value })}
            placeholder="OPENAI_API_KEY"
          />
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Rate Limit (RPM)"
              type="number"
              value={form.rate_limit_rpm}
              onChange={(e) => setForm({ ...form, rate_limit_rpm: e.target.value })}
              placeholder="Optional"
            />
            <Input
              label="Rate Limit (TPM)"
              type="number"
              value={form.rate_limit_tpm}
              onChange={(e) => setForm({ ...form, rate_limit_tpm: e.target.value })}
              placeholder="Optional"
            />
          </div>
        </div>

        <div className="flex justify-between gap-3 mt-6 pt-4 border-t border-border">
          <div>
            {editingProvider && (
              <Button
                variant="secondary"
                onClick={() => {
                  setModalOpen(false);
                  setTimeout(() => openFetchModelsModal(editingProvider), 300);
                }}
                disabled={saving}
              >
                <Download className="w-4 h-4" />
                Fetch Models
              </Button>
            )}
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={() => setModalOpen(false)}>
              Cancel
            </Button>
            {!editingProvider && (
              <Button
                variant="secondary"
                onClick={() => handleSave(true)}
                disabled={saving || !form.name || !form.base_url}
              >
                {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                Save & Fetch Models
              </Button>
            )}
            <Button onClick={() => handleSave(false)} disabled={saving}>
              {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : null}
              {editingProvider ? 'Update' : 'Create'} Provider
            </Button>
          </div>
        </div>
      </Modal>

      {/* Fetch Models Modal */}
      <Modal
        isOpen={modelsModalOpen}
        onClose={() => setModelsModalOpen(false)}
        title={`Import Models from ${modelsProvider?.name || 'Provider'}`}
      >
        {fetchingModels ? (
          <div className="flex flex-col items-center justify-center py-12">
            <LoadingSpinner size="lg" />
            <p className="text-muted-foreground mt-4">Fetching available models...</p>
          </div>
        ) : availableModels.length === 0 ? (
          <div className="text-center py-12">
            <Server className="w-12 h-12 mx-auto mb-3 text-muted-foreground opacity-50" />
            <p className="text-muted-foreground">No models found</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-muted-foreground">
                {selectedModels.length} of {availableModels.length} models selected
              </p>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={selectAllModels}>
                  Select All
                </Button>
                <Button variant="secondary" size="sm" onClick={deselectAllModels}>
                  Deselect All
                </Button>
              </div>
            </div>

            <div className="max-h-80 overflow-y-auto space-y-2 border border-border rounded-xl p-2">
              {availableModels.map((model, index) => (
                <button
                  key={`${index}-${model.id}`}
                  onClick={() => toggleModelSelection(model.id)}
                  className={cn(
                    'w-full flex items-center justify-between p-3 rounded-lg transition-all text-left',
                    selectedModels.includes(model.id)
                      ? 'bg-white text-black'
                      : 'bg-secondary text-foreground hover:bg-accent'
                  )}
                >
                  <div>
                    <p className="font-mono text-sm">{model.id}</p>
                    {model.owned_by && (
                      <p className={cn(
                        'text-xs',
                        selectedModels.includes(model.id) ? 'text-black/60' : 'text-muted-foreground'
                      )}>
                        {model.owned_by}
                      </p>
                    )}
                  </div>
                  {selectedModels.includes(model.id) && (
                    <Check className="w-5 h-5" />
                  )}
                </button>
              ))}
            </div>
          </>
        )}

        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-border">
          <Button variant="secondary" onClick={() => setModelsModalOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleImportModels}
            disabled={importingModels || selectedModels.length === 0}
          >
            {importingModels ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            Import {selectedModels.length} Model{selectedModels.length !== 1 ? 's' : ''}
          </Button>
        </div>
      </Modal>
    </div>
  );
}
