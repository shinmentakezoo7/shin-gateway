'use client';

import { useEffect, useState, useCallback } from 'react';
import { Plus, Trash2, Power, Copy, Check, Key, Shield, Clock, RefreshCw } from 'lucide-react';
import { Card, CardContent, Button, Badge, Input, PageHeader, LoadingSpinner, EmptyState } from '@/components/ui';
import { Modal } from '@/components/Modal';
import { useToast } from '@/components/Toast';
import {
  getApiKeys,
  getModels,
  createApiKey,
  deleteApiKey,
  toggleApiKey,
  ApiKey,
  ModelAlias,
  formatNumber,
  formatDate,
  getRelativeTime,
  ApiError,
} from '@/lib/api';
import { cn } from '@/lib/utils';

type ApiKeyForm = {
  name: string;
  rate_limit_rpm: string;
  rate_limit_tpm: string;
  expires_in_days: string;
  allowed_models: string[];
};

const emptyForm: ApiKeyForm = {
  name: '',
  rate_limit_rpm: '',
  rate_limit_tpm: '',
  expires_in_days: '',
  allowed_models: [],
};

export default function ApiKeysPage() {
  const { success, error: showError } = useToast();
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [models, setModels] = useState<ModelAlias[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [form, setForm] = useState<ApiKeyForm>(emptyForm);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadData = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    try {
      const [keysData, modelsData] = await Promise.all([
        getApiKeys(),
        getModels(),
      ]);
      setApiKeys(keysData.api_keys || []);
      setModels(modelsData.models || []);
    } catch (err) {
      console.error('Failed to load API keys:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const openAddModal = () => {
    setForm(emptyForm);
    setModalOpen(true);
  };

  const handleCreate = async () => {
    if (!form.name) {
      showError('Validation Error', 'Name is required');
      return;
    }

    setSaving(true);
    try {
      const data = {
        name: form.name,
        rate_limit_rpm: form.rate_limit_rpm ? parseInt(form.rate_limit_rpm) : undefined,
        rate_limit_tpm: form.rate_limit_tpm ? parseInt(form.rate_limit_tpm) : undefined,
        expires_in_days: form.expires_in_days ? parseInt(form.expires_in_days) : undefined,
        allowed_models: form.allowed_models.length > 0 ? form.allowed_models : undefined,
      };

      const result = await createApiKey(data);
      setNewApiKey(result.key);
      success('API Key Created', `${form.name} has been generated`);
      setModalOpen(false);
      loadData();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to create API key', err.detail);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (key: ApiKey) => {
    if (!confirm(`Delete API key "${key.name}"? This action cannot be undone.`)) return;
    try {
      await deleteApiKey(key.id);
      success('API Key Deleted', `${key.name} has been removed`);
      loadData();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to delete API key', err.detail);
      }
    }
  };

  const handleToggle = async (key: ApiKey) => {
    try {
      await toggleApiKey(key.id);
      success(
        key.enabled ? 'API Key Disabled' : 'API Key Enabled',
        `${key.name} has been ${key.enabled ? 'disabled' : 'enabled'}`
      );
      loadData();
    } catch (err) {
      if (err instanceof ApiError) {
        showError('Failed to toggle API key', err.detail);
      }
    }
  };

  const copyKey = () => {
    if (newApiKey) {
      navigator.clipboard.writeText(newApiKey);
      setCopied(true);
      success('Copied', 'API key copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const closeKeyDisplay = () => {
    setNewApiKey(null);
    setCopied(false);
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
        title="API Keys"
        description="Manage access tokens for your gateway"
        action={
          <div className="flex items-center gap-2">
            <Button variant="secondary" onClick={() => loadData(true)} disabled={refreshing}>
              <RefreshCw className={cn('w-4 h-4', refreshing && 'animate-spin')} />
            </Button>
            <Button onClick={openAddModal} icon={Plus}>
              Generate Key
            </Button>
          </div>
        }
      />

      {/* New Key Display */}
      {newApiKey && (
        <div className="mb-6 rounded-xl border border-white/30 bg-white/5 p-6 animate-fade-in">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-white/20 text-white flex items-center justify-center flex-shrink-0">
              <Shield className="w-6 h-6" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-semibold text-foreground mb-1">
                New API Key Generated!
              </h3>
              <p className="text-sm text-muted-foreground mb-3">
                Copy this key now - it won&apos;t be shown again
              </p>
              <div className="bg-secondary rounded-xl p-4 font-mono text-sm text-foreground break-all border border-border">
                {newApiKey}
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <Button onClick={copyKey} size="sm">
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                {copied ? 'Copied!' : 'Copy'}
              </Button>
              <Button variant="secondary" onClick={closeKeyDisplay} size="sm">
                Close
              </Button>
            </div>
          </div>
        </div>
      )}

      {apiKeys.length === 0 ? (
        <Card>
          <CardContent>
            <EmptyState
              icon={Key}
              title="No API keys created"
              description="Generate API keys to authenticate requests to your gateway."
              action={
                <Button onClick={openAddModal} icon={Plus}>
                  Generate Key
                </Button>
              }
            />
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {apiKeys.map((key, index) => (
            <div
              key={key.id}
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
                        key.enabled
                          ? 'bg-white/20 text-white'
                          : 'bg-secondary text-muted-foreground'
                      )}
                    >
                      <Key className="w-6 h-6" />
                    </div>
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-lg font-semibold text-foreground">{key.name}</h3>
                        <Badge variant={key.enabled ? 'default' : 'muted'} dot online={key.enabled}>
                          {key.enabled ? 'Active' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="font-mono text-muted-foreground">{key.key_prefix}...</span>
                        {key.last_used_at && (
                          <span className="flex items-center gap-1 text-muted-foreground">
                            <Clock className="w-3 h-3" />
                            Used {getRelativeTime(key.last_used_at)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => handleToggle(key)}
                      className={cn(
                        'w-9 h-9 rounded-lg flex items-center justify-center transition-all',
                        key.enabled
                          ? 'bg-white/20 text-white hover:bg-white/30'
                          : 'bg-secondary text-muted-foreground hover:bg-accent'
                      )}
                      title={key.enabled ? 'Disable' : 'Enable'}
                    >
                      <Power className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(key)}
                      className="w-9 h-9 rounded-lg bg-white/10 text-white/70 hover:bg-white/20 flex items-center justify-center transition-all"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Key Details */}
                <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-border">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Rate Limits:</span>
                    {key.rate_limit_rpm ? (
                      <Badge variant="secondary">{key.rate_limit_rpm} RPM</Badge>
                    ) : null}
                    {key.rate_limit_tpm ? (
                      <Badge variant="secondary">{formatNumber(key.rate_limit_tpm)} TPM</Badge>
                    ) : null}
                    {!key.rate_limit_rpm && !key.rate_limit_tpm && (
                      <Badge variant="muted">Unlimited</Badge>
                    )}
                  </div>
                  {key.expires_at && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Expires:</span>
                      <span className="text-sm text-foreground">{formatDate(key.expires_at)}</span>
                    </div>
                  )}
                  {key.allowed_models && key.allowed_models.length > 0 && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">Models:</span>
                      <span className="text-sm text-foreground">{key.allowed_models.length} allowed</span>
                    </div>
                  )}
                  {key.created_at && (
                    <div className="flex items-center gap-2 ml-auto">
                      <span className="text-xs text-muted-foreground">Created:</span>
                      <span className="text-sm text-foreground">{formatDate(key.created_at)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* API Key Modal */}
      <Modal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Generate API Key"
      >
        <div className="space-y-4">
          <Input
            label="Name / Description"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="Production API Key"
            required
          />
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Rate Limit (RPM)"
              type="number"
              value={form.rate_limit_rpm}
              onChange={(e) => setForm({ ...form, rate_limit_rpm: e.target.value })}
              placeholder="Unlimited"
            />
            <Input
              label="Rate Limit (TPM)"
              type="number"
              value={form.rate_limit_tpm}
              onChange={(e) => setForm({ ...form, rate_limit_tpm: e.target.value })}
              placeholder="Unlimited"
            />
          </div>
          <Input
            label="Expires In (days, leave empty for no expiry)"
            type="number"
            value={form.expires_in_days}
            onChange={(e) => setForm({ ...form, expires_in_days: e.target.value })}
            placeholder="Never"
          />
          {models.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-muted-foreground">
                Allowed Models (leave empty for all)
              </label>
              <div className="flex flex-wrap gap-2">
                {models.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() => {
                      const allowed = form.allowed_models.includes(model.alias)
                        ? form.allowed_models.filter((m) => m !== model.alias)
                        : [...form.allowed_models, model.alias];
                      setForm({ ...form, allowed_models: allowed });
                    }}
                    className={cn(
                      'px-3 py-1.5 rounded-lg text-sm font-mono transition-all',
                      form.allowed_models.includes(model.alias)
                        ? 'bg-white text-black'
                        : 'bg-secondary text-muted-foreground hover:bg-accent'
                    )}
                  >
                    {model.alias}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-border">
          <Button variant="secondary" onClick={() => setModalOpen(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={saving}>
            {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : null}
            Generate Key
          </Button>
        </div>
      </Modal>
    </div>
  );
}
