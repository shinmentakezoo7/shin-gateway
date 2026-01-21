"""
Shin Gateway Configuration Settings

Pydantic-based configuration loading from YAML and environment variables.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml


# =============================================================================
# Gateway Configuration
# =============================================================================

class GatewayConfig(BaseModel):
    """Gateway server configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    require_api_key: bool = True
    api_key_header: str = "x-api-key"
    api_keys: list[str] = Field(default_factory=list)
    request_timeout: int = 120
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_request_size: int = 100 * 1024 * 1024  # 100MB for large contexts


# =============================================================================
# Rate Limit Configuration
# =============================================================================

class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_second: int = 10
    burst_size: int = 5


# =============================================================================
# Provider Configuration
# =============================================================================

class ProviderConfig(BaseModel):
    """LLM Provider configuration"""
    type: Literal["openai_compat", "litellm", "anthropic"] = "openai_compat"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 120
    rate_limit: Optional[RateLimitConfig] = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from direct value or environment variable"""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


# =============================================================================
# Model Defaults
# =============================================================================

class ModelDefaults(BaseModel):
    """Default parameters for a model"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None


# =============================================================================
# Fallback Configuration
# =============================================================================

class FallbackConfig(BaseModel):
    """Fallback provider configuration"""
    provider: str
    model: str


# =============================================================================
# Model Alias Configuration
# =============================================================================

class ModelConfig(BaseModel):
    """Model alias configuration"""
    provider: str
    model: str
    defaults: Optional[ModelDefaults] = None
    fallbacks: list[FallbackConfig] = Field(default_factory=list)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None


# =============================================================================
# Circuit Breaker Configuration
# =============================================================================

class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 1


# =============================================================================
# Retry Configuration
# =============================================================================

class RetrySettings(BaseModel):
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True


# =============================================================================
# Complete Settings
# =============================================================================

class Settings(BaseModel):
    """Complete application settings"""
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file"""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider by name"""
        return self.providers.get(name)

    def get_model(self, alias: str) -> Optional[ModelConfig]:
        """Get model config by alias"""
        return self.models.get(alias)

    def resolve_model(self, model_identifier: str) -> tuple[Optional[ProviderConfig], Optional[ModelConfig]]:
        """
        Resolve model identifier (alias or ID) to provider and model config.

        Supports lookup by:
        1. Model alias (the primary key in models dict)
        2. Model ID (matches against target_model field)
        3. Direct model name (if it matches a target_model in any config)
        4. Dynamic routing with provider/model format (e.g., openai/gpt-4o, groq/llama-3.3-70b)
        """
        # First try direct alias lookup
        model_config = self.get_model(model_identifier)
        if model_config:
            provider_config = self.get_provider(model_config.provider)
            return provider_config, model_config

        # If not found by alias, try to find by target_model (the actual model ID)
        for alias, config in self.models.items():
            if config.model == model_identifier:
                provider_config = self.get_provider(config.provider)
                return provider_config, config

        # Dynamic routing: support provider/model format (e.g., openai/gpt-4o, groq/llama-3.3-70b)
        if "/" in model_identifier:
            parts = model_identifier.split("/", 1)
            if len(parts) == 2:
                provider_name, target_model = parts
                provider_config = self.get_provider(provider_name)
                if provider_config:
                    # Create a dynamic model config for this request
                    dynamic_model_config = ModelConfig(
                        provider=provider_name,
                        model=target_model,
                    )
                    return provider_config, dynamic_model_config

        return None, None


# =============================================================================
# Environment Settings
# =============================================================================

class EnvSettings(BaseSettings):
    """Environment-based settings"""
    config_path: Path = Path("config/config.yaml")
    gateway_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    log_level: str = "INFO"
    debug: bool = False

    class Config:
        env_prefix = "SHIN_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# =============================================================================
# Settings Loader
# =============================================================================

_settings: Optional[Settings] = None
_env_settings: Optional[EnvSettings] = None


def load_env_settings() -> EnvSettings:
    """Load environment settings"""
    global _env_settings
    if _env_settings is None:
        _env_settings = EnvSettings()
    return _env_settings


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load application settings from YAML, admin database, and environment"""
    global _settings

    if _settings is not None and config_path is None:
        return _settings

    _settings = _load_settings_impl(config_path)
    return _settings


def get_settings() -> Settings:
    """Get loaded settings (must call load_settings first)"""
    if _settings is None:
        return load_settings()
    return _settings


def reload_settings(config_path: Optional[Path] = None) -> Settings:
    """Force reload settings atomically"""
    global _settings
    # Load new settings first, then swap atomically
    new_settings = _load_settings_impl(config_path)
    _settings = new_settings
    return _settings


def _load_settings_impl(config_path: Optional[Path] = None) -> Settings:
    """Internal implementation of settings loading"""
    env = load_env_settings()
    path = config_path or env.config_path

    # Load from YAML
    settings = Settings.from_yaml(path)

    # Override with environment variables
    if env.gateway_api_key:
        settings.gateway.api_keys.append(env.gateway_api_key)

    if env.log_level:
        settings.gateway.log_level = env.log_level

    # Set API keys from environment if not in config
    for name, provider in settings.providers.items():
        if provider.api_key_env and not provider.api_key:
            provider.api_key = os.getenv(provider.api_key_env)

    # Merge providers and models from admin database
    try:
        from admin.models import get_admin_db
        db = get_admin_db()

        # Load providers from admin database
        # Build a mapping of provider_id -> provider_name for model alias resolution
        provider_id_to_name = {}

        for provider in db.list_providers(enabled_only=False):  # Load ALL providers, not just enabled
            provider_name = provider.name  # Use name as provider key (user-friendly)
            provider_id_to_name[provider.id] = provider_name

            # Only add enabled providers to settings
            if not provider.enabled:
                continue

            # Get API key: prefer api_key, fallback to api_key_env value or env var
            api_key = provider.api_key
            if not api_key and provider.api_key_env:
                # Check if api_key_env is an actual key (starts with common prefixes) or env var name
                if provider.api_key_env.startswith(('sk-', 'nvapi-', 'gsk_', 'xai-')):
                    api_key = provider.api_key_env
                else:
                    api_key = os.getenv(provider.api_key_env)

            # Map provider type
            provider_type = "openai_compat"
            if provider.type == "anthropic":
                provider_type = "anthropic"
            elif provider.type == "openai":
                provider_type = "openai_compat"

            settings.providers[provider_name] = ProviderConfig(
                type=provider_type,
                base_url=provider.base_url,
                api_key=api_key,
                timeout=provider.timeout,
                rate_limit=RateLimitConfig(
                    requests_per_minute=provider.rate_limit_rpm or 60,
                    requests_per_second=10,
                    burst_size=5
                ) if provider.rate_limit_rpm else None,
                extra_headers=provider.extra_headers or {},
            )

        # Load model aliases from admin database
        for model in db.list_model_aliases(enabled_only=True):
            # Resolve provider_id to provider name
            provider_name = provider_id_to_name.get(model.provider_id, model.provider_id)

            # Only add model if its provider exists in settings (is enabled)
            if provider_name not in settings.providers:
                import logging
                logging.getLogger("shin-gateway").warning(
                    f"Model alias '{model.alias}' references disabled provider '{provider_name}', skipping"
                )
                continue

            settings.models[model.alias] = ModelConfig(
                provider=provider_name,  # Use resolved provider name
                model=model.target_model,
                defaults=ModelDefaults(
                    temperature=model.default_temperature,
                    max_tokens=model.default_max_tokens,
                ) if model.default_temperature or model.default_max_tokens else None,
            )
    except Exception as e:
        import logging
        logging.getLogger("shin-gateway").warning(f"Failed to load admin database settings: {e}")

    return settings
