"""
Admin API Router

REST API endpoints for managing providers, models, API keys, and viewing statistics.
"""

from __future__ import annotations
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from admin.models import (
    get_admin_db, AdminDatabase, Provider, ModelAlias, ApiKey
)
from admin.stats import get_stats_collector, StatsCollector
from config.settings import reload_settings


router = APIRouter(prefix="/admin", tags=["admin"])


# =============================================================================
# Pydantic Models for API
# =============================================================================

class ProviderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    type: str = Field(..., pattern="^(openai|anthropic)$")
    base_url: str = Field(..., min_length=1)
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = Field(default=120, ge=1, le=600)
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1)
    rate_limit_tpm: Optional[int] = Field(default=None, ge=1)
    extra_headers: dict = Field(default_factory=dict)


class ProviderUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    type: Optional[str] = Field(default=None, pattern="^(openai|anthropic)$")
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: Optional[int] = Field(default=None, ge=1, le=600)
    enabled: Optional[bool] = None
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1)
    rate_limit_tpm: Optional[int] = Field(default=None, ge=1)
    extra_headers: Optional[dict] = None


class ModelAliasCreate(BaseModel):
    alias: str = Field(..., min_length=1, max_length=100)
    provider_id: str = Field(..., min_length=1)
    target_model: str = Field(..., min_length=1)
    default_temperature: Optional[float] = Field(default=None, ge=0, le=2)
    default_max_tokens: Optional[int] = Field(default=None, ge=1)


class ModelAliasUpdate(BaseModel):
    alias: Optional[str] = Field(default=None, min_length=1, max_length=100)
    provider_id: Optional[str] = None
    target_model: Optional[str] = None
    enabled: Optional[bool] = None
    default_temperature: Optional[float] = Field(default=None, ge=0, le=2)
    default_max_tokens: Optional[int] = Field(default=None, ge=1)


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1)
    rate_limit_tpm: Optional[int] = Field(default=None, ge=1)
    allowed_models: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(default=None, ge=1)


class ApiKeyUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    enabled: Optional[bool] = None
    rate_limit_rpm: Optional[int] = Field(default=None, ge=1)
    rate_limit_tpm: Optional[int] = Field(default=None, ge=1)
    allowed_models: Optional[List[str]] = None


# =============================================================================
# Dependencies
# =============================================================================

def get_db() -> AdminDatabase:
    return get_admin_db()


def get_stats() -> StatsCollector:
    return get_stats_collector()


# =============================================================================
# Provider Endpoints
# =============================================================================

@router.get("/providers")
async def list_providers(
    enabled_only: bool = Query(False),
    db: AdminDatabase = Depends(get_db)
):
    """List all providers"""
    providers = db.list_providers(enabled_only=enabled_only)
    return {
        "providers": [p.to_dict() for p in providers],
        "total": len(providers)
    }


@router.post("/providers")
async def create_provider(
    data: ProviderCreate,
    db: AdminDatabase = Depends(get_db)
):
    """Create a new provider"""
    provider_id = secrets.token_urlsafe(8)

    provider = Provider(
        id=provider_id,
        name=data.name,
        type=data.type,
        base_url=data.base_url,
        api_key=data.api_key,
        api_key_env=data.api_key_env,
        timeout=data.timeout,
        enabled=True,
        rate_limit_rpm=data.rate_limit_rpm,
        rate_limit_tpm=data.rate_limit_tpm,
        extra_headers=data.extra_headers
    )

    try:
        created = db.create_provider(provider)
        reload_settings()  # Reload settings so new provider is available
        return {"provider": created.to_dict(), "message": "Provider created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/providers/{provider_id}")
async def get_provider(
    provider_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Get provider by ID"""
    provider = db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return {"provider": provider.to_dict()}


@router.patch("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    data: ProviderUpdate,
    db: AdminDatabase = Depends(get_db)
):
    """Update a provider"""
    updates = data.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated = db.update_provider(provider_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Provider not found")

    reload_settings()  # Reload settings so provider changes are available
    return {"provider": updated.to_dict(), "message": "Provider updated successfully"}


@router.delete("/providers/{provider_id}")
async def delete_provider(
    provider_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Delete a provider and its associated model aliases"""
    # First, delete all model aliases associated with this provider
    deleted_models = db.delete_model_aliases_by_provider(provider_id)

    # Then delete the provider
    if db.delete_provider(provider_id):
        reload_settings()  # Reload settings so provider removal is reflected
        return {
            "message": f"Provider deleted successfully. {deleted_models} model alias(es) also removed."
        }
    raise HTTPException(status_code=404, detail="Provider not found")


@router.post("/providers/{provider_id}/toggle")
async def toggle_provider(
    provider_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Toggle provider enabled/disabled"""
    provider = db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    updated = db.update_provider(provider_id, {"enabled": not provider.enabled})
    reload_settings()  # Reload settings so provider state change is reflected
    return {
        "provider": updated.to_dict(),
        "message": f"Provider {'enabled' if updated.enabled else 'disabled'}"
    }


@router.get("/providers/{provider_id}/fetch-models")
async def fetch_provider_models(
    provider_id: str,
    request: Request,
    db: AdminDatabase = Depends(get_db)
):
    """Fetch available models from the provider's API"""
    import httpx
    import os

    provider = db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    # Get API key
    api_key = provider.api_key
    if not api_key and provider.api_key_env:
        api_key = os.environ.get(provider.api_key_env)

    if not api_key:
        raise HTTPException(status_code=400, detail="No API key configured for this provider")

    models = []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if provider.type == "openai":
                # OpenAI-compatible API
                base_url = provider.base_url.rstrip('/')
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                response.raise_for_status()
                data = response.json()

                # Handle OpenAI format
                if "data" in data:
                    for model in data["data"]:
                        model_id = model.get("id", "")
                        if model_id:
                            models.append({
                                "id": model_id,
                                "owned_by": model.get("owned_by", "unknown"),
                                "created": model.get("created", 0)
                            })

            elif provider.type == "anthropic":
                # Anthropic doesn't have a models endpoint, return known models
                anthropic_models = [
                    {"id": "claude-3-5-sonnet-20241022", "owned_by": "anthropic"},
                    {"id": "claude-3-5-haiku-20241022", "owned_by": "anthropic"},
                    {"id": "claude-3-opus-20240229", "owned_by": "anthropic"},
                    {"id": "claude-3-sonnet-20240229", "owned_by": "anthropic"},
                    {"id": "claude-3-haiku-20240307", "owned_by": "anthropic"},
                ]
                models = anthropic_models

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Provider API error: {e.response.text}"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to provider: {str(e)}"
        )

    return {
        "provider_id": provider_id,
        "provider_name": provider.name,
        "models": models,
        "total": len(models)
    }


class BulkModelCreate(BaseModel):
    models: List[str] = Field(..., min_length=1)
    alias_prefix: Optional[str] = None


@router.post("/providers/{provider_id}/import-models")
async def import_provider_models(
    provider_id: str,
    data: BulkModelCreate,
    db: AdminDatabase = Depends(get_db)
):
    """Import multiple models from provider as model aliases"""
    provider = db.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    created = []
    skipped = []

    for model_id in data.models:
        # Generate alias - use prefix if provided, otherwise use model id as-is
        alias = f"{data.alias_prefix}{model_id}" if data.alias_prefix else model_id

        # Check if alias already exists
        existing = db.get_model_alias(alias)
        if existing:
            skipped.append({"model": model_id, "alias": alias, "reason": "Alias already exists"})
            continue

        model = ModelAlias(
            id=secrets.token_urlsafe(8),
            alias=alias,
            provider_id=provider_id,
            target_model=model_id,
            enabled=True
        )

        try:
            db.create_model_alias(model)
            created.append({"model": model_id, "alias": alias})
        except Exception as e:
            skipped.append({"model": model_id, "alias": alias, "reason": str(e)})

    if created:
        reload_settings()  # Reload settings so new models are available

    return {
        "created": created,
        "skipped": skipped,
        "total_created": len(created),
        "total_skipped": len(skipped),
        "message": f"Imported {len(created)} models, skipped {len(skipped)}"
    }


# =============================================================================
# Model Alias Endpoints
# =============================================================================

@router.get("/models")
async def list_models(
    enabled_only: bool = Query(False),
    db: AdminDatabase = Depends(get_db)
):
    """List all model aliases"""
    models = db.list_model_aliases(enabled_only=enabled_only)

    # Build provider ID to name mapping for display
    providers = db.list_providers()
    provider_names = {p.id: p.name for p in providers}

    return {
        "models": [
            {
                "id": m.id,
                "alias": m.alias,
                "provider_id": m.provider_id,
                "provider_name": provider_names.get(m.provider_id, m.provider_id),
                "target_model": m.target_model,
                "enabled": m.enabled,
                "default_temperature": m.default_temperature,
                "default_max_tokens": m.default_max_tokens,
                "created_at": m.created_at,
                "updated_at": m.updated_at
            }
            for m in models
        ],
        "total": len(models)
    }


@router.post("/models")
async def create_model(
    data: ModelAliasCreate,
    db: AdminDatabase = Depends(get_db)
):
    """Create a new model alias"""
    # Verify provider exists
    if not db.get_provider(data.provider_id):
        raise HTTPException(status_code=400, detail="Provider not found")

    model = ModelAlias(
        id=secrets.token_urlsafe(8),
        alias=data.alias,
        provider_id=data.provider_id,
        target_model=data.target_model,
        enabled=True,
        default_temperature=data.default_temperature,
        default_max_tokens=data.default_max_tokens
    )

    try:
        created = db.create_model_alias(model)
        reload_settings()  # Reload settings so new model is available
        return {"model": created.__dict__, "message": "Model alias created successfully"}
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=400, detail="Model alias already exists")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Get model by ID"""
    model = db.get_model_alias_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get provider name for display
    provider = db.get_provider(model.provider_id)
    provider_name = provider.name if provider else model.provider_id

    return {
        "model": {
            **model.__dict__,
            "provider_name": provider_name
        }
    }


@router.patch("/models/{model_id}")
async def update_model(
    model_id: str,
    data: ModelAliasUpdate,
    db: AdminDatabase = Depends(get_db)
):
    """Update a model alias"""
    updates = data.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    # Validate provider exists if provider_id is being updated
    if "provider_id" in updates:
        if not db.get_provider(updates["provider_id"]):
            raise HTTPException(status_code=400, detail="Provider not found")

    updated = db.update_model_alias(model_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Model not found")

    reload_settings()  # Reload settings so model changes are available
    return {"model": updated.__dict__, "message": "Model updated successfully"}


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Delete a model alias"""
    if db.delete_model_alias(model_id):
        reload_settings()  # Reload settings so model removal is reflected
        return {"message": "Model deleted successfully"}
    raise HTTPException(status_code=404, detail="Model not found")


@router.post("/models/{model_id}/toggle")
async def toggle_model(
    model_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Toggle model alias enabled/disabled"""
    model = db.get_model_alias_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    updated = db.update_model_alias(model_id, {"enabled": not model.enabled})
    reload_settings()  # Reload settings so model state change is reflected
    return {
        "model": updated.__dict__,
        "message": f"Model {'enabled' if updated.enabled else 'disabled'}"
    }


# =============================================================================
# API Key Endpoints
# =============================================================================

@router.get("/api-keys")
async def list_api_keys(db: AdminDatabase = Depends(get_db)):
    """List all API keys"""
    keys = db.list_api_keys()
    return {
        "api_keys": [k.to_dict() for k in keys],
        "total": len(keys)
    }


@router.post("/api-keys")
async def create_api_key(
    data: ApiKeyCreate,
    db: AdminDatabase = Depends(get_db)
):
    """Create a new API key"""
    expires_at = None
    if data.expires_in_days:
        expires_at = (datetime.utcnow() + timedelta(days=data.expires_in_days)).isoformat()

    api_key, plain_key = db.create_api_key(
        name=data.name,
        rate_limit_rpm=data.rate_limit_rpm,
        rate_limit_tpm=data.rate_limit_tpm,
        allowed_models=data.allowed_models,
        expires_at=expires_at
    )

    return {
        "api_key": api_key.to_dict(),
        "key": plain_key,  # Only returned once!
        "message": "API key created. Save this key - it won't be shown again!"
    }


@router.get("/api-keys/{key_id}")
async def get_api_key(
    key_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Get API key by ID"""
    keys = db.list_api_keys()
    for key in keys:
        if key.id == key_id:
            return {"api_key": key.to_dict()}
    raise HTTPException(status_code=404, detail="API key not found")


@router.patch("/api-keys/{key_id}")
async def update_api_key(
    key_id: str,
    data: ApiKeyUpdate,
    db: AdminDatabase = Depends(get_db)
):
    """Update an API key"""
    updates = data.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updated = db.update_api_key(key_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="API key not found")

    return {"api_key": updated.to_dict(), "message": "API key updated successfully"}


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Delete an API key"""
    if db.delete_api_key(key_id):
        return {"message": "API key deleted successfully"}
    raise HTTPException(status_code=404, detail="API key not found")


@router.post("/api-keys/{key_id}/toggle")
async def toggle_api_key(
    key_id: str,
    db: AdminDatabase = Depends(get_db)
):
    """Toggle API key enabled/disabled"""
    keys = db.list_api_keys()
    for key in keys:
        if key.id == key_id:
            updated = db.update_api_key(key_id, {"enabled": not key.enabled})
            return {
                "api_key": updated.to_dict(),
                "message": f"API key {'enabled' if updated.enabled else 'disabled'}"
            }
    raise HTTPException(status_code=404, detail="API key not found")


# =============================================================================
# Statistics Endpoints
# =============================================================================

@router.get("/stats/overview")
async def get_stats_overview(stats: StatsCollector = Depends(get_stats)):
    """Get overall statistics overview"""
    return stats.get_overview()


@router.get("/stats/providers")
async def get_provider_stats(
    provider_id: Optional[str] = Query(None),
    stats: StatsCollector = Depends(get_stats)
):
    """Get provider statistics"""
    return {"providers": stats.get_provider_stats(provider_id)}


@router.get("/stats/models")
async def get_model_stats(
    model_alias: Optional[str] = Query(None),
    stats: StatsCollector = Depends(get_stats)
):
    """Get model statistics"""
    return {"models": stats.get_model_stats(model_alias)}


@router.get("/stats/live")
async def get_live_stats(stats: StatsCollector = Depends(get_stats)):
    """Get live metrics for real-time dashboard"""
    return stats.get_live_metrics()


@router.get("/stats/timeseries")
async def get_timeseries(
    minutes: int = Query(60, ge=1, le=1440),
    stats: StatsCollector = Depends(get_stats)
):
    """Get timeseries data for charts"""
    return {"timeseries": stats.get_timeseries(minutes)}


@router.get("/stats/usage")
async def get_usage_stats(
    hours: int = Query(24, ge=1, le=720),
    provider_id: Optional[str] = Query(None),
    model_alias: Optional[str] = Query(None),
    db: AdminDatabase = Depends(get_db)
):
    """Get historical usage statistics"""
    start_time = datetime.utcnow() - timedelta(hours=hours)
    return db.get_usage_stats(
        start_time=start_time,
        provider_id=provider_id,
        model_alias=model_alias
    )


@router.get("/stats/usage/by-provider")
async def get_usage_by_provider(
    hours: int = Query(24, ge=1, le=720),
    db: AdminDatabase = Depends(get_db)
):
    """Get usage stats grouped by provider"""
    start_time = datetime.utcnow() - timedelta(hours=hours)
    return {"providers": db.get_usage_by_provider(start_time=start_time)}


@router.get("/stats/usage/by-model")
async def get_usage_by_model(
    hours: int = Query(24, ge=1, le=720),
    db: AdminDatabase = Depends(get_db)
):
    """Get usage stats grouped by model"""
    start_time = datetime.utcnow() - timedelta(hours=hours)
    return {"models": db.get_usage_by_model(start_time=start_time)}


@router.post("/stats/reset")
async def reset_stats(stats: StatsCollector = Depends(get_stats)):
    """Reset all real-time statistics"""
    stats.reset()
    return {"message": "Statistics reset successfully"}


# =============================================================================
# Dashboard Endpoint
# =============================================================================

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Serve admin dashboard HTML"""
    from pathlib import Path

    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text())

    # Fallback: redirect to API docs
    return HTMLResponse(content="""
        <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/admin/api-docs" />
        </head>
        <body>
            <p>Redirecting to API documentation...</p>
        </body>
        </html>
    """)
