"""
Admin Data Models

SQLite-backed storage for providers, API keys, and usage statistics.
"""

from __future__ import annotations
import json
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import threading


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Provider:
    """Provider configuration"""
    id: str
    name: str
    type: str  # openai, anthropic
    base_url: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: int = 120
    enabled: bool = True
    rate_limit_rpm: Optional[int] = None  # Requests per minute
    rate_limit_tpm: Optional[int] = None  # Tokens per minute
    extra_headers: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Don't expose API key in responses
        if d.get("api_key"):
            d["api_key"] = "***"
        return d


@dataclass
class ModelAlias:
    """Model alias configuration"""
    id: str
    alias: str  # The name clients use
    provider_id: str
    target_model: str  # Actual model name
    enabled: bool = True
    default_temperature: Optional[float] = None
    default_max_tokens: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ApiKey:
    """API key for gateway access"""
    id: str
    name: str
    key_hash: str
    key_prefix: str  # First 8 chars for identification
    enabled: bool = True
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    allowed_models: List[str] = field(default_factory=list)  # Empty = all
    created_at: str = ""
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        del d["key_hash"]  # Never expose hash
        return d


@dataclass
class UsageRecord:
    """Single usage record"""
    id: int
    timestamp: str
    provider_id: str
    model_alias: str
    api_key_id: Optional[str]
    request_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str  # success, error
    error_type: Optional[str] = None


# =============================================================================
# Database Manager
# =============================================================================

class AdminDatabase:
    """SQLite database manager for admin data"""

    _instance: Optional["AdminDatabase"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: str = "data/admin.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @classmethod
    def get_instance(cls, db_path: str = "data/admin.db") -> "AdminDatabase":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
        return cls._instance

    @contextmanager
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_schema(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Providers table
                CREATE TABLE IF NOT EXISTS providers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    api_key TEXT,
                    api_key_env TEXT,
                    timeout INTEGER DEFAULT 120,
                    enabled INTEGER DEFAULT 1,
                    rate_limit_rpm INTEGER,
                    rate_limit_tpm INTEGER,
                    extra_headers TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- Model aliases table
                CREATE TABLE IF NOT EXISTS model_aliases (
                    id TEXT PRIMARY KEY,
                    alias TEXT UNIQUE NOT NULL,
                    provider_id TEXT NOT NULL,
                    target_model TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    default_temperature REAL,
                    default_max_tokens INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (provider_id) REFERENCES providers(id)
                );

                -- API keys table
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    key_prefix TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    rate_limit_rpm INTEGER,
                    rate_limit_tpm INTEGER,
                    allowed_models TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    expires_at TEXT
                );

                -- Usage records table (partitioned by date)
                CREATE TABLE IF NOT EXISTS usage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    model_alias TEXT NOT NULL,
                    api_key_id TEXT,
                    request_id TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_type TEXT
                );

                -- Indexes for usage queries
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_records(provider_id);
                CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_records(model_alias);
                CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage_records(api_key_id);

                -- Real-time stats table (rolling window)
                CREATE TABLE IF NOT EXISTS realtime_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    requests INTEGER DEFAULT 0,
                    tokens INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    latency_sum REAL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_realtime_timestamp ON realtime_stats(timestamp);
                CREATE INDEX IF NOT EXISTS idx_realtime_provider ON realtime_stats(provider_id);
            """)
            conn.commit()

    # =========================================================================
    # Provider CRUD
    # =========================================================================

    def create_provider(self, provider: Provider) -> Provider:
        """Create a new provider"""
        now = datetime.utcnow().isoformat()
        provider.created_at = now
        provider.updated_at = now

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO providers (
                    id, name, type, base_url, api_key, api_key_env,
                    timeout, enabled, rate_limit_rpm, rate_limit_tpm,
                    extra_headers, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                provider.id, provider.name, provider.type, provider.base_url,
                provider.api_key, provider.api_key_env, provider.timeout,
                int(provider.enabled), provider.rate_limit_rpm, provider.rate_limit_tpm,
                json.dumps(provider.extra_headers), provider.created_at, provider.updated_at
            ))
            conn.commit()
        return provider

    def get_provider(self, provider_id: str) -> Optional[Provider]:
        """Get provider by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM providers WHERE id = ?", (provider_id,)
            ).fetchone()

        if not row:
            return None

        return Provider(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            base_url=row["base_url"],
            api_key=row["api_key"],
            api_key_env=row["api_key_env"],
            timeout=row["timeout"],
            enabled=bool(row["enabled"]),
            rate_limit_rpm=row["rate_limit_rpm"],
            rate_limit_tpm=row["rate_limit_tpm"],
            extra_headers=json.loads(row["extra_headers"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def list_providers(self, enabled_only: bool = False) -> List[Provider]:
        """List all providers"""
        with self.get_connection() as conn:
            query = "SELECT * FROM providers"
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY name"

            rows = conn.execute(query).fetchall()

        return [
            Provider(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                base_url=row["base_url"],
                api_key=row["api_key"],
                api_key_env=row["api_key_env"],
                timeout=row["timeout"],
                enabled=bool(row["enabled"]),
                rate_limit_rpm=row["rate_limit_rpm"],
                rate_limit_tpm=row["rate_limit_tpm"],
                extra_headers=json.loads(row["extra_headers"] or "{}"),
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            for row in rows
        ]

    def update_provider(self, provider_id: str, updates: Dict[str, Any]) -> Optional[Provider]:
        """Update provider"""
        updates["updated_at"] = datetime.utcnow().isoformat()

        if "extra_headers" in updates:
            updates["extra_headers"] = json.dumps(updates["extra_headers"])
        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [provider_id]

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE providers SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

        return self.get_provider(provider_id)

    def delete_provider(self, provider_id: str) -> bool:
        """Delete provider"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM providers WHERE id = ?", (provider_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Model Alias CRUD
    # =========================================================================

    def create_model_alias(self, model: ModelAlias) -> ModelAlias:
        """Create a new model alias"""
        now = datetime.utcnow().isoformat()
        model.created_at = now
        model.updated_at = now

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO model_aliases (
                    id, alias, provider_id, target_model, enabled,
                    default_temperature, default_max_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.id, model.alias, model.provider_id, model.target_model,
                int(model.enabled), model.default_temperature, model.default_max_tokens,
                model.created_at, model.updated_at
            ))
            conn.commit()
        return model

    def get_model_alias(self, alias: str) -> Optional[ModelAlias]:
        """Get model alias by alias name"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_aliases WHERE alias = ?", (alias,)
            ).fetchone()

        if not row:
            return None

        return ModelAlias(
            id=row["id"],
            alias=row["alias"],
            provider_id=row["provider_id"],
            target_model=row["target_model"],
            enabled=bool(row["enabled"]),
            default_temperature=row["default_temperature"],
            default_max_tokens=row["default_max_tokens"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def get_model_alias_by_id(self, model_id: str) -> Optional[ModelAlias]:
        """Get model alias by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM model_aliases WHERE id = ?", (model_id,)
            ).fetchone()

        if not row:
            return None

        return ModelAlias(
            id=row["id"],
            alias=row["alias"],
            provider_id=row["provider_id"],
            target_model=row["target_model"],
            enabled=bool(row["enabled"]),
            default_temperature=row["default_temperature"],
            default_max_tokens=row["default_max_tokens"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def list_model_aliases(self, enabled_only: bool = False) -> List[ModelAlias]:
        """List all model aliases"""
        with self.get_connection() as conn:
            query = "SELECT * FROM model_aliases"
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY alias"

            rows = conn.execute(query).fetchall()

        return [
            ModelAlias(
                id=row["id"],
                alias=row["alias"],
                provider_id=row["provider_id"],
                target_model=row["target_model"],
                enabled=bool(row["enabled"]),
                default_temperature=row["default_temperature"],
                default_max_tokens=row["default_max_tokens"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            for row in rows
        ]

    def update_model_alias(self, model_id: str, updates: Dict[str, Any]) -> Optional[ModelAlias]:
        """Update model alias"""
        updates["updated_at"] = datetime.utcnow().isoformat()

        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [model_id]

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE model_aliases SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

            # Fetch by id - must be inside the context manager
            row = conn.execute(
                "SELECT * FROM model_aliases WHERE id = ?", (model_id,)
            ).fetchone()

        if row:
            return ModelAlias(
                id=row["id"],
                alias=row["alias"],
                provider_id=row["provider_id"],
                target_model=row["target_model"],
                enabled=bool(row["enabled"]),
                default_temperature=row["default_temperature"],
                default_max_tokens=row["default_max_tokens"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
        return None

    def delete_model_alias(self, model_id: str) -> bool:
        """Delete model alias"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM model_aliases WHERE id = ?", (model_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_model_aliases_by_provider(self, provider_id: str) -> int:
        """Delete all model aliases for a provider. Returns count of deleted models."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM model_aliases WHERE provider_id = ?", (provider_id,)
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # API Key CRUD
    # =========================================================================

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """Generate a new API key, returns (key, hash, prefix)"""
        key = f"sk-shin-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_prefix = key[:12]
        return key, key_hash, key_prefix

    @staticmethod
    def hash_api_key(key: str) -> str:
        """Hash an API key"""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_api_key(self, name: str, rate_limit_rpm: Optional[int] = None,
                       rate_limit_tpm: Optional[int] = None,
                       allowed_models: Optional[List[str]] = None,
                       expires_at: Optional[str] = None) -> tuple[ApiKey, str]:
        """Create a new API key, returns (ApiKey, plain_key)"""
        key, key_hash, key_prefix = self.generate_api_key()
        now = datetime.utcnow().isoformat()

        api_key = ApiKey(
            id=secrets.token_urlsafe(16),
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            enabled=True,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_tpm=rate_limit_tpm,
            allowed_models=allowed_models or [],
            created_at=now,
            expires_at=expires_at
        )

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO api_keys (
                    id, name, key_hash, key_prefix, enabled,
                    rate_limit_rpm, rate_limit_tpm, allowed_models,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key.id, api_key.name, api_key.key_hash, api_key.key_prefix,
                int(api_key.enabled), api_key.rate_limit_rpm, api_key.rate_limit_tpm,
                json.dumps(api_key.allowed_models), api_key.created_at, api_key.expires_at
            ))
            conn.commit()

        return api_key, key

    def validate_api_key(self, key: str) -> Optional[ApiKey]:
        """Validate an API key and return the ApiKey if valid"""
        key_hash = self.hash_api_key(key)

        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND enabled = 1",
                (key_hash,)
            ).fetchone()

        if not row:
            return None

        # Check expiration
        if row["expires_at"]:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.utcnow() > expires:
                return None

        # Update last used
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), row["id"])
            )
            conn.commit()

        return ApiKey(
            id=row["id"],
            name=row["name"],
            key_hash=row["key_hash"],
            key_prefix=row["key_prefix"],
            enabled=bool(row["enabled"]),
            rate_limit_rpm=row["rate_limit_rpm"],
            rate_limit_tpm=row["rate_limit_tpm"],
            allowed_models=json.loads(row["allowed_models"] or "[]"),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            expires_at=row["expires_at"]
        )

    def list_api_keys(self) -> List[ApiKey]:
        """List all API keys"""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys ORDER BY created_at DESC"
            ).fetchall()

        return [
            ApiKey(
                id=row["id"],
                name=row["name"],
                key_hash=row["key_hash"],
                key_prefix=row["key_prefix"],
                enabled=bool(row["enabled"]),
                rate_limit_rpm=row["rate_limit_rpm"],
                rate_limit_tpm=row["rate_limit_tpm"],
                allowed_models=json.loads(row["allowed_models"] or "[]"),
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                expires_at=row["expires_at"]
            )
            for row in rows
        ]

    def update_api_key(self, key_id: str, updates: Dict[str, Any]) -> Optional[ApiKey]:
        """Update API key"""
        if "enabled" in updates:
            updates["enabled"] = int(updates["enabled"])
        if "allowed_models" in updates:
            updates["allowed_models"] = json.dumps(updates["allowed_models"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [key_id]

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE api_keys SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()

            row = conn.execute(
                "SELECT * FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()

        if row:
            return ApiKey(
                id=row["id"],
                name=row["name"],
                key_hash=row["key_hash"],
                key_prefix=row["key_prefix"],
                enabled=bool(row["enabled"]),
                rate_limit_rpm=row["rate_limit_rpm"],
                rate_limit_tpm=row["rate_limit_tpm"],
                allowed_models=json.loads(row["allowed_models"] or "[]"),
                created_at=row["created_at"],
                last_used_at=row["last_used_at"],
                expires_at=row["expires_at"]
            )
        return None

    def delete_api_key(self, key_id: str) -> bool:
        """Delete API key"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM api_keys WHERE id = ?", (key_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Usage Records
    # =========================================================================

    def record_usage(
        self,
        provider_id: str,
        model_alias: str,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        status: str,
        api_key_id: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """Record a usage entry"""
        now = datetime.utcnow().isoformat()

        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO usage_records (
                    timestamp, provider_id, model_alias, api_key_id,
                    request_id, input_tokens, output_tokens, latency_ms,
                    status, error_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, provider_id, model_alias, api_key_id, request_id,
                input_tokens, output_tokens, latency_ms, status, error_type
            ))
            conn.commit()

    def get_usage_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        provider_id: Optional[str] = None,
        model_alias: Optional[str] = None,
        api_key_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get aggregated usage statistics"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        query = """
            SELECT
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                AVG(latency_ms) as avg_latency_ms,
                MIN(latency_ms) as min_latency_ms,
                MAX(latency_ms) as max_latency_ms,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_requests
            FROM usage_records
            WHERE timestamp >= ? AND timestamp <= ?
        """
        params = [start_time.isoformat(), end_time.isoformat()]

        if provider_id:
            query += " AND provider_id = ?"
            params.append(provider_id)
        if model_alias:
            query += " AND model_alias = ?"
            params.append(model_alias)
        if api_key_id:
            query += " AND api_key_id = ?"
            params.append(api_key_id)

        with self.get_connection() as conn:
            row = conn.execute(query, params).fetchone()

        return {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_requests": row["total_requests"] or 0,
            "total_input_tokens": row["total_input_tokens"] or 0,
            "total_output_tokens": row["total_output_tokens"] or 0,
            "total_tokens": (row["total_input_tokens"] or 0) + (row["total_output_tokens"] or 0),
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
            "min_latency_ms": round(row["min_latency_ms"] or 0, 2),
            "max_latency_ms": round(row["max_latency_ms"] or 0, 2),
            "successful_requests": row["successful_requests"] or 0,
            "failed_requests": row["failed_requests"] or 0,
            "success_rate": round(
                (row["successful_requests"] or 0) / max(row["total_requests"] or 1, 1) * 100, 2
            )
        }

    def get_usage_by_provider(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get usage stats grouped by provider"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        query = """
            SELECT
                provider_id,
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM usage_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY provider_id
            ORDER BY total_requests DESC
        """

        with self.get_connection() as conn:
            rows = conn.execute(
                query, [start_time.isoformat(), end_time.isoformat()]
            ).fetchall()

        return [
            {
                "provider_id": row["provider_id"],
                "total_requests": row["total_requests"],
                "total_input_tokens": row["total_input_tokens"] or 0,
                "total_output_tokens": row["total_output_tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
                "errors": row["errors"] or 0
            }
            for row in rows
        ]

    def get_usage_by_model(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get usage stats grouped by model"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        query = """
            SELECT
                model_alias,
                provider_id,
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                AVG(latency_ms) as avg_latency_ms
            FROM usage_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY model_alias, provider_id
            ORDER BY total_requests DESC
        """

        with self.get_connection() as conn:
            rows = conn.execute(
                query, [start_time.isoformat(), end_time.isoformat()]
            ).fetchall()

        return [
            {
                "model_alias": row["model_alias"],
                "provider_id": row["provider_id"],
                "total_requests": row["total_requests"],
                "total_input_tokens": row["total_input_tokens"] or 0,
                "total_output_tokens": row["total_output_tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2)
            }
            for row in rows
        ]

    def get_usage_timeseries(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "hour"  # minute, hour, day
    ) -> List[Dict[str, Any]]:
        """Get usage over time"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.utcnow()

        # SQLite date truncation
        if interval == "minute":
            group_expr = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif interval == "hour":
            group_expr = "strftime('%Y-%m-%d %H:00', timestamp)"
        else:  # day
            group_expr = "strftime('%Y-%m-%d', timestamp)"

        query = f"""
            SELECT
                {group_expr} as time_bucket,
                COUNT(*) as requests,
                SUM(input_tokens + output_tokens) as tokens,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM usage_records
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY time_bucket
            ORDER BY time_bucket
        """

        with self.get_connection() as conn:
            rows = conn.execute(
                query, [start_time.isoformat(), end_time.isoformat()]
            ).fetchall()

        return [
            {
                "time": row["time_bucket"],
                "requests": row["requests"],
                "tokens": row["tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
                "errors": row["errors"] or 0
            }
            for row in rows
        ]

    def cleanup_old_records(self, days: int = 30):
        """Clean up old usage records"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM usage_records WHERE timestamp < ?",
                (cutoff,)
            )
            conn.commit()


# =============================================================================
# Global Access
# =============================================================================

_db: Optional[AdminDatabase] = None


def get_admin_db() -> AdminDatabase:
    """Get the admin database instance"""
    global _db
    if _db is None:
        _db = AdminDatabase.get_instance()
    return _db


def set_admin_db(db: AdminDatabase):
    """Set the admin database instance"""
    global _db
    _db = db
