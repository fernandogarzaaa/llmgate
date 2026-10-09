"""Tenants, virtual API keys and usage records (SQLite or Postgres via SQLAlchemy Core)."""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

metadata = MetaData()

tenants = Table(
    "tenants", metadata,
    Column("id", String(64), primary_key=True),
    Column("name", String(128), nullable=False, unique=True),
    Column("rpm", Integer, nullable=False, default=0),
    Column("tpm", Integer, nullable=False, default=0),
    Column("budget_usd", Float, nullable=False, default=0.0),
    Column("budget_period", String(16), nullable=False, default="monthly"),
    Column("allowed_models", Text, nullable=True),  # JSON list or NULL = all
    Column("created_at", DateTime(timezone=True), nullable=False),
)

api_keys = Table(
    "api_keys", metadata,
    Column("id", String(64), primary_key=True),
    Column("tenant_id", String(64), ForeignKey("tenants.id"), nullable=False),
    Column("name", String(128), nullable=False, default=""),
    Column("prefix", String(16), nullable=False),
    Column("key_hash", String(64), nullable=False, unique=True),
    Column("revoked", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

usage = Table(
    "usage", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("request_id", String(64), nullable=False),
    Column("tenant_id", String(64), nullable=False),
    Column("key_id", String(64), nullable=True),
    Column("model", String(128), nullable=False),
    Column("provider", String(128), nullable=False),
    Column("route", String(32), nullable=False, default="direct"),
    Column("cache", String(16), nullable=False, default="miss"),
    Column("status", Integer, nullable=False),
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("saved_usd", Float, nullable=False, default=0.0),
    Column("latency_ms", Float, nullable=False, default=0.0),
)
Index("ix_usage_tenant_ts", usage.c.tenant_id, usage.c.ts)

KEY_PREFIX = "lg-"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass
class Tenant:
    id: str
    name: str
    rpm: int = 0
    tpm: int = 0
    budget_usd: float = 0.0
    budget_period: str = "monthly"
    allowed_models: Optional[List[str]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "rpm": self.rpm, "tpm": self.tpm, "budget_usd": self.budget_usd,
                "budget_period": self.budget_period, "allowed_models": self.allowed_models}


@dataclass
class KeyRecord:
    id: str
    tenant_id: str
    name: str
    prefix: str
    revoked: bool
    created_at: datetime


def _tenant(row: Any) -> Tenant:
    allowed = json.loads(row.allowed_models) if row.allowed_models else None
    return Tenant(row.id, row.name, row.rpm, row.tpm, row.budget_usd, row.budget_period, allowed)


def normalize_url(url: str) -> str:
    """Accept ``postgres://`` / ``postgresql://`` and use the psycopg 3 driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Store:
    def __init__(self, url: str) -> None:
        url = normalize_url(url)
        kw: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kw["poolclass"] = StaticPool  # one shared connection, or each thread sees an empty DB
        self.url = url
        self.engine: Engine = create_engine(url, **kw)
        metadata.create_all(self.engine)

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True

    # ------------------------------------------------------------------ tenants
    def create_tenant(self, name: str, rpm: int = 0, tpm: int = 0, budget_usd: float = 0.0,
                      budget_period: str = "monthly", allowed_models: Optional[Sequence[str]] = None) -> Tenant:
        if budget_period not in ("daily", "monthly", "total"):
            raise ValueError("budget_period must be daily, monthly or total")
        t = Tenant(uuid.uuid4().hex[:16], name, rpm, tpm, budget_usd, budget_period,
                   list(allowed_models) if allowed_models else None)
        with self.engine.begin() as c:
            c.execute(tenants.insert().values(
                id=t.id, name=name, rpm=rpm, tpm=tpm, budget_usd=budget_usd, budget_period=budget_period,
                allowed_models=json.dumps(t.allowed_models) if t.allowed_models else None, created_at=_now()))
        return t

    def get_tenant(self, id_or_name: str) -> Optional[Tenant]:
        with self.engine.connect() as c:
            row = c.execute(select(tenants).where(
                (tenants.c.id == id_or_name) | (tenants.c.name == id_or_name))).first()
        return _tenant(row) if row else None

    def list_tenants(self) -> List[Tenant]:
        with self.engine.connect() as c:
            return [_tenant(r) for r in c.execute(select(tenants).order_by(tenants.c.name))]

    def update_tenant(self, id_or_name: str, **fields: Any) -> Tenant:
        t = self.get_tenant(id_or_name)
        if t is None:
            raise KeyError(id_or_name)
        allowed = {"rpm", "tpm", "budget_usd", "budget_period", "allowed_models"}
        vals = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if "allowed_models" in vals:
            vals["allowed_models"] = json.dumps(list(vals["allowed_models"])) if vals["allowed_models"] else None
        if vals:
            with self.engine.begin() as c:
                c.execute(update(tenants).where(tenants.c.id == t.id).values(**vals))
        t2 = self.get_tenant(t.id)
        assert t2 is not None
        return t2

    # --------------------------------------------------------------------- keys
    def create_key(self, tenant: str, name: str = "", key: Optional[str] = None) -> "tuple[str, KeyRecord]":
        """Create a key for ``tenant`` (id or name). Returns ``(plaintext, record)``; only the hash is stored."""
        t = self.get_tenant(tenant)
        if t is None:
            raise KeyError(f"unknown tenant {tenant!r}")
        plain = key or new_key()
        rec = KeyRecord(uuid.uuid4().hex[:16], t.id, name, plain[:10], False, _now())
        with self.engine.begin() as c:
            c.execute(api_keys.insert().values(id=rec.id, tenant_id=t.id, name=name, prefix=rec.prefix,
                                               key_hash=hash_key(plain), revoked=False, created_at=rec.created_at))
        return plain, rec

    def resolve_key(self, plain: str) -> Optional["tuple[KeyRecord, Tenant]"]:
        with self.engine.connect() as c:
            row = c.execute(select(api_keys, tenants.c.name.label("tenant_name"))
                            .join(tenants, tenants.c.id == api_keys.c.tenant_id)
                            .where(api_keys.c.key_hash == hash_key(plain))).first()
            if row is None or row.revoked:
                return None
            trow = c.execute(select(tenants).where(tenants.c.id == row.tenant_id)).first()
        rec = KeyRecord(row.id, row.tenant_id, row.name, row.prefix, row.revoked, _aware(row.created_at))
        return rec, _tenant(trow)

    def key_exists(self, plain: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(select(api_keys.c.id).where(api_keys.c.key_hash == hash_key(plain))).first() is not None

    def list_keys(self, tenant: Optional[str] = None) -> List[KeyRecord]:
        q = select(api_keys).order_by(api_keys.c.created_at)
        if tenant:
            t = self.get_tenant(tenant)
            if t is None:
                return []
            q = q.where(api_keys.c.tenant_id == t.id)
        with self.engine.connect() as c:
            return [KeyRecord(r.id, r.tenant_id, r.name, r.prefix, r.revoked, _aware(r.created_at))
                    for r in c.execute(q)]

    def revoke_key(self, key_id: str) -> bool:
        with self.engine.begin() as c:
            res = c.execute(update(api_keys).where(api_keys.c.id == key_id).values(revoked=True))
        return bool(res.rowcount)

    # -------------------------------------------------------------------- usage
    def record_usage(self, rows: Sequence[Dict[str, Any]]) -> None:
        if rows:
            with self.engine.begin() as c:
                c.execute(usage.insert(), list(rows))

    def tenant_spend(self, tenant_id: str, since: datetime) -> float:
        with self.engine.connect() as c:
            v = c.execute(select(func.coalesce(func.sum(usage.c.cost_usd), 0.0))
                          .where(usage.c.tenant_id == tenant_id, usage.c.ts >= since)).scalar()
        return float(v or 0.0)

    def usage_summary(self, since: Optional[datetime] = None, by: Sequence[str] = ("tenant", "model"),
                      tenant: Optional[str] = None) -> List[Dict[str, Any]]:
        cols = {"tenant": tenants.c.name.label("tenant"), "model": usage.c.model, "provider": usage.c.provider,
                "cache": usage.c.cache, "route": usage.c.route}
        unknown = set(by) - set(cols)
        if unknown:
            raise ValueError(f"cannot group by {sorted(unknown)}; choose from {sorted(cols)}")
        group = [cols[b] for b in by]
        q = (select(*group, func.count().label("requests"),
                    func.sum(usage.c.prompt_tokens).label("prompt_tokens"),
                    func.sum(usage.c.completion_tokens).label("completion_tokens"),
                    func.sum(usage.c.cost_usd).label("cost_usd"),
                    func.sum(usage.c.saved_usd).label("saved_usd"))
             .select_from(usage.outerjoin(tenants, tenants.c.id == usage.c.tenant_id))
             .group_by(*group).order_by(*group))
        if since is not None:
            q = q.where(usage.c.ts >= since)
        if tenant:
            t = self.get_tenant(tenant)
            q = q.where(usage.c.tenant_id == (t.id if t else tenant))
        with self.engine.connect() as c:
            out = []
            for r in c.execute(q):
                d = dict(r._mapping)
                for k in ("prompt_tokens", "completion_tokens"):
                    d[k] = int(d[k] or 0)
                for k in ("cost_usd", "saved_usd"):
                    d[k] = round(float(d[k] or 0.0), 8)
                out.append(d)
            return out

    def close(self) -> None:
        self.engine.dispose()
