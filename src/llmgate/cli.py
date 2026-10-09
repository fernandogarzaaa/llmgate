"""Admin CLI: ``llmgate serve | tenants | keys | usage | mock-upstream | check-config``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import __version__


def _store(args: argparse.Namespace) -> Any:
    from .store import Store

    url = args.database_url or os.environ.get("LLMGATE_DATABASE_URL")
    if not url:
        from .config import load_config

        url = load_config(args.config).database_url
    return Store(url)


def _print_table(rows: List[Dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, default=str))
        return
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def parse_since(value: str) -> datetime:
    """``7d``, ``12h``, ``30m`` or an ISO date/datetime (UTC assumed)."""
    v = value.strip()
    units = {"d": "days", "h": "hours", "m": "minutes"}
    if v and v[-1] in units and v[:-1].replace(".", "", 1).isdigit():
        return datetime.now(timezone.utc) - timedelta(**{units[v[-1]]: float(v[:-1])})
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from .app import create_app
    from .config import load_config

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.server.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=args.host or cfg.server.host,
        port=args.port or cfg.server.port,
        log_level=cfg.server.log_level,
        access_log=args.access_log,
        workers=1,
    )
    return 0


def cmd_mock(args: argparse.Namespace) -> int:
    import uvicorn

    from .mock_upstream import create_mock_app

    app = create_mock_app(args.name, args.latency_ms, args.fail_rate)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .config import load_config

    cfg = load_config(args.config)
    print(f"ok: {len(cfg.providers)} providers, {len(cfg.models)} models, {len(cfg.routing.rules)} rules")
    return 0


def cmd_tenants(args: argparse.Namespace) -> int:
    st = _store(args)
    if args.action == "create":
        allowed = [m for m in (args.models or "").split(",") if m] or None
        t = st.create_tenant(args.name, args.rpm, args.tpm, args.budget, args.period, allowed)
        _print_table([t.as_dict()], args.json)
    elif args.action == "update":
        allowed = [m for m in args.models.split(",") if m] if args.models is not None else None
        t = st.update_tenant(
            args.name,
            rpm=args.rpm,
            tpm=args.tpm,
            budget_usd=args.budget,
            budget_period=args.period,
            allowed_models=allowed,
        )
        _print_table([t.as_dict()], args.json)
    else:
        _print_table([t.as_dict() for t in st.list_tenants()], args.json)
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    st = _store(args)
    if args.action == "create":
        if st.get_tenant(args.tenant) is None:
            if not args.create_tenant:
                print(f"error: unknown tenant {args.tenant!r} (use --create-tenant)", file=sys.stderr)
                return 2
            st.create_tenant(args.tenant)
        plain, rec = st.create_key(args.tenant, args.name or "")
        if args.json:
            print(json.dumps({"id": rec.id, "tenant_id": rec.tenant_id, "key": plain}))
        else:
            print(plain)
            print(f"# key id {rec.id} for tenant {args.tenant}; store it now, it is not shown again", file=sys.stderr)
    elif args.action == "revoke":
        if not st.revoke_key(args.key_id):
            print(f"error: no key {args.key_id!r}", file=sys.stderr)
            return 1
        print(f"revoked {args.key_id}")
    else:
        rows = [
            {
                "id": k.id,
                "tenant_id": k.tenant_id,
                "name": k.name,
                "prefix": k.prefix + "...",
                "revoked": k.revoked,
                "created_at": k.created_at.isoformat(timespec="seconds"),
            }
            for k in st.list_keys(args.tenant)
        ]
        _print_table(rows, args.json)
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    st = _store(args)
    rows = st.usage_summary(parse_since(args.since), [b for b in args.by.split(",") if b], args.tenant)
    _print_table(rows, args.json)
    if not args.json and rows:
        print(
            f"\ntotal cost ${sum(r['cost_usd'] for r in rows):.6f}, saved by cache "
            f"${sum(r['saved_usd'] for r in rows):.6f}, requests {sum(r['requests'] for r in rows)}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmgate", description="OpenAI-compatible LLM gateway")
    p.add_argument("--version", action="version", version=f"llmgate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", "-c", default=os.environ.get("LLMGATE_CONFIG"), help="config YAML")
        sp.add_argument("--database-url", default=None, help="overrides the config's database_url")
        sp.add_argument("--json", action="store_true", help="JSON output")

    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--config", "-c", default=os.environ.get("LLMGATE_CONFIG"))
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--access-log", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mock-upstream", help="run a mock OpenAI/Anthropic upstream")
    s.add_argument("--name", default="mock")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=9000)
    s.add_argument("--latency-ms", type=float, default=50.0)
    s.add_argument("--fail-rate", type=float, default=0.0)
    s.set_defaults(fn=cmd_mock)

    s = sub.add_parser("check-config", help="validate a config file")
    s.add_argument("--config", "-c", default=os.environ.get("LLMGATE_CONFIG"))
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("tenants", help="manage tenants")
    s.add_argument("action", choices=["list", "create", "update"])
    s.add_argument("name", nargs="?")
    s.add_argument("--rpm", type=int, default=None)
    s.add_argument("--tpm", type=int, default=None)
    s.add_argument("--budget", type=float, default=None, help="USD per period (0 = unlimited)")
    s.add_argument("--period", choices=["daily", "monthly", "total"], default=None)
    s.add_argument("--models", default=None, help="comma-separated allow-list (empty = all)")
    common(s)
    s.set_defaults(fn=cmd_tenants)

    s = sub.add_parser("keys", help="manage virtual API keys")
    s.add_argument("action", choices=["list", "create", "revoke"])
    s.add_argument("key_id", nargs="?", help="key id (revoke)")
    s.add_argument("--tenant", default=None)
    s.add_argument("--name", default=None)
    s.add_argument("--create-tenant", action="store_true", help="create the tenant if it does not exist")
    common(s)
    s.set_defaults(fn=cmd_keys)

    s = sub.add_parser("usage", help="cost and token usage")
    s.add_argument("--since", default="30d", help="7d, 12h, 30m or an ISO date")
    s.add_argument("--by", default="tenant,model", help="comma list of tenant,model,provider,cache,route")
    s.add_argument("--tenant", default=None)
    common(s)
    s.set_defaults(fn=cmd_usage)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "tenants":
        if args.action in ("create", "update") and not args.name:
            parser.error(f"tenants {args.action} needs a name")
        if args.action == "create":
            args.rpm, args.tpm = args.rpm or 0, args.tpm or 0
            args.budget, args.period = args.budget or 0.0, args.period or "monthly"
    if args.cmd == "keys":
        if args.action == "create" and not args.tenant:
            parser.error("keys create needs --tenant")
        if args.action == "revoke" and not args.key_id:
            parser.error("keys revoke needs a key id")
    from .config import ConfigError

    try:
        return int(args.fn(args))
    except (ConfigError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
