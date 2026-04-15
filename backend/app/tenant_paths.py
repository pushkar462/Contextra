"""Per-tenant filesystem layout under data/tenants/{id}/ with legacy fallback for default.

Data root comes from Settings.data_dir (env DATA_ROOT, or repo-root ../data by default).
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import get_settings

_TENANT_RE = re.compile(r"^[\w.-]{1,64}$")


def normalize_tenant_id(raw: str | None) -> str:
    tid = (raw or "").strip() or "default"
    if not _TENANT_RE.match(tid):
        raise ValueError("Tenant id must be 1–64 chars: letters, digits, _, -, .")
    if ".." in tid or "/" in tid or "\\" in tid:
        raise ValueError("Invalid tenant id")
    return tid


def tenant_data_dir(tenant_id: str) -> Path:
    """
    Root directory for a tenant's registry, uploads, FAISS, and job store.

    If tenant is `default` and a pre-multi-tenant `data/registry.json` exists while
    `data/tenants/default/registry.json` does not, use the legacy `data/` root so
    existing installs keep working.
    """
    settings = get_settings()
    tid = normalize_tenant_id(tenant_id)
    tenants_path = settings.data_dir / "tenants" / tid
    if tid == "default":
        legacy_reg = settings.data_dir / "registry.json"
        new_reg = tenants_path / "registry.json"
        if legacy_reg.exists() and not new_reg.exists():
            return settings.data_dir
    tenants_path.mkdir(parents=True, exist_ok=True)
    return tenants_path


def tenant_upload_dir(tenant_id: str) -> Path:
    d = tenant_data_dir(tenant_id) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d
