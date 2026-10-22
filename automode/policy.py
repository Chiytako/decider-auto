"""The auto-mode policy (rules) and per-user overrides, in the shape of Claude Code's `autoMode` settings.

Defaults: spec/automode_defaults.json (`claude auto-mode defaults`), with the environment replaced by an environment id from
spec/envs.json plus spec/envs.local.json (private, if present: your own environments). Overrides are lists for `environment`, `allow` and `soft_deny`; a list
replaces the defaults unless it contains the string "$defaults", which splices the defaults in at that position (the same
rule as Claude Code). `hard_deny` cannot be overridden.
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "spec")
KEYS = ("environment", "allow", "soft_deny")
_DEFAULTS = None


def _envs() -> dict:
    envs = json.load(open(os.path.join(SPEC, "envs.json")))
    local = os.path.join(SPEC, "envs.local.json")
    if os.path.exists(local):
        envs.update(json.load(open(local)))
    return envs


def defaults(env_id: str = "default") -> dict:
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = json.load(open(os.path.join(SPEC, "automode_defaults.json")))
    envs = _envs()
    return {"environment": list(envs.get(env_id) or _DEFAULTS["environment"]), "allow": list(_DEFAULTS["allow"]),
            "soft_deny": list(_DEFAULTS["soft_deny"]), "hard_deny": list(_DEFAULTS["hard_deny"])}


def splice(default: list, override) -> list:
    if override is None:
        return list(default)
    out = []
    for item in override:
        if item == "$defaults":
            out.extend(default)
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def effective(overrides: dict | None, env_id: str = "default") -> dict:
    d = defaults(env_id)
    o = overrides or {}
    eff = {k: splice(d[k], o.get(k)) for k in KEYS}
    eff["hard_deny"] = d["hard_deny"]
    return eff


def customized(overrides: dict | None, env_id: str = "default") -> dict:
    """Which parts differ from the defaults (the stage-1 model only learned the default allow / soft_deny rules)."""
    d, e = defaults(env_id), effective(overrides, env_id)
    return {k: e[k] != d[k] for k in KEYS}


def rule_name(text: str) -> str:
    return re.split(r"[:\[]", text, maxsplit=1)[0].strip()
