"""Backwards-compatible re-export of the named-domain table, which now lives in ``base``.

The table moved to :mod:`digitalearth.base.domains` so every tier can resolve a name, not just this one: a
region name is data, and keeping it here meant ``base`` could not read it without importing a renderer.
Import from ``digitalearth.base.domains`` in new code; this module stays so existing imports keep working.
"""

from digitalearth.base.domains import DOMAINS, DomainLike, resolve_domain

__all__ = ["DOMAINS", "DomainLike", "resolve_domain"]
