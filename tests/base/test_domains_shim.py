"""The old import path for the named-domain table still works.

The table moved from `digitalearth.static.domains` to `digitalearth.base.domains` so that `base` could
resolve a name without importing a renderer. The old module stays as a re-export, because it is a public
import path a caller outside this repository may already use, and this is what says so.
"""

from digitalearth.base import domains as base_domains
from digitalearth.static import domains as static_domains


class TestTheOldImportPathStillResolves:
    """`digitalearth.static.domains` re-exports what moved to `base`."""

    def test_the_shim_hands_back_the_same_objects(self):
        """Not equal copies — the same table and the same function, so there is one source of truth."""
        assert static_domains.DOMAINS is base_domains.DOMAINS, (
            "two tables would drift apart"
        )
        assert static_domains.resolve_domain is base_domains.resolve_domain

    def test_a_name_resolves_through_either_path(self):
        """What a caller on the old path asked for still comes back."""
        through_the_shim = static_domains.resolve_domain("europe")
        assert through_the_shim == base_domains.resolve_domain("europe")
        assert through_the_shim == (-25.0, 34.0, 45.0, 72.0), through_the_shim

    def test_the_shim_names_what_it_exports(self):
        """So `from digitalearth.static.domains import *` keeps working too."""
        assert sorted(static_domains.__all__) == [
            "DOMAINS",
            "DomainLike",
            "resolve_domain",
        ]
