"""C9 — the default basemap provider is one shared constant, not a literal per tier.

Before this, each rendering backend hard-coded its own fallback provider: the interactive tier reached for
``"CartoLight"`` and the web tier for ``"CartoDark"``, so the same ``basemap=True`` produced a light map on one
backend and a dark one on the other. :data:`digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` is the single
answer both now read, and it lives in ``base/`` because a provider name is pure data — nothing about it needs a
renderer, which is exactly the test the engine-neutrality guard applies to this package.
"""

import ast
import pathlib

from digitalearth.base import basemaps
from digitalearth.base.basemaps import DEFAULT_BASEMAP_PROVIDER


class TestDefaultBasemapProvider:
    """Tests for the shared default-provider constant."""

    def test_it_is_cartolight(self):
        """The constant holds the agreed default provider name.

        Test scenario:
            The tiers only agree if the value is fixed, so the contract names it outright: ``"CartoLight"``,
            a token-free provider every backend can resolve, and light because data is drawn on top of it.
        """
        assert DEFAULT_BASEMAP_PROVIDER == "CartoLight", (
            f"the shared default must be 'CartoLight', got {DEFAULT_BASEMAP_PROVIDER!r}"
        )

    def test_it_is_exported_from_the_module(self):
        """It is reachable as a module attribute, which is how the backends import it.

        Test scenario:
            Two other tiers import this name directly. A constant that existed only inside a function body,
            or under a different spelling, would leave them hard-coding their own literal again.
        """
        assert getattr(basemaps, "DEFAULT_BASEMAP_PROVIDER", None) == "CartoLight", (
            "the constant must be importable from digitalearth.base.basemaps"
        )

    def test_it_is_documented(self):
        """The constant carries a ``#:`` doc comment, so the docs and the IDE explain the choice.

        Test scenario:
            A bare string assignment tells a reader the value but not why it is that value, or that the other
            tiers are meant to read it rather than pick their own. The doc comment is checked by parsing the
            source, since ``#:`` comments are not available at runtime.
        """
        source = pathlib.Path(basemaps.__file__).read_text(encoding="utf-8")
        lines = source.splitlines()
        assignment = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("DEFAULT_BASEMAP_PROVIDER")
        )
        preceding = [
            line
            for line in reversed(lines[:assignment])
            if line.startswith("#:") or not line.strip()
        ]
        assert any(line.startswith("#:") for line in preceding), (
            "DEFAULT_BASEMAP_PROVIDER must carry a #: doc comment explaining the choice"
        )

    def test_it_stays_pure_data(self):
        """The constant is a plain string literal — no renderer is consulted to produce it.

        Test scenario:
            ``base/`` must import no rendering engine, so the default provider has to be a *name* rather than
            a built provider object. Parsing the assignment (instead of reading the value) is what catches a
            later change to, say, an ``xyzservices`` lookup, which would be the start of that drift.
        """
        tree = ast.parse(pathlib.Path(basemaps.__file__).read_text(encoding="utf-8"))
        assigned = [
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "DEFAULT_BASEMAP_PROVIDER"
        ]
        assert len(assigned) == 1, "expected one annotated DEFAULT_BASEMAP_PROVIDER"
        assert isinstance(assigned[0].value, ast.Constant), (
            "the default provider must be a plain string literal, not a constructed object"
        )
