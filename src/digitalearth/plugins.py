"""plugins — discover third-party extensions registered via Python entry points (RP.11).

Digital-Earth exposes a small set of extension points so *other* packages can plug in without forking it:
an extra autostyle library, a new source adapter, and so on. A plugin package advertises itself with an
entry point in one of the ``digitalearth.*`` groups (declared in its own ``pyproject.toml``), e.g.::

    [project.entry-points."digitalearth.styles"]
    my_styles = "my_pkg.styles:LIBRARY"

At runtime :func:`load_plugins` discovers and loads those entry points. This is pure
``importlib.metadata`` plumbing — no third-party dependency — and is the mechanism behind earthkit-plots'
``_plugins.py``, scoped here to the extension points Digital-Earth actually offers.
"""
import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Dict, Iterator, Optional, Sequence

__all__ = ["GROUPS", "iter_plugins", "load_plugins"]

logger = logging.getLogger(__name__)

#: The entry-point groups Digital-Earth looks up. ``styles`` extend the autostyle library; ``sources``
#: register new input adapters. A plugin package targets one of these group names.
GROUPS = ("digitalearth.styles", "digitalearth.sources")


def iter_plugins(group: str, *, eps: Optional[Sequence[EntryPoint]] = None) -> Iterator[EntryPoint]:
    """Yield the entry points registered under ``group`` (without loading them).

    Args:
        group: The entry-point group to query (e.g. ``"digitalearth.styles"``).
        eps: Override the discovered entry points (mainly for testing); when ``None`` the installed
            environment is queried via ``importlib.metadata.entry_points``.

    Yields:
        Each :class:`importlib.metadata.EntryPoint` advertised under ``group``.

    Examples:
        - An unknown / unused group yields nothing:
            ```python
            >>> from digitalearth.plugins import iter_plugins
            >>> list(iter_plugins("digitalearth.nonexistent"))
            []

            ```
        - Inspect a supplied entry point without importing its target:
            ```python
            >>> from importlib.metadata import EntryPoint
            >>> from digitalearth.plugins import iter_plugins
            >>> ep = EntryPoint("demo", "my_pkg.styles:LIBRARY", "digitalearth.styles")
            >>> [e.name for e in iter_plugins("digitalearth.styles", eps=[ep])]
            ['demo']

            ```
    """
    selected = entry_points(group=group) if eps is None else eps
    for ep in selected:
        yield ep


def load_plugins(group: str, *, eps: Optional[Sequence[EntryPoint]] = None) -> Dict[str, Any]:
    """Discover and **load** every plugin registered under ``group``.

    Each entry point is imported via ``EntryPoint.load()`` and collected by its name. Loading is what
    actually runs the plugin's target (e.g. resolves ``my_pkg.styles:LIBRARY`` to the object it names). A
    plugin whose ``load()`` raises is skipped (logged at ``WARNING``) so one broken third-party plugin cannot
    abort discovery of the healthy ones.

    Args:
        group: The entry-point group to load (e.g. ``"digitalearth.styles"``).
        eps: Override the discovered entry points (mainly for testing); when ``None`` the installed
            environment is queried.

    Returns:
        Mapping of entry-point name to the loaded object. Empty when nothing is registered.

    Examples:
        - With nothing installed under the group, the result is empty:
            ```python
            >>> from digitalearth.plugins import load_plugins
            >>> load_plugins("digitalearth.nonexistent")
            {}

            ```
        - A fake entry point whose target is a dict is loaded by name (no real install needed):
            ```python
            >>> class FakeEP:
            ...     name = "extra"
            ...     def load(self):
            ...         return {"cmap": "magma"}
            >>> from digitalearth.plugins import load_plugins
            >>> loaded = load_plugins("digitalearth.styles", eps=[FakeEP()])
            >>> loaded["extra"]["cmap"]
            'magma'

            ```
    """
    loaded: Dict[str, Any] = {}
    for ep in iter_plugins(group, eps=eps):
        try:
            loaded[ep.name] = ep.load()
        except Exception as exc:  # one broken plugin must not abort discovery of the rest
            logger.warning("skipping plugin %r in group %r: %s", ep.name, group, exc)
    return loaded
