"""WebMap leaves a persistent, viewable representation in a saved/exported notebook (#559).

``ds.plot(var, backend="web")`` returns a ``WebMap`` whose inline display was the live MapLibre widget only:
``_repr_mimebundle_`` delegated to the widget, so ``application/vnd.jupyter.widget-view+json`` was the only rich
output and a saved/exported notebook (``nbconvert --execute``, ``mkdocs-jupyter`` with ``execute: false``) showed
nothing once the widget state was gone. ``_repr_mimebundle_`` now also carries a self-contained ``text/html``
fallback (the standalone map embedded in an isolating ``<iframe srcdoc>``), so a static export still renders the
map while an interactive front-end keeps rendering the live widget. Gated on the ``web`` extra.
"""

# ruff: noqa: E402
import pytest

pytest.importorskip("maplibre")

from digitalearth.web import WebMap


def _bundle_data(bundle):
    """Return the mimebundle's data dict for both protocol forms.

    ``ipywidgets`` returns the ``(data, metadata)`` tuple form; an older hook returns a bare dict.
    """
    return bundle[0] if isinstance(bundle, tuple) else bundle


def test_repr_carries_a_static_html_fallback_for_export():
    """The repr adds a ``text/html`` fallback, so a saved/exported notebook still renders the map.

    Test scenario:
        A bare ``WebMap`` reprs to a bundle whose ``text/html`` is a self-contained MapLibre page wrapped in an
        ``<iframe srcdoc>`` — the persistent output ``nbconvert``/mkdocs keeps, with no caller-side ``to_html``.
    """
    data = _bundle_data(WebMap(center=(0.0, 0.0), zoom=2)._repr_mimebundle_())
    assert "text/html" in data, "expected a static HTML fallback for export"
    html = data["text/html"]
    assert "<iframe" in html and "srcdoc=" in html, (
        "the fallback must isolate the standalone page in an iframe so its scripts do not collide"
    )
    assert "maplibre" in html.lower(), "the fallback must embed the actual MapLibre page"


def test_repr_still_carries_the_live_widget_view():
    """The live widget-view survives, so an interactive session keeps the interactive map.

    Test scenario:
        The same repr still carries ``application/vnd.jupyter.widget-view+json`` alongside the HTML fallback, so a
        front-end with the widget manager prefers the live widget (ranked above ``text/html``).
    """
    data = _bundle_data(WebMap()._repr_mimebundle_())
    assert "application/vnd.jupyter.widget-view+json" in data
    assert "text/html" in data


def test_empty_map_repr_is_safe():
    """An undrawn map still reprs with a static fallback, so a freshly built map is always displayable.

    Test scenario:
        A ``WebMap`` with no layers produces a ``text/html`` fallback rather than raising or omitting it.
    """
    data = _bundle_data(WebMap()._repr_mimebundle_())
    assert "text/html" in data and "<iframe" in data["text/html"]


def test_repr_never_raises_when_the_fallback_fails(monkeypatch):
    """A failing HTML fallback must not break the repr; the live widget-view still carries the cell.

    Test scenario:
        ``to_html`` raising (e.g. an engine hiccup while serialising) leaves the widget-view intact and omits the
        fallback, instead of propagating out of the repr and blanking the cell.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("to_html exploded")

    monkeypatch.setattr(WebMap, "to_html", _boom)
    data = _bundle_data(WebMap()._repr_mimebundle_())
    assert "application/vnd.jupyter.widget-view+json" in data
    assert "text/html" not in data, "a failed fallback is omitted, not half-written"
