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
    assert "maplibre" in html.lower(), (
        "the fallback must embed the actual MapLibre page"
    )


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


class _FakeWidget:
    """A stand-in widget whose ``_repr_mimebundle_`` returns a scripted bundle and records its kwargs."""

    def __init__(self, bundle):
        self._bundle = bundle
        self.calls = []

    def _repr_mimebundle_(self, include=None, exclude=None):
        self.calls.append({"include": include, "exclude": exclude})
        return self._bundle


class _HooklessWidget:
    """A stand-in widget with no ``_repr_mimebundle_`` at all (the ``hook is None`` branch)."""


def test_repr_preserves_the_tuple_form_and_metadata(monkeypatch):
    """A ``(data, metadata)`` bundle keeps its metadata and gains the HTML fallback, still as a tuple.

    Test scenario:
        A widget whose hook returns the tuple protocol form is merged into without losing the metadata half, and
        the result stays a 2-tuple so Jupyter still receives the display metadata.
    """
    widget = _FakeWidget(({"application/x-demo": 1}, {"meta": "kept"}))
    monkeypatch.setattr(WebMap, "render", lambda self: widget)
    bundle = WebMap()._repr_mimebundle_()
    assert isinstance(bundle, tuple), "the tuple protocol form must be preserved"
    data, metadata = bundle
    assert metadata == {"meta": "kept"}, f"metadata dropped: {metadata}"
    assert data["application/x-demo"] == 1 and "text/html" in data


def test_repr_does_not_overwrite_an_existing_html_representation(monkeypatch):
    """When the widget already offers ``text/html``, the fallback leaves it untouched.

    Test scenario:
        A hook that already supplies ``text/html`` must win; the export fallback only fills a gap, never clobbers
        a richer representation the engine produced.
    """
    widget = _FakeWidget(
        {"text/html": "ENGINE-HTML", "application/vnd.jupyter.widget-view+json": {}}
    )
    monkeypatch.setattr(WebMap, "render", lambda self: widget)
    called = []
    monkeypatch.setattr(
        WebMap, "_export_fallback_html", lambda self: called.append(1) or "FALLBACK"
    )
    data = _bundle_data(WebMap()._repr_mimebundle_())
    assert data["text/html"] == "ENGINE-HTML", (
        "an existing text/html must not be overwritten"
    )
    assert called == [], (
        "the fallback must not even be built when text/html already exists"
    )


def test_repr_forwards_include_and_exclude_to_the_widget(monkeypatch):
    """``include``/``exclude`` are passed through to the widget's hook as keywords.

    Test scenario:
        Jupyter's mime negotiation reaches the underlying widget unchanged, so the map honours a front-end that
        asks for or excludes specific mime types.
    """
    widget = _FakeWidget({"application/vnd.jupyter.widget-view+json": {}})
    monkeypatch.setattr(WebMap, "render", lambda self: widget)
    WebMap()._repr_mimebundle_(include=["a"], exclude=["b"])
    assert widget.calls == [{"include": ["a"], "exclude": ["b"]}], (
        f"kwargs not forwarded: {widget.calls}"
    )


def test_repr_handles_a_widget_without_the_hook(monkeypatch):
    """A widget with no ``_repr_mimebundle_`` still yields a bundle carrying the HTML fallback.

    Test scenario:
        The ``hook is None`` branch starts from an empty bundle and the export fallback alone makes the cell
        displayable, rather than returning nothing.
    """
    monkeypatch.setattr(WebMap, "render", lambda self: _HooklessWidget())
    monkeypatch.setattr(
        WebMap, "_export_fallback_html", lambda self: "<iframe srcdoc='x'></iframe>"
    )
    data = _bundle_data(WebMap()._repr_mimebundle_())
    assert data == {"text/html": "<iframe srcdoc='x'></iframe>"}, (
        f"unexpected bundle: {data}"
    )


def test_export_fallback_html_escapes_the_document():
    """The standalone page is HTML-escaped inside ``srcdoc`` so it is a valid, non-injecting attribute.

    Test scenario:
        A raw ``<!DOCTYPE html>`` is never emitted unescaped into the fragment; its ``<`` and ``"`` become
        ``&lt;``/``&quot;`` entities, which is what makes the iframe srcdoc self-contained.
    """
    fragment = WebMap(center=(0.0, 0.0), zoom=2)._export_fallback_html()
    assert fragment.startswith('<iframe srcdoc="') and fragment.rstrip().endswith(
        "></iframe>"
    )
    assert "&lt;!DOCTYPE html&gt;" in fragment, (
        "the document body must be entity-escaped inside srcdoc"
    )
    assert "<!DOCTYPE html>" not in fragment, (
        "an unescaped document would break out of the srcdoc attribute"
    )


def test_export_fallback_html_uses_the_configured_height():
    """The iframe is sized to the map's configured pixel height.

    Test scenario:
        A map built with ``height=321`` produces an iframe styled ``height:321px``, so the embedded map matches
        the widget's footprint.
    """
    fragment = WebMap(height=321)._export_fallback_html()
    assert "height:321px" in fragment, (
        f"configured height not applied: {fragment[:120]}"
    )


def test_export_fallback_html_defaults_height_when_unset():
    """A map with ``height=None`` falls back to a sensible default so the iframe is never zero-height.

    Test scenario:
        ``height=None`` (MapLibre's own default) yields ``height:500px`` in the iframe style rather than an
        empty or invalid length.
    """
    fragment = WebMap(height=None)._export_fallback_html()
    assert "height:500px" in fragment, f"default height not applied: {fragment[:120]}"


def test_export_fallback_html_propagates_missing_engine(monkeypatch):
    """Without the ``web`` extra, building the fallback raises ``ImportError`` (which the repr then swallows).

    Test scenario:
        ``to_html`` raising ``ImportError`` when maplibre is absent propagates out of ``_export_fallback_html`` —
        the contract the repr relies on to omit the fallback and keep only the widget-view.
    """

    def _no_engine(self, *args, **kwargs):
        raise ImportError("install digitalearth[web]")

    monkeypatch.setattr(WebMap, "to_html", _no_engine)
    with pytest.raises(ImportError, match=r"digitalearth\[web\]"):
        WebMap()._export_fallback_html()


def test_repr_respects_exclude_and_include_for_the_fallback(monkeypatch):
    """The injected fallback honours Jupyter's ``exclude``/``include`` mime negotiation, like the widget hook.

    Test scenario:
        ``exclude=["text/html"]`` and a non-empty ``include`` that omits ``text/html`` both suppress the
        fallback; ``include`` naming ``text/html`` (or no hints) keeps it. The real fallback is stubbed so the
        test isolates the negotiation, not the HTML build.
    """
    widget = _FakeWidget({"application/vnd.jupyter.widget-view+json": {}})
    monkeypatch.setattr(WebMap, "render", lambda self: widget)
    monkeypatch.setattr(WebMap, "_export_fallback_html", lambda self: "FALLBACK")

    excluded = _bundle_data(WebMap()._repr_mimebundle_(exclude=["text/html"]))
    assert "text/html" not in excluded, "exclude must suppress the injected fallback"

    narrowed = _bundle_data(
        WebMap()._repr_mimebundle_(include=["application/vnd.jupyter.widget-view+json"])
    )
    assert "text/html" not in narrowed, (
        "a non-empty include that omits text/html must suppress it"
    )

    asked = _bundle_data(WebMap()._repr_mimebundle_(include=["text/html"]))
    assert asked.get("text/html") == "FALLBACK", (
        "include naming text/html must keep the fallback"
    )

    default = _bundle_data(WebMap()._repr_mimebundle_())
    assert default.get("text/html") == "FALLBACK", "no hints must keep the fallback"


def test_repr_keeps_a_tuple_even_when_metadata_is_none(monkeypatch):
    """A widget that returns the tuple form keeps that shape, so the repr round-trips the protocol faithfully.

    Test scenario:
        A hook returning ``(data, None)`` yields a tuple back (not a bare dict), preserving exactly what the
        widget declared while still carrying the injected fallback.
    """
    widget = _FakeWidget(({"application/vnd.jupyter.widget-view+json": {}}, None))
    monkeypatch.setattr(WebMap, "render", lambda self: widget)
    monkeypatch.setattr(WebMap, "_export_fallback_html", lambda self: "FALLBACK")
    bundle = WebMap()._repr_mimebundle_()
    assert isinstance(bundle, tuple), (
        "a tuple-returning widget must round-trip as a tuple"
    )
    data, metadata = bundle
    assert metadata is None and data["text/html"] == "FALLBACK"


def test_repr_logs_a_swallowed_fallback_error(monkeypatch):
    """A failed fallback is logged at debug level, so a silently-blank export is at least discoverable.

    Test scenario:
        ``to_html`` raising leaves the widget-view intact and omits ``text/html`` (the repr never raises), and a
        debug log records the swallowed error rather than it vanishing with a bare ``pass``.
    """
    from loguru import logger

    records = []
    sink_id = logger.add(records.append, level="DEBUG")
    try:
        monkeypatch.setattr(
            WebMap,
            "to_html",
            lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom-xyz")),
        )
        data = _bundle_data(WebMap()._repr_mimebundle_())
    finally:
        logger.remove(sink_id)
    assert "text/html" not in data, "a failed fallback is still omitted"
    assert any("boom-xyz" in str(record) for record in records), (
        "the swallowed fallback error must be logged, not discarded silently"
    )


def test_temporal_map_repr_falls_back_to_a_single_frame():
    """A temporal map still gets a viewable export fallback — a single-frame snapshot, not the live slider.

    Test scenario:
        A ``timeslider`` map (whose live view is a slider composite) reprs with exactly one ``<iframe>`` export
        fallback, confirming the static export is a single map frame (an interactive slider cannot survive a
        kernel-less export) rather than nothing at all.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1)]) for i in range(6)]
    gdf = gpd.GeoDataFrame(
        {"time": [2000, 2000, 2010, 2010, 2020, 2020], "pop": range(6)},
        geometry=geoms,
        crs=4326,
    )
    m = WebMap().timeslider(gdf, kdim="time", column="pop")
    assert m._temporal_times() == [2000, 2010, 2020], (
        "fixture must be a genuinely temporal map"
    )
    data = _bundle_data(m._repr_mimebundle_())
    assert "text/html" in data, (
        "a temporal map must still leave a viewable export fallback"
    )
    assert data["text/html"].count("<iframe") == 1, (
        "the static fallback is a single map frame"
    )


def test_export_fallback_html_escapes_a_breakout_payload(monkeypatch):
    """A document containing a ``srcdoc`` breakout payload is entity-escaped, closing the injection vector.

    Test scenario:
        ``to_html`` returning a page with ``"></iframe><script>`` yields a fragment with a single ``srcdoc="..."``
        boundary and the payload entity-escaped, so the embedded document cannot break out of the attribute.
    """
    payload = '"></iframe><script>alert(1)</script>'
    monkeypatch.setattr(
        WebMap, "to_html", lambda self, *a, **k: f"<html>{payload}</html>"
    )
    fragment = WebMap()._export_fallback_html()
    assert fragment.count('srcdoc="') == 1, "exactly one srcdoc attribute boundary"
    assert "</iframe>" == fragment[-len("</iframe>") :], (
        "the only </iframe> is the real closer"
    )
    assert fragment.count("<script>") == 0, (
        "the breakout payload must be entity-escaped, not live"
    )
    assert "&lt;script&gt;" in fragment and "&quot;" in fragment
