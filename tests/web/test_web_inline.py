"""web.inline (#575) — render the static HTML map inline instead of the live widget.

The option and its resolution are engine-neutral and tested without maplibre; the repr-behaviour tests
``importorskip`` maplibre because they build a ``WebMap`` widget. ``web.inline="auto"`` is driven by the
doctor's verdict, which these tests supply synthetically through ``_report_once`` so no live JupyterLab is
needed.
"""

import sys

import pytest

import digitalearth
from digitalearth.web.doctor import inline_mode

_DOCTOR = sys.modules["digitalearth.web.doctor"]

_WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"


def _raises_export_error(_self):
    """Stand in for a failing ``_export_fallback_html`` (the static HTML export raised)."""
    raise RuntimeError("static export boom")


@pytest.fixture(autouse=True)
def _reset_state():
    """Restore the option default and clear the doctor's process caches around each test."""
    _DOCTOR._cached_report = None
    _DOCTOR._warned_once = False
    yield
    digitalearth.set_option("web.inline", "auto")
    _DOCTOR._cached_report = None
    _DOCTOR._warned_once = False


class TestWebInlineOption:
    """The ``web.inline`` package option — its values and default (#575 item 1)."""

    def test_default_is_auto(self):
        """Fresh from the registry, web.inline defaults to ``"auto"`` (the live widget stays default)."""
        from digitalearth.base.options import _reset_options

        _reset_options()
        assert digitalearth.get_option("web.inline") == "auto"

    @pytest.mark.parametrize("value", ["widget", "static", "auto"])
    def test_accepts_the_three_modes(self, value):
        """Each documented mode round-trips through set/get."""
        digitalearth.set_option("web.inline", value)
        assert digitalearth.get_option("web.inline") == value

    def test_rejects_an_unknown_value(self):
        """An unlisted value is refused with a message naming web.inline's OWN valid values (review M1)."""
        with pytest.raises(ValueError) as excinfo:
            digitalearth.set_option("web.inline", "nope")
        message = str(excinfo.value)
        assert "not a valid value for option 'web.inline'" in message
        assert "'web.inline' takes 'widget', 'static' or 'auto'" in message
        assert "three_d.off_screen" not in message, (
            "the remedy must describe the failing option, not a different one"
        )


class TestInlineModeResolution:
    """``inline_mode`` resolves the option to a concrete ``"static"``/``"widget"`` (#575)."""

    def test_explicit_static_passes_through(self):
        digitalearth.set_option("web.inline", "static")
        assert inline_mode() == "static"

    def test_explicit_widget_passes_through(self):
        digitalearth.set_option("web.inline", "widget")
        assert inline_mode() == "widget"

    def test_auto_is_static_when_the_stack_is_broken(self, monkeypatch):
        """auto picks static when the doctor reports a detectably-broken stack."""
        monkeypatch.setattr(
            _DOCTOR,
            "_report_once",
            lambda: _DOCTOR.WidgetStackReport(
                checked=True, problems=("x is incompatible",)
            ),
        )
        digitalearth.set_option("web.inline", "auto")
        assert inline_mode() == "static"

    def test_auto_is_widget_when_the_stack_is_healthy(self, monkeypatch):
        """auto keeps the live widget when the doctor reports a healthy stack."""
        monkeypatch.setattr(
            _DOCTOR,
            "_report_once",
            lambda: _DOCTOR.WidgetStackReport(checked=True, problems=()),
        )
        digitalearth.set_option("web.inline", "auto")
        assert inline_mode() == "widget"

    def test_auto_is_widget_when_the_stack_is_unchecked(self, monkeypatch):
        """An unchecked verdict (could not inspect) is not 'broken', so auto keeps the widget."""
        monkeypatch.setattr(
            _DOCTOR,
            "_report_once",
            lambda: _DOCTOR.WidgetStackReport(checked=False),
        )
        digitalearth.set_option("web.inline", "auto")
        assert inline_mode() == "widget"


class TestReprHonoursInlineMode:
    """The mimebundle emits the widget-view only when the resolved mode is ``"widget"`` (#575 item 2)."""

    def test_static_mode_drops_the_widget_view(self):
        """web.inline='static' serves the self-contained text/html map with no widget-view to fail to bind."""
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        digitalearth.set_option("web.inline", "static")
        bundle = WebMap().basemap("OSM")._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert "text/html" in data, sorted(data)
        assert "application/vnd.jupyter.widget-view+json" not in data, sorted(data)

    def test_widget_mode_keeps_the_widget_view(self):
        """web.inline='widget' keeps today's behaviour — the live widget-view plus the html fallback."""
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        digitalearth.set_option("web.inline", "widget")
        bundle = WebMap().basemap("OSM")._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert "application/vnd.jupyter.widget-view+json" in data, sorted(data)
        assert "text/html" in data, sorted(data)

    def test_a_failing_mode_resolution_degrades_to_the_widget(self, monkeypatch):
        """If inline_mode() raises, the repr falls back to the live widget rather than raising out of it.

        Test scenario:
            A diagnostic must never break the map's repr. Patch inline_mode to raise; the mimebundle must
            still come back (degraded to the widget-view path), not propagate the exception.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        def boom():
            raise RuntimeError("inline-mode blew up")

        monkeypatch.setattr(_DOCTOR, "inline_mode", boom)
        bundle = WebMap().basemap("OSM")._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert "application/vnd.jupyter.widget-view+json" in data, sorted(data)

    def test_a_failing_preflight_warning_does_not_break_the_repr(self, monkeypatch):
        """If the preflight _warn_once() raises, the repr still returns the widget rather than propagating.

        Test scenario:
            The repr-time doctor warning is a diagnostic; a failure in it must be logged and swallowed, never
            escape _repr_mimebundle_. Patch _warn_once to raise and assert the mimebundle still comes back.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        def boom():
            raise RuntimeError("warn-once blew up")

        monkeypatch.setattr(_DOCTOR, "_warn_once", boom)
        bundle = WebMap().basemap("OSM")._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert _WIDGET_VIEW in data, sorted(data)

    def test_static_export_failure_restores_the_widget_view(self, monkeypatch):
        """In static mode, a failing HTML export restores the live widget-view instead of an empty cell (M2).

        Test scenario:
            static mode drops the widget-view and serves text/html; if _export_fallback_html raises, the cell
            must not degrade to a text/plain-only stand-in (strictly worse than widget mode) — the widget-view
            is restored as a last resort.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        m = WebMap().basemap("OSM")
        monkeypatch.setattr(type(m), "_export_fallback_html", _raises_export_error)
        digitalearth.set_option("web.inline", "static")
        bundle = m._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert _WIDGET_VIEW in data, sorted(data)

    def test_static_with_failed_export_and_empty_widget_degrades_to_the_text_stand_in(
        self, monkeypatch
    ):
        """If the static export fails AND the widget yields nothing, the repr returns the text stand-in, no crash.

        Test scenario:
            The static last-resort restores the widget-view only when the widget actually produced one. With
            both the HTML export raising and an empty widget bundle, the repr returns just the text/plain
            stand-in rather than raising.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        m = WebMap().basemap("OSM")
        monkeypatch.setattr(type(m), "_export_fallback_html", _raises_export_error)
        monkeypatch.setattr(
            type(m), "_widget_mimebundle", lambda self, w, inc, exc: ({}, None, False)
        )
        digitalearth.set_option("web.inline", "static")
        bundle = m._repr_mimebundle_()
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert data == {"text/plain": "<WebMap: static inline map>"}, sorted(data)

    def test_static_mode_with_html_excluded_restores_the_widget_view(self):
        """When the front-end excludes text/html, static mode falls back to the widget-view, not an empty cell (M2).

        Test scenario:
            A front-end negotiating include={widget-view} makes _wants_mime drop text/html, so the static map
            cannot be served; the repr must then restore the widget-view rather than return only text/plain.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        digitalearth.set_option("web.inline", "static")
        bundle = WebMap().basemap("OSM")._repr_mimebundle_(include={_WIDGET_VIEW})
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert _WIDGET_VIEW in data, sorted(data)

    def test_static_mode_honours_a_text_plain_exclusion(self):
        """Static mode routes its text/plain stand-in through include/exclude like any other representation (L1).

        Test scenario:
            A front-end excluding text/plain still gets the static map (text/html) but not the stand-in — static
            mode is the only path that injects a mime itself, so it must respect the negotiation too.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        digitalearth.set_option("web.inline", "static")
        bundle = WebMap().basemap("OSM")._repr_mimebundle_(exclude={"text/plain"})
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert "text/plain" not in data, sorted(data)
        assert "text/html" in data, sorted(data)

    def test_static_auto_downgrade_does_not_warn_may_not_render_inline(
        self, monkeypatch
    ):
        """auto→static (broken stack) renders the map inline, so the 'may not render inline' warning is skipped (M1).

        Test scenario:
            On default 'auto' with a broken-stack verdict, inline_mode()='static' and the HTML map renders in the
            cell; emitting a preflight warning that the widget 'may not render inline' would contradict what just
            happened, so no such warning is raised.
        """
        pytest.importorskip("maplibre")
        import warnings

        from digitalearth.web import WebMap

        monkeypatch.setattr(
            _DOCTOR,
            "_report_once",
            lambda: _DOCTOR.WidgetStackReport(
                checked=True, problems=("x is incompatible",)
            ),
        )
        digitalearth.set_option("web.inline", "auto")
        map_ = WebMap().basemap("OSM")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            map_._repr_mimebundle_()
        offenders = [
            str(w.message) for w in caught if "may not render inline" in str(w.message)
        ]
        assert not offenders, offenders

    def test_forced_widget_on_a_broken_stack_still_warns(self, monkeypatch):
        """Forcing web.inline='widget' on a broken stack keeps the 'may not render inline' warning (M1 complement).

        Test scenario:
            When the user opts into the live widget on a broken stack, the widget really may fail to bind, so the
            preflight warning must still fire — only the static path suppresses it.
        """
        pytest.importorskip("maplibre")
        from digitalearth.web import WebMap

        monkeypatch.setattr(
            _DOCTOR,
            "_report_once",
            lambda: _DOCTOR.WidgetStackReport(
                checked=True, problems=("x is incompatible",)
            ),
        )
        digitalearth.set_option("web.inline", "widget")
        map_ = WebMap().basemap("OSM")
        with pytest.warns(UserWarning, match="may not render inline"):
            map_._repr_mimebundle_()
