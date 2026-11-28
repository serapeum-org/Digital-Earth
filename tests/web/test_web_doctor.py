"""The web-tier widget-stack doctor (#576) — detect a broken/shadowing JupyterLab extension stack.

The analysis is a pure function of a gathered :class:`WidgetStackState`, so every verdict is tested against a
synthetic state (a clean stack, an incompatible extension, a user-level shadow, a missing widget manager, a
below-floor version) with no live browser or JupyterLab build. One test also runs the real ``doctor()`` against
this env to prove the gather path does not raise.
"""

import pytest

from digitalearth.web.doctor import (
    ExtensionInfo,
    WidgetStackState,
    _analyze,
    doctor,
)

_ENV = "/env/share/jupyter/labextensions"


def _state(exts=(), disabled=frozenset(), versions=None, jlver="4.6.6", err=None):
    """Build a WidgetStackState for a verdict test, defaulting the parts a case does not exercise."""
    return WidgetStackState(
        extensions=tuple(exts),
        disabled=frozenset(disabled),
        env_labextensions_dir=_ENV,
        jupyterlab_version=jlver,
        versions=versions
        or {
            "ipywidgets": "8.1.9",
            "jupyterlab_widgets": "3.0.17",
            "anywidget": "0.11.0",
        },
        inspect_error=err,
    )


def _healthy_exts():
    """The two extensions the web widget needs, both compatible and installed in the env dir."""
    return (
        ExtensionInfo("@jupyter-widgets/jupyterlab-manager", "5.0.16", _ENV, False, ""),
        ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),
    )


class TestAnalyzeFlagsTheRealFailures:
    """Each verdict the doctor must reach, from a crafted state — the #576 incident and its neighbours."""

    def test_a_clean_stack_is_ok(self):
        report = _analyze(_state(exts=_healthy_exts()))
        assert report.checked is True
        assert report.ok is True, report.problems
        assert report.problems == ()

    def test_an_incompatible_extension_is_flagged(self):
        """The exact #576 incident: a stale labextension marked incompatible with the running JupyterLab."""
        exts = _healthy_exts() + (
            ExtensionInfo(
                "@jupyter-notebook/lab-extension",
                "7.5.6",
                "/user/share/jupyter/labextensions",
                True,
                "requires @jupyterlab/application ^4.5 (have 4.6.6)",
            ),
        )
        report = _analyze(_state(exts=exts))
        assert report.ok is False
        joined = "\n".join(report.problems)
        assert "@jupyter-notebook/lab-extension" in joined
        assert "incompatible" in joined.lower()

    def test_a_user_level_shadow_is_flagged(self):
        """An extension loaded from a user dir (searched before the env) shadows the env's copy."""
        exts = _healthy_exts() + (
            ExtensionInfo(
                "jupyterlab_pygments",
                "0.3.0",
                "/user/share/jupyter/labextensions",
                False,
                "",
            ),
        )
        report = _analyze(_state(exts=exts))
        assert report.ok is False
        assert any(
            "shadow" in p.lower() and "jupyterlab_pygments" in p
            for p in report.problems
        )

    def test_a_missing_widget_manager_is_flagged(self):
        report = _analyze(
            _state(exts=(ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),))
        )
        assert report.ok is False
        assert any("jupyterlab-manager" in p for p in report.problems)

    def test_a_disabled_widget_manager_is_flagged(self):
        report = _analyze(
            _state(
                exts=_healthy_exts(), disabled={"@jupyter-widgets/jupyterlab-manager"}
            )
        )
        assert report.ok is False
        assert any("disabled" in p.lower() for p in report.problems)

    def test_a_below_floor_ipywidgets_is_flagged(self):
        report = _analyze(
            _state(
                exts=_healthy_exts(),
                versions={"ipywidgets": "7.7.0", "anywidget": "0.11.0"},
            )
        )
        assert report.ok is False
        assert any("ipywidgets" in p for p in report.problems)

    def test_an_inspect_error_reports_unchecked_not_broken(self):
        """When JupyterLab can't be introspected, the doctor says so — it does not invent problems."""
        report = _analyze(_state(err="jupyterlab not importable"))
        assert report.checked is False
        assert report.ok is True, (
            "an un-inspectable stack must not be reported as broken"
        )
        assert "could not inspect" in str(report).lower()

    def test_the_summary_points_at_the_static_fallback(self):
        report = _analyze(
            _state(exts=(ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),))
        )
        assert "web.inline" in str(report)


class TestDoctorRunsLive:
    """The gather path must survive the real environment without raising, whatever it finds."""

    def test_doctor_does_not_raise_and_returns_a_report(self):
        pytest.importorskip("jupyterlab")
        report = doctor()
        assert hasattr(report, "ok")
        assert isinstance(report.problems, tuple)
