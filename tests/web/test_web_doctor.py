"""The web-tier widget-stack doctor (#576) — detect a broken/shadowing JupyterLab extension stack.

The analysis is a pure function of a gathered :class:`WidgetStackState`, so every verdict is tested against a
synthetic state (a clean stack, an incompatible extension, a missing/disabled widget manager, a below-floor
version) with no live browser or JupyterLab build. One test also runs the real ``doctor()`` against this env to
prove the gather path does not raise.
"""

import sys

import pytest

from digitalearth.web.doctor import (
    ExtensionInfo,
    WidgetStackState,
    _analyze,
    doctor,
)

_ENV = "/env/share/jupyter/labextensions"


def _state(exts=(), disabled=None, versions=None, jlver="4.6.6", err=None):
    """Build a WidgetStackState for a verdict test, defaulting the parts a case does not exercise."""
    return WidgetStackState(
        extensions=tuple(exts),
        disabled=dict(disabled or {}),
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

    def test_a_user_only_compatible_extension_is_not_flagged(self):
        """A compatible extension in a user dir is NOT a problem — a path mismatch is not a shadow (review M1).

        Test scenario:
            The old heuristic flagged any non-env ``ext_dir`` as a shadow, which false-flags a legitimate
            user-only extension. A compatible extension, wherever it loads from, must leave the report clean.
        """
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
        assert report.ok is True, report.problems

    def test_a_missing_widget_manager_is_flagged(self):
        report = _analyze(
            _state(exts=(ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),))
        )
        assert report.ok is False
        assert any("jupyterlab-manager" in p for p in report.problems)

    def test_a_disabled_widget_manager_is_flagged(self):
        report = _analyze(
            _state(
                exts=_healthy_exts(),
                disabled={"@jupyter-widgets/jupyterlab-manager": True},
            )
        )
        assert report.ok is False
        assert any("disabled" in p.lower() for p in report.problems)

    def test_a_plugin_granular_disable_is_detected(self):
        """A disable at ``name:plugin`` granularity is matched, not only the bare name (review M2, FN)."""
        report = _analyze(
            _state(
                exts=_healthy_exts(),
                disabled={"@jupyter-widgets/jupyterlab-manager:plugin": True},
            )
        )
        assert report.ok is False
        assert any("disabled" in p.lower() for p in report.problems)

    def test_an_explicit_reenable_is_not_treated_as_disabled(self):
        """A key present with value ``False`` (a re-enable) must NOT be read as disabled (review M2, FP)."""
        report = _analyze(
            _state(
                exts=_healthy_exts(),
                disabled={"@jupyter-widgets/jupyterlab-manager": False},
            )
        )
        assert report.ok is True, report.problems

    def test_a_regex_pattern_disable_is_detected(self):
        """JupyterLab treats disabled keys as regexes; a pattern matching the manager is honoured (review L2)."""
        report = _analyze(
            _state(exts=_healthy_exts(), disabled={"@jupyter-widgets/.*": True})
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

    def test_the_remedy_points_at_a_mechanism_that_exists_today(self):
        """The advertised remedy must be a command that works now, not a set_option that raises (review H1).

        Test scenario:
            The report names the instance ``.save("map.html")`` call (which renders a standalone page with no
            widget manager today), and does NOT instruct a bare ``set_option(...)`` call — ``web.inline`` is
            #575's unshipped option, so advising it would raise ValueError.
        """
        report = _analyze(
            _state(exts=(ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),))
        )
        text = str(report)
        assert '.save("map.html")' in text
        assert "set_option(" not in text, (
            "must not advertise a set_option call that raises today"
        )

    def test_the_report_caveats_the_kernel_environment(self):
        """A problem report warns that it read the kernel env, which may differ from the server's (review L1)."""
        report = _analyze(
            _state(exts=(ExtensionInfo("anywidget", "0.11.0", _ENV, False, ""),))
        )
        assert "kernel's environment" in str(report)


class TestDoctorRunsLive:
    """The gather path must survive the real environment without raising, whatever it finds."""

    def test_doctor_does_not_raise_and_returns_a_report(self):
        pytest.importorskip("jupyterlab")
        report = doctor()
        assert hasattr(report, "ok")
        assert isinstance(report.problems, tuple)


class TestBelowFloor:
    """`_below_floor` — the version-floor comparison, including the inputs that must NOT count as a breach."""

    @pytest.mark.parametrize(
        "version, floor, expected",
        [
            (None, "8", True),
            ("7.7.0", "8", True),
            ("8", "8", False),
            ("8.1.9", "8", False),
            ("0.9.21", "0.9", False),
            ("not-a-pep440", "8", False),
        ],
    )
    def test_below_floor(self, version, floor, expected):
        """A missing or older version is below the floor; equal/newer/unparseable are not.

        Args:
            version: The installed version string (or None for missing).
            floor: The minimum acceptable version.
            expected: Whether `_below_floor` should report a breach.

        Test scenario:
            None and an older version breach; equal and newer do not; a non-PEP440 string cannot be proven a
            breach, so it returns False rather than raising.
        """
        from digitalearth.web.doctor import _below_floor

        assert _below_floor(version, floor) is expected, f"{version} vs {floor}"


class TestReportStr:
    """`WidgetStackReport.__str__` — the three shapes: healthy, unchecked, and a problem list."""

    def test_a_healthy_checked_report_says_healthy(self):
        """A checked report with no problems renders the single healthy line."""
        report = _analyze(_state(exts=_healthy_exts()))
        assert "healthy" in str(report).lower()


class TestGatherDegradesGracefully:
    """`_gather` must never raise; an introspection failure becomes an unchecked report."""

    def test_a_failing_registry_read_sets_inspect_error(self, monkeypatch):
        """When the JupyterLab registry read raises, `_gather` returns a state flagged inspect_error.

        Test scenario:
            Patch `jupyterlab.commands._AppHandler` (which `_gather` now constructs directly) to raise;
            `_gather` catches it and the resulting state carries an `inspect_error`, which `_analyze` turns
            into an unchecked (not broken) report.
        """
        pytest.importorskip("jupyterlab")
        import jupyterlab.commands as jc

        from digitalearth.web.doctor import _gather

        monkeypatch.setattr(
            jc,
            "_AppHandler",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        state = _gather()
        assert state.inspect_error is not None
        assert _analyze(state).checked is False

    def test_a_missing_package_version_is_none_not_an_error(self, monkeypatch):
        """`_gather` records None for a widget package whose version lookup raises, without failing."""
        pytest.importorskip("jupyterlab")
        import importlib.metadata as md

        d = sys.modules["digitalearth.web.doctor"]

        real = md.version

        def fake(pkg):
            if pkg == "ipywidgets":
                raise md.PackageNotFoundError(pkg)
            return real(pkg)

        monkeypatch.setattr(d, "md", md, raising=False)
        monkeypatch.setattr(md, "version", fake)
        state = d._gather()
        assert state.versions.get("ipywidgets") is None


class TestWarnOnce:
    """`_warn_once` — fires once when the stack is broken, caches, and never raises."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        """Clear the once-per-process latch before and after each test so cases don't bleed."""
        d = sys.modules["digitalearth.web.doctor"]

        if hasattr(d._warn_once, "_done"):
            delattr(d._warn_once, "_done")
        yield
        if hasattr(d._warn_once, "_done"):
            delattr(d._warn_once, "_done")

    def test_warns_when_broken_and_caches(self, monkeypatch):
        """A broken report logs one warning; a second call is a no-op (the gather must not run per-plot)."""
        d = sys.modules["digitalearth.web.doctor"]

        calls = {"doctor": 0, "warn": 0}
        broken = d.WidgetStackReport(checked=True, problems=("x is incompatible",))

        def fake_doctor():
            calls["doctor"] += 1
            return broken

        monkeypatch.setattr(d, "doctor", fake_doctor)
        monkeypatch.setattr(
            d.logger,
            "warning",
            lambda *a, **k: calls.__setitem__("warn", calls["warn"] + 1),
        )
        d._warn_once()
        d._warn_once()
        assert calls["doctor"] == 1, "gather must run at most once per process"
        assert calls["warn"] == 1, "exactly one warning for a broken stack"

    def test_does_not_warn_when_healthy(self, monkeypatch):
        """A healthy report emits no warning."""
        d = sys.modules["digitalearth.web.doctor"]

        warned = {"n": 0}
        monkeypatch.setattr(
            d, "doctor", lambda: d.WidgetStackReport(checked=True, problems=())
        )
        monkeypatch.setattr(
            d.logger,
            "warning",
            lambda *a, **k: warned.__setitem__("n", warned["n"] + 1),
        )
        d._warn_once()
        assert warned["n"] == 0, "a healthy stack must not warn"

    def test_swallows_a_raising_doctor(self, monkeypatch):
        """A diagnostic that raises must never propagate out of the repr hook."""
        d = sys.modules["digitalearth.web.doctor"]

        def boom():
            raise RuntimeError("doctor blew up")

        monkeypatch.setattr(d, "doctor", boom)
        d._warn_once()

    def test_warns_via_python_warnings_when_broken(self, monkeypatch):
        """A broken stack also raises a Python warning, so the notice survives a removed loguru sink (L3)."""
        d = sys.modules["digitalearth.web.doctor"]
        monkeypatch.setattr(
            d,
            "doctor",
            lambda: d.WidgetStackReport(checked=True, problems=("x is incompatible",)),
        )
        monkeypatch.setattr(d.logger, "warning", lambda *a, **k: None)
        with pytest.warns(UserWarning, match="may not render inline"):
            d._warn_once()


class TestDisabledMapHelpers:
    """`_as_disabled_map` / `_is_ext_disabled` — the disabled-extension normalisation (review M2)."""

    def test_as_disabled_map_normalises_each_shape(self):
        """A mapping is kept (bools preserved); a list becomes all-True; anything else is empty."""
        d = sys.modules["digitalearth.web.doctor"]
        assert d._as_disabled_map({"a": True, "b": False}) == {"a": True, "b": False}
        assert d._as_disabled_map(["a", "b"]) == {"a": True, "b": True}
        assert d._as_disabled_map(None) == {}

    def test_is_ext_disabled_matches_name_and_plugin_and_honours_bool(self):
        """A match is the bare name or a ``name:plugin`` id, and only when the value is truthy."""
        d = sys.modules["digitalearth.web.doctor"]
        assert d._is_ext_disabled("ext", {"ext": True}) is True
        assert d._is_ext_disabled("ext", {"ext:plugin": True}) is True
        assert d._is_ext_disabled("ext", {"ext": False}) is False
        assert d._is_ext_disabled("ext", {"other": True}) is False
        assert d._is_ext_disabled("@scope/ext", {"@scope/.*": True}) is True
        assert d._is_ext_disabled("ext", {"(": True}) is False
