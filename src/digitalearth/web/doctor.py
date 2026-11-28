"""Diagnose the JupyterLab widget stack behind ``plot(backend="web")`` (#576).

``plot(backend="web")`` renders a live ``maplibre`` (anywidget) widget in a notebook. When the front-end widget
manager cannot bind the model it shows **"model not found"** and a blank cell — and the cause is almost always a
JupyterLab extension problem the kernel can *see on disk* even though it cannot watch the browser fail: a stale
extension marked incompatible with the running JupyterLab, one shadowing the env from a user-level install, or a
missing/disabled widget manager.

:func:`doctor` inspects that state and returns a :class:`WidgetStackReport` naming each problem with its remedy.
It is **best-effort and advisory**: it reads the extension registry and installed versions, never the browser,
so it detects the common on-disk *causes* but can never assert the widget *will* render — and it never changes
the environment. When it cannot introspect JupyterLab at all it says so rather than inventing problems. The
guaranteed-render path when a stack is broken is to write the map to a standalone page —
:meth:`~digitalearth.web.map.WebMap.save` — which renders in any browser with no widget manager. (A static
*inline* representation, ``web.inline="static"``, is tracked separately in #575, not yet shipped.)

The split is deliberate: :func:`_gather` does the impure introspection (best-effort, wrapped), and :func:`_analyze`
is a pure function of the gathered :class:`WidgetStackState`, so every verdict is unit-tested against a synthetic
state with no live JupyterLab (#576 DoD).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

from loguru import logger

__all__ = ["ExtensionInfo", "WidgetStackReport", "WidgetStackState", "doctor"]

#: The widget manager and the anywidget host that ``maplibre``'s widget needs on the front-end.
_WIDGET_MANAGER = "@jupyter-widgets/jupyterlab-manager"
_ANYWIDGET = "anywidget"

#: Minimum versions of the kernel-side widget packages for a JupyterLab-4 / Notebook-7 front-end. Below these,
#: the kernel side is a plausible cause rather than a known-good baseline.
_FLOORS: dict[str, str] = {
    "ipywidgets": "8",
    "jupyterlab_widgets": "3",
    "anywidget": "0.9",
}

#: The one-line escape hatch appended to every non-clean report: a remedy that works today, with no widget
#: manager. (It points at ``WebMap.save`` rather than ``web.inline="static"`` because the latter is #575's
#: not-yet-shipped inline option — advising a command that raises would be worse than no advice, H1.)
_STATIC_HINT = (
    "To see the map regardless of the widget stack, write it to a standalone page — "
    'WebMap.save("map.html") — and open that file in a browser; it renders with no widget manager. '
    '(A static inline representation, web.inline="static", is tracked in #575 but not yet shipped.)'
)

#: Appended to a problem report: the doctor reads the *kernel's* environment, but a JupyterLab server is often
#: launched from a different one, where "model not found" actually lives — so a user is not sent to fix the
#: wrong environment (review L1).
_KERNEL_ENV_CAVEAT = (
    "Note: this inspects the kernel's environment; if JupyterLab is served from a different environment, "
    "check the extensions there too."
)


@dataclass(frozen=True)
class ExtensionInfo:
    """One JupyterLab federated (prebuilt) extension, as the registry reports it.

    Attributes:
        name: The extension's package name, e.g. ``"@jupyter-widgets/jupyterlab-manager"``.
        version: Its installed version.
        ext_dir: The ``labextensions`` directory it loads from — the key to spotting a user-level shadow.
        incompatible: Whether the registry flags it incompatible with the running JupyterLab (the ``X`` in
            ``jupyter labextension list``).
        detail: The incompatibility detail when there is one, else ``""``.
    """

    name: str
    version: str
    ext_dir: str
    incompatible: bool
    detail: str


@dataclass(frozen=True)
class WidgetStackState:
    """The raw facts :func:`_gather` reads, and :func:`_analyze` reasons over.

    Attributes:
        extensions: Every federated extension found, in registry order.
        disabled: JupyterLab's merged ``disabledExtensions`` mapping — key (extension name or ``name:plugin``
            id) to a bool, where ``False`` is an explicit re-enable. Not a bare name set, so the boolean and
            plugin granularity are honoured (see :func:`_is_ext_disabled`).
        env_labextensions_dir: This interpreter's own ``share/jupyter/labextensions``, reported for context.
            It is informational only: the doctor does **not** infer a "shadow" from a path mismatch, because
            the search order (``prefer_environment_over_user``) and true-duplicate detection are not reliably
            available here, and a user-only extension is not a shadow (review M1).
        jupyterlab_version: The running JupyterLab version, or ``None`` when it could not be read.
        versions: Installed versions of the kernel-side widget packages (``None`` for a missing one).
        inspect_error: ``None`` on a successful gather, else why introspection failed — which makes the report
            say "could not inspect" instead of claiming a healthy or broken stack.
    """

    extensions: tuple[ExtensionInfo, ...]
    disabled: Mapping[str, bool]
    env_labextensions_dir: str
    jupyterlab_version: str | None
    versions: dict[str, str | None]
    inspect_error: str | None = None


@dataclass(frozen=True)
class WidgetStackReport:
    """The doctor's verdict: whether the stack looks able to render a web widget, and what to fix if not.

    Attributes:
        checked: Whether the stack could be introspected at all. ``False`` means "no opinion" — not "broken".
        problems: One actionable line per detected problem, each ending in its remedy. Empty when clean.

    Examples:
        - A clean, checked stack is ``ok`` and prints the single healthy line:
            ```python
            >>> from digitalearth.web.doctor import WidgetStackReport
            >>> report = WidgetStackReport(checked=True, problems=())
            >>> report.ok
            True
            >>> print(report)
            digitalearth web widget-stack doctor: the JupyterLab widget stack looks healthy.

            ```
        - A problem makes it not ``ok``; the summary lists it and points at the static-render remedy:
            ```python
            >>> from digitalearth.web.doctor import WidgetStackReport
            >>> report = WidgetStackReport(checked=True, problems=("anywidget is disabled",))
            >>> report.ok
            False
            >>> "anywidget is disabled" in str(report)
            True
            >>> "web.inline" in str(report)
            True

            ```
        - An unchecked report is treated as ``ok`` (no opinion), and says it could not inspect:
            ```python
            >>> from digitalearth.web.doctor import WidgetStackReport
            >>> report = WidgetStackReport(checked=False)
            >>> report.ok
            True
            >>> "could not inspect" in str(report)
            True

            ```
    """

    checked: bool
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """``True`` when no problem was detected (including when the stack could not be inspected)."""
        return not self.problems

    def __str__(self) -> str:
        """A human-readable report — the healthy line, the problem list, or the could-not-inspect note."""
        if not self.checked:
            return (
                "digitalearth web widget-stack doctor: could not inspect JupyterLab "
                "(is the 'web' extra installed?). " + _STATIC_HINT
            )
        if not self.problems:
            return "digitalearth web widget-stack doctor: the JupyterLab widget stack looks healthy."
        lines = [f"  - {p}" for p in self.problems]
        return (
            "digitalearth web widget-stack doctor found "
            f'{len(self.problems)} problem(s) that can stop plot(backend="web") rendering:\n'
            + "\n".join(lines)
            + "\n"
            + _KERNEL_ENV_CAVEAT
            + "\n"
            + _STATIC_HINT
        )


def _below_floor(version: str | None, floor: str) -> bool:
    """Whether ``version`` is older than ``floor`` (missing counts as below; unparseable never does)."""
    if version is None:
        return True
    try:
        from packaging.version import Version

        return Version(version) < Version(floor)
    except Exception:  # noqa: BLE001 - a non-PEP440 version is not a floor breach we can prove
        return False


def _as_disabled_map(disabled: object) -> dict[str, bool]:
    """Normalise JupyterLab's ``disabledExtensions`` to a ``{key: bool}`` map, whatever shape it arrives in.

    ``get_app_info`` returns a mapping ``{key: bool}``, but older/other shapes hand back a list of names. A
    list is read as "all True"; a mapping is kept as-is so an explicit ``False`` (a re-enable) is preserved.
    """
    if isinstance(disabled, Mapping):
        return {str(k): bool(v) for k, v in disabled.items()}
    if isinstance(disabled, (list, tuple, set, frozenset)):
        return {str(k): True for k in disabled}
    return {}


def _is_ext_disabled(name: str, disabled: Mapping[str, bool]) -> bool:
    """Whether ``name`` is disabled, the way JupyterLab's own ``_is_disabled`` reads it.

    A match is the bare extension name or any ``name:plugin`` id (partition on ``":"``), and only when the
    value is truthy — so a key present with ``False`` (an explicit re-enable) is **not** treated as disabled.
    """
    return any(key.partition(":")[0] == name and off for key, off in disabled.items())


def _analyze(state: WidgetStackState) -> WidgetStackReport:
    """Turn gathered facts into a verdict — a pure function, so it is tested without a live JupyterLab.

    Args:
        state: The facts :func:`_gather` read (or a synthetic one in a test).

    Returns:
        The report: unchecked when ``state`` carries an ``inspect_error``, else one problem line per detected
        fault (incompatible extension, missing/disabled widget manager, below-floor package).
    """
    if state.inspect_error is not None:
        return WidgetStackReport(checked=False)

    problems: list[str] = []
    names = {e.name for e in state.extensions}

    for ext in state.extensions:
        if ext.incompatible:
            detail = f" ({ext.detail})" if ext.detail else ""
            problems.append(
                f"{ext.name} {ext.version} is incompatible with JupyterLab "
                f"{state.jupyterlab_version or '?'}{detail}, at {ext.ext_dir} — update or remove it; a broken "
                "extension stops the widget manager activating, which is what 'model not found' is."
            )

    for needed, why in (
        (_WIDGET_MANAGER, "the ipywidgets widget manager"),
        (_ANYWIDGET, "anywidget, which maplibre's widget is built on"),
    ):
        if _is_ext_disabled(needed, state.disabled):
            problems.append(
                f"{needed} ({why}) is disabled — re-enable it so the web widget can bind."
            )
        elif needed not in names:
            problems.append(
                f"{needed} ({why}) is not installed as a JupyterLab extension — install it into the "
                "environment JupyterLab runs from."
            )

    for pkg, floor in _FLOORS.items():
        if _below_floor(state.versions.get(pkg), floor):
            have = state.versions.get(pkg) or "missing"
            problems.append(
                f"{pkg} is {have}; the web widget needs >= {floor} on a JupyterLab-4 / Notebook-7 front-end."
            )

    return WidgetStackReport(checked=True, problems=tuple(problems))


def _gather() -> WidgetStackState:
    """Read the JupyterLab extension registry and the installed widget-package versions — best-effort.

    Returns:
        A populated :class:`WidgetStackState`, or one whose ``inspect_error`` is set when JupyterLab is not
        importable or its registry cannot be read — the doctor never raises out of a diagnostic.
    """
    import importlib.metadata as md

    def _ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001 - a missing package is a fact, not an error to raise
            return None

    versions = {pkg: _ver(pkg) for pkg in _FLOORS}
    env_dir = os.path.join(sys.prefix, "share", "jupyter", "labextensions")

    try:
        from jupyter_core.paths import jupyter_path
        from jupyterlab.commands import AppOptions, _AppHandler, get_app_info

        options = AppOptions(labextensions_path=jupyter_path("labextensions"))
        info = get_app_info(options)
        # A second handler for the compat map is deliberate (review L3): `get_app_info` post-processes the
        # `disabled` map that a bare `_AppHandler(...).info` lacks, so we keep it for `info`, and the compat
        # read needs a handler of its own. `_get_extension_compat()` returns {name: [compat-error tuples]} —
        # a non-empty list means incompatible (the `X`). It is a private API; any shape change is absorbed by
        # the broad `except` below, which turns it into an "unchecked" report rather than a raise (review N1).
        compat = _AppHandler(options)._get_extension_compat()
        federated = info.get("federated_extensions", {})
        exts = tuple(
            ExtensionInfo(
                name=name,
                version=str(meta.get("version", "?")),
                ext_dir=str(meta.get("ext_dir", "")),
                incompatible=bool(compat.get(name)),
                detail="; ".join(str(d) for d in compat.get(name, [])),
            )
            for name, meta in sorted(federated.items())
        )
        return WidgetStackState(
            extensions=exts,
            disabled=_as_disabled_map(info.get("disabled")),
            env_labextensions_dir=env_dir,
            jupyterlab_version=_ver("jupyterlab"),
            versions=versions,
        )
    except Exception as error:  # noqa: BLE001 - any introspection failure becomes an "unchecked" report
        return WidgetStackState(
            extensions=(),
            disabled={},
            env_labextensions_dir=env_dir,
            jupyterlab_version=_ver("jupyterlab"),
            versions=versions,
            inspect_error=f"{type(error).__name__}: {error}",
        )


def doctor() -> WidgetStackReport:
    """Diagnose the JupyterLab widget stack behind ``plot(backend="web")`` and return the verdict.

    Reads the extension registry and installed versions (never the browser), so it flags the on-disk causes of
    a "model not found" failure — an incompatible or shadowing extension, a missing/disabled widget manager, a
    below-floor package — without claiming the widget will render. It changes nothing.

    Returns:
        A :class:`WidgetStackReport`; print it (``print(digitalearth.web.doctor())``) for the remedies. When the
        stack cannot be inspected the report says so rather than reporting a problem.

    Examples:
        - The report is always safe to build whatever the environment finds, and its ``ok`` flag reads back
          as a plain bool you can branch on:
            ```python
            >>> from digitalearth.web.doctor import doctor
            >>> report = doctor()
            >>> report.ok in (True, False)
            True

            ```
    """
    return _analyze(_gather())


def _warn_once() -> None:
    """Warn once per process if the widget stack looks broken — called from the web map's repr.

    Kept cheap and cached (the gather scans the extension registry, so it must never run per-plot), and fully
    guarded: a diagnostic may never raise out of a ``_repr_mimebundle_``.
    """
    if getattr(_warn_once, "_done", False):
        return
    _warn_once._done = True  # type: ignore[attr-defined]
    try:
        report = doctor()
        if report.checked and not report.ok:
            logger.warning(
                'plot(backend="web") may not render inline — {}', str(report)
            )
    except Exception:  # noqa: BLE001 - the repr must survive a failing doctor
        pass
