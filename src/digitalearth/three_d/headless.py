"""Headless rendering for the 3-D tier — a virtual X display for Linux machines that have no screen.

``Scene3D(off_screen=True)`` stops PyVista opening a window, but VTK still needs an OpenGL context to draw into.
On Windows and macOS the platform provides one; on Linux the stock VTK wheel gets it from an X server, so a
server, a container or a remote Jupyter kernel with no ``DISPLAY`` cannot render a scene at all.

There are two ways round that, and this module serves the first:

- **A virtual X server (Xvfb).** :func:`start_xvfb` launches one from Python and points ``DISPLAY`` at it, so a
  notebook or script can render without being wrapped in ``xvfb-run``. PyVista shipped a helper of the same name
  and has since dropped it (the installed 0.48 has none), so the tier carries its own. CI does not need it: the
  3-D jobs already run under ``xvfb-run -a``, which sets ``DISPLAY`` before Python starts, and the helper leaves
  an existing display alone.
- **A VTK build that needs no X server** — EGL (GPU, headless) or OSMesa (software). Those replace the ``vtk``
  package itself and are an installation choice, not something code can switch at run time; the docs page on
  headless rendering covers them.

Only the standard library is used here: starting an X server needs no renderer, and keeping PyVista out means
calling :func:`start_xvfb` before the first scene is drawn costs nothing.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from math import isfinite

__all__ = ["start_xvfb"]

#: An X display name as ``DISPLAY`` spells it: an optional host, a colon, a display number and an optional screen.
_DISPLAY_NAME = re.compile(r"^[\w.\-]*:\d+(\.\d+)?$")

#: Seconds :func:`_reap` waits for a terminated server to go, and again for a killed one.
#:
#: Bounded because the reaper runs from an :mod:`atexit` hook, where an open-ended wait is a hung
#: interpreter; generous because an X server that has not released its display is the failure the hook
#: exists to prevent. Xvfb exits on ``SIGTERM`` in milliseconds, so this is a ceiling, not a cost.
_REAP_TIMEOUT: float = 5.0


def start_xvfb(
    display: str = ":99",
    *,
    window_size: Sequence[int] = (1024, 768),
    wait: float = 0.5,
) -> subprocess.Popen | None:
    """Start a virtual X server and point ``DISPLAY`` at it, so the 3-D tier can render with no screen.

    Call it once, before the first scene is drawn. It does nothing — and returns ``None`` — when there is
    nothing to do: ``DISPLAY`` is already set (a desktop session, or a process run under ``xvfb-run``), or the
    platform is Windows or macOS, where VTK renders without an X server. That makes it safe to call
    unconditionally at the top of a script meant to run everywhere.

    On Linux with no display it launches ``Xvfb <display> -screen 0 <width>x<height>x24 -nolisten tcp``,
    waits ``wait`` seconds to see that the server stays up, and only then sets ``DISPLAY``.

    **Stopping it.** Call ``.terminate()`` on the returned process when you are done, or leave it: the
    helper registers an :mod:`atexit` hook that terminates the server, waits (briefly, and killing it if
    it will not go) for it to actually exit, and unsets the ``DISPLAY`` it set, so a script that forgets
    leaves no stray server behind. The OS does **not** do this for you — a
    ``subprocess.Popen`` child outlives the interpreter that started it, and ``Popen.__del__``
    deliberately does not terminate a running one (dropping the last reference to a live child only warns
    ``subprocess <pid> is still running``). Until the hook existed, a forgotten server went on holding
    ``:99``; the next run then found no ``DISPLAY``, asked for ``:99`` again, and failed with the
    ``RuntimeError`` below — so "safe to call unconditionally" was false from the second run onward
    (review M3).

    Args:
        display: The X display to create, such as ``":99"``. Pick another number if that one is taken.
        window_size: ``(width, height)`` of the virtual screen in pixels. It bounds the largest window or
            screenshot VTK can render, so make it at least the ``window_size`` of the scenes you draw.
        wait: Seconds to wait for the server to come up before declaring it started. A server that exits
            within that time (a display number already in use, for instance) is reported as an error.
            It must be a positive finite number, and anything else is refused by name: ``0`` and any
            negative make the check pass instantly — the same as not checking at all — ``nan`` disables it
            the same way by comparing false against every bound, and ``inf`` turns it into a hang.

    Returns:
        The ``subprocess.Popen`` of the server that was started — call ``.terminate()`` on it to stop it
        early, or leave it to the ``atexit`` hook — or ``None`` when no server was needed.

    Raises:
        ValueError: when ``display`` is not an X display name, ``window_size`` is not two positive ints,
            or ``wait`` is not a positive finite number of seconds.
        FileNotFoundError: when ``Xvfb`` is not on ``PATH`` (``apt install xvfb`` on Debian/Ubuntu,
            ``dnf install xorg-x11-server-Xvfb`` on Fedora/RHEL).
        RuntimeError: when the server exits during the startup wait; ``DISPLAY`` is left unset.

    Examples:
        - Start a display before drawing, on a headless Linux box (skipped here, since it launches a real
          X server):
            ```python
            >>> from digitalearth.three_d import Scene3D, start_xvfb  # doctest: +SKIP
            >>> server = start_xvfb()                                  # doctest: +SKIP
            >>> with Scene3D(off_screen=True) as scene:                # doctest: +SKIP
            ...     scene.screenshot("frame.png")
            >>> if server is not None:                                 # doctest: +SKIP
            ...     server.terminate()

            ```
        - A size Xvfb would reject is refused before anything is launched, on every platform:
            ```python
            >>> from digitalearth.three_d import start_xvfb
            >>> try:
            ...     start_xvfb(window_size=(0, 768))
            ... except ValueError as error:
            ...     print(error)
            start_xvfb() needs window_size as two positive ints (width, height); got (0, 768)

            ```
        - So is a startup wait that would not wait: the argument is checked before the platform is, so
          these answer the same way on Linux, Windows and macOS:
            ```python
            >>> from digitalearth.three_d import start_xvfb
            >>> for wait in (0, -1, float("inf"), float("nan")):
            ...     try:
            ...         start_xvfb(wait=wait)
            ...     except ValueError as error:
            ...         print(error)
            start_xvfb() needs wait as a positive finite number of seconds; got 0
            start_xvfb() needs wait as a positive finite number of seconds; got -1
            start_xvfb() needs wait as a positive finite number of seconds; got inf
            start_xvfb() needs wait as a positive finite number of seconds; got nan

            ```

    See Also:
        digitalearth.three_d.base.Scene3DBase: ``off_screen=`` is the other half of rendering headless.
    """
    _check_request(display, window_size, wait)
    if os.environ.get("DISPLAY") or not sys.platform.startswith("linux"):
        return None
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise FileNotFoundError(
            "start_xvfb() needs the Xvfb binary on PATH and found none; install it (apt install xvfb on "
            "Debian/Ubuntu, dnf install xorg-x11-server-Xvfb on Fedora/RHEL), run under `xvfb-run`, or use "
            "an EGL/OSMesa build of VTK, which needs no X server"
        )
    # Not `int(side)`: `_check_request` above has already refused anything that is not an `int`, so the
    # coercion could only ever have been applied to an `int` (review N3).
    width, height = window_size
    server = subprocess.Popen(
        [xvfb, display, "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        code = server.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        # Still running after the wait: the server is up, and VTK will find it through DISPLAY.
        os.environ["DISPLAY"] = display
        # And it is stopped when the interpreter exits, because nothing else does: a `Popen` child
        # outlives its parent, and `Popen.__del__` deliberately does not terminate a running one —
        # dropping the last reference to a live child only warns `subprocess <pid> is still running`.
        # `_reap` waits for the child it terminates, which is what keeps that warning from firing here.
        # Left behind, the server kept holding the display, and the next run found no DISPLAY, asked for
        # the same number and raised the startup RuntimeError below (review M3).
        atexit.register(_reap, server, display)
        return server
    raise RuntimeError(
        f"Xvfb on display {display} exited with code {code} while starting; the display may already be in "
        f"use — pass another one, e.g. start_xvfb(':100')"
    )


def _reap(server: subprocess.Popen, display: str) -> None:
    """Stop a server :func:`start_xvfb` started, and give ``DISPLAY`` back if it is still ours.

    Registered with :mod:`atexit` at the end of a successful start, and safe to run over an already-stopped
    server: a caller who called ``server.terminate()`` themselves leaves the terminate here nothing to do,
    and running it twice over raises nothing. The ``DISPLAY`` half still runs in that case, which is the
    point — a stopped server's display number should not go on being advertised.

    ``terminate()`` only *asks*, so the child is then waited for — ``SIGKILL`` after
    :data:`_REAP_TIMEOUT` seconds for one that will not go, and one more bounded wait to collect its
    status. Both halves matter: until the wait existed, ``Popen.returncode`` stayed ``None``, so dropping
    the hook's own reference at shutdown made ``Popen.__del__`` warn ``subprocess <pid> is still
    running`` — the very warning this hook is here to remove (review L1) — and the hook returned before
    Xvfb had released the display, so a wrapper that re-execed at once could still find it taken. The
    waits are bounded rather than open-ended because this runs during interpreter shutdown, where a
    blocked hook is a hung process.

    Args:
        server: The X server that was started.
        display: The display it was put on, which ``start_xvfb`` also wrote into ``DISPLAY``.

    Examples:
        - The reaper does not return while the child is still running, which is what sets ``returncode``
          and so keeps ``Popen.__del__`` from warning; running it again over the stopped child is a no-op
          (a display nothing here set, so ``DISPLAY`` is left alone):
            ```python
            >>> import subprocess, sys
            >>> from digitalearth.three_d.headless import _reap
            >>> child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            >>> _reap(child, ":4242")
            >>> stopped = child.returncode
            >>> stopped is None
            False
            >>> _reap(child, ":4242")
            >>> child.returncode == stopped
            True

            ```
    """
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=_REAP_TIMEOUT)
        except subprocess.TimeoutExpired:
            # It ignored the polite request, so stop asking: `kill` is a signal the process cannot
            # handle, and the second wait only collects the status `returncode` is read from.
            server.kill()
            try:
                server.wait(timeout=_REAP_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Unreapable within the bound. Nothing further can be done from an `atexit` hook, and
                # raising out of one only prints a traceback over the interpreter's shutdown.
                pass
    if os.environ.get("DISPLAY") == display:
        # Only when it still names *this* server: something else may have pointed DISPLAY elsewhere
        # since, and that display is not ours to take away.
        del os.environ["DISPLAY"]


def _check_request(display: str, window_size: Sequence[int], wait: float) -> None:
    """Refuse a display name, screen size or startup wait Xvfb would reject, before anything is launched.

    Args:
        display: The requested X display name.
        window_size: The requested ``(width, height)``.
        wait: The requested startup wait, in seconds.

    Raises:
        ValueError: naming the argument that is wrong and what was passed.

    Note:
        ``wait`` was the one argument not checked here, and it is the one that decides whether the startup
        check happens at all: ``Popen.wait(timeout=0)`` and any negative raise ``TimeoutExpired`` at once,
        which :func:`start_xvfb` reads as "the server is up", so a server that dies immediately was
        reported as started and ``DISPLAY`` was pointed at a dead display — the one failure the argument
        exists to catch. ``nan`` compares false against every bound and disables it the same way, ``inf``
        turns the wait into a hang, and a non-number raised from inside :mod:`subprocess` instead of by
        name (review L2).
    """
    if not isinstance(display, str) or not _DISPLAY_NAME.match(display):
        raise ValueError(
            f"start_xvfb() needs display as an X display name such as ':99'; got {display!r}"
        )
    sides = tuple(window_size)
    valid = len(sides) == 2 and all(
        isinstance(side, int) and not isinstance(side, bool) and side > 0
        for side in sides
    )
    if not valid:
        raise ValueError(
            f"start_xvfb() needs window_size as two positive ints (width, height); got {sides}"
        )
    waited = (
        isinstance(wait, (int, float))
        and not isinstance(wait, bool)
        and isfinite(wait)
        and wait > 0
    )
    if not waited:
        raise ValueError(
            f"start_xvfb() needs wait as a positive finite number of seconds; got {wait!r}"
        )
