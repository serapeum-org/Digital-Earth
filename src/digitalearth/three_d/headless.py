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
from typing import Optional, Sequence

__all__ = ["start_xvfb"]

#: An X display name as ``DISPLAY`` spells it: an optional host, a colon, a display number and an optional screen.
_DISPLAY_NAME = re.compile(r"^[\w.\-]*:\d+(\.\d+)?$")


def start_xvfb(
    display: str = ":99",
    *,
    window_size: Sequence[int] = (1024, 768),
    wait: float = 0.5,
) -> Optional[subprocess.Popen]:
    """Start a virtual X server and point ``DISPLAY`` at it, so the 3-D tier can render with no screen.

    Call it once, before the first scene is drawn. It does nothing — and returns ``None`` — when there is
    nothing to do: ``DISPLAY`` is already set (a desktop session, or a process run under ``xvfb-run``), or the
    platform is Windows or macOS, where VTK renders without an X server. That makes it safe to call
    unconditionally at the top of a script meant to run everywhere.

    On Linux with no display it launches ``Xvfb <display> -screen 0 <width>x<height>x24 -nolisten tcp``,
    waits ``wait`` seconds to see that the server stays up, and only then sets ``DISPLAY``.

    **Stopping it.** Call ``.terminate()`` on the returned process when you are done, or leave it: the
    helper registers an :mod:`atexit` hook that terminates the server and unsets the ``DISPLAY`` it set,
    so a script that forgets leaves no stray server behind. The OS does **not** do this for you — a
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

    Returns:
        The ``subprocess.Popen`` of the server that was started — call ``.terminate()`` on it to stop it
        early, or leave it to the ``atexit`` hook — or ``None`` when no server was needed.

    Raises:
        ValueError: when ``display`` is not an X display name, or ``window_size`` is not two positive ints.
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

    See Also:
        digitalearth.three_d.base.Scene3DBase: ``off_screen=`` is the other half of rendering headless.
    """
    _check_request(display, window_size)
    if os.environ.get("DISPLAY") or not sys.platform.startswith("linux"):
        return None
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        raise FileNotFoundError(
            "start_xvfb() needs the Xvfb binary on PATH and found none; install it (apt install xvfb on "
            "Debian/Ubuntu, dnf install xorg-x11-server-Xvfb on Fedora/RHEL), run under `xvfb-run`, or use "
            "an EGL/OSMesa build of VTK, which needs no X server"
        )
    width, height = (int(side) for side in window_size)
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

    Registered with :mod:`atexit` at the end of a successful start. Idempotent, so a caller who already
    called ``server.terminate()`` themselves leaves this nothing to do.

    Args:
        server: The X server that was started.
        display: The display it was put on, which ``start_xvfb`` also wrote into ``DISPLAY``.
    """
    if server.poll() is None:
        server.terminate()
    if os.environ.get("DISPLAY") == display:
        # Only when it still names *this* server: something else may have pointed DISPLAY elsewhere
        # since, and that display is not ours to take away.
        del os.environ["DISPLAY"]


def _check_request(display: str, window_size: Sequence[int]) -> None:
    """Refuse a display name or screen size Xvfb would reject, before anything is launched.

    Args:
        display: The requested X display name.
        window_size: The requested ``(width, height)``.

    Raises:
        ValueError: naming the argument that is wrong and what was passed.
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
