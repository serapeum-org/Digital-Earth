"""``start_xvfb`` — a virtual X display for rendering the 3-D tier on a display-less Linux box (#364).

PyVista used to ship this helper and no longer does, so the tier carries its own. Nothing here starts a real X
server: the platform, the ``PATH`` lookup and the process launch are replaced with fakes, so the tests run the
same on every OS and record exactly what would have been launched.
"""

import subprocess

import pytest

pytest.importorskip("pyvista")

from digitalearth.three_d import headless, start_xvfb  # noqa: E402

XVFB = "/usr/bin/Xvfb"


class _FakeServer:
    """A stand-in for the ``Popen`` of an X server, which either keeps running or has already exited."""

    def __init__(self, args, exit_code=None, **_kwargs):
        """Record the command line it was launched with.

        Args:
            args: The argv the helper launched.
            exit_code: ``None`` for a server that keeps running, else the code it exited with.
            **_kwargs: The remaining ``Popen`` keywords, which a fake has no use for.
        """
        self.args = args
        self.exit_code = exit_code

    def wait(self, timeout=None):
        """Return the exit code of a server that died, or time out on one that is still running.

        Args:
            timeout: How long the caller is willing to wait.

        Returns:
            The exit code, when the server has exited.

        Raises:
            subprocess.TimeoutExpired: when the server is still running.
        """
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.exit_code


@pytest.fixture
def linux_without_display(monkeypatch):
    """Make the helper see a Linux host with ``Xvfb`` installed and no display, and record launches.

    Args:
        monkeypatch: pytest's patcher, which also restores ``DISPLAY`` afterwards.

    Returns:
        A list that gains every fake server launched.
    """
    launched = []
    monkeypatch.setattr(headless.sys, "platform", "linux")
    # Set before deleting: `delenv` records nothing for a variable that is absent, so a `DISPLAY` the helper
    # sets would outlive the test. `setenv` records the original state, absence included, and restores it.
    monkeypatch.setenv("DISPLAY", "")
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setattr(
        headless.shutil, "which", lambda name: XVFB if name == "Xvfb" else None
    )

    def popen(args, **kwargs):
        server = _FakeServer(args, **kwargs)
        launched.append(server)
        return server

    monkeypatch.setattr(headless.subprocess, "Popen", popen)
    return launched


class TestStartXvfb:
    """What the helper launches, and when it launches nothing."""

    def test_it_launches_xvfb_on_the_display_at_the_size_asked(
        self, linux_without_display
    ):
        """The command line is the one an operator would type, with a 24-bit screen of the requested size."""
        server = start_xvfb(display=":42", window_size=(800, 600), wait=0.01)
        assert server is linux_without_display[0]
        assert server.args == [
            XVFB,
            ":42",
            "-screen",
            "0",
            "800x600x24",
            "-nolisten",
            "tcp",
        ]

    def test_it_points_display_at_the_new_server(self, linux_without_display):
        """VTK finds the X server through ``DISPLAY``, so the helper sets it once the server is up."""
        import os

        start_xvfb(display=":42", wait=0.01)
        assert os.environ["DISPLAY"] == ":42"

    def test_an_existing_display_is_left_alone(
        self, linux_without_display, monkeypatch
    ):
        """Under ``xvfb-run`` or a desktop session a server is already there, so nothing is started."""
        monkeypatch.setenv("DISPLAY", ":0")
        assert start_xvfb(wait=0.01) is None
        assert linux_without_display == []

    @pytest.mark.parametrize("platform", ["win32", "darwin"])
    def test_windows_and_macos_need_no_x_server(
        self, linux_without_display, monkeypatch, platform
    ):
        """VTK renders natively there, so the call is a no-op and a portable script can make it anyway.

        Args:
            linux_without_display: The fixture, overridden here for the platform.
            monkeypatch: pytest's patcher.
            platform: The ``sys.platform`` value to pretend to run on.
        """
        monkeypatch.setattr(headless.sys, "platform", platform)
        assert start_xvfb(wait=0.01) is None
        assert linux_without_display == []

    def test_a_missing_xvfb_says_what_to_install(
        self, linux_without_display, monkeypatch
    ):
        """No binary on ``PATH`` is reported by name, with the package that provides it."""
        monkeypatch.setattr(headless.shutil, "which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="xvfb"):
            start_xvfb(wait=0.01)

    def test_a_server_that_exits_at_once_is_an_error(
        self, linux_without_display, monkeypatch
    ):
        """A server that dies during the wait (a taken display number, say) is not reported as started."""
        import os

        monkeypatch.setattr(
            headless.subprocess,
            "Popen",
            lambda args, **kwargs: _FakeServer(args, exit_code=1),
        )
        with pytest.raises(RuntimeError, match="exited with code 1"):
            start_xvfb(display=":42", wait=0.01)
        assert "DISPLAY" not in os.environ

    @pytest.mark.parametrize("window_size", [(0, 600), (800, -1), (800,)])
    def test_a_window_size_that_is_not_two_positive_ints_is_refused(
        self, linux_without_display, window_size
    ):
        """A size Xvfb would reject is refused before anything is launched.

        Args:
            linux_without_display: The fixture, recording launches.
            window_size: An invalid ``(width, height)``.
        """
        with pytest.raises(ValueError, match="window_size"):
            start_xvfb(window_size=window_size, wait=0.01)
        assert linux_without_display == []

    def test_a_display_that_is_not_a_display_name_is_refused(
        self, linux_without_display
    ):
        """``display`` is an X display name such as ``:99``; anything else is refused before a launch."""
        with pytest.raises(ValueError, match="display"):
            start_xvfb(display="99", wait=0.01)
