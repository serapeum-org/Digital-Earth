"""``start_xvfb`` — a virtual X display for rendering the 3-D tier on a display-less Linux box (#364).

PyVista used to ship this helper and no longer does, so the tier carries its own. Nothing here starts a real X
server: the platform, the ``PATH`` lookup and the process launch are replaced with fakes, so the tests run the
same on every OS and record exactly what would have been launched.
"""

import gc
import subprocess
import sys
import warnings

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
        self.terminated = False
        #: One entry per ``wait()``, holding the timeout it was given — the startup check's first, the
        #: reaper's next.
        self.waits = []

    def wait(self, timeout=None):
        """Return the exit code of a server that died, or time out on one that is still running.

        Args:
            timeout: How long the caller is willing to wait.

        Returns:
            The exit code, when the server has exited.

        Raises:
            subprocess.TimeoutExpired: when the server is still running.
        """
        self.waits.append(timeout)
        if self.exit_code is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.exit_code

    def poll(self):
        """Report whether the server has exited, the way ``Popen.poll`` does.

        Returns:
            ``None`` while it runs, else its exit code.
        """
        return self.exit_code

    def terminate(self):
        """Record that something asked the server to stop."""
        self.terminated = True
        self.exit_code = -15


class _StubbornServer(_FakeServer):
    """A server that ignores ``terminate()`` and dies only on ``kill()``, as a hung X server would.

    Every call is appended to :attr:`calls`, so the order the reaper makes them in can be pinned.
    """

    def __init__(self, args, **kwargs):
        """Start out running, with an empty call log.

        Args:
            args: The argv the helper launched.
            **kwargs: Forwarded to :class:`_FakeServer`.
        """
        super().__init__(args, **kwargs)
        self.calls = []
        self.killed = False

    def wait(self, timeout=None):
        """Log the wait, then behave as :class:`_FakeServer` does.

        Args:
            timeout: How long the caller is willing to wait.

        Returns:
            The exit code, when the server has exited.
        """
        self.calls.append(("wait", timeout))
        return super().wait(timeout)

    def terminate(self):
        """Log the request and keep running: ``SIGTERM`` is what this server ignores."""
        self.calls.append(("terminate", None))
        self.terminated = True

    def kill(self):
        """Log the kill and die, the way ``SIGKILL`` cannot be refused."""
        self.calls.append(("kill", None))
        self.killed = True
        self.exit_code = -9


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


@pytest.fixture
def registered_hooks(monkeypatch):
    """Capture what the helper hands ``atexit.register`` instead of really registering it.

    A hook left on the real ``atexit`` would run when pytest exits and terminate a fake, so the module's
    own ``atexit`` is replaced for the duration of the test.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        A list of ``(function, args)`` pairs, one per registration.
    """
    hooks = []

    def register(function, *args, **kwargs):
        hooks.append((function, args, kwargs))
        return function

    monkeypatch.setattr(headless.atexit, "register", register)
    return hooks


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

    @pytest.mark.parametrize(
        "wait",
        [0, -5, 0.0, -0.5, float("nan"), float("inf"), True, "x", None],
        ids=[
            "zero",
            "negative",
            "zero-float",
            "negative-float",
            "nan",
            "inf",
            "bool",
            "str",
            "none",
        ],
    )
    def test_a_wait_that_is_not_a_positive_finite_number_is_refused(
        self, linux_without_display, wait
    ):
        """``wait`` is the startup check's whole length, so a value that disables it is refused by name.

        Args:
            linux_without_display: The fixture, recording launches.
            wait: An invalid number of seconds to wait.

        Test scenario:
            - ``display`` and ``window_size`` are both refused by name and ``wait`` was not, so every one
              of these was accepted: measured against a fake that never exits, ``wait=-5``, ``wait=0``,
              ``wait='x'``, ``wait=None``, ``wait=nan`` and ``wait=inf`` all returned a started server
              with ``DISPLAY`` set.
            - It is not cosmetic for the non-positive ones. ``Popen.wait(timeout=0)`` and any negative
              raise ``TimeoutExpired`` at once, which the code reads as "the server is up": measured
              against a real child that exits with code 2, ``wait=0`` reported it as started and pointed
              ``DISPLAY`` at the dead display, while ``wait=0.5`` on the same launch raised the startup
              ``RuntimeError``. ``nan`` compares false against every bound, so it disables the check the
              same way.
            - Nothing is launched, since the check runs before the ``PATH`` lookup.
        """
        with pytest.raises(ValueError, match="wait"):
            start_xvfb(display=":42", wait=wait)
        assert linux_without_display == []

    def test_a_display_that_is_not_a_display_name_is_refused(
        self, linux_without_display
    ):
        """``display`` is an X display name such as ``:99``; anything else is refused before a launch."""
        with pytest.raises(ValueError, match="display"):
            start_xvfb(display="99", wait=0.01)


class TestStoppingTheServer:
    """A server this module started is stopped when the interpreter exits, not left holding its display."""

    def test_a_started_server_is_registered_for_cleanup(
        self, linux_without_display, registered_hooks
    ):
        """The helper hands the server it started to ``atexit``, with the display it was put on.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.

        Test scenario:
            - Nothing was registered before: a ``subprocess.Popen`` child is not killed when the parent
              interpreter exits, and ``Popen.__del__`` deliberately does not terminate a running child
              (CPython says so itself — dropping the last reference to a live child warns ``subprocess
              <pid> is still running``).
            - A forgotten ``server.terminate()`` therefore left ``Xvfb`` holding ``:99``, and the next run
              found no ``DISPLAY``, tried ``:99`` again and raised the startup ``RuntimeError``.
        """
        server = start_xvfb(display=":42", wait=0.01)
        assert [(hook.__name__, args) for hook, args, _ in registered_hooks] == [
            ("_reap", (server, ":42"))
        ], registered_hooks

    def test_the_cleanup_stops_the_server_and_puts_display_back(
        self, linux_without_display, registered_hooks
    ):
        """Running the registered hook terminates the server and leaves ``DISPLAY`` as it was found.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.
        """
        import os

        server = start_xvfb(display=":42", wait=0.01)
        assert os.environ["DISPLAY"] == ":42"
        hook, args, kwargs = registered_hooks[0]
        hook(*args, **kwargs)
        assert server.terminated is True
        assert "DISPLAY" not in os.environ

    def test_the_cleanup_leaves_a_display_someone_else_set(
        self, linux_without_display, registered_hooks, monkeypatch
    ):
        """``DISPLAY`` is only unset when it still names the server being stopped.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.
            monkeypatch: pytest's patcher, which restores ``DISPLAY`` afterwards.

        Test scenario:
            - Start a server on ``:42``, then have something else point ``DISPLAY`` at ``:7``.
            - Stopping the server must still terminate it, but must not take away a display it does not
              own.
        """
        import os

        server = start_xvfb(display=":42", wait=0.01)
        monkeypatch.setenv("DISPLAY", ":7")
        hook, args, kwargs = registered_hooks[0]
        hook(*args, **kwargs)
        assert server.terminated is True
        assert os.environ["DISPLAY"] == ":7"

    def test_the_cleanup_waits_for_the_child_it_terminated(
        self, linux_without_display, registered_hooks
    ):
        """``terminate()`` only asks; the hook waits for the child to actually go.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.

        Test scenario:
            - One ``wait`` has happened by the time the server is started: the startup check, with the
              ``wait=`` the caller gave.
            - Running the hook adds a second one, with a timeout of its own — bounded, so a server that
              will not die cannot hang the interpreter's exit.
            - Without it ``Popen.returncode`` stayed ``None`` and ``Popen.__del__`` warned
              ``subprocess <pid> is still running`` at shutdown, which is the symptom the hook exists to
              remove.
        """
        server = start_xvfb(display=":42", wait=0.01)
        assert server.waits == [0.01], server.waits
        hook, args, kwargs = registered_hooks[0]
        hook(*args, **kwargs)
        assert server.terminated is True
        assert len(server.waits) == 2, server.waits
        assert isinstance(server.waits[1], (int, float)) and server.waits[1] > 0, (
            f"the reaper must wait with a positive bound, got {server.waits[1]!r}"
        )

    def test_a_server_that_ignores_terminate_is_killed(
        self, linux_without_display, registered_hooks, monkeypatch
    ):
        """A child that survives ``SIGTERM`` is killed rather than waited for forever.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.
            monkeypatch: pytest's patcher.

        Test scenario:
            - The fake ignores ``terminate()``, so the reaper's first wait times out.
            - It then calls ``kill()``, which the fake cannot refuse, and waits once more to reap the
              exit status — four calls in that order after the startup check.
        """
        monkeypatch.setattr(
            headless.subprocess, "Popen", lambda args, **kwargs: _StubbornServer(args)
        )
        server = start_xvfb(display=":42", wait=0.01)
        hook, args, kwargs = registered_hooks[0]
        hook(*args, **kwargs)
        assert [name for name, _ in server.calls] == [
            "wait",
            "terminate",
            "wait",
            "kill",
            "wait",
        ], server.calls
        assert server.killed is True

    def test_a_reaped_child_warns_nothing_when_its_last_reference_goes(self):
        """The one measurable symptom: a real child, reaped, then dropped, and no ``ResourceWarning``.

        Test scenario:
            - A real ``subprocess.Popen`` of a sleeping interpreter, which the reaper is run over
              directly — no X server and no ``DISPLAY`` of its own (``":nobody"`` is not the one set).
            - ``returncode`` is set afterwards, which is what ``Popen.__del__`` reads: before the wait it
              stayed ``None`` and dropping the last reference warned ``subprocess <pid> is still
              running`` — measured, the exact warning the module's docstring cites.
        """
        server = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        headless._reap(server, ":nobody")
        assert server.returncode is not None, (
            "the reaper left the child unreaped, so Popen.__del__ will warn at shutdown"
        )
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            del server
            gc.collect()
        leaked = [
            str(entry.message)
            for entry in seen
            if "still running" in str(entry.message)
        ]
        assert leaked == [], leaked

    def test_a_server_already_stopped_is_not_terminated_twice(
        self, linux_without_display, registered_hooks
    ):
        """A caller who stopped the server themselves leaves the hook nothing to do.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.
        """
        server = start_xvfb(display=":42", wait=0.01)
        server.terminated = False
        server.exit_code = 0
        hook, args, kwargs = registered_hooks[0]
        hook(*args, **kwargs)
        assert server.terminated is False

    def test_a_server_that_never_started_registers_nothing(
        self, linux_without_display, registered_hooks, monkeypatch
    ):
        """There is nothing to clean up when no server was launched.

        Args:
            linux_without_display: The fixture, recording launches.
            registered_hooks: What was handed to ``atexit.register``.
            monkeypatch: pytest's patcher.
        """
        monkeypatch.setattr(
            headless.subprocess,
            "Popen",
            lambda args, **kwargs: _FakeServer(args, exit_code=1),
        )
        with pytest.raises(RuntimeError, match="exited with code 1"):
            start_xvfb(display=":42", wait=0.01)
        assert registered_hooks == []
