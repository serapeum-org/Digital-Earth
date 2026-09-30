"""DE-24 — the backend seam behind ``quickmap``/``quickplot`` (Wave 8, order 29).

These pin the structural collapse of ``api.py`` behind its stable façade:

* importing ``digitalearth.api`` — and a ``web``/``interactive``/``3d`` ``quickmap`` — no longer drags in the
  matplotlib tier (``digitalearth.static`` imports ``Map`` eagerly, so loading it is expensive);
* the input-type→kind decision is made in **one** place (:func:`~digitalearth.api._input_kind`), which every
  backend routes through, rather than four hand-written ladders;
* the four per-tier colour-key impls are reached through **one** seam
  (:func:`~digitalearth.api._add_key` / :data:`~digitalearth.api._KEY_ADDERS`), not four copy-pasted call
  sites.

The lazy-import checks run in a **subprocess**, because ``sys.modules`` is process-wide: any earlier test in
this session that built a static map would already have loaded the tier, so the check is only meaningful in an
interpreter that imported nothing but ``api``.
"""

import subprocess
import sys
import textwrap


def _run_fresh(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter and return the completed process.

    Args:
        body: Python source to execute with ``python -c``. It must print ``OK`` on success and raise
            otherwise, so the caller can assert on both the return code and the output.

    Returns:
        The finished :class:`subprocess.CompletedProcess`, with ``stdout``/``stderr`` captured as text.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


class TestImportingApiDoesNotLoadTheMatplotlibTier:
    """DE-24 invariant 2 — ``import digitalearth.api`` must not import ``digitalearth.static``."""

    def test_a_fresh_import_of_api_leaves_the_static_tier_unloaded(self):
        """Importing only ``api`` leaves ``digitalearth.static`` out of ``sys.modules``.

        Test scenario:
            ``api`` used to import ``Map``, the static ``capabilities`` and the static ``guides`` at module
            top, so importing it pulled in matplotlib even for ``backend="web"``. The three are lazy now.
            The assertion goes red the moment any of them is made eager again — which is the mutation that
            proves this test can fail.
        """
        completed = _run_fresh(
            """
            import sys
            import digitalearth.api  # noqa: F401

            leaked = sorted(m for m in sys.modules if m.startswith("digitalearth.static"))
            assert "digitalearth.static" not in sys.modules, (
                "importing digitalearth.api loaded the matplotlib tier: " + repr(leaked)
            )
            assert "matplotlib" not in sys.modules, "importing digitalearth.api loaded matplotlib"
            print("OK")
            """
        )
        assert completed.returncode == 0, (
            f"the fresh-import check failed:\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
        assert completed.stdout.strip().endswith("OK"), completed.stdout

    def test_a_web_quickmap_does_not_load_the_matplotlib_tier(self):
        """Dispatching ``quickmap(..., backend="web")`` reaches the web builder without loading static.

        Test scenario:
            The refusal gate and the backend switch run before any builder, and both must read the capability
            table without resolving the matplotlib row. With the web builder mocked, the whole dispatch runs
            and ``digitalearth.static`` must still be absent — the guarantee the docstring makes for a
            ``web``/``interactive``/``3d`` call.
        """
        completed = _run_fresh(
            """
            import sys
            from unittest import mock

            import digitalearth.api as qp

            with mock.patch.object(qp, "_quickmap_web") as built:
                qp.quickmap(object(), backend="web")
            assert built.called, "the web builder was not reached"
            assert "digitalearth.static" not in sys.modules, (
                "a web quickmap loaded the matplotlib tier"
            )
            print("OK")
            """
        )
        assert completed.returncode == 0, (
            f"the web-dispatch check failed:\nstdout={completed.stdout!r}\nstderr={completed.stderr!r}"
        )
        assert completed.stdout.strip().endswith("OK"), completed.stdout
