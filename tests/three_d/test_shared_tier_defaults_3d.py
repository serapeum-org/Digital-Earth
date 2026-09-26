"""M9/L11 — the 3-D tier's half of the shared-constant regressions, with PyVista present.

``tests/base/test_shared_tier_defaults.py`` can only read this tier's source, because its modules import
PyVista at import time and the lean ``dev`` environment has none. This file closes that gap in the ``viz3d``
environment: the frame rate its animation entry points start from really is the object ``base/animation.py``
declares, and its classified layers really are coloured by the shared sampler.

Run it with ``pixi run -e viz3d test-3d``; without the extra the whole module skips.
"""

import pytest

pytest.importorskip("pyvista")

from digitalearth.base.animation import DEFAULT_FPS  # noqa: E402
from digitalearth.base.symbology import sample_cmap  # noqa: E402
from digitalearth.three_d import animation as animation3d  # noqa: E402
from digitalearth.three_d import base as base3d  # noqa: E402


class TestTheAnimationRateIsTheSharedOne:
    """M9 — this tier reads the one rate rather than declaring a third copy of it."""

    def test_the_tier_constant_is_the_shared_object(self):
        """``three_d.animation.DEFAULT_FPS`` *is* the constant from ``base/animation.py``.

        Test scenario:
            The tier declared its own ``DEFAULT_FPS: float = 3.0``. ``is`` rather than ``==`` is what tells a
            shared constant from a re-declared literal that happens to match today.
        """
        assert animation3d.DEFAULT_FPS is DEFAULT_FPS

    @pytest.mark.parametrize("method", ["orbit", "record"])
    def test_both_entry_points_default_to_the_shared_rate(self, method):
        """``orbit`` and ``record`` start from the shared rate when ``fps`` is omitted.

        Args:
            method: The animation entry point to inspect.

        Test scenario:
            The default is the number a caller actually animates at, so it is what has to be the shared
            object rather than a literal that matches it today. This tier used to answer 12 for ``orbit``
            and 8 for its callback loop, which is ``record`` since #299.
        """
        import inspect

        default = (
            inspect.signature(getattr(animation3d.AnimationMixin, method))
            .parameters["fps"]
            .default
        )
        assert default is DEFAULT_FPS, (
            f"{method}() falls back to {default!r}, not the shared rate"
        )


class TestTheColormapSamplerIsShared:
    """L11 — classified 3-D layers take their colours from ``base/symbology.py``."""

    def test_the_tier_defines_no_sampler_of_its_own(self):
        """``three_d.base`` exposes no private ``_sample_cmap`` any more.

        Test scenario:
            Its copy was one of three implementations of the same evenly-spaced sampling, each free to drift
            from the others while their docstrings promised they matched.
        """
        assert not hasattr(base3d, "_sample_cmap")

    def test_a_graduated_layer_is_coloured_by_the_shared_sampler(self):
        """The colours a classified layer hands PyVista are exactly the shared helper's.

        Test scenario:
            The sampler is reached through ``classified_scalars``, which is what every 3-D builder colours
            through — so this checks the wiring, not just the import.
        """
        style = base3d.classified_scalars(
            [1.0, 2.0, 3.0, 40.0], scheme="quantiles", k=2, cmap="viridis"
        )
        assert style["cmap"] == sample_cmap("viridis", 2)

    def test_an_explicit_colour_sequence_is_still_honoured(self):
        """A ready-made list of colours passes through, as this tier's own copy allowed.

        Test scenario:
            The 3-D copy accepted a sequence as well as a name; the shared helper had to keep that, or moving
            to it would have narrowed what this tier's ``cmap`` accepts.
        """
        colours = ["#ff0000", "#00ff00"]
        style = base3d.classified_scalars(
            [1.0, 2.0, 3.0, 40.0], scheme="quantiles", k=2, cmap=colours
        )
        assert style["cmap"] == colours


class TestTheBigDataCutoffDefaultsToTheCellBudget:
    """Contract C8's per-unit default, asserted live for the tier that measures in cells.

    ``tests/base/test_shared_tier_defaults.py`` asserts the 50 000-row constant is shared between the
    interactive and web tiers; it cannot ask this one, because a scene needs PyVista. Nothing checked the
    clause's words here, which is how it went on claiming a single shared default for ten times the number
    (review M5).
    """

    def test_the_scene_starts_from_the_cell_budget(self):
        """A fresh scene's attribute *is* ``DEFAULT_CELL_BUDGET``, not a literal that matches it today.

        Test scenario:
            The attribute is what every builder falls back to when the call passes no cutoff, so it is the
            value a caller actually reduces at. ``is`` rather than ``==`` is what tells the shared constant
            from a re-declared copy.
        """
        from digitalearth.three_d import Scene3D
        from digitalearth.three_d.bigdata import DEFAULT_CELL_BUDGET

        scene = Scene3D(off_screen=True)
        try:
            started_from = scene.big_data_threshold
        finally:
            scene.close()
        assert started_from is DEFAULT_CELL_BUDGET, (
            f"the scene starts from {started_from!r}, not the module's cell budget"
        )

    def test_it_is_not_the_row_cutoff_the_2d_tiers_share(self):
        """The two numbers count different things, so this tier must not read the row cutoff.

        Test scenario:
            Reading `DEFAULT_BIG_DATA_THRESHOLD` here would reduce an ordinary DEM tile — a 708x708 one is
            499 849 cells against a 50 000-row cutoff — which is the picture the separate budget exists to
            protect.
        """
        from digitalearth.base.bigdata import DEFAULT_BIG_DATA_THRESHOLD
        from digitalearth.three_d.bigdata import DEFAULT_CELL_BUDGET

        assert DEFAULT_CELL_BUDGET != DEFAULT_BIG_DATA_THRESHOLD, (
            "the cell budget and the row cutoff have converged, which contract C8 no longer claims"
        )
