"""What the interactive tier declares, and one defect its style records caused (DE-23c + U-1, #300).

The tier's refusals lived wherever they were raised, and `api.py` held one table about it. These cover the
declaration that replaces that, and the bug the old per-type restyle caused: a dashboard widget gave every
raster the style of whichever one was added last.
"""

import subprocess
import sys

import numpy as np
import pytest

from digitalearth.base.capabilities import Capabilities
from digitalearth.base.sources import get_source

pytest.importorskip("holoviews")

from digitalearth.interactive import InteractiveMap  # noqa: E402
from digitalearth.interactive.capabilities import CAPABILITIES  # noqa: E402


def _dem():
    """Return a small ramped raster.

    Returns:
        A `Source` over the ramp.
    """
    return get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))


class TestTheDeclaration:
    """`capabilities.py` says what the tier can draw (#294)."""

    def test_the_declaration_is_this_backend(self):
        """The row is named as `quickmap(backend=...)` spells it."""
        assert CAPABILITIES.backend == "interactive", CAPABILITIES.backend

    def test_the_channels_are_the_ones_the_fold_measured(self):
        """#298 measured them against HoloViews, so the declaration repeats no guess."""
        from digitalearth.interactive.style_fold import CHANNEL_OPTIONS

        folded = {entry.channel for entry in CHANNEL_OPTIONS}
        assert CAPABILITIES.channels == folded, (
            f"declared {sorted(CAPABILITIES.channels)}, folds {sorted(folded)}"
        )

    def test_the_channel_holoviews_cannot_express_says_so(self):
        """`height` has no HoloViews option at all, and the declaration gives that as the reason."""
        assert "z-height" in CAPABILITIES.reason("height"), CAPABILITIES.absent

    def test_a_text_layer_is_drawn_even_though_the_text_channel_does_not_fold(self):
        """One name cannot be both claimed and disclaimed: the kind is declared, the channel is not.

        Test scenario:
            `Capabilities` refuses a name that is listed as supported and absent at once, which is what
            caught this: the tier draws `text` layers and cannot style a `text` channel, and those are two
            different questions.
        """
        from digitalearth.interactive.style_fold import UNEXPRESSIBLE

        assert "text" in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)
        assert "text" in UNEXPRESSIBLE, sorted(UNEXPRESSIBLE)
        assert CAPABILITIES.reason("text") is None, CAPABILITIES.absent

    def test_the_declaration_loads_without_holoviews(self):
        """A dispatcher reads it before it decides which backend to build.

        Test scenario:
            Run in a subprocess, since this session has HoloViews loaded already.
        """
        code = (
            "import sys; from digitalearth.interactive.capabilities import CAPABILITIES;"
            "print('holoviews' in sys.modules, CAPABILITIES.backend)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "False interactive", (
            result.stdout or result.stderr
        )

    def test_the_dispatcher_reads_the_declaration(self):
        """`api.BACKEND_CAPABILITIES["interactive"]` is derived, and says what it always said."""
        from digitalearth.api import BACKEND_CAPABILITIES

        assert sorted(BACKEND_CAPABILITIES["interactive"]) == [
            "basemap",
            "coastlines",
            "colorbar",
            "crs",
            "kind",
        ], BACKEND_CAPABILITIES["interactive"]

    def test_a_refusal_carries_the_declared_reason(self):
        """A caller is told what the tier does instead of what they asked for."""
        from digitalearth import quickmap

        with pytest.raises(ValueError, match="pans and zooms"):
            quickmap(_dem(), backend="interactive", domain="europe")

    def test_the_declaration_is_a_capabilities_value(self):
        """The shared type, so one support matrix can be built from every tier's row."""
        assert isinstance(CAPABILITIES, Capabilities), type(CAPABILITIES)


class TestEachLayerKeepsItsOwnStyle:
    """A widget move restyles each layer with its own recorded style (#300)."""

    def _styles(self, composed) -> list:
        """Return the style each element of a composed figure carries.

        Args:
            composed: The HoloViews object a render produced.

        Returns:
            One style dict per element, in draw order.
        """
        import holoviews as hv

        # A one-layer figure is the element itself, which does not iterate; an overlay does.
        elements = list(composed) if isinstance(composed, hv.Overlay) else [composed]
        return [
            hv.Store.lookup_options("bokeh", element, "style").kwargs
            for element in elements
        ]

    def test_two_rasters_keep_their_colormaps_through_an_alpha_change(self):
        """The measured defect: both rasters came back in the second one's colormap and limits.

        Test scenario:
            `image(dem, cmap="magma", clim=(0, 10))` then `image(dem, cmap="Blues", clim=(0, 80))`, then the
            opacity slider moved. The styles were merged into one dict — last layer winning — and applied per
            element *type*, which is the only way `.opts()` can be applied to an overlay.
        """
        m = InteractiveMap()
        m.image(_dem(), cmap="magma", clim=(0.0, 10.0))
        m.image(_dem(), cmap="Blues", clim=(0.0, 80.0))
        styles = self._styles(m._render_with_overrides({"alpha": 0.5}))
        assert [style.get("cmap") for style in styles] == ["magma", "Blues"], styles

    def test_the_widget_value_reaches_every_layer(self):
        """Each layer takes the override; what it does not take is its own recorded style."""
        m = InteractiveMap()
        m.image(_dem(), cmap="magma")
        m.image(_dem(), cmap="Blues")
        styles = self._styles(m._render_with_overrides({"alpha": 0.25}))
        assert [style.get("alpha") for style in styles] == [0.25, 0.25], styles

    def test_a_colormap_widget_overrides_every_raster(self):
        """A widget that names a colormap means that colormap, on each layer it claims."""
        m = InteractiveMap()
        m.image(_dem(), cmap="magma")
        m.image(_dem(), cmap="Blues")
        styles = self._styles(m._render_with_overrides({"cmap": "viridis"}))
        assert [style.get("cmap") for style in styles] == ["viridis", "viridis"], styles

    def test_a_map_with_no_overrides_renders_as_it_was_built(self):
        """No widget moved, so nothing is restyled and the figure is the rendered one."""
        m = InteractiveMap()
        m.image(_dem(), cmap="magma")
        styles = self._styles(m._render_with_overrides({}))
        assert styles[0].get("cmap") == "magma", styles
