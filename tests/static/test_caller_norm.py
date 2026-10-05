"""A caller's own matplotlib ``Normalize`` is drawn with as given (ST-12).

cleopatra >=0.38 takes a built ``Normalize`` on ``plot(norm=...)`` and, as an escape hatch, on the ``color=``
group parameter. The static tier forwards both untouched, so any norm matplotlib can build — ``LogNorm``,
``BoundaryNorm``, a ``FuncNorm`` — reaches the drawn artist without a dedicated ``color_scale`` variant. That
worked before these tests; they pin it, and pin ``norm`` as a declared style key so a caller can find it.
"""

import pytest
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize

from digitalearth.static import Map
from digitalearth.static.render_compat import STATIC_STYLE_SCHEMA, route_flat_style

RASTER = "examples/data/acc4000.tif"


@pytest.fixture(scope="module")
def raster():
    """Return the small accumulation raster the render tests draw.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(RASTER)


class TestNormIsDeclared:
    """``norm`` is part of the static tier's declared style vocabulary."""

    def test_norm_is_a_declared_key(self):
        """The schema lists ``norm``, so a discovery surface can show it."""
        assert "norm" in STATIC_STYLE_SCHEMA.names(), sorted(
            STATIC_STYLE_SCHEMA.names()
        )

    def test_norm_says_what_it_controls(self):
        """The declaration carries a description that names a matplotlib ``Normalize``."""
        doc = STATIC_STYLE_SCHEMA.keys["norm"].doc
        assert "Normalize" in doc, doc

    def test_norm_drives_no_channel(self):
        """A norm is how values map onto the ramp, not a visual channel of the layer.

        Test scenario:
            The same reason ``color`` (the ColorScaling group) declares none: the colour comes from ``cmap``
            plus the data, and the norm is the scaling between them.
        """
        assert STATIC_STYLE_SCHEMA.keys["norm"].channel is None, (
            STATIC_STYLE_SCHEMA.keys["norm"]
        )

    def test_a_misspelt_norm_is_nameable(self):
        """A near miss on ``norm`` now has an answer."""
        assert STATIC_STYLE_SCHEMA.suggest("nrom") == "norm", (
            STATIC_STYLE_SCHEMA.suggest("nrom")
        )

    def test_norm_routes_as_a_property_holding_the_same_object(self):
        """Routing keeps the caller's object; it does not copy or rebuild it."""
        caller_norm = LogNorm(vmin=1.0, vmax=100.0)
        sym, rest = route_flat_style({"norm": caller_norm})
        assert sym.props["norm"] is caller_norm, dict(sym.props)
        assert not rest, rest


class TestTheCallersNormIsDrawn:
    """The drawn artist colours through the very object the caller passed."""

    def test_norm_keyword_on_a_field(self, raster):
        """``Map.field(ds, norm=LogNorm(...))`` draws with that ``LogNorm``.

        Args:
            raster: The accumulation raster.
        """
        caller_norm = LogNorm(vmin=1.0, vmax=1000.0)
        with Map() as canvas:
            canvas.field(raster, norm=caller_norm)
            artist = canvas.artist()
        assert artist.norm is caller_norm, artist.norm

    def test_norm_through_the_color_group_on_a_field(self, raster):
        """``color=`` a ``Normalize`` is cleopatra's escape hatch, and it survives the colour fold.

        Args:
            raster: The accumulation raster.

        Test scenario:
            The static fold builds a ``ColorScaling`` only from flat colour keys; a built object under
            ``color=`` must pass through rather than be rebuilt into a linear scale.
        """
        caller_norm = LogNorm(vmin=1.0, vmax=1000.0)
        with Map() as canvas:
            canvas.field(raster, color=caller_norm)
            artist = canvas.artist()
        assert artist.norm is caller_norm, artist.norm

    def test_the_limits_drawn_are_the_norms_own(self, raster):
        """A plain ``Normalize`` with explicit limits fixes the colour limits, rather than the data range.

        Args:
            raster: The accumulation raster.
        """
        caller_norm = Normalize(vmin=-5.0, vmax=5.0)
        with Map() as canvas:
            canvas.field(raster, norm=caller_norm)
            artist = canvas.artist()
        assert artist.get_clim() == (-5.0, 5.0), artist.get_clim()

    def test_a_boundary_norm_on_value_coloured_points(self, raster):
        """Point layers forward the norm too.

        Args:
            raster: The accumulation raster.

        Test scenario:
            ``grid_points`` reaches a different cleopatra glyph (``ScatterGlyph``) through the same
            pass-through, so the pin is not field-only.
        """
        caller_norm = BoundaryNorm([0, 10, 100, 1000], 256)
        with Map() as canvas:
            canvas.grid_points(raster, norm=caller_norm)
            artist = canvas.artist()
        assert artist.norm is caller_norm, artist.norm

    def test_a_norm_that_is_not_a_normalize_is_refused(self, raster):
        """A string under ``norm=`` is an error, not a silently ignored keyword.

        Args:
            raster: The accumulation raster.
        """
        canvas = Map()
        with pytest.raises(TypeError, match="Normalize"):
            canvas.field(raster, norm="log")
