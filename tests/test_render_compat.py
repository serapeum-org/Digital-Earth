"""Unit tests for :mod:`digitalearth._render_compat` — the flat-kwarg -> cleopatra group-object translation."""
import numpy as np
import pytest
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import ColorScale, ColorScaling

from digitalearth._render_compat import (
    FLAT_STYLE_KEYS,
    _coerce_color_scale,
    group_render_kwargs,
    prepare_plot_kwargs,
    relocate_flat_style,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("linear", ColorScale.LINEAR),
        ("power", ColorScale.POWER),
        ("sym_log", ColorScale.SYM_LOGNORM),
        ("symlog", ColorScale.SYM_LOGNORM),
        ("sym_lognorm", ColorScale.SYM_LOGNORM),
        ("sym-lognorm", ColorScale.SYM_LOGNORM),  # exact enum value
        ("boundary", ColorScale.BOUNDARY_NORM),
        ("boundary_norm", ColorScale.BOUNDARY_NORM),
        ("MidPoint", ColorScale.MIDPOINT),  # case-insensitive
        (ColorScale.POWER, ColorScale.POWER),  # already an enum
    ],
)
def test_coerce_color_scale_maps_friendly_spellings(value, expected):
    """Friendly color_scale spellings (and a real enum) coerce to the matching ColorScale member."""
    assert _coerce_color_scale(value) is expected


@pytest.mark.parametrize("value", ["lognorm", "bogus", "log"])
def test_coerce_color_scale_rejects_unknown(value):
    """An unrecognised color_scale raises a clear error rather than crashing opaquely at render time."""
    with pytest.raises(ValueError, match="not a recognised colour scale"):
        _coerce_color_scale(value)


def test_group_render_kwargs_folds_each_group():
    """Each family of flat kwargs folds into its typed cleopatra group object."""
    out = group_render_kwargs(
        {"levels": 5, "scheme": "quantiles", "k": 4, "style": "terrain", "display_cell_value": True,
         "color_scale": "power", "gamma": 0.3, "cmap": "viridis"}
    )
    assert isinstance(out["contour"], Contour)
    assert out["contour"].levels == 5
    assert isinstance(out["classify"], Classify)
    assert out["classify"].scheme == "quantiles"
    assert out["classify"].k == 4
    assert isinstance(out["data_style"], DataStyle)
    assert out["data_style"].style == "terrain"
    assert isinstance(out["cells"], CellValues)
    assert isinstance(out["color"], ColorScaling)
    assert out["color"].kind is ColorScale.POWER
    assert out["cmap"] == "viridis"  # non-styling kwargs pass through untouched


def test_group_render_kwargs_is_idempotent():
    """Folding an already-folded dict is a no-op (safe to apply centrally)."""
    once = group_render_kwargs({"levels": 3, "scheme": "quantiles"})
    twice = group_render_kwargs(once)
    assert twice["contour"] is once["contour"]
    assert twice["classify"] is once["classify"]


def test_group_render_kwargs_wraps_points_overlay():
    """A bare points array with marker styling becomes a PointOverlay carrying that styling."""
    arr = np.zeros((2, 3))
    out = group_render_kwargs({"points": arr, "point_color": "red", "point_size": 40})
    assert isinstance(out["points"], PointOverlay)
    assert out["points"].color == "red"
    assert out["points"].size == 40


def test_group_render_kwargs_respects_accepted_groups():
    """With an ``accepted`` set, groups the glyph lacks are not folded — their flat members are left alone."""
    out = group_render_kwargs({"scheme": "quantiles", "alpha": 0.5}, accepted={"color", "contour", "classify"})
    assert isinstance(out["classify"], Classify)  # accepted -> folded
    assert out["alpha"] == 0.5
    assert "data_style" not in out  # data_style not accepted -> left flat


def test_prepare_plot_kwargs_defers_alpha_for_vector_glyph():
    """A vector glyph (no data_style parameter) hands alpha back for post-hoc application instead of folding it."""
    glyph = ScatterGlyph(np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2]))
    kwargs, alpha = prepare_plot_kwargs(glyph, {"scheme": "quantiles", "alpha": 0.5})
    assert alpha == 0.5
    assert isinstance(kwargs["classify"], Classify)
    assert "alpha" not in kwargs
    assert "data_style" not in kwargs


def test_prepare_plot_kwargs_folds_alpha_for_array_glyph():
    """An array glyph accepts DataStyle, so alpha folds into it and nothing is deferred."""
    glyph = ArrayGlyph(np.arange(12, dtype=float).reshape(3, 4))
    kwargs, alpha = prepare_plot_kwargs(glyph, {"alpha": 0.5})
    assert alpha is None
    assert isinstance(kwargs["data_style"], DataStyle)
    assert kwargs["data_style"].alpha == 0.5


def test_prepare_plot_kwargs_rejects_unsupported_styling():
    """A raster-only styling kwarg on a vector glyph raises a clear ValueError naming it."""
    glyph = ScatterGlyph(np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2]))
    with pytest.raises(ValueError, match=r"does not support the styling option\(s\) \['style'\]"):
        prepare_plot_kwargs(glyph, {"style": "terrain"})


def test_prepare_plot_kwargs_rejects_points_overlay_on_unsupported_glyph():
    """A points overlay on a glyph with no ``points`` parameter raises a clear ValueError, not a TypeError."""
    glyph = ScatterGlyph(np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2]))
    with pytest.raises(ValueError, match=r"does not support the styling option\(s\).*points"):
        prepare_plot_kwargs(glyph, {"points": np.zeros((2, 3))})


def test_group_render_kwargs_keeps_flat_member_when_group_object_present():
    """A flat member passed alongside a built group object of the same group is left in place, not dropped."""
    scaling = ColorScaling()
    out = group_render_kwargs({"color": scaling, "gamma": 0.3})
    assert out["color"] is scaling
    assert out["gamma"] == 0.3


def test_relocate_flat_style_pops_styling_leaves_constructor_options():
    """relocate_flat_style removes the flat members and group params, leaving constructor-safe options."""
    opts = {"scheme": "quantiles", "levels": 5, "color": ColorScaling(), "cmap": "viridis", "add_colorbar": False}
    moved = relocate_flat_style(opts)
    assert set(moved) == {"scheme", "levels", "color"}
    assert opts == {"cmap": "viridis", "add_colorbar": False}


def test_flat_style_keys_covers_group_members_and_params():
    """FLAT_STYLE_KEYS spans every flat member, the point_* aliases, and the typed group parameter names."""
    for key in ("levels", "scheme", "style", "color_scale", "points", "point_color", "color", "contour"):
        assert key in FLAT_STYLE_KEYS


def test_scatter_alpha_applies_to_the_rendered_artist():
    """Map().scatter(fc, alpha=) reaches the artist (regression: it used to raise TypeError on vector glyphs)."""
    import geopandas as gpd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    from digitalearth.scene import Map

    fc = FeatureCollection(
        gpd.GeoDataFrame({"v": [1.0, 2.0, 3.0]}, geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs="EPSG:4326")
    )
    artist = Map(crs=4326).scatter(fc, alpha=0.5)
    assert artist.get_alpha() == 0.5
