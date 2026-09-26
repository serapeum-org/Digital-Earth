"""Unit tests for :mod:`digitalearth.static.render_compat` — the flat-kwarg -> cleopatra group-object translation."""

import numpy as np
import pytest
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, PointOverlay
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import ColorScale, ColorScaling

from digitalearth.static.render_compat import (
    FLAT_STYLE_KEYS,
    group_render_kwargs,
    prepare_plot_kwargs,
    relocate_flat_style,
)

# The colour group is folded by `style_fold` and covered in `test_style_fold.py`; what it is asserted for
# here is that it still arrives beside the groups this module folds itself.


def test_group_render_kwargs_folds_each_group():
    """Each family of flat kwargs folds into its typed cleopatra group object."""
    out = group_render_kwargs(
        {
            "levels": 5,
            "scheme": "quantiles",
            "k": 4,
            "style": "terrain",
            "display_cell_value": True,
            "color_scale": "power",
            "gamma": 0.3,
            "cmap": "viridis",
        }
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
    out = group_render_kwargs(
        {"scheme": "quantiles", "alpha": 0.5}, accepted={"color", "contour", "classify"}
    )
    assert isinstance(out["classify"], Classify)  # accepted -> folded
    assert out["alpha"] == 0.5
    assert "data_style" not in out  # data_style not accepted -> left flat


def test_prepare_plot_kwargs_defers_alpha_for_vector_glyph():
    """A vector glyph (no data_style parameter) hands alpha back for post-hoc application instead of folding it."""
    glyph = ScatterGlyph(
        np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2])
    )
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
    glyph = ScatterGlyph(
        np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2])
    )
    with pytest.raises(
        ValueError, match=r"does not support the styling option\(s\) \['style'\]"
    ):
        prepare_plot_kwargs(glyph, {"style": "terrain"})


def test_prepare_plot_kwargs_rejects_points_overlay_on_unsupported_glyph():
    """A points overlay on a glyph with no ``points`` parameter raises a clear ValueError, not a TypeError."""
    glyph = ScatterGlyph(
        np.array([0.0, 1]), np.array([0.0, 1]), values=np.array([1.0, 2])
    )
    zeros = np.zeros((2, 3))
    with pytest.raises(
        ValueError, match=r"does not support the styling option\(s\).*points"
    ):
        prepare_plot_kwargs(glyph, {"points": zeros})


def test_group_render_kwargs_keeps_flat_member_when_group_object_present():
    """A flat member passed alongside a built group object of the same group is left in place, not dropped.

    The colour group is the exception: it refuses the pair instead, because forwarding both made cleopatra
    answer with advice the caller had already followed (see `test_style_fold.py`).
    """
    contour = Contour(levels=3)
    out = group_render_kwargs({"contour": contour, "labels": True})
    assert out["contour"] is contour
    assert out["labels"] is True


def test_relocate_flat_style_pops_styling_leaves_constructor_options():
    """relocate_flat_style removes the flat members and group params, leaving constructor-safe options."""
    opts = {
        "scheme": "quantiles",
        "levels": 5,
        "color": ColorScaling(),
        "cmap": "viridis",
        "add_colorbar": False,
    }
    moved = relocate_flat_style(opts)
    assert set(moved) == {"scheme", "levels", "color"}
    assert opts == {"cmap": "viridis", "add_colorbar": False}


def test_flat_style_keys_covers_group_members_and_params():
    """FLAT_STYLE_KEYS spans every flat member, the point_* aliases, and the typed group parameter names."""
    for key in (
        "levels",
        "scheme",
        "style",
        "color_scale",
        "points",
        "point_color",
        "color",
        "contour",
    ):
        assert key in FLAT_STYLE_KEYS


def test_points_alpha_applies_to_the_rendered_artist():
    """Map().points(fc, alpha=) reaches the artist (regression: it used to raise TypeError on vector glyphs)."""
    import geopandas as gpd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    from digitalearth.static import Map

    fc = FeatureCollection(
        gpd.GeoDataFrame(
            {"v": [1.0, 2.0, 3.0]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:4326",
        )
    )
    artist = Map(crs=4326).points(fc, alpha=0.5)
    assert artist.get_alpha() == 0.5


def test_an_already_built_point_overlay_is_left_alone():
    """A caller who passes a built PointOverlay keeps it, and stray point_* keys are not silently dropped.

    Test scenario:
        The fold is applied centrally and must be idempotent on its own output. Rebuilding the overlay would
        discard whatever the caller configured on it; dropping the leftover point_* keys instead of leaving
        them for `prepare_plot_kwargs` to report would lose styling with no error.
    """
    overlay = PointOverlay(np.array([[0.0, 0.0]]))
    out = group_render_kwargs({"points": overlay, "point_color": "red"})
    assert out["points"] is overlay, "a built overlay must be kept as given"
    assert out["point_color"] == "red", (
        "a stray point_* key must be left in place, not silently dropped"
    )


def test_marker_styling_with_no_points_array_is_refused():
    """point_* styling passed without any `points` names the keys rather than vanishing.

    Test scenario:
        This is the module's own thesis applied to itself: dropping the keys meant `point_color="red"` on a
        layer with no `points=` did nothing and said nothing, which is the silence the declared schema exists
        to remove. Contrast `test_an_already_built_point_overlay_is_left_alone`, where a stray `point_*` key
        *is* left in place — there the overlay exists, so `prepare_plot_kwargs` can report the key against
        the glyph that could not take it. Here there is nothing to report it against, so it is named now.
    """
    with pytest.raises(ValueError, match="no points= array was given"):
        group_render_kwargs({"point_color": "red", "point_size": 8})


def test_an_explicit_points_none_is_still_a_no_op():
    """Forwarding `points=None` does nothing, as it did before the styling guard was added.

    Test scenario:
        `points=` is a public styling kwarg on the raster builders, and forwarding an optional variable
        (`points=points_or_none`) is the ordinary way to write a wrapper around them. The round-1 fix that
        names orphaned point styling raised unconditionally once `points` was present-but-None, so it
        reported an empty list of keys and broke every such wrapper.
    """
    out = group_render_kwargs({"points": None, "cmap": "viridis"})
    assert out == {"cmap": "viridis"}, f"points=None must fold to nothing, got {out}"


def test_orphaned_point_styling_names_the_keyword_the_caller_wrote():
    """The error says `point_color`, not cleopatra's internal field name `color`.

    Test scenario:
        The keys were collected by `PointOverlay` *field* name, so the message listed names the caller never
        typed. `color` is worse than merely unfamiliar — it is itself a declared style key here, meaning
        cleopatra's ColorScaling group, so the message pointed at a real keyword that does something else.
    """
    with pytest.raises(ValueError, match=r"\['point_color'\]"):
        group_render_kwargs({"points": None, "point_color": "red"})


def test_a_group_with_no_members_is_not_built():
    """A fold that built an empty group would hand cleopatra a spec nobody asked for.

    Test scenario:
        The other arm of `_fold_group`'s early return, beside the built-group one: a call that names none
        of a group's flat members leaves that group out entirely.
    """
    out = group_render_kwargs({"cmap": "viridis"})
    assert "contour" not in out, out
    assert "classify" not in out, out
    assert out == {"cmap": "viridis"}, out
