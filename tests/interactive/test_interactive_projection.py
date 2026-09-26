"""DI.9 — projections beyond Web-Mercator via the matplotlib backend (projection / graticule).

Renders Orthographic/Robinson through the HoloViews matplotlib backend (no browser); asserts the
backend switch, the projection on the rendered object, the tiles-vs-projection guards, and a headless
PNG export. Runs in the ``interactive`` pixi env.
"""

import pytest

from digitalearth.interactive import InteractiveMap
from digitalearth.interactive.projection import _element_extent

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


def _graticule_parameters():
    """Return both 2-D tiers' `graticule` parameters, so the two signatures can be compared.

    Returns:
        `(interactive, static)`, each a mapping of parameter name to `inspect.Parameter`.
    """
    import inspect

    from digitalearth.static.maps.projection import ProjectionMixin as StaticMixin

    return (
        inspect.signature(InteractiveMap.graticule).parameters,
        inspect.signature(StaticMixin.graticule).parameters,
    )


class TestProjection:
    """``projection`` — arbitrary display projections via the matplotlib backend."""

    def test_projection_makes_rasters_gv_with_crs(self, m, dataset):
        """Under a projection, image() emits a gv.Image (crs-aware) so GeoViews can reproject it."""
        m.projection("Robinson")
        m.image(dataset)
        assert isinstance(m.layers[0], gv.Image), (
            f"expected gv.Image under a projection, got {type(m.layers[0])}"
        )

    def test_no_projection_keeps_plain_hv_image(self, m, dataset):
        m.image(dataset)
        assert isinstance(m.layers[0], hv.Image) and not isinstance(
            m.layers[0], gv.Image
        )

    def test_render_sets_projection_on_the_object(self, m, dataset):
        m.projection("Orthographic").image(dataset)
        obj = m.render()
        proj = hv.Store.lookup_options("matplotlib", obj, "plot").kwargs.get(
            "projection"
        )
        assert proj is not None, (
            "render() must pass the projection through to the mpl backend"
        )
        assert "Orthographic" in type(proj).__name__

    def test_orthographic_png_export(self, m, dataset, tmp_path):
        out = tmp_path / "globe.png"
        m.projection("Orthographic").image(dataset).save(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_projection_none_returns_to_bokeh_path(self, m, dataset):
        m.projection("Robinson")
        m.projection(None)
        m.image(dataset)
        assert isinstance(m.layers[0], hv.Image) and not isinstance(
            m.layers[0], gv.Image
        )

    def test_chains(self, m):
        assert m.projection("Robinson") is m

    def test_epsg_int_projection(self, m, dataset):
        """An EPSG int projection resolves via process_crs and renders through the mpl backend."""
        m.projection(3857).image(dataset)
        obj = m.render()
        proj = hv.Store.lookup_options("matplotlib", obj, "plot").kwargs.get(
            "projection"
        )
        assert proj is not None, "EPSG-int projection must reach the mpl backend"

    def test_cartopy_object_passes_through(self, m, dataset):
        """A pre-built cartopy projection object is used verbatim (no name resolution)."""
        import cartopy.crs as ccrs

        proj = ccrs.Mollweide()
        m.projection(proj).image(dataset)
        assert m._projection is proj, "a cartopy object must pass straight through"

    def test_unknown_projection_name_raises(self, m):
        with pytest.raises(ValueError, match="unknown projection"):
            m.projection("NotAProjection")

    def test_graticule_opts_forwarded(self, m):
        m.graticule(line_width=0.5)
        assert isinstance(m.layers[-1], gv.element.Feature)


class TestProjectionTileGuards:
    """tiles and a non-Mercator projection are mutually exclusive (both directions)."""

    def test_projection_after_tiles_raises(self, m, dataset):
        m.image(dataset).tiles("CartoLight")
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.projection("Robinson")

    def test_tiles_after_projection_raises(self, m, dataset):
        m.projection("Robinson").image(dataset)
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.tiles("CartoLight")

    def test_constructor_tiles_block_projection(self):
        m = InteractiveMap(tiles="CartoLight")
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.projection("Robinson")


class TestGraticule:
    """``graticule`` — lon/lat grid feature."""

    def test_graticule_adds_feature(self, m):
        m.graticule()
        assert isinstance(m.layers[0], gv.element.Feature), f"got {type(m.layers[0])}"

    def test_chains(self, m):
        assert m.graticule() is m


class TestGraticuleSpacing:
    """#252 — ``graticule`` takes the same lon_step/lat_step the static tier does."""

    def test_default_is_the_thirty_degree_grid(self, m):
        """Unchanged behaviour: no arguments still draws the 30-degree Natural-Earth grid."""
        m.graticule()
        assert m.layers[-1].data.name == "graticules_30", (
            f"default spacing changed: {m.layers[-1].data.name}"
        )

    @pytest.mark.parametrize("step", [1, 5, 10, 15, 20, 30])
    def test_requested_spacing_is_honoured(self, step):
        """The requested spacing selects the matching Natural-Earth graticule layer."""
        m = InteractiveMap().graticule(lon_step=step, lat_step=step)
        assert m.layers[-1].data.name == f"graticules_{step}", (
            f"lon_step/lat_step ignored: {m.layers[-1].data.name}"
        )

    def test_both_tiers_take_the_same_spacing_arguments(self):
        """The three spellings of a spacing are on both tiers' signatures (#324).

        Test scenario:
            `graticule` is a TIER2 name, so wherever it exists it means one thing: `lon_step`,
            `lat_step` and `spacing` are the keywords `base/contract.py` records for it, and a caller
            moving between tiers writes the same call. This half of the old signature check is unchanged
            — it is the half that is still literally true.
        """
        interactive, static = _graticule_parameters()
        for name in ("lon_step", "lat_step", "spacing"):
            assert name in interactive, f"interactive graticule must accept {name}"
            assert name in static, f"static graticule must accept {name}"

    def test_both_tiers_draw_the_same_grid_when_no_step_is_named(self):
        """A caller who names no spacing gets the same grid on either tier.

        Test scenario:
            This used to be checked by comparing the two signatures' default *values*, which held only
            while both tiers spelled 30 degrees as a literal. The static tier now defaults its two steps
            to `None`, so it can tell a step the caller wrote from one they did not and say when
            `spacing=` has discarded one (review R2-L3). The defaults are no longer the same value; the
            grid they draw still is, and the grid is what a caller moving between tiers is promised.

            So each side is read from what that tier actually drew rather than from its signature, which
            is a stronger claim than the one it replaces: either tier moving its default fails this,
            including by resolving `None` to something other than 30.
        """
        from digitalearth.static import Map

        drawn = InteractiveMap().graticule()
        interactive_step = int(drawn.layers[-1].data.name.rsplit("_", 1)[-1])
        built = Map(crs=4326)
        try:
            built.graticule(name="grid")
            props = built.figure_spec.layers.get("grid").symbology.props
        finally:
            built.close()
        assert (props["lon_step"], props["lat_step"]) == (
            interactive_step,
            interactive_step,
        ), (props, interactive_step)

    def test_only_the_static_tier_defaults_its_steps_to_a_sentinel(self):
        """The divergence itself, pinned exactly, so it can neither widen nor close unnoticed.

        Test scenario:
            The two tiers spell "the caller named no step" differently, and only one of them needs to.
            The static tier reports a discarded step, so it has to see that a step was written; the
            interactive tier makes no such report, so it keeps the literal. Stating that here — rather
            than deleting the comparison that caught it — is what keeps the divergence tracked, the same
            shape as `KEYWORD_SHORTFALLS` in `tests/test_contract_names.py`: the pair is compared
            exactly, so this fails the moment either tier changes its mind, including the day the
            interactive tier grows the same report and takes `None` too.

            It would grow it for a reason: `InteractiveMap.graticule` discards `lon_step`/`lat_step`
            under `spacing=` exactly as the static tier did, and says nothing. That is the same defect on
            the other tier, and it is out of R2-L3's scope, which names the static module alone.
        """
        from digitalearth.static.maps.projection import DEFAULT_GRATICULE_STEP

        interactive, static = _graticule_parameters()
        sentinels = {
            name: (interactive[name].default, static[name].default)
            for name in ("lon_step", "lat_step")
        }
        assert sentinels == {
            "lon_step": (DEFAULT_GRATICULE_STEP, None),
            "lat_step": (DEFAULT_GRATICULE_STEP, None),
        }, sentinels

    def test_asymmetric_spacing_raises(self, m):
        """Natural Earth ships symmetric graticules only — say so, do not ignore the request."""
        with pytest.raises(ValueError, match="symmetric"):
            m.graticule(lon_step=10, lat_step=20)

    def test_unsupported_spacing_raises(self, m):
        with pytest.raises(ValueError, match="exist only at"):
            m.graticule(lon_step=7, lat_step=7)

    def test_style_opts_still_forwarded_alongside_spacing(self, m):
        m.graticule(lon_step=10, lat_step=10, line_width=0.5)
        assert isinstance(m.layers[-1], gv.element.Feature)
        assert m.layers[-1].data.name == "graticules_10"

    def test_chains(self, m):
        assert m.graticule(lon_step=15, lat_step=15) is m


class _RangesTo:
    """An object that answers ``range`` with fixed edges, the way a caller's own element may.

    ``_element_extent`` asks the element for its range rather than reading the data behind it, so an object
    that answers ``range`` is a supported input — the docstring says so. It is also the only way to reach the
    non-finite guard: HoloViews drops an infinity out of its own elements before ``range`` ever sees it, so a
    real ``hv.Scatter`` carrying ``inf`` reports a finite range instead.
    """

    def __init__(self, edges):
        """Store the edges to report.

        Args:
            edges: The ``(xmin, ymin, xmax, ymax)`` this element should claim to cover.
        """
        self._xmin, self._ymin, self._xmax, self._ymax = edges

    def range(self, dimension):
        """Return the low/high pair for one dimension, as a HoloViews element would.

        Args:
            dimension: ``0`` for x, ``1`` for y.

        Returns:
            The ``(low, high)`` pair for that dimension.
        """
        return (self._xmin, self._xmax) if dimension == 0 else (self._ymin, self._ymax)


class TestElementExtent:
    """``_element_extent`` — the rectangle one element covers, or ``None`` when it has no frame to give."""

    def test_a_finite_element_reports_its_four_edges(self):
        """A scatter with real coordinates answers ``(xmin, ymin, xmax, ymax)``.

        Test scenario:
            Two points spanning x 0..4 and y 1..9. Every edge is a different number, so a transposed or
            reordered pair could not pass this.
        """
        extent = _element_extent(hv.Scatter([(0.0, 1.0), (4.0, 9.0)]))
        assert extent == (0.0, 1.0, 4.0, 9.0), f"expected the four edges in order, got {extent}"

    def test_no_element_has_no_extent(self):
        """``None`` in place of an element answers ``None`` rather than raising.

        Test scenario:
            A panel with no layer to measure passes ``None``; a frame cannot be set from it.
        """
        assert _element_extent(None) is None, "None in place of an element must not be measured"

    def test_an_element_built_from_no_data_has_no_extent(self):
        """An empty element ranges to ``(None, None)``, which is not a rectangle.

        Test scenario:
            ``hv.Scatter([])`` — HoloViews answers ``None`` for both ends of both dimensions, so the
            edges are present as values but name no region.
        """
        assert _element_extent(hv.Scatter([])) is None, "an element built from no data must not yield a frame"

    @pytest.mark.parametrize(
        "edges, which",
        [
            ((0.0, 1.0, float("inf"), 9.0), "an infinite east edge"),
            ((float("-inf"), 1.0, 4.0, 9.0), "an infinite west edge"),
            ((0.0, 1.0, 4.0, float("nan")), "a NaN north edge"),
            ((0.0, float("nan"), 4.0, 9.0), "a NaN south edge"),
        ],
    )
    def test_a_non_finite_edge_has_no_extent(self, edges, which):
        """A range that is present but not finite is refused instead of reaching the frame.

        Args:
            edges: The ``(xmin, ymin, xmax, ymax)`` the element reports.
            which: Which edge is not finite, so a failure names it.

        Test scenario:
            ``inf`` and ``nan`` both have to be refused, and on either end of either dimension: each one
            becomes an axes limit that cannot be drawn, and ``nan`` additionally compares false against
            everything, so a bounds check downstream would silently pass it.
        """
        extent = _element_extent(_RangesTo(edges))
        assert extent is None, f"{which} must not yield a frame, got {extent}"
