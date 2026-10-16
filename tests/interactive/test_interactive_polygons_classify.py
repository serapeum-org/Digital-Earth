"""#331 — InteractiveMap.polygons() classifies by scheme/k, the way choropleth() and the web tier already do.

The Core contract declares ``scheme`` and ``k`` on ``polygons``; the web tier's ``polygons`` classifies and
this tier's ``choropleth`` classifies, but this tier's ``polygons`` had neither keyword nor a code path. A
caller who wanted graduated polygon fills had to route through ``choropleth``. These tests pin the new
capability and that it cuts a column into exactly the breaks ``choropleth`` does, so the two builders cannot
drift.
"""

import inspect

import pytest

from digitalearth.interactive import InteractiveMap


@pytest.fixture
def polygons_gdf():
    """Four unit squares carrying a numeric ``val`` and a categorical ``cls`` column (EPSG:4326)."""
    pytest.importorskip("geoviews")
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {
            "val": [1.0, 5.0, 9.0, 13.0],
            "cls": ["a", "b", "a", "c"],
            "geometry": [box(i, 0.0, i + 1.0, 1.0) for i in range(4)],
        },
        crs="EPSG:4326",
    )


class TestPolygonsClassifies:
    """``polygons()`` gained ``scheme=``/``k=`` (#331); it must cut the column the way ``choropleth()`` does."""

    def test_the_signature_now_takes_scheme_and_k(self):
        """The capability is real — both keywords are on the signature, so the #331 gap is closed.

        Test scenario:
            #331 is a missing capability, not a spelling: the allowlist row vouched for ``polygons`` lacking
            ``scheme``/``k`` entirely. With the row gone, the signature itself must carry them.
        """
        params = set(inspect.signature(InteractiveMap.polygons).parameters)
        assert {"scheme", "k"} <= params, sorted(params)

    def test_graduated_polygons_match_choropleth_breaks(self, polygons_gdf):
        """``polygons(column, scheme, k)`` cuts the same class edges ``choropleth(column, scheme, k)`` does.

        Test scenario:
            Both route through the one classifier, so the breaks published on ``last_breaks`` are identical —
            the parity #331 asks for, tying ``polygons`` to the ``choropleth`` the cross-tier conformance
            probe already holds to a shared constant.
        """
        via_polygons = InteractiveMap().polygons(
            polygons_gdf, column="val", scheme="quantiles", k=3
        )
        via_choropleth = InteractiveMap().choropleth(
            polygons_gdf, "val", scheme="quantiles", k=3
        )
        assert via_polygons.last_breaks == via_choropleth.last_breaks
        assert via_polygons.last_breaks, (
            "a graduated scheme must publish its class edges"
        )

    def test_categorical_polygons_match_choropleth_categories(self, polygons_gdf):
        """``scheme="categorical"`` keys the distinct values identically through either builder (#331)."""
        via_polygons = InteractiveMap().polygons(
            polygons_gdf, column="cls", scheme="categorical"
        )
        via_choropleth = InteractiveMap().choropleth(
            polygons_gdf, "cls", scheme="categorical"
        )
        assert via_polygons.last_breaks == via_choropleth.last_breaks

    def test_a_scheme_without_a_column_is_refused(self, polygons_gdf):
        """A scheme styles a column, so naming one without a column is refused, not ignored (as ``points``).

        Test scenario:
            A dropped styling request is exactly what this tier stopped doing elsewhere; ``polygons`` must
            refuse ``scheme=`` with no ``column=`` rather than drawing plain outlines in silence.
        """
        m = InteractiveMap()
        with pytest.raises(ValueError, match="scheme= classifies column="):
            m.polygons(polygons_gdf, scheme="quantiles")

    def test_outlines_without_a_scheme_still_build(self, polygons_gdf):
        """The default path is untouched: no scheme draws a plain polygon layer, which still registers."""
        m = InteractiveMap().polygons(polygons_gdf)
        assert m.layers, "a plain polygon layer must still register"

    def test_rasterize_true_with_a_scheme_warns_at_the_caller(self, polygons_gdf):
        """``rasterize=True`` with a scheme is dropped (a classified layer draws raw glyphs), so it warns.

        Test scenario:
            A classified layer is never datashaded, so forcing ``rasterize=True`` is a dropped request —
            named rather than discarded, like ``graticule()``'s over-specification warning. The warning must
            point at THIS caller, not the ``@_skips_off_limb`` wrapper, so its ``filename`` is checked (the
            builder is decorated, so the warn uses ``stacklevel=3``); the layer still classifies.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = InteractiveMap().polygons(
                polygons_gdf, column="val", scheme="quantiles", k=3, rasterize=True
            )
        dropped = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "rasterize=True" in str(w.message)
        ]
        assert dropped, [str(w.message) for w in caught]
        assert dropped[0].filename.endswith("test_interactive_polygons_classify.py"), (
            f"the warning must point at the caller, not the wrapper: {dropped[0].filename}"
        )
        assert m.last_breaks, (
            "the layer must still classify despite the ignored rasterize=True"
        )

    def test_big_data_threshold_with_a_scheme_warns(self, polygons_gdf):
        """An explicit ``big_data_threshold=`` with a scheme is dropped too — a classified layer is not routed."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            InteractiveMap().polygons(
                polygons_gdf,
                column="val",
                scheme="quantiles",
                k=3,
                big_data_threshold=1,
            )
        assert [w for w in caught if "big_data_threshold" in str(w.message)], [
            str(w.message) for w in caught
        ]

    def test_rasterize_false_with_a_scheme_is_silent(self, polygons_gdf):
        """``rasterize=False`` asks for raw glyphs, exactly what a classified layer draws, so it does not warn.

        Test scenario:
            Warning that ``rasterize=False`` was "dropped" would contradict the ImportError hint that tells
            callers to pass ``rasterize=False`` for raw glyphs; a classified layer already draws them, so the
            request is honoured, not dropped.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            InteractiveMap().polygons(
                polygons_gdf, column="val", scheme="quantiles", k=3, rasterize=False
            )
        assert [str(w.message) for w in caught if "rasterize" in str(w.message)] == []

    def test_auto_rasterize_with_a_scheme_is_silent(self, polygons_gdf):
        """The default ``rasterize="auto"`` names no datashading request, so a classified call stays silent."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            InteractiveMap().polygons(
                polygons_gdf, column="val", scheme="quantiles", k=3
            )
        assert [str(w.message) for w in caught if "rasterize" in str(w.message)] == []

    def test_a_scheme_with_a_missing_column_is_named(self, polygons_gdf):
        """A scheme naming an absent column raises a named ``KeyError``, as ``choropleth`` does — not a bare one.

        Test scenario:
            Without the check the classifier reaches into the frame and surfaces a bare pandas ``KeyError``;
            ``polygons`` names the column and the builder instead, matching ``choropleth``.
        """
        m = InteractiveMap()
        with pytest.raises(KeyError, match="polygons column 'nope' not found"):
            m.polygons(polygons_gdf, column="nope", scheme="quantiles")
