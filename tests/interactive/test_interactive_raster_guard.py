"""#175 — every interactive vector builder rejects a pyramids raster, naming the builder that was called.

A raster handed to a vector builder used to die on whatever expression the builder ran first — ``has no
len()`` from ``points``/``polygons``, ``argument of type 'int' is not iterable`` from ``choropleth`` (#152's
exact message) — never with a hint that the method wanted a vector. The guard now lives in
``digitalearth.base.guards`` and is shared with the web tier (which fixed the same failure under #163); this
covers the interactive half.
"""

import pytest

from digitalearth.interactive import InteractiveMap

# (method, kwargs, named): `named` is the builder the guard quotes — the method's own name, except `flow`,
# a thin alias that routes its rejection through `graph`.
_BUILDERS = [
    ("points", {}, "points"),
    ("lines", {}, "lines"),
    ("polygons", {}, "polygons"),
    ("choropleth", {"column": "fid"}, "choropleth"),
    ("hexbin", {}, "hexbin"),
    ("kde", {}, "kde"),
    ("graph", {"edges": []}, "graph"),
    ("flow", {"edges": []}, "graph"),
    ("trimesh", {}, "trimesh"),
    ("rasterize", {}, "rasterize"),
    ("datashade", {}, "datashade"),
    ("trajectory", {}, "trajectory"),
]


class TestEveryVectorBuilderRejectsARaster:
    """A pyramids raster is turned away by every vector builder, naming the builder and the raster route."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("geoviews")

    @pytest.fixture
    def point_fc(self):
        """A point ``FeatureCollection`` — valid vector input for the point-taking builders."""
        from pyramids.feature import FeatureCollection

        return FeatureCollection.read_file("tests/data/points.geojson")

    @pytest.mark.parametrize(
        "method, kwargs, named", _BUILDERS, ids=[builder[0] for builder in _BUILDERS]
    )
    def test_raster_is_rejected_by_name(self, dataset, method, kwargs, named):
        """Each builder rejects a ``Dataset`` with a message naming it and the tier's raster entry points.

        Test scenario:
            Some builders touch the input at build time (``points``/``polygons``/``choropleth``/``trimesh``)
            and reject on the call; the rest record their source and reject when the layer is drawn. Either
            way the message quotes the builder and routes the caller to ``field()``/``timecube()`` instead of
            leaking the incidental ``has no len()`` / ``argument of type 'int'`` failure.
        """
        m = InteractiveMap()
        with pytest.raises(TypeError) as excinfo:
            built = getattr(m, method)(dataset, **kwargs)
            built.render()
        message = str(excinfo.value)
        assert message.startswith(f"{named}()"), (
            f"{method} did not name the builder: {message}"
        )
        assert "does not take a raster" in message, message
        assert "argument of type" not in message, (
            f"{method} still leaks #152's incidental error: {message}"
        )

    def test_the_quoted_raster_entry_points_are_real_methods(self, dataset):
        """The message routes to ``field()``/``timecube()``, both real ``InteractiveMap`` methods (#175 DoD).

        Test scenario:
            An invented method name in the guidance would mislead the caller, so each quoted entry point is
            asserted to exist — a wrong name fails this test rather than shipping.
        """
        with pytest.raises(TypeError) as excinfo:
            InteractiveMap().points(dataset)
        message = str(excinfo.value)
        for entry in ("field", "timecube"):
            assert f"{entry}()" in message, f"{entry} not named: {message}"
            assert hasattr(InteractiveMap, entry), (
                f"the guard quotes {entry!r}, which is not a real InteractiveMap method"
            )

    def test_a_geometry_less_table_is_told_to_set_geometry(self):
        """A table whose active geometry was never set gets the set_geometry message, not raster advice.

        Test scenario:
            The guard separates a real table missing its active geometry from an actual raster, so each gets
            its own accurate instruction — the raster-vs-table split carried over from the web tier (#163).
        """
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import Point

        table = gpd.GeoDataFrame({"value": [1.0], "geom": [Point(0, 0)]})
        with pytest.raises(TypeError, match="set_geometry"):
            InteractiveMap().points(table)

    def test_trimesh_still_accepts_a_non_vector_mesh(self):
        """``trimesh`` takes a UGRID mesh (no ``geometry``), so it uses the raster-only guard, not the vector.

        Test scenario:
            A mesh exposes ``node_x``/``node_y`` rather than ``geometry``; :meth:`_require_vector` would
            wrongly reject it, so ``trimesh`` uses :meth:`_reject_raster`, which lets a non-raster object
            through (returns ``None``) while still refusing a raster.
        """
        mesh = type(
            "Mesh", (), {"node_x": (0.0, 1.0, 0.0), "node_y": (0.0, 0.0, 1.0)}
        )()
        assert InteractiveMap()._reject_raster(mesh, "trimesh") is None

    def test_a_valid_vector_layer_still_builds(self, point_fc):
        """The guard must not disturb the supported path: a point ``FeatureCollection`` still builds a layer.

        Test scenario:
            Valid vector input reaches each point-taking builder unchanged, so none of them may raise and
            each must register its layer.
        """
        for method in ("points", "hexbin", "kde"):
            m = getattr(InteractiveMap(), method)(point_fc)
            assert m.layers, f"{method} accepted the layer but registered nothing"
