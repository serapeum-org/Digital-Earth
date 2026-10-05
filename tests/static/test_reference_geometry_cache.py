"""The reference geography is projected once per CRS, in one PROJ call (ST-15).

Two costs, both measured on this branch's parent (7e26ca42), on an orthographic globe with the 110m
Natural-Earth layers:

```
project_line_features(coastline)    0.7254s  0.7487s  0.6667s   (134 parts, 5128 vertices -> 76 segments)
project_polygon_features(land)      0.6887s  0.7563s  0.7959s   (127 rings, 7600 vertices -> 70 rings)
```

Every draw paid it again, and nearly all of it was per-*call* PROJ overhead rather than the transform:

```
per-part reproject (134 calls)            0.6749s
batched reproject  (1 call, 5128 pts)     0.0090s
split_finite over all parts               0.0020s
```

So the row is two things — one PROJ call per layer instead of one per part, and a cache so a second draw of
the same layer into the same CRS pays neither. The claims here are **call counts and object identity**,
because a timing is not a claim a test can make reliably; the timings live in the commit that made them.

The pre-existing ``lru_cache`` in ``decoration.py`` covers neither: it is
``_families_matplotlib_has(len(fontManager.ttflist))``, matplotlib's registered *font families*. Executed on
the parent: ``hasattr(cleopatra.basemap.reference.natural_earth, "cache_info")`` is ``False``, and the only
``lru_cache`` in the module is that one.
"""

import numpy as np
import pytest
from cleopatra.basemap.reference import natural_earth
from pyramids.base.crs import reproject_coordinates

from digitalearth.static import Map, projections
from digitalearth.static.maps import decoration

#: The layer every test here draws. 110m is the coarsest Natural-Earth resolution and still 134 parts, which
#: is the point: the cost measured above is per part, not per vertex.
RESOLUTION = "110m"

#: An orthographic globe, so the reference geometry really is projected here (a flat map hands the job to
#: cleopatra's ``add_features``, which reprojects for itself).
GLOBE = projections.orthographic(20, 10)

#: A second globe CRS, to show the cache keys on the display CRS rather than on the layer alone.
OTHER_GLOBE = projections.orthographic(-9, 39)


@pytest.fixture(autouse=True)
def empty_cache():
    """Start and leave every test here with an empty projected-reference cache.

    The cache is process-wide by design — two maps in one CRS should share it — so a test that measured a
    miss would otherwise depend on which test ran before it.

    Yields:
        None.
    """
    decoration.PROJECTED_REFERENCE.clear()
    yield
    decoration.PROJECTED_REFERENCE.clear()


@pytest.fixture
def globe():
    """Yield an orthographic globe with its projection frame already computed.

    The frame (the convex hull of the projected sphere) is what ``_project_polygon_features`` re-closes
    rings against, and computing it is expensive for its own reasons (#381) — warming it here keeps it out
    of what these tests measure.

    Yields:
        A :class:`~digitalearth.static.map.Map` on a globe frame.
    """
    scene = Map(crs=GLOBE, globe=True)
    scene._frame()
    yield scene
    scene.close()


def _part_by_part(crs, parts, step_deg=None):
    """Project lon/lat parts one call at a time — the behaviour that has to be preserved.

    This is the shape the two projectors had before they were batched, written out here so the batched
    result is compared against an independent computation rather than against itself.

    Args:
        crs: The display CRS to project into.
        parts: The raw ``(N, 2)`` lon/lat arrays Natural Earth hands over.
        step_deg: Densify each part to this spacing first, as the polygon path does; ``None`` to leave the
            vertices as they came, as the line path does.

    Returns:
        The projected x/y of each part, in order, as a list of ``(M, 2)`` arrays.
    """
    out = []
    for part in parts:
        xy = np.asarray(part, dtype=float)
        if xy.size == 0:
            continue
        if step_deg is not None:
            xy = projections.densify_lonlat(xy, step_deg=step_deg)
        x, y = reproject_coordinates(
            xy[:, 0].tolist(), xy[:, 1].tolist(), from_crs=4326, to_crs=crs
        )
        out.append(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    return out


class TestOnePROJCallPerLayer:
    """134 calls and one call transform the same points; only one of them costs 0.67 s."""

    def test_a_line_layer_is_projected_in_one_call(self, globe, mocker):
        """However many parts the layer has, PROJ is asked once.

        Args:
            globe: The map being drawn on.
            mocker: To count the calls.

        Test scenario:
            The 110m coastline is 134 parts. The count is the structural half of the measurement in this
            module's docstring: it is what makes the 75-fold difference, and unlike a duration it is the
            same on every machine.
        """
        counted = mocker.spy(decoration, "reproject_coordinates")
        globe._project_line_features(natural_earth("coastline", RESOLUTION))
        assert counted.call_count == 1, counted.call_count

    def test_a_polygon_layer_is_projected_in_one_call(self, globe, mocker):
        """The fill path densifies before projecting, and still asks PROJ once.

        Args:
            globe: The map being drawn on.
            mocker: To count the calls.
        """
        counted = mocker.spy(decoration, "reproject_coordinates")
        globe._project_polygon_features(natural_earth("land", RESOLUTION))
        assert counted.call_count == 1, counted.call_count

    def test_the_batched_segments_are_what_part_by_part_projection_gave(self, globe):
        """Batching is a change of cost, not of geometry.

        Args:
            globe: The map being drawn on.

        Test scenario:
            The expected segments are built by :func:`_part_by_part`, which is the loop the projector used
            to be, and split with the same splitter — so what is compared is the projection call pattern
            and nothing else.
        """
        parts = natural_earth("coastline", RESOLUTION)
        expected = []
        for projected in _part_by_part(GLOBE, parts):
            expected += projections._split_finite(projected[:, 0], projected[:, 1])
        drawn = globe._project_line_features(parts)
        assert len(drawn) == len(expected), (len(drawn), len(expected))
        assert all(
            np.allclose(left, right, equal_nan=True)
            for left, right in zip(drawn, expected)
        ), "a batched segment differs from the part-by-part one"

    def test_the_batched_rings_are_what_part_by_part_projection_gave(self, globe):
        """The same for the fill path, densification included.

        Args:
            globe: The map being drawn on.
        """
        parts = natural_earth("land", RESOLUTION)
        boundary = globe._frame()[0]
        expected = []
        for projected in _part_by_part(GLOBE, parts, step_deg=1.0):
            expected += projections.close_visible_runs(
                projected[:, 0], projected[:, 1], boundary
            )
        drawn = globe._project_polygon_features(parts)
        assert len(drawn) == len(expected), (len(drawn), len(expected))
        assert all(
            np.allclose(left, right, equal_nan=True)
            for left, right in zip(drawn, expected)
        ), "a batched ring differs from the part-by-part one"

    def test_a_layer_with_nothing_in_it_asks_proj_nothing(self, globe, mocker):
        """An empty layer is not a reason to call PROJ with an empty array.

        Args:
            globe: The map being drawn on.
            mocker: To count the calls.
        """
        counted = mocker.spy(decoration, "reproject_coordinates")
        drawn = globe._project_line_features([np.empty((0, 2))])
        assert drawn == [], drawn
        assert counted.call_count == 0, counted.call_count


class TestTheProjectedGeometryIsKept:
    """Reference geography is the same every draw, and so is its projection into one display CRS."""

    def test_a_second_draw_of_the_same_layer_projects_nothing(self, mocker):
        """The whole point: a redraw — an animation frame — pays no reprojection at all.

        Args:
            mocker: To count the calls.

        Test scenario:
            Two separate maps in the same display CRS, because the cache is keyed on the CRS and not on the
            scene — which is what makes an animation's frames, and a second figure, free.
        """
        first = Map(crs=GLOBE, globe=True)
        first.coastlines(resolution=RESOLUTION)
        first.close()
        second = Map(crs=GLOBE, globe=True)
        counted = mocker.spy(decoration, "reproject_coordinates")
        second.coastlines(resolution=RESOLUTION)
        second.close()
        assert counted.call_count == 0, counted.call_count

    def test_a_hit_hands_back_the_very_geometry_it_kept(self, globe):
        """Identity, not equality: an equal result could have been recomputed.

        Args:
            globe: The map being drawn on.
        """
        held = decoration.PROJECTED_REFERENCE.projected(globe, "coastline", RESOLUTION)
        again = decoration.PROJECTED_REFERENCE.projected(globe, "coastline", RESOLUTION)
        assert again is held, "the second ask recomputed the geometry"

    def test_the_kept_geometry_cannot_be_written_through(self, globe):
        """A shared cache that hands out writable arrays is a cache a drawer can corrupt.

        Args:
            globe: The map being drawn on.
        """
        held = decoration.PROJECTED_REFERENCE.projected(globe, "coastline", RESOLUTION)
        with pytest.raises(ValueError, match="read-only"):
            held[0][0, 0] = 0.0

    def test_another_display_crs_is_another_entry(self, globe):
        """The same layer projected into a second CRS is different geometry, and is projected again.

        Args:
            globe: The map being drawn on.

        Test scenario:
            A rotation swaps the display CRS between animation frames, so a cache keyed on the layer alone
            would hand the second frame the first frame's projection — the whole globe drawn in the wrong
            place.
        """
        elsewhere = Map(crs=OTHER_GLOBE, globe=True)
        here = decoration.PROJECTED_REFERENCE.projected(globe, "coastline", RESOLUTION)
        there = decoration.PROJECTED_REFERENCE.projected(
            elsewhere, "coastline", RESOLUTION
        )
        elsewhere.close()
        assert there is not here, "two display CRSs shared one projection"

    def test_a_line_layer_and_a_fill_of_the_same_name_are_two_entries(self, globe):
        """``land`` as lines and ``land`` as rings are different geometry under one name.

        Args:
            globe: The map being drawn on.
        """
        lines = decoration.PROJECTED_REFERENCE.projected(globe, "land", RESOLUTION)
        rings = decoration.PROJECTED_REFERENCE.projected(
            globe, "land", RESOLUTION, polygon=True
        )
        assert rings is not lines, "a fill was handed the line layer's projection"

    def test_the_cache_holds_a_bounded_number_of_entries(self):
        """A 10m layer is megabytes, so the cache has to forget the oldest rather than grow.

        Test scenario:
            One more display CRS than the cache keeps, each asked for the same layer. Nothing is asserted
            about *which* entry went — only that the cache stopped growing, which is what keeps a long
            rotation from holding every frame's projection at once.
        """
        keep = decoration.PROJECTED_REFERENCE.keep
        for lon in range(keep + 1):
            scene = Map(crs=projections.orthographic(float(lon * 10), 0.0), globe=True)
            decoration.PROJECTED_REFERENCE.projected(scene, "coastline", RESOLUTION)
            scene.close()
        assert len(decoration.PROJECTED_REFERENCE) == keep, len(
            decoration.PROJECTED_REFERENCE
        )

    def test_geometry_that_changed_under_one_name_is_not_served_from_the_cache(
        self, globe, mocker
    ):
        """A cache keyed on the layer's *name* alone would hand back the geometry it no longer has.

        Args:
            globe: The map being drawn on.
            mocker: To stand a different coastline in place of Natural Earth's.

        Test scenario:
            The case that forced the key to carry a fingerprint of the raw parts: ``test_globe.py`` stubs
            ``decoration.natural_earth`` in several tests, and the cache is process-wide, so an entry under
            ``("coastline", "110m", …)`` alone would serve the stub's geometry to the next real draw —
            cross-test contamination out of a performance change. The read is remade every draw (5 ms
            against the 730 ms this removes) precisely so this can be checked.
        """
        real = decoration.PROJECTED_REFERENCE.projected(globe, "coastline", RESOLUTION)
        mocker.patch.object(
            decoration,
            "natural_earth",
            return_value=[np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])],
        )
        stubbed = decoration.PROJECTED_REFERENCE.projected(
            globe, "coastline", RESOLUTION
        )
        assert stubbed is not real, "the stubbed read was served the real geometry"
        assert len(stubbed) == 1, stubbed

    def test_a_globe_fill_draws_from_the_kept_rings(self):
        """The rings are handed straight to a ``PolyCollection``, so they have to be drawable as kept.

        Test scenario:
            The cached arrays are read-only, which a collection that wanted to write through them would
            refuse — so the fill path is drawn here rather than assumed, twice over so the second draw is
            the one reading from the cache.
        """
        first = Map(crs=GLOBE, globe=True)
        first.land(resolution=RESOLUTION)
        first.close()
        second = Map(crs=GLOBE, globe=True)
        second.land(resolution=RESOLUTION, name="fill")
        paths = len(second.artist("fill").get_paths())
        second.close()
        assert paths > 0, "the globe fill drew no polygons from the kept rings"


class TestTheFingerprintOfAGeometryWithNothingInIt:
    """The cache key reads the first and last coordinate of the geometry — which may not exist."""

    def test_a_layer_whose_every_part_is_empty_fingerprints_to_nothing(self):
        """A reference layer with no coordinate at all has an empty fingerprint, not an `IndexError`.

        Test scenario:
            The fingerprint exists so that geometry changed under one name is not served from the cache,
            and it reads `drawn[0][0]` and `drawn[-1][-1]`. A layer whose parts are all empty — a
            resolution that clipped everything away, or an upstream read that answered nothing — has no
            such coordinate, so the reader stops at the count instead of indexing into nothing.
        """
        fingerprint = decoration._ProjectedReference._fingerprint(
            [np.empty((0, 2)), np.empty((0, 2))]
        )
        assert fingerprint == (), (
            f"an all-empty layer should fingerprint to nothing; got {fingerprint!r}"
        )

    def test_a_layer_with_one_coordinate_still_fingerprints_to_something(self):
        """One point is enough, so the empty answer above is about emptiness and not about shape.

        Test scenario:
            The pair of tests is what pins where the line is. A reader that returned `()` for any layer
            would pass the case above and lose the cache's only protection against stale geometry.
        """
        fingerprint = decoration._ProjectedReference._fingerprint(
            [np.array([[1.0, 2.0]]), np.empty((0, 2))]
        )
        assert fingerprint == ((1, 0), (1.0, 2.0), (1.0, 2.0)), (
            f"one coordinate should fingerprint to its counts and ends; got {fingerprint!r}"
        )
