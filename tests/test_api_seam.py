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

import pytest

from digitalearth import api as qp


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


def _points():
    """Return a small point ``FeatureCollection`` (the repo's shared points fixture)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


def _polygons():
    """Return a small polygon ``FeatureCollection`` (the points fixture, buffered)."""
    fc = _points()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


def _lines():
    """Return a small line ``FeatureCollection``, built inline so the test owns its geometry."""
    import geopandas as gpd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {"v": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


def _mixed():
    """Return a collection whose geometry types disagree (a point and a polygon)."""
    import pandas as pd

    return pd.concat([_points().iloc[:1], _polygons().iloc[:1]])


class TestOneInputToKindDecision:
    """DE-24 invariant 1 — ``_input_kind`` is the single input-type→kind decision every backend routes through.

    The ``isinstance`` ladder, the empty guard and the geometry check used to be written out once per backend.
    These pin the one decision they collapsed into, in the shared kind registry's vocabulary.
    """

    def test_a_raster_is_the_raster_kind(self, dataset):
        """A pyramids ``Dataset`` classifies as ``"raster"``.

        Args:
            dataset: The raster fixture.
        """
        assert qp._input_kind(dataset, "quickmap") == "raster"

    @pytest.mark.parametrize(
        ("build", "expected"),
        [(_points, "points"), (_polygons, "polygons"), (_lines, "lines")],
    )
    def test_a_uniform_collection_classifies_by_its_geometry(self, build, expected):
        """A collection whose geometries agree classifies as that geometry's family.

        Args:
            build: Builds the collection under test.
            expected: The family it should be reported as.

        Test scenario:
            Points, polygons and lines are three distinct families now — the old ``_vector_kind`` only told
            polygons from "everything else", so a line collection read as points there. The distinction is
            what lets the 3-D tier refuse a line collection by name while the 2-D tiers still draw it.
        """
        assert qp._input_kind(build(), "quickmap") == expected

    def test_a_collection_of_disagreeing_geometries_is_named_mixed_not_guessed(self):
        """A point+polygon collection is ``"mixed"`` — no family is guessed for it.

        Test scenario:
            The 3-D tier needs to tell a genuinely mixed collection from a uniform one, because it refuses
            the first with a message listing the geometry types. Folding mixed into "points" (as the 2-D
            tiers do downstream) would have lost that.
        """
        assert qp._input_kind(_mixed(), "quickplot") == "mixed"

    @pytest.mark.parametrize("caller", ["quickmap", "quickplot"])
    def test_an_empty_collection_is_refused_naming_the_caller(self, caller):
        """An empty collection raises, and the message names the caller it was asked for.

        Args:
            caller: The public function name threaded into the refusal.

        Test scenario:
            The matplotlib path spells this ``quickmap`` and the other three ``quickplot``; the shared
            decision must carry whichever it was handed, or the refusal names the wrong entry point.
        """
        empty = _points().iloc[0:0]
        with pytest.raises(
            ValueError, match=rf"{caller} got an empty FeatureCollection"
        ):
            qp._input_kind(empty, caller)

    def test_an_unsupported_type_is_refused_naming_the_caller(self):
        """Neither a ``Dataset`` nor a ``FeatureCollection`` raises ``TypeError`` naming the caller."""
        with pytest.raises(TypeError, match="quickmap cannot draw a str"):
            qp._input_kind("not data", "quickmap")

    def test_the_uniform_families_are_the_shared_registry_s_data_families(self):
        """The families ``_input_kind`` reports for a uniform input are the kind registry's data families.

        Test scenario:
            The dispatch is meant to speak the registry's vocabulary, not a private one — so every uniform
            family it can return must be a member of :data:`~digitalearth.base.registry.KIND_TAKES`. A
            renamed family here that the registry does not know goes red.
        """
        from digitalearth.base.registry import KIND_TAKES

        assert {"raster", "points", "lines", "polygons"} <= set(KIND_TAKES), sorted(
            KIND_TAKES
        )


class _RecordingScene:
    """A scene stand-in that records which builder the dispatch reached, without any engine.

    Only the builder names the 2-D dispatch can call are provided; a family routed to a method not here
    raises ``AttributeError``, which is itself a useful failure.
    """

    def __init__(self):
        #: The builder names reached, in order.
        self.calls = []

    def field(self, data, **kwargs):
        """Record a raster field draw."""
        self.calls.append("field")

    def points(self, data, **kwargs):
        """Record a marker draw."""
        self.calls.append("points")

    def polygons(self, data, **kwargs):
        """Record an outline-polygon draw."""
        self.calls.append("polygons")

    def choropleth(self, data, column, **kwargs):
        """Record a choropleth draw."""
        self.calls.append("choropleth")


class TestTheMatplotlibDispatchReachesTheRightBuilder:
    """Each family the one decision reports reaches this tier's own builder — proven without matplotlib.

    ``_draw`` is the matplotlib backend's drawing, keyed by :func:`~digitalearth.api._input_kind`. A recording
    stand-in is enough to prove the routing; the actual rendering is the tier suites' job (run in CI).
    """

    def test_a_raster_reaches_field(self, dataset):
        """A ``Dataset`` is drawn with ``field`` (the ``"auto"`` raster renderer).

        Args:
            dataset: The raster fixture.
        """
        scene = _RecordingScene()
        qp._draw(scene, dataset, "auto")
        assert scene.calls == ["field"], scene.calls

    def test_polygons_without_a_column_reach_polygons(self):
        """A polygon collection with no ``column`` is drawn as outlines."""
        scene = _RecordingScene()
        qp._draw(scene, _polygons(), "auto")
        assert scene.calls == ["polygons"], scene.calls

    def test_polygons_with_a_column_reach_choropleth(self):
        """A ``column`` turns the polygon draw into a choropleth."""
        scene = _RecordingScene()
        qp._draw(scene, _polygons(), "auto", column="fid")
        assert scene.calls == ["choropleth"], scene.calls

    def test_points_reach_points(self):
        """A point collection is drawn as a marker map."""
        scene = _RecordingScene()
        qp._draw(scene, _points(), "auto")
        assert scene.calls == ["points"], scene.calls

    def test_a_line_collection_is_drawn_as_points_on_this_tier(self):
        """Lines reach ``points`` here, as this tier has always drawn every non-polygon vector.

        Test scenario:
            The one decision now distinguishes lines from points, but the matplotlib tier still has no line
            builder in ``quickmap``'s dispatch, so it maps the line family onto its marker draw — the
            behaviour ``_vector_kind`` gave when it lumped lines in with points. The 3-D tier reads the same
            family and refuses it; that difference is each backend supplying its own drawing.
        """
        scene = _RecordingScene()
        qp._draw(scene, _lines(), "auto")
        assert scene.calls == ["points"], scene.calls

    def test_a_column_on_points_is_refused_before_any_builder(self):
        """A ``column`` on point input is refused by name rather than forwarded into ``points``.

        Test scenario:
            ``Map.points`` sizes markers by ``size_column`` and has no fill column, so ``column`` there was
            an opaque cleopatra error before it was named (review M19). The refusal must still happen on the
            unified path, and before the builder is reached.
        """
        scene = _RecordingScene()
        # `_points()` built above the block, leaving only the refused `_draw` call inside it — a fixture that
        # raised here would otherwise satisfy `pytest.raises` while `_draw` never ran (tree guard
        # `test_refusal_blocks.py`).
        points = _points()
        with pytest.raises(ValueError, match="fills polygons"):
            qp._draw(scene, points, "auto", column="fid")
        assert scene.calls == [], scene.calls


class TestOneKeySeam:
    """DE-24 invariant 3 — the four per-tier colour-key impls are reached through one seam, ``_add_key``.

    The four ``_quickmap_*`` builders used to each name their own key impl at their own call site. They now
    all go through ``_add_key(backend, scene, visible=)``, which dispatches on :data:`~digitalearth.api.
    _KEY_ADDERS`. Each impl stays per-tier; only the call point is shared.
    """

    def test_the_seam_maps_each_backend_to_its_own_key_impl(self):
        """``_KEY_ADDERS`` points each backend at its own tier's key impl, not another's.

        Test scenario:
            The seam is only correct if the table wires each backend to the impl that speaks that engine's
            key; a swapped entry would key a web map through matplotlib's colorbar. Asserted per entry by
            identity so a swap fails loudly.
        """
        assert qp._KEY_ADDERS["matplotlib"] is qp._add_static_key
        assert qp._KEY_ADDERS["interactive"] is qp._add_interactive_key
        assert qp._KEY_ADDERS["web"] is qp._add_web_legend
        assert qp._KEY_ADDERS["3d"] is qp._add_3d_key

    def test_every_dispatchable_backend_has_a_key_adder(self):
        """The seam covers exactly the backends ``quickmap`` dispatches to.

        Test scenario:
            The two tables are built independently — one from the key impls, one from the tier capability
            declarations — so a backend added to one and not the other would dispatch to a key adder that
            is not there (a ``KeyError`` inside ``_add_key``) or leave a backend unkeyed.
        """
        assert set(qp._KEY_ADDERS) == set(qp.BACKEND_CAPABILITIES), sorted(
            set(qp._KEY_ADDERS).symmetric_difference(qp.BACKEND_CAPABILITIES)
        )

    @pytest.mark.parametrize("backend", ["matplotlib", "interactive", "web", "3d"])
    def test_add_key_calls_the_backend_s_impl_carrying_the_flag(
        self, backend, monkeypatch
    ):
        """``_add_key`` invokes the table's impl for the backend, passing ``visible`` through.

        Args:
            backend: The backend whose key adder is exercised.
            monkeypatch: Swaps in a recording impl for that backend only.

        Test scenario:
            The seam has to reach the *named* backend's impl and carry the ``colorbar=`` decision as
            ``visible=``. Both are asserted — a seam that dropped the flag, or dispatched to a fixed impl,
            would go red here.
        """
        recorded = []

        def _spy(scene, *, visible):
            recorded.append((scene, visible))

        monkeypatch.setitem(qp._KEY_ADDERS, backend, _spy)
        sentinel = object()
        assert qp._add_key(backend, sentinel, visible=False) is None
        assert recorded == [(sentinel, False)], recorded


class TestLazyBackendMap:
    """The backend table defers its matplotlib row yet resolves eager rows and refuses unknown ones."""

    def test_eager_resolves_directly_and_a_lazy_row_builds_once(self):
        """An eager row returns its stored value; a lazy row runs its builder once and caches the result.

        Test scenario:
            The matplotlib row is the only deferred one, so this pins the class's contract on a fake table:
            the eager row needs no builder, and reading a lazy row twice must call its builder a single time
            (a rebuild on every read would re-import the matplotlib tier on every quickmap).
        """
        builds = []

        def _build():
            builds.append(1)
            return "L"

        table = qp._LazyBackendMap(eager={"e": "E"}, lazy={"l": _build})
        reads = [table["e"], table["l"], table["l"]]
        assert reads == ["E", "L", "L"], f"unexpected resolutions: {reads}"
        assert builds == [1], (
            f"the lazy builder must run once and cache, ran {len(builds)}x"
        )

    def test_an_unknown_backend_raises_keyerror_naming_it(self):
        """A key in neither the eager nor the lazy table raises KeyError, not a silently-built row.

        Test scenario:
            Membership and iteration name only the declared backends, so a read of anything else is a
            programming error the table surfaces rather than a row it fabricates.
        """
        table = qp._LazyBackendMap(eager={"e": "E"}, lazy={"l": lambda: "L"})
        with pytest.raises(KeyError, match="nope"):
            table["nope"]


class TestThe3DTierRefusesGeometryItCannotDraw:
    """backend='3d' names the families it has no builder for, before a VTK plotter is ever opened (DE-24)."""

    @pytest.mark.parametrize("build", [_lines, _mixed])
    def test_a_line_or_mixed_collection_is_refused_by_type(self, build):
        """A line or geometrically-mixed collection is refused with a TypeError listing its geometry.

        Args:
            build: Builds the unsupported collection under test.

        Test scenario:
            The 2-D tiers draw these as markers, but the 3-D tier has only point/polygon/raster builders, so
            the shared decision names the family and this tier refuses it in its own words — and does so
            before Scene3D constructs a plotter, which is why the refusal is reachable without the 3-D engine.
        """
        data = build()
        with pytest.raises(
            TypeError, match="needs a uniformly point, polygon, or raster"
        ):
            qp._quickmap_3d(data)
