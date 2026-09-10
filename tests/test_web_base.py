"""DW.0 — packaging + ``WebMap`` skeleton + display-CRS plumbing.

Covers the web tier's foundation: the lazy engine import (the package imports without the ``web`` extra;
engine-touching methods raise an actionable ``ImportError``), the layer registry, the render/save lifecycle
over the ``maplibre`` widget, the ``_to_display_source`` reproject-through-pyramids choke point, and the proof
that the DX.3 import guard now covers ``src/digitalearth/web/``.

Engine-dependent tests ``importorskip`` maplibre — they run in the ``web`` pixi env
(``pixi run -e web test-web``); everything else runs in the lean ``dev`` env too.
"""

import datetime
import importlib
import pathlib
import sys

import numpy as np
import pytest

from digitalearth.base.sources.source import Source
from digitalearth.web import WebMap
from digitalearth.web.base import _require_maplibre, _resolve_style
from tests.test_no_competitor_imports import (
    FORBIDDEN,
    _top_level_imports,
    test_tiers_import_no_gis_competitor,
)

_WEB_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "digitalearth" / "web"


class TestLazyImport:
    """The package and constructor work without the engine; builders fail actionably without it."""

    def test_package_imports_without_engine_loaded(self):
        """``import digitalearth.web`` must not itself import maplibre/lonboard.

        Only **module-top-level** statements count: the engine imports inside ``_require_maplibre``
        (and other methods) are exactly the lazy pattern the tier promises, so the guard's whole-tree
        AST walk would be the wrong tool here.
        """
        import ast

        engine = {"maplibre", "lonboard"}
        for mod in _WEB_ROOT.rglob("*.py"):
            tree = ast.parse(mod.read_text(encoding="utf-8"))
            top_level = set()
            for (
                node
            ) in tree.body:  # module body only — function-local imports stay lazy
                if isinstance(node, ast.Import):
                    top_level.update(alias.name.split(".")[0] for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                ):
                    top_level.add(node.module.split(".")[0])
            assert not top_level & engine, (
                f"{mod.name} imports the MapLibre engine at module top — the tier promises a lazy "
                "engine import (engine imports belong inside methods via _require_maplibre)"
            )

    def test_constructing_needs_no_engine(self):
        m = WebMap(center=(8.0, 47.0), zoom=5, style="light", crs=4326, height=300)
        assert m.layers == []
        assert (m.center, m.zoom, m.style, m.crs, m.height) == (
            (8.0, 47.0),
            5,
            "light",
            4326,
            300,
        )

    def test_missing_engine_raises_actionable_error(self, monkeypatch):
        """With maplibre unimportable, the lazy import points at the install command."""
        monkeypatch.setitem(
            sys.modules, "maplibre", None
        )  # makes `import maplibre` raise
        with pytest.raises(ImportError, match=r"digitalearth\[web\]"):
            _require_maplibre()

    def test_render_without_engine_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "maplibre", None)
        with pytest.raises(ImportError, match=r"digitalearth\[web\]"):
            WebMap().render()

    def test_repr_mimebundle_degrades_gracefully_without_engine(self, monkeypatch):
        """A bare repr in a notebook must not raise when the extra is missing."""
        monkeypatch.setitem(sys.modules, "maplibre", None)
        assert WebMap()._repr_mimebundle_() == {}


class TestImportGuardCoversWeb:
    """DX.3 negative fixture: the guard now polices ``web/``."""

    def test_temp_forbidden_import_fails_the_guard(self):
        """Dropping an xarray-importing module under ``web/`` makes the guard test fail."""
        bad = _WEB_ROOT / "_tmp_guard_negative_fixture.py"
        bad.write_text("import xarray\n", encoding="utf-8")
        try:
            with pytest.raises(AssertionError, match="xarray"):
                test_tiers_import_no_gis_competitor()
        finally:
            bad.unlink()
        test_tiers_import_no_gis_competitor()  # tree is clean again

    def test_web_modules_are_clean(self):
        offenders = {
            mod.name: sorted(FORBIDDEN & set(_top_level_imports(mod)))
            for mod in _WEB_ROOT.rglob("*.py")
            if FORBIDDEN & set(_top_level_imports(mod))
        }
        assert offenders == {}


class TestDisplaySource:
    """``_to_display_source`` — the single reproject-through-pyramids choke point."""

    def test_reprojects_to_display_crs(self, dataset):
        m = WebMap(crs=3857)
        assert dataset.epsg != 3857, (
            "fixture must start in a non-display CRS for this test"
        )
        src = m._to_display_source(dataset)
        assert src.crs == 3857

    def test_same_crs_passes_through_without_warp(self, dataset, monkeypatch):
        m = WebMap(crs=dataset.epsg)

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError(
                "to_crs must not be called when data is already in the display CRS"
            )

        monkeypatch.setattr(type(dataset), "to_crs", _boom)
        src = m._to_display_source(dataset)
        assert src.crs == dataset.epsg

    def test_source_passes_through_untouched(self):
        import numpy as np

        from digitalearth.base.sources.dimension import DimensionInfo

        src = Source(
            DimensionInfo(np.zeros((2, 2)), "z"),
            DimensionInfo(np.arange(2.0), "x"),
            DimensionInfo(np.arange(2.0), "y"),
            crs=4326,
        )
        assert WebMap(crs=3857)._to_display_source(src) is src


class TestRegistryAndRender:
    """The layer registry composes into one MapLibre widget (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_add_layer_chains(self):
        m = WebMap()
        out = m.add_layer("a").add_layer("b")
        assert out is m
        assert m.layers == ["a", "b"]

    def test_render_returns_maplibre_widget(self):
        from maplibre.ipywidget import MapWidget

        widget = WebMap(center=(0.0, 0.0), zoom=3).render()
        assert isinstance(widget, MapWidget)

    def test_empty_map_still_renders_a_widget(self):
        from maplibre.ipywidget import MapWidget

        assert isinstance(WebMap().render(), MapWidget)

    def test_height_lands_on_the_widget(self):
        widget = WebMap(height=321).render()
        # maplibre stores height as a CSS length string ("321px").
        assert str(widget.height).startswith("321")

    def test_save_html_writes_a_file(self, tmp_path):
        out = tmp_path / "m.html"
        assert WebMap(center=(0.0, 0.0), zoom=2).save(str(out)) == str(out)
        assert out.stat().st_size > 1_000
        assert "maplibre" in out.read_text(encoding="utf-8").lower()

    def test_show_returns_the_widget(self, monkeypatch):
        from maplibre.ipywidget import MapWidget

        shown = []
        import IPython.display

        monkeypatch.setattr(IPython.display, "display", shown.append)
        widget = WebMap().show()
        assert isinstance(widget, MapWidget)
        assert shown == [widget], (
            "show() must push the widget through IPython display once"
        )

    def test_repr_mimebundle_delegates_to_widget(self):
        bundle = WebMap()._repr_mimebundle_()
        # ipywidgets returns the (data, metadata) tuple form of the protocol; older hooks return a bare dict.
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert isinstance(data, dict) and data, (
            "expected the widget's non-empty mimebundle"
        )
        assert "application/vnd.jupyter.widget-view+json" in data


class TestUtf8Shim:
    """The Windows-cp1252 UTF-8 workaround (`_patch_maplibre_html_encoding` + UTF-8 `save`)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_shim_rebinds_every_maplibre_submodule(self, monkeypatch):
        """On a platform that needs it, the shim rebinds the reader on *every* maplibre submodule.

        Simulates the cp1252 read failure so the patch installs regardless of the host OS, and injects a
        fake maplibre submodule that imported ``read_internal_file`` by name — it must be rebound too, so no
        stale bare-``open`` reader survives.
        """
        import builtins
        import types

        import maplibre._utils as _utils

        from digitalearth.web import base

        real_open = builtins.open

        def _sentinel(
            *args, **kwargs
        ):  # a fresh, un-shimmed reader (no _digitalearth_utf8 flag)
            raise AssertionError("this reader should have been rebound by the shim")

        fake = types.ModuleType("maplibre._fake_reader_holder")
        fake.read_internal_file = _sentinel
        monkeypatch.setitem(sys.modules, "maplibre._fake_reader_holder", fake)
        # start from an un-patched reader (monkeypatch auto-restores after the test)
        monkeypatch.setattr(_utils, "read_internal_file", _sentinel)

        def _fake_open(
            file, *args, **kwargs
        ):  # fail the probe like Windows cp1252 does
            if "pywidget.js" in str(file) and not kwargs.get("encoding"):
                raise UnicodeDecodeError("charmap", b"\x9d", 0, 1, "simulated cp1252")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _fake_open)

        base._patch_maplibre_html_encoding()

        shim = _utils.read_internal_file
        assert getattr(shim, "_digitalearth_utf8", False), (
            "the canonical _utils reader must be shimmed"
        )
        assert fake.read_internal_file is shim, (
            "a submodule holding the reader by name must be rebound too"
        )

    def test_save_writes_utf8_non_ascii_title(self, tmp_path):
        """`save` writes the HTML as UTF-8 so a non-ASCII title round-trips (the write-side cp1252 fix)."""
        out = tmp_path / "u.html"
        WebMap(center=(0.0, 0.0), zoom=2).save(str(out), title="façade ′ café —")
        assert "façade ′ café —" in out.read_text(encoding="utf-8"), (
            "unicode title must survive the write"
        )


class TestStyleResolution:
    """``_resolve_style`` maps aliases to CartoCDN URLs and passes URLs/dicts through (no engine)."""

    @pytest.mark.parametrize(
        "alias, slug",
        [("dark", "dark-matter"), ("light", "positron"), ("voyager", "voyager")],
    )
    def test_alias_resolves_to_cartocdn_url(self, alias, slug):
        url = _resolve_style(alias)
        assert url == f"https://basemaps.cartocdn.com/gl/{slug}-gl-style/style.json"

    def test_url_and_dict_pass_through(self):
        url = "https://example.com/style.json"
        assert _resolve_style(url) is url
        spec = {"version": 8, "layers": []}
        assert _resolve_style(spec) is spec


class TestConstructionDefaults:
    """Constructor defaults and the reproject predicate (no engine needed)."""

    def test_default_configuration(self):
        """A bare ``WebMap()`` defaults to EPSG:4326 (MapLibre lon/lat), zoom 2, dark style, height 500."""
        m = WebMap()
        assert (m.center, m.zoom, m.style, m.crs, m.height) == (
            None,
            2,
            "dark",
            4326,
            500,
        )
        assert m.layers == []

    @pytest.mark.parametrize(
        "crs, data_epsg, expected",
        [
            (3857, 3857, False),  # already in the display CRS
            (3857, 4326, True),  # differing EPSG -> reproject
            ("ESRI:54009", 3857, True),  # string display CRS -> always reproject
            (3857, None, True),  # no epsg attribute -> reproject path
        ],
    )
    def test_needs_reproject_matrix(self, crs, data_epsg, expected):
        """``_needs_reproject`` only short-circuits on an exact int-EPSG match."""

        class _Data:
            pass

        data = _Data()
        if data_epsg is not None:
            data.epsg = data_epsg
        assert WebMap(crs=crs)._needs_reproject(data) is expected

    def test_composition_includes_all_mixins(self):
        """``WebMap`` composes the base plus all seven capability mixins."""
        from digitalearth.web.base import WebMapBase
        from digitalearth.web.bigdata import BigDataMixin
        from digitalearth.web.decoration import DecorationMixin
        from digitalearth.web.export import ExportMixin
        from digitalearth.web.raster import RasterMixin
        from digitalearth.web.temporal import TemporalMixin
        from digitalearth.web.threed import ThreeDMixin
        from digitalearth.web.vector import VectorMixin

        mro = WebMap.__mro__
        for cls in (
            WebMapBase,
            RasterMixin,
            VectorMixin,
            BigDataMixin,
            ThreeDMixin,
            TemporalMixin,
            DecorationMixin,
            ExportMixin,
        ):
            assert cls in mro, f"{cls.__name__} missing from WebMap MRO"


def test_reimport_is_stable():
    """The package re-imports cleanly (no import-time engine side effects)."""
    mod = importlib.reload(importlib.import_module("digitalearth.web"))
    assert hasattr(mod, "WebMap")


class TestJsonSafeDatetimes:
    """Date-like columns are ISO-encoded before the frame reaches GeoJSON (issue #156)."""

    @staticmethod
    def _frame(**extra):
        """A point frame with a tz-naive datetime column and a NaT, plus any extra columns."""
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        columns = {
            "name": ["a", "b", "c"],
            "score": [1.0, 2.0, 3.0],
            "from_date": pd.to_datetime(["2026-01-01 06:00", "2026-02-01 18:30", None]),
        }
        columns.update(extra)
        return gpd.GeoDataFrame(
            columns,
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs=4326,
        )

    def test_timestamps_become_iso_strings(self):
        """A Timestamp column is replaced by ISO-8601 text, which json can serialise."""
        import json

        out = WebMap()._json_safe(self._frame())
        assert out["from_date"].iloc[0].startswith("2026-01-01T06:00:00")
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_nat_becomes_null_not_the_string(self):
        """A missing timestamp serialises as null rather than the text 'NaT'."""
        out = WebMap()._json_safe(self._frame())
        assert out["from_date"].iloc[2] is None

    def test_a_missing_value_serialises_as_json_null_not_nan(self):
        """The encoded column must be ``object`` dtype so the gap survives as JSON ``null``.

        Test scenario:
            pandas 3 infers a ``str`` dtype for ``.map`` / ``.where`` results, whose missing marker is a
            float ``nan`` — and ``json.dumps`` writes that as the bare literal ``NaN``, which is not valid
            JSON and is not what a GeoJSON consumer expects. Building an explicit ``object`` Series is
            what keeps it a real ``None``.
        """
        import json

        out = WebMap()._json_safe(self._frame())
        assert out["from_date"].dtype == object, (
            f"encoded column must stay object, got {out['from_date'].dtype}"
        )
        payload = json.dumps(out.drop(columns="geometry").to_dict(orient="records"))
        assert "null" in payload, (
            f"the missing timestamp did not become null: {payload}"
        )
        assert "NaN" not in payload, (
            f"invalid JSON NaN leaked into the payload: {payload}"
        )

    def test_iso_text_still_sorts_chronologically(self):
        """The encoding keeps timeslider's ordering valid — lexicographic == chronological."""
        out = WebMap()._json_safe(self._frame())
        stamps = [s for s in out["from_date"] if s is not None]
        assert stamps == sorted(stamps)

    def test_frames_without_datetimes_are_passed_through(self):
        """No date-like column means no copy — the caller's frame is returned as-is."""
        frame = self._frame().drop(columns="from_date")
        assert WebMap()._json_safe(frame) is frame

    def test_the_callers_frame_is_not_mutated(self):
        """Coercion happens on a copy; the frame the caller still holds keeps its dtype."""
        import pandas as pd

        frame = self._frame()
        WebMap()._json_safe(frame)
        assert pd.api.types.is_datetime64_any_dtype(frame["from_date"])

    def test_timezone_aware_columns_keep_their_offset(self):
        """A tz-aware column encodes with its UTC offset, not silently as naive local time."""
        frame = self._frame()
        frame["from_date"] = frame["from_date"].dt.tz_localize("UTC")
        out = WebMap()._json_safe(frame)
        assert out["from_date"].iloc[0] == "2026-01-01T06:00:00+00:00", (
            f"expected a UTC offset in the encoding, got {out['from_date'].iloc[0]!r}"
        )

    def test_object_columns_of_dates_are_encoded_too(self):
        """A column of ``datetime.date`` objects has dtype ``object``, so the dtype checks miss it.

        It still fails ``json.dumps`` with "Object of type date is not JSON serializable", so it has to
        be covered — the original fix for this issue keyed only on ``is_datetime64_any_dtype``.
        """
        import datetime as dt
        import json

        frame = self._frame(day=[dt.date(2026, 1, 1), dt.date(2026, 2, 1), None])
        out = WebMap()._json_safe(frame)
        assert out["day"].iloc[0] == "2026-01-01", (
            f"date not ISO-encoded: {out['day'].iloc[0]!r}"
        )
        assert out["day"].iloc[2] is None, "a missing date must become null"
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_timedelta_columns_become_iso_durations(self):
        """``timedelta64`` also fails json.dumps; it encodes as an ISO-8601 duration."""
        import json

        import pandas as pd

        frame = self._frame(age=pd.to_timedelta([1, 2, None], unit="D"))
        out = WebMap()._json_safe(frame)
        assert out["age"].iloc[0] == "P1DT0H0M0S", (
            f"unexpected duration: {out['age'].iloc[0]!r}"
        )
        assert out["age"].iloc[2] is None, "a missing duration must become null"
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_non_date_object_columns_are_left_alone(self):
        """A plain object column (strings) must not be touched by the date encoding."""
        import pandas as pd

        frame = self._frame(label=pd.Series(["x", "y", "z"], dtype=object))
        assert frame["label"].dtype == object, (
            "the fixture must be object dtype to enter the scan"
        )
        out = WebMap()._json_safe(frame)
        assert list(out["label"]) == ["x", "y", "z"], "a string column was altered"

    def test_the_geometry_column_is_never_encoded(self):
        """Geometry is what MapLibre actually needs; it must survive untouched."""
        frame = self._frame()
        out = WebMap()._json_safe(frame)
        assert out.geometry.equals(frame.geometry), (
            "geometry was altered by the encoding"
        )

    def test_a_mixed_object_column_encodes_only_its_dates(self):
        """A column holding dates *and* other values must not crash on the non-dates.

        Test scenario:
            The dtype checks cannot see inside an object column, so the encoder scans its values. If it
            then called ``isoformat()`` on every element, a column of ``[date, "n/a"]`` would raise
            ``AttributeError`` — turning a frame that merely rendered oddly into one that cannot render
            at all.
        """
        import datetime as dt
        import json

        frame = self._frame(mixed=[dt.date(2026, 1, 1), "n/a", 5])
        out = WebMap()._json_safe(frame)
        assert list(out["mixed"]) == ["2026-01-01", "n/a", 5], (
            f"mixed column mangled: {list(out['mixed'])}"
        )
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_period_columns_are_encoded(self):
        """``period`` is date-like and fails json.dumps just like the rest.

        Test scenario:
            A monthly/quarterly series is a natural fit for a period column, and it raised
            "Object of type Period is not JSON serializable" before being covered.
        """
        import json

        import pandas as pd

        frame = self._frame(month=pd.period_range("2026-01", periods=3, freq="M"))
        out = WebMap()._json_safe(frame)
        assert list(out["month"]) == ["2026-01", "2026-02", "2026-03"], (
            f"periods mangled: {list(out['month'])}"
        )
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_an_all_null_datetime_column_becomes_all_null(self):
        """Every value missing is still a datetime column, and must encode to nulls, not 'NaT'.

        Test scenario:
            A datetime column with no value at all still matches on dtype, so it must encode to nulls
            rather than slipping through as raw NaT.
        """
        import json

        import pandas as pd

        frame = self._frame()
        frame["from_date"] = pd.to_datetime([None, None, None])
        out = WebMap()._json_safe(frame)
        assert list(out["from_date"]) == [None, None, None], (
            f"expected all null, got {list(out['from_date'])}"
        )
        payload = json.dumps(out.drop(columns="geometry").to_dict(orient="records"))
        assert "NaN" not in payload, f"a NaN leaked into the payload: {payload}"
        assert "NaT" not in payload, f"a NaT leaked into the payload: {payload}"

    def test_an_empty_frame_is_handled(self):
        """A zero-row frame with a datetime column must not raise.

        Test scenario:
            A filtered-to-empty event feed is a normal thing to hand a map.
        """
        import geopandas as gpd
        import pandas as pd

        empty = gpd.GeoDataFrame(
            {"from_date": pd.to_datetime([])}, geometry=[], crs=4326
        )
        out = WebMap()._json_safe(empty)
        assert len(out) == 0, "an empty frame must stay empty"

    def test_sub_second_precision_survives(self):
        """Microseconds are kept, so an ordering by timestamp is not silently coarsened.

        Test scenario:
            A strftime pattern of second resolution would truncate these, collapsing distinct events to
            the same instant; ``isoformat()`` keeps them apart.
        """
        import pandas as pd

        frame = self._frame()
        frame["from_date"] = pd.to_datetime(
            [
                "2026-01-01 00:00:00.123456",
                "2026-01-01 00:00:00.654321",
                "2026-01-01 00:00:01.000000",
            ]
        )
        out = WebMap()._json_safe(frame)
        assert out["from_date"].iloc[0] == "2026-01-01T00:00:00.123456", (
            f"sub-second precision lost: {out['from_date'].iloc[0]!r}"
        )
        assert len(set(out["from_date"])) == 3, (
            "distinct instants collapsed to the same string"
        )

    def test_encoding_is_idempotent(self):
        """Encoding an already-encoded frame is a no-op, so a double pass cannot corrupt it.

        Test scenario:
            The second call sees only string columns, finds nothing date-like, and returns the frame
            unchanged — which is what makes the choke point safe to call more than once.
        """
        once = WebMap()._json_safe(self._frame())
        twice = WebMap()._json_safe(once)
        assert twice is once, "a second pass should return the same object"

    def test_every_date_like_column_is_encoded_not_just_the_first(self):
        """A frame with several date-like columns has all of them converted.

        Test scenario:
            Event feeds routinely carry both a start and an end timestamp (GDACS `from_date`/`to_date`).
        """
        import datetime as dt
        import json

        import pandas as pd

        frame = self._frame(
            to_date=pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"]),
            day=[dt.date(2026, 1, 1), dt.date(2026, 2, 1), dt.date(2026, 3, 1)],
        )
        out = WebMap()._json_safe(frame)
        for column in ("from_date", "to_date", "day"):
            assert out[column].dtype == object, f"{column} was not encoded"
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    @pytest.mark.parametrize(
        "values, expected",
        [
            (["n/a", "DATE"], ["n/a", "2026-01-01"]),
            ([None, "n/a", "DATE"], [None, "n/a", "2026-01-01"]),
            ([1, "DATE"], [1, "2026-01-01"]),
            (["DATE", "n/a"], ["2026-01-01", "n/a"]),
        ],
    )
    def test_a_date_anywhere_in_an_object_column_triggers_encoding(
        self, values, expected
    ):
        """Encoding must not depend on which row the first date happens to sit in.

        Args:
            values: The object column, with "DATE" standing in for a ``datetime.date``.
            expected: The encoded column.

        Test scenario:
            Sampling only the first non-null value left ``["n/a", date(...)]`` unconverted, so it still
            died on "Object of type date is not JSON serializable" — the very error being fixed. The two
            orderings are equally unserialisable and must behave the same.
        """
        import datetime as dt
        import json

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        column = [dt.date(2026, 1, 1) if v == "DATE" else v for v in values]
        frame = gpd.GeoDataFrame(
            {"c": pd.Series(column, dtype=object)},
            geometry=[Point(i, i) for i in range(len(column))],
            crs=4326,
        )
        out = WebMap()._json_safe(frame)
        assert list(out["c"]) == expected, f"expected {expected}, got {list(out['c'])}"
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    def test_a_plain_timedelta_is_encoded(self):
        """``datetime.timedelta`` has no ``isoformat()`` — only pandas' subclass does.

        Test scenario:
            An object column holding a plain timedelta raised
            ``AttributeError: 'datetime.timedelta' object has no attribute 'isoformat'`` out of the
            encoder itself, which is worse than the serialisation error it was meant to prevent.
        """
        import datetime as dt
        import json

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"gap": pd.Series([dt.timedelta(days=1), "x"], dtype=object)},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        out = WebMap()._json_safe(frame)
        assert list(out["gap"]) == ["P1DT0H0M0S", "x"], (
            f"unexpected encoding: {list(out['gap'])}"
        )
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    @pytest.mark.parametrize(
        "value, expected",
        [
            (datetime.time(6, 30), "06:30:00"),
            (np.datetime64("2026-01-01"), "2026-01-01T00:00:00"),
            (np.timedelta64(1, "D"), "P1DT0H0M0S"),
        ],
    )
    def test_scalar_types_the_dtype_checks_miss_are_encoded(self, value, expected):
        """Scalars that fail ``json.dumps`` but match no pandas date dtype are covered.

        Args:
            value: The scalar placed in an object column.
            expected: Its ISO encoding.

        Test scenario:
            None of these is caught by the datetime64/timedelta64 dtype checks — ``time`` is not a
            ``date`` subclass, and a numpy scalar in an object column has no pandas dtype to match. The
            numpy scalars also lack ``isoformat()``, so they route through pandas' wrappers.
        """
        import json

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        scalar = value
        frame = gpd.GeoDataFrame(
            {"c": pd.Series([scalar], dtype=object)}, geometry=[Point(0, 0)], crs=4326
        )
        out = WebMap()._json_safe(frame)
        assert out["c"].iloc[0] == expected, (
            f"expected {expected!r}, got {out['c'].iloc[0]!r}"
        )
        json.dumps(
            out.drop(columns="geometry").to_dict(orient="records")
        )  # must not raise

    @pytest.mark.parametrize(
        "categories, encoded",
        [
            (["2026-01-01", "2026-01-01", "2026-02-01"], True),
            (["a", "b", "a"], False),
        ],
    )
    def test_a_categorical_column_is_unwrapped_before_the_dtype_checks(
        self, categories, encoded
    ):
        """A categorical wraps its real dtype in ``.categories`` and matched none of the dtype tests.

        Args:
            categories: The column values, timestamps or plain text.
            encoded: Whether that column should be converted.

        Test scenario:
            Storing a repeated timestamp column as a category is routine for an event feed, and it still
            raised issue #156's exact TypeError. A categorical of text must stay untouched, so the
            unwrapping cannot simply convert every categorical.
        """
        import json

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        values = pd.to_datetime(categories) if encoded else pd.Series(categories)
        frame = gpd.GeoDataFrame(
            {"c": pd.Series(values).astype("category")},
            geometry=[Point(i, i) for i in range(len(categories))],
            crs=4326,
        )
        assert isinstance(frame["c"].dtype, pd.CategoricalDtype), (
            "the fixture must be categorical"
        )
        out = WebMap()._json_safe(frame)
        if encoded:
            assert out["c"].iloc[0] == "2026-01-01T00:00:00", (
                f"not encoded: {out['c'].iloc[0]!r}"
            )
            json.dumps(
                out.drop(columns="geometry").to_dict(orient="records")
            )  # must not raise
        else:
            assert out is frame, "a categorical of text must not be copied"

    def test_a_one_element_array_holding_nan_is_not_treated_as_missing(self):
        """``pd.isna`` judges a container element-wise, which would swallow the whole value.

        Test scenario:
            ``bool(pd.isna(np.array([nan])))`` is True, so a one-element array holding NaN was silently
            replaced by None — losing a real cell value rather than encoding a date.
        """
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {
                "c": pd.Series(
                    [np.array([np.nan]), pd.Timestamp("2026-01-01")], dtype=object
                )
            },
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        out = WebMap()._json_safe(frame)
        assert isinstance(out["c"].iloc[0], np.ndarray), (
            f"the array was replaced: {out['c'].iloc[0]!r}"
        )
        assert out["c"].iloc[1] == "2026-01-01T00:00:00", (
            "the date alongside it was not encoded"
        )

    def test_a_text_column_is_returned_without_a_copy(self):
        """Scanning an object column for dates must not convert one that holds none.

        Test scenario:
            The scan looks at every value now, so a plain text column is the case most at risk of being
            copied or coerced for nothing.
        """
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"label": pd.Series(["a", "b"], dtype=object)},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        assert frame["label"].dtype == object, (
            "the fixture must be object dtype to enter the scan"
        )
        assert WebMap()._json_safe(frame) is frame, (
            "a text-only frame must not be copied"
        )

    def test_a_numeric_column_with_nan_is_left_alone(self):
        """Only date-like columns are touched; a float column keeps its dtype and its NaN.

        Test scenario:
            The null handling is per date-like column, not global — converting a numeric column to
            object would change what the classification builders read downstream.
        """
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"score": [1.0, float("nan")]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        out = WebMap()._json_safe(frame)
        assert out is frame, "a numeric column must not be encoded"
        assert pd.api.types.is_float_dtype(out["score"]), "the float dtype was changed"

    def test_a_list_valued_element_does_not_break_the_null_check(self):
        """``pd.isna`` raises on a list rather than judging it; such a value counts as present.

        Test scenario:
            Without the guard the whole call would die on
            ``ValueError: The truth value of an array ... is ambiguous`` for a frame that merely holds a
            list-valued attribute alongside a date.
        """
        import datetime as dt

        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"c": pd.Series([[1, 2], dt.date(2026, 1, 1)], dtype=object)},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        out = WebMap()._json_safe(frame)
        assert out["c"].iloc[0] == [1, 2], (
            f"the list value was altered: {out['c'].iloc[0]!r}"
        )
        assert out["c"].iloc[1] == "2026-01-01", (
            f"the date was not encoded: {out['c'].iloc[1]!r}"
        )

    def test_a_geoseries_is_returned_untouched(self):
        """A GeoSeries passes the vector guard but has no columns to encode.

        Test scenario:
            ``_json_safe`` reads ``.columns``, which ``_require_vector`` does not guarantee — only
            ``.geometry`` is.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        series = gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs=4326)
        assert WebMap()._json_safe(series) is series, (
            "a GeoSeries must pass straight through"
        )

    def test_the_feature_collection_branch_encodes_too(self, tmp_path):
        """A pyramids ``FeatureCollection`` takes its own branch of ``_display_gdf`` and must be encoded.

        Args:
            tmp_path: pytest's per-test directory, holding the written GeoJSON.

        Test scenario:
            ``_display_gdf`` branches on ``epsg``/``to_crs``, which a plain GeoDataFrame does not have —
            so reaching that path needs a real ``FeatureCollection``. It is a GeoDataFrame subclass, so
            the encoding copy has to preserve both the subclass and its pyramids surface.
        """
        gpd = pytest.importorskip("geopandas")
        feature = pytest.importorskip("pyramids.feature")
        import pandas as pd
        from shapely.geometry import Point

        source = gpd.GeoDataFrame(
            {"when": pd.to_datetime(["2026-01-01", "2026-02-01"])},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        path = tmp_path / "events.geojson"
        source.to_file(str(path), driver="GeoJSON")
        collection = feature.FeatureCollection.read_file(str(path))
        assert hasattr(collection, "epsg"), (
            "fixture must expose .epsg or the branch is not exercised"
        )

        out = WebMap()._display_gdf(collection, method="points")
        assert out["when"].iloc[0].startswith("2026-01-01T"), (
            "the FeatureCollection branch did not encode"
        )
        assert isinstance(out, type(collection)), (
            f"the copy lost the subclass: {type(out).__name__}"
        )
        assert out.epsg == 4326, (
            f"the pyramids surface was lost: {getattr(out, 'epsg', None)}"
        )

    def test_a_renamed_geometry_column_is_still_excluded(self):
        """Geometry is found by its active name, not by the literal string "geometry".

        Test scenario:
            `rename_geometry` is common when a layer comes from a GeoPackage or PostGIS. Excluding the
            wrong column would send geometry through the encoder.
        """
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"when": pd.to_datetime(["2026-01-01", "2026-02-01"])},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        ).rename_geometry("shape")
        out = WebMap()._json_safe(frame)
        assert out.geometry.equals(frame.geometry), (
            "the renamed geometry column was altered"
        )
        assert out["when"].iloc[0] == "2026-01-01T00:00:00", (
            "the date column was not encoded"
        )

    def test_a_non_default_index_is_preserved(self):
        """The encoded column keeps the frame's index, so it aligns on assignment.

        Test scenario:
            A filtered frame carries a gapped index; building the replacement Series with a fresh
            RangeIndex would misalign the values or produce NaN.
        """
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"when": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])},
            geometry=[Point(i, i) for i in range(3)],
            crs=4326,
        )
        filtered = frame.iloc[[0, 2]]
        out = WebMap()._json_safe(filtered)
        assert list(out.index) == [0, 2], f"index not preserved: {list(out.index)}"
        assert list(out["when"]) == ["2026-01-01T00:00:00", "2026-03-01T00:00:00"], (
            f"values misaligned against the index: {list(out['when'])}"
        )

    def test_the_reprojection_branch_encodes_too(self):
        """A frame in another CRS takes the reproject branch of ``_display_gdf`` and must be encoded.

        Test scenario:
            ``_display_gdf`` has three return paths; this is the one a caller in a projected CRS hits, and
            it applies the encoding to the *reprojected* frame rather than the original.
        """
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"when": pd.to_datetime(["2026-01-01", "2026-02-01"])},
            geometry=[Point(500000, 5800000), Point(510000, 5810000)],
            crs=3857,
        )
        out = WebMap()._display_gdf(frame, method="points")
        assert out["when"].iloc[0] == "2026-01-01T00:00:00", (
            "the reprojected frame was not encoded"
        )
        assert out.crs.to_epsg() == 4326, (
            f"expected the display CRS, got {out.crs.to_epsg()}"
        )


class TestDatetimeFramesReachTheMap:
    """End-to-end: a dated frame now renders and saves through every vector builder (issue #156)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    @pytest.fixture
    def dated_points(self):
        """An event-feed-shaped point layer: a value column and a datetime column."""
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import Point

        return gpd.GeoDataFrame(
            {
                "score": [1.0, 2.0],
                "from_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            },
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )

    @pytest.fixture
    def dated_polygons(self):
        """An area layer carrying a value column and a datetime column."""
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import Polygon

        return gpd.GeoDataFrame(
            {
                "score": [1.0, 2.0],
                "from_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            },
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3)]),
            ],
            crs=4326,
        )

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("polygons", {}),
            ("choropleth", {"column": "score"}),
        ],
    )
    def test_area_builders_save_a_dated_frame(
        self, tmp_path, dated_polygons, method, kwargs
    ):
        """The polygon builders serialise a datetime column instead of raising.

        Args:
            tmp_path: pytest's per-test directory.
            dated_polygons: The dated area layer.
            method: The builder under test.
            kwargs: Extra required arguments for that builder.

        Test scenario:
            `choropleth` is the builder that reproduced issue #156's exact message before the fix.
        """
        out = tmp_path / f"{method}.html"
        getattr(WebMap(), method)(dated_polygons, **kwargs).save(str(out))
        assert out.stat().st_size > 1_000, f"{method} wrote an empty page"

    def test_lines_save_a_dated_frame(self, tmp_path):
        """The line builder serialises a datetime column too.

        Args:
            tmp_path: pytest's per-test directory.

        Test scenario:
            `lines` needs its own fixture — a line layer is not interchangeable with points or areas.
        """
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import LineString

        frame = gpd.GeoDataFrame(
            {"from_date": pd.to_datetime(["2026-01-01"])},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=4326,
        )
        out = tmp_path / "lines.html"
        WebMap().lines(frame).save(str(out))
        assert out.stat().st_size > 1_000, "lines wrote an empty page"

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("points", {"column": "score"}),
            ("heatmap", {}),
            ("cluster", {}),
        ],
    )
    def test_point_builders_save_a_dated_frame(
        self, tmp_path, dated_points, method, kwargs
    ):
        """Each point builder serialises a datetime column instead of raising.

        Args:
            tmp_path: pytest's per-test directory.
            dated_points: The dated point layer.
            method: The builder under test.
            kwargs: Extra required arguments for that builder.

        Test scenario:
            Every one of these previously died on
            ``TypeError: Object of type Timestamp is not JSON serializable``.
        """
        out = tmp_path / f"{method}.html"
        getattr(WebMap(), method)(dated_points, **kwargs).save(str(out))
        assert out.stat().st_size > 1_000, f"{method} wrote an empty page"

    def test_timeslider_scrubs_the_datetime_column_it_encoded(
        self, tmp_path, dated_points
    ):
        """The slider still orders its steps after the column becomes ISO text.

        Test scenario:
            timeslider is the builder this bug hit hardest — the kdim it scrubs is usually the very
            column that broke serialisation. The steps must stay in chronological order, which is why
            ISO-8601 (lexicographically sortable) is the right encoding.
        """
        m = WebMap().timeslider(dated_points, kdim="from_date")
        assert m._temporal_times() == ["2026-01-01T00:00:00", "2026-02-01T00:00:00"], (
            f"expected the encoded steps in chronological order, got {m._temporal_times()}"
        )
        out = tmp_path / "slider.html"
        m.save(str(out))
        page = out.read_text(encoding="utf-8", errors="replace")
        assert '"from_date": "2026-01-01T00:00:00"' in page, (
            "the encoded value never reached the page"
        )
        assert "Timestamp(" not in page, "a raw Timestamp leaked into the page"

    def test_a_missing_time_value_does_not_break_the_slider(self, tmp_path):
        """A NaT in the kdim drops that step instead of raising when the steps are sorted.

        Args:
            tmp_path: pytest's per-test directory.

        Test scenario:
            Encoding maps NaT to None, and sorting a mix of str and None raises
            "'<' not supported between instances of 'NoneType' and 'str'". A feature with no time cannot
            sit at any step, so it is dropped — an event feed with an open-ended `to_date` is normal.
        """
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"from_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"])},
            geometry=[Point(i, i) for i in range(3)],
            crs=4326,
        )
        frame.loc[1, "from_date"] = pd.NaT
        m = WebMap().timeslider(frame, kdim="from_date")
        assert m._temporal_times() == ["2026-01-01T00:00:00", "2026-03-01T00:00:00"], (
            f"the missing step should be dropped, got {m._temporal_times()}"
        )
        out = tmp_path / "slider.html"
        m.save(str(out))
        assert out.stat().st_size > 1_000, "the saved slider page looks empty"

    def test_a_kdim_of_only_missing_values_is_rejected(self, tmp_path):
        """Dropping the missing steps must not turn an all-empty series into a silent empty slider.

        Args:
            tmp_path: pytest's per-test directory (unused; keeps the signature uniform).

        Test scenario:
            The empty check runs after the drop, so a column of nothing but NaT still raises the
            actionable ValueError rather than building a slider with no stops.
        """
        gpd = pytest.importorskip("geopandas")
        import pandas as pd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"from_date": pd.to_datetime([None, None])},
            geometry=[Point(0, 0), Point(1, 1)],
            crs=4326,
        )
        m = WebMap()
        with pytest.raises(ValueError, match="at least one time step"):
            m.timeslider(frame, kdim="from_date")
