"""`PanelSpec` and `FigureSpec` — a figure that round-trips with no renderer (DE-22, #283).

Stage S4's exit criterion is that `FigureSpec.from_dict(fig.to_dict())` reproduces the figure with no renderer
imported. These cover the two types, the references between a figure's parts, and that criterion — asserted in a fresh
interpreter rather than assumed.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from digitalearth.base.spec import (
    SCHEMA_VERSION,
    Bounds,
    Camera,
    DataRef,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Selection,
    Symbology,
    Viewport,
)

REPO = Path(__file__).resolve().parents[2]


def _neutrality_guard():
    """Load the engine-neutrality test module, so this file checks the same banned set it does."""
    spec = importlib.util.spec_from_file_location(
        "_neutrality", REPO / "tests" / "test_base_is_engine_neutral.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _two_panel_figure() -> FigureSpec:
    """Return a figure exercising every part: two sources, a drape, a group, two panels in two kinds of view."""
    layers = (
        LayerTree()
        .add(
            LayerSpec(
                "dem", "raster", source_id="srtm", symbology=Symbology.of(opacity=0.9)
            )
        )
        .add(
            LayerSpec(
                "imagery",
                "rgb",
                source_id="s2",
                selection=Selection.of((4, 3, 2)),
                z_source="layer:dem",
            )
        )
        .add(LayerSpec("grid", "graticule", group="reference"))
        .set_group_visible("reference", False)
    )
    return FigureSpec(
        panels=(
            PanelSpec(
                "map",
                Viewport(4326, bounds=Bounds(-10.0, 35.0, 30.0, 60.0, crs=4326)),
                layers=("dem", "grid"),
                title="Elevation",
            ),
            PanelSpec(
                "scene",
                Camera.look_at(
                    (10.0, 47.0, 0.0), azimuth=225.0, elevation=30.0, distance=5.0
                ),
                layers=("dem", "imagery"),
            ),
        ),
        sources={
            "srtm": DataRef("data/srtm.tif"),
            "s2": DataRef("s3://bucket/s2.tif", driver="COG"),
        },
        layers=layers,
        size=(12.0, 6.0),
        title="Alps",
    )


class TestPanelSpec:
    """One panel: a view and the layers it names."""

    def test_a_panel_defaults_to_an_unframed_web_mercator_view(self):
        """The default view is the static `Map`'s."""
        assert PanelSpec("main").view == Viewport(3857)

    def test_a_string_of_layers_is_refused_rather_than_split_into_letters(self):
        """`tuple("t2m")` would be three one-letter layer ids."""
        with pytest.raises(ValueError, match="got the string 't2m'"):
            PanelSpec("main", layers="t2m")

    def test_a_layer_listed_twice_is_refused(self):
        """Showing a layer twice in one panel is a mistake, not a double draw."""
        with pytest.raises(ValueError, match=r"lists layers \['dem'\] more than once"):
            PanelSpec("main", layers=("dem", "dem"))

    @pytest.mark.parametrize("layer_id", ["", 3, None])
    def test_a_layer_id_that_is_not_a_non_empty_string_is_refused(self, layer_id):
        """Layer ids are strings.

        Args:
            layer_id: The id under test.
        """
        with pytest.raises(ValueError, match="non-empty layer ids"):
            PanelSpec("main", layers=(layer_id,))

    def test_a_view_that_is_neither_a_viewport_nor_a_camera_is_refused(self):
        """A bare CRS is not a view."""
        with pytest.raises(ValueError, match="view must be a Viewport or a Camera"):
            PanelSpec("main", view=4326)

    def test_a_non_string_title_is_refused(self):
        """A title is text."""
        with pytest.raises(ValueError, match="title must be a string"):
            PanelSpec("main", title=5)

    def test_from_dict_needs_exactly_one_kind_of_view(self):
        """Both or neither of `viewport` and `camera` cannot say what the panel is."""
        both = {
            "id": "p",
            "viewport": {"crs": 4326},
            "camera": {"position": [0, -1, 0]},
        }
        neither = {"id": "p"}
        with pytest.raises(ValueError, match="exactly one of 'viewport'"):
            PanelSpec.from_dict(both)
        with pytest.raises(ValueError, match="exactly one of 'viewport'"):
            PanelSpec.from_dict(neither)

    @pytest.mark.parametrize(
        "panel",
        [
            PanelSpec(
                "flat", Viewport(3857, domain="europe"), layers=("a", "b"), title="Flat"
            ),
            PanelSpec("3d", Camera((0.0, -10.0, 5.0), vertical_exaggeration=2.0)),
        ],
        ids=["viewport", "camera"],
    )
    def test_a_panel_survives_a_json_round_trip(self, panel):
        """Either kind of view comes back, under the key that names it.

        Args:
            panel: The panel under test.
        """
        rebuilt = PanelSpec.from_dict(json.loads(json.dumps(panel.to_dict())))
        assert rebuilt == panel, f"the panel changed in a round trip: {rebuilt!r}"


class TestFigureReferences:
    """A figure's parts refer to each other by id, and every reference must resolve."""

    def test_a_figure_needs_a_panel(self):
        """A figure with nothing to draw in is refused."""
        with pytest.raises(ValueError, match="at least one panel"):
            FigureSpec(panels=())

    def test_a_panel_that_is_not_a_panel_spec_is_refused(self):
        """A panel's stored dict form is not a panel; only `from_dict` turns one into a `PanelSpec`.

        Test scenario:
            Passing the dict straight to the constructor would otherwise fail later on `panel.id`, naming neither
            the field nor the type that was wrong.
        """
        panels = ({"id": "p", "viewport": {"crs": 3857}},)
        with pytest.raises(
            ValueError, match="panels must be PanelSpec values; got dict"
        ):
            FigureSpec(panels=panels)

    def test_panel_ids_must_be_unique(self):
        """Two panels named alike cannot be told apart."""
        panels = (PanelSpec("p"), PanelSpec("p"))
        with pytest.raises(ValueError, match=r"\['p'\] appear more than once"):
            FigureSpec(panels=panels)

    def test_a_layer_reading_a_missing_source_is_refused(self):
        """The message names the layer, the missing source and the sources that exist."""
        panels = (PanelSpec("p", layers=("dem",)),)
        layers = LayerTree((LayerSpec("dem", "raster", source_id="srtm"),))
        sources = {"copernicus": DataRef("dem.tif")}
        with pytest.raises(
            ValueError, match=r"layer 'dem' reads source 'srtm'.*\['copernicus'\]"
        ):
            FigureSpec(panels=panels, layers=layers, sources=sources)

    def test_an_elevation_source_that_is_not_a_layer_must_be_a_source(self):
        """`z_source` naming a source is checked like `source_id`; a `layer:` reference is the tree's to check."""
        panels = (PanelSpec("p"),)
        layers = LayerTree((LayerSpec("mesh", "points", z_source="lidar"),))
        with pytest.raises(ValueError, match="reads source 'lidar'"):
            FigureSpec(panels=panels, layers=layers)

    def test_a_panel_naming_a_layer_that_is_not_in_the_figure_is_refused(self):
        """The message names the panel, the missing layers and the layers that exist."""
        panels = (PanelSpec("p", layers=("dem", "roads")),)
        layers = LayerTree((LayerSpec("dem", "raster"),))
        with pytest.raises(
            ValueError, match=r"panel 'p' shows layers \['roads'\].*\['dem'\]"
        ):
            FigureSpec(panels=panels, layers=layers)

    def test_a_source_that_is_not_a_dataref_is_refused(self):
        """A path string is not an address a resolver can open without a driver hint."""
        panels = (PanelSpec("p"),)
        with pytest.raises(ValueError, match="source 'srtm' must be a DataRef"):
            FigureSpec(panels=panels, sources={"srtm": "dem.tif"})

    def test_an_empty_source_id_is_refused(self):
        """An empty key names no source."""
        panels = (PanelSpec("p"),)
        sources = {"": DataRef("dem.tif")}
        with pytest.raises(ValueError, match="needs an id"):
            FigureSpec(panels=panels, sources=sources)

    def test_a_padded_source_id_is_kept_exactly(self):
        """Ids are compared exactly everywhere in a figure, so a padded source id is a distinct, valid key."""
        layers = LayerTree((LayerSpec("dem", "raster", source_id=" srtm"),))
        figure = FigureSpec(
            panels=(PanelSpec("p", layers=("dem",)),),
            sources={" srtm": DataRef("dem.tif")},
            layers=layers,
        )
        assert list(figure.sources) == [" srtm"], list(figure.sources)

    def test_layers_must_be_a_layer_tree(self):
        """A bare tuple of layers has no ordering or group rules."""
        panels = (PanelSpec("p"),)
        layers = (LayerSpec("a", "points"),)
        with pytest.raises(ValueError, match="layers must be a LayerTree"):
            FigureSpec(panels=panels, layers=layers)

    @pytest.mark.parametrize(
        "size",
        [(10.0,), (0.0, 5.0), (-1.0, 5.0), (float("inf"), 5.0), "12x6", (True, 5.0)],
    )
    def test_a_size_that_is_not_two_positive_finite_numbers_is_refused(self, size):
        """One number, zero, a negative, an infinity, a string and a boolean are each refused.

        Args:
            size: The size under test.
        """
        panels = (PanelSpec("p"),)
        with pytest.raises(ValueError, match="size must be"):
            FigureSpec(panels=panels, size=size)

    def test_a_non_string_figure_title_is_refused(self):
        """A figure's title is text, as a panel's is."""
        panels = (PanelSpec("p"),)
        with pytest.raises(
            ValueError, match="FigureSpec title must be a string or None"
        ):
            FigureSpec(panels=panels, title=5)

    def test_sources_cannot_be_changed_after_construction(self):
        """The sources mapping is read-only, so a figure cannot be re-pointed from outside."""
        figure = FigureSpec(
            panels=(PanelSpec("p"),), sources={"srtm": DataRef("dem.tif")}
        )
        other = DataRef("x.tif")
        with pytest.raises(TypeError):
            figure.sources["other"] = other

    def test_panel_lookup_names_the_panels_that_exist(self):
        """A miss lists what is there."""
        figure = FigureSpec(panels=(PanelSpec("left"), PanelSpec("right")))
        with pytest.raises(KeyError, match=r"no panel 'centre'.*\['left', 'right'\]"):
            figure.panel("centre")

    def test_layers_of_follows_the_tree_order_not_the_panel_listing(self):
        """Reordering the tree reorders every panel that shows those layers."""
        figure = _two_panel_figure()
        moved = FigureSpec(
            panels=figure.panels,
            sources=dict(figure.sources),
            layers=figure.layers.move("dem", -1),
        )
        assert [layer.id for layer in figure.layers_of("scene")] == ["dem", "imagery"]
        assert [layer.id for layer in moved.layers_of("scene")] == ["imagery", "dem"]

    def test_two_panels_can_hold_different_crss(self):
        """The view belongs to the panel, so a figure is no longer limited to one projection per grid."""
        figure = FigureSpec(
            panels=(
                PanelSpec("a", Viewport(3857)),
                PanelSpec("b", Viewport("EPSG:3413")),
            )
        )
        assert [panel.view.crs for panel in figure.panels] == [3857, "EPSG:3413"]


class TestTheSchemaVersion:
    """The version is written from the start, and an unknown one is refused by name."""

    def test_the_version_is_always_written(self):
        """Even the smallest figure records which schema it was written in."""
        assert (
            FigureSpec(panels=(PanelSpec("p"),)).to_dict()["schema_version"]
            == SCHEMA_VERSION
        )

    @pytest.mark.parametrize("version", [2, 0, True, "1", 1.0])
    def test_the_constructor_refuses_a_version_it_does_not_read(self, version):
        """Only the current version is accepted.

        Args:
            version: The version under test.
        """
        panels = (PanelSpec("p"),)
        with pytest.raises(
            ValueError, match="is not one this version of digitalearth reads"
        ):
            FigureSpec(panels=panels, schema_version=version)

    def test_from_dict_refuses_a_newer_version_before_reading_anything_else(self):
        """A newer figure is refused by version, not by whatever field changed meaning.

        Test scenario:
            The stored dict also carries a key this version does not know. If the version were checked after the
            keys, the caller would be told about an unknown key — the symptom — instead of the version, the cause.
        """
        stored = {"schema_version": 2, "panels": [], "legend": {}}
        with pytest.raises(ValueError, match="got schema_version 2"):
            FigureSpec.from_dict(stored)

    def test_from_dict_refuses_a_float_version(self):
        """`1.0` equals `1` but is not the version spelling a writer produces, so a reader refuses it.

        Test scenario:
            A plain `==` let `1.0` through, and a figure built with it wrote `"schema_version": 1.0` back out.
        """
        stored = {
            "schema_version": 1.0,
            "panels": [{"id": "p", "viewport": {"crs": 3857}}],
        }
        with pytest.raises(ValueError, match="got schema_version 1.0"):
            FigureSpec.from_dict(stored)

    def test_from_dict_needs_a_version(self):
        """A stored figure with no version cannot be read safely."""
        stored = {"panels": [{"id": "p", "viewport": {"crs": 3857}}]}
        with pytest.raises(ValueError, match="needs 'schema_version'"):
            FigureSpec.from_dict(stored)

    def test_from_dict_refuses_something_that_is_not_a_mapping(self):
        """A list is not a stored figure."""
        with pytest.raises(TypeError, match="needs a mapping"):
            FigureSpec.from_dict([])

    def test_from_dict_refuses_an_unknown_key(self):
        """A key a newer writer added under the same version is refused, not dropped."""
        stored = {
            "schema_version": 1,
            "panels": [{"id": "p", "viewport": {"crs": 3857}}],
            "theme": "dark",
        }
        with pytest.raises(ValueError, match=r"unknown keys \['theme'\]"):
            FigureSpec.from_dict(stored)


class TestTheRoundTrip:
    """Stage S4's exit criterion."""

    def test_a_whole_figure_survives_a_json_round_trip(self):
        """Every part of a two-panel figure comes back equal through JSON text.

        Test scenario:
            Built through constructors and tree changes on one side, through `from_dict` on parsed JSON on the other.
        """
        figure = _two_panel_figure()
        rebuilt = FigureSpec.from_dict(json.loads(json.dumps(figure.to_dict())))
        assert rebuilt == figure, (
            f"the figure changed in a round trip: {rebuilt.to_dict()}"
        )

    def test_the_round_trip_imports_no_renderer(self):
        """`FigureSpec.from_dict(fig.to_dict())` in a fresh interpreter leaves no renderer in `sys.modules`.

        Test scenario:
            Run in a new process, because this interpreter has already imported renderers for other tests. The banned
            set is read from `tests/test_base_is_engine_neutral.py`, so the two checks cannot drift apart; the
            renderer-backed subpackages of `digitalearth` itself and `matplotlib.pyplot` are checked too.
        """
        guard = _neutrality_guard()
        program = textwrap.dedent(
            """
            import json, sys
            from digitalearth.base.spec import (
                Bounds, Camera, DataRef, FigureSpec, LayerSpec, LayerTree, PanelSpec, Selection, Symbology, Viewport,
            )

            layers = (
                LayerTree()
                .add(LayerSpec("dem", "raster", source_id="srtm", symbology=Symbology.of(opacity=0.9)))
                .add(LayerSpec("imagery", "rgb", source_id="s2", selection=Selection.of((4, 3, 2)),
                               z_source="layer:dem"))
            )
            figure = FigureSpec(
                panels=(
                    PanelSpec("map", Viewport(4326, bounds=Bounds(-10.0, 35.0, 30.0, 60.0, crs=4326)),
                              layers=("dem",)),
                    PanelSpec("scene", Camera.look_at((10.0, 47.0, 0.0), azimuth=225.0, elevation=30.0,
                                                      distance=5.0), layers=("dem", "imagery")),
                ),
                sources={"srtm": DataRef("data/srtm.tif"), "s2": DataRef("s3://bucket/s2.tif")},
                layers=layers,
            )
            rebuilt = FigureSpec.from_dict(json.loads(json.dumps(figure.to_dict())))
            print(json.dumps({"equal": rebuilt == figure, "modules": sorted(sys.modules)}))
            """
        )
        env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report["equal"] is True, (
            "the figure did not survive the round trip in a fresh interpreter"
        )
        loaded = report["modules"]
        renderers = sorted(
            name
            for name in loaded
            if name.split(".")[0] in guard.BANNED_PACKAGES
            or name == "matplotlib.pyplot"
            or any(
                name == sub or name.startswith(sub + ".")
                for sub in guard.BANNED_SUBPACKAGES
            )
        )
        assert renderers == [], f"the round trip imported renderer modules: {renderers}"
