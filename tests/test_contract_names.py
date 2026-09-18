"""The Core contract, held against each facade (U-3, #299).

Four tiers grew four vocabularies: a raster field was `imshow`, `image`, `add_raster` and `terrain`; a colour
key was a builder on two tiers and a toggle on a third; `render` returned the engine's object on two and `None`
on one; and eight names meant two different things depending on which facade you held. `base/contract.py` is
the agreement that ends that, and these are what hold the tiers to it — including the honest half, which is
what a tier has *not* built and which order builds it.

The tiers whose seams have landed (web, 3-D) must answer to every Core name they can draw; interactive and
static carry pending lists their own seams (#300, #303) empty.
"""

import inspect
import warnings

import pytest

from digitalearth.base.contract import (
    ALIASES,
    CORE,
    PENDING,
    TIER2,
    alias_table,
    core_method,
    pending_for,
)

#: The facades, by the backend name `quickmap` spells them with. Each is imported lazily, so a missing extra
#: skips that tier rather than failing the file.
FACADES = {
    "web": ("digitalearth.web", "WebMap"),
    "3d": ("digitalearth.three_d", "Scene3D"),
    "interactive": ("digitalearth.interactive", "InteractiveMap"),
    "matplotlib": ("digitalearth.static", "Map"),
}

#: The tiers whose renderer seam has landed, and which therefore answer to the Core now.
SEAMED = ("web", "3d")


def _facade(backend: str):
    """Return a tier's facade class, skipping the test when its extra is not installed.

    Args:
        backend: The tier to load.

    Returns:
        The facade class.
    """
    module, name = FACADES[backend]
    return getattr(pytest.importorskip(module), name)


class TestTheContractIsWellFormed:
    """The declaration itself, before any tier is consulted."""

    def test_every_core_name_is_declared_once(self):
        """One row per name, or a tier could satisfy two different promises under one spelling."""
        names = [method.name for method in CORE]
        assert len(names) == len(set(names)), f"repeated Core names: {names}"

    def test_every_tier_2_name_is_declared_once(self):
        """The same, for the names that must agree wherever they appear."""
        names = [method.name for method in TIER2]
        assert len(names) == len(set(names)), f"repeated Tier 2 names: {names}"

    def test_a_name_is_never_both_core_and_tier_2(self):
        """A name is either answered everywhere or agreed wherever it appears, not both."""
        overlap = {method.name for method in CORE} & {method.name for method in TIER2}
        assert overlap == set(), (
            f"names in both tiers of the contract: {sorted(overlap)}"
        )

    def test_every_alias_points_at_a_name_the_contract_declares(self):
        """An alias to nowhere would deprecate a spelling for a name nothing promises."""
        declared = {method.name for method in CORE} | {method.name for method in TIER2}
        # `record` and `terrain_tiles` are tier-native names a collision freed up, not contract names.
        native = {"record", "terrain_tiles", "grid_points"}
        for backend, table in ALIASES.items():
            unknown = sorted(set(table.values()) - declared - native)
            assert unknown == [], f"{backend} aliases point at {unknown}"

    def test_every_pending_entry_names_a_core_method(self):
        """A pending list is about the Core, so an entry outside it is a stale note."""
        declared = {method.name for method in CORE}
        for backend, table in PENDING.items():
            unknown = sorted(set(table) - declared)
            assert unknown == [], (
                f"{backend} lists {unknown} as pending, which is not Core"
            )

    def test_every_pending_entry_says_why_or_when(self):
        """The honest half: a name absent by design reads differently from one nobody has written."""
        for backend, table in PENDING.items():
            silent = [name for name, reason in table.items() if not reason.strip()]
            assert silent == [], f"{backend} gives no reason for {silent}"


class TestTheSeamedTiersAnswerToTheCore:
    """web and 3-D, whose seams have landed."""

    @pytest.mark.parametrize("backend", SEAMED)
    def test_every_core_name_is_present_or_pending(self, backend):
        """A Core name is either on the facade or declared pending, with a reason.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        pending = pending_for(backend)
        missing = [
            method.name
            for method in CORE
            if not hasattr(facade, method.name) and method.name not in pending
        ]
        assert missing == [], (
            f"{backend} answers to neither the name nor a reason: {missing}"
        )

    @pytest.mark.parametrize("backend", SEAMED)
    def test_nothing_pending_is_quietly_present(self, backend):
        """A name listed as pending that the tier does have is a stale list.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        built = [name for name in pending_for(backend) if hasattr(facade, name)]
        assert built == [], f"{backend} lists {built} as pending, but has them"

    @pytest.mark.parametrize("backend", SEAMED)
    def test_every_old_spelling_still_works(self, backend):
        """An alias is a promise to the caller who already wrote the old name.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        broken = [old for old in alias_table(backend) if not hasattr(facade, old)]
        assert broken == [], f"{backend} dropped {broken} instead of deprecating them"

    @pytest.mark.parametrize("backend", SEAMED)
    def test_every_alias_warns_and_names_its_replacement(self, backend):
        """The other half of the promise: the old name must not linger silently.

        Args:
            backend: The tier under test.

        Test scenario:
            The alias is read off the class rather than called, since calling one draws a map. What is checked
            is that it is a deprecation shim at all — a plain method assigned to two names would pass every
            other test here and warn nobody.
        """
        facade = _facade(backend)
        for old, new in alias_table(backend).items():
            held = inspect.getattr_static(facade, old)
            source = inspect.getsource(held) if inspect.isfunction(held) else ""
            assert "DeprecationWarning" in source, f"{backend}.{old} does not warn"
            assert new in source or new in (held.__doc__ or ""), (
                f"{backend}.{old} does not name {new} as its replacement"
            )


class TestTheUnseamedTiersDeclareTheirGap:
    """interactive and static, whose seams are #300 and #303."""

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_what_is_missing_is_listed_rather_than_discovered(self, backend):
        """Every Core name the facade lacks is in its pending list, naming what will build it.

        Args:
            backend: The tier under test.
        """
        facade = _facade(backend)
        pending = pending_for(backend)
        missing = [
            method.name
            for method in CORE
            if not hasattr(facade, method.name) and method.name not in pending
        ]
        assert missing == [], f"{backend} is missing {missing} without saying so"

    @pytest.mark.parametrize("backend", ["interactive", "matplotlib"])
    def test_each_pending_name_says_which_seam_or_order_builds_it(self, backend):
        """A pending entry is a plan, not a shrug.

        Args:
            backend: The tier under test.
        """
        vague = [
            name
            for name, reason in pending_for(backend).items()
            if "#" not in reason and "order" not in reason
        ]
        assert vague == [], f"{backend} does not say what builds {vague}"


class TestOneNamePerMeaning:
    """The collisions the contract settles, checked where they were live."""

    def test_the_web_tier_draws_a_field_under_that_name(self):
        """`add_raster` is the alias now; `field` is the name every tier answers to."""
        web = _facade("web")
        assert callable(web.field), "web must draw a field"
        assert inspect.getattr_static(web, "add_raster") is not inspect.getattr_static(
            web, "field"
        ), "the alias must be a shim, not the same function under two names"

    def test_the_web_tier_switches_projection_by_name(self):
        """`globe(True)` was a projection switch spelled as a layer builder."""
        web = _facade("web")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert web().projection("globe").viewport.globe is True, "globe by name"

    def test_the_web_tier_writes_an_animation_under_the_shared_name(self):
        """`save_animation` is what static and interactive already called it."""
        web = _facade("web")
        assert callable(web.save_animation), "web must write an animation"

    def test_the_3d_tier_records_under_its_own_name(self):
        """A callback loop is not what `animate` means on the other tiers, so it is `record`."""
        scene = _facade("3d")
        assert callable(scene.record), "the 3-D callback loop is record()"

    def test_terrain_means_one_thing_per_tier(self):
        """A pyramids DEM in 3-D, a terrain-RGB tile URL on web — under different names."""
        web = _facade("web")
        scene = _facade("3d")
        assert callable(web.terrain_tiles), "the web tier reads terrain tiles"
        assert callable(scene.terrain), "the 3-D tier draws a DEM"


class TestTheAttachedIssues:
    """#260 and #263, whose web halves close with this contract."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip the class when MapLibre is not installed: two of these draw a map."""
        pytest.importorskip("maplibre")

    def test_web_text_takes_the_string_as_s(self):
        """#260: the third argument is `s`, as it is on the static and interactive tiers."""
        web = _facade("web")
        parameters = inspect.signature(web.text).parameters
        assert "s" in parameters, sorted(parameters)

    def test_web_text_takes_a_crs(self):
        """#260: a point in another CRS is placed, not drawn in the wrong ocean."""
        web = _facade("web")
        assert "crs" in inspect.signature(web.text).parameters, "text() must take crs="

    def test_web_text_still_accepts_the_old_keyword(self):
        """The rename is a promise to the caller who already wrote `string=`."""
        web = _facade("web")
        with pytest.warns(DeprecationWarning, match="string"):
            drawn = web().text(4.9, 52.4, string="Amsterdam")
        assert drawn.layer_ids, drawn.layer_ids

    def test_web_graticule_takes_two_steps(self):
        """#263: meridians and parallels are spaced separately, as on the interactive tier."""
        web = _facade("web")
        parameters = inspect.signature(web.graticule).parameters
        assert {"lon_step", "lat_step"} <= set(parameters), sorted(parameters)

    def test_web_graticule_still_takes_one_spacing(self):
        """`spacing=` is the short way to ask for a square grid, and still works."""
        web = _facade("web")
        assert web().graticule(spacing=20.0).layer_ids, "a square grid still draws"

    def test_web_text_places_a_point_given_in_another_crs(self):
        """#260: a Web-Mercator coordinate lands where it belongs, not in the Atlantic.

        Test scenario:
            The tier places data in EPSG:4326, so a point given in another CRS has to be reprojected — which
            it goes through `Bounds.to_crs`, keeping the CRS work in pyramids.
        """
        web = _facade("web")
        placed = web().text(500000.0, 6800000.0, "Utrecht", crs=3857)
        assert placed.layer_ids, placed.layer_ids

    def test_web_text_needs_a_string(self):
        """A call with no string names what is missing rather than drawing an empty label."""
        web = _facade("web")
        with pytest.raises(TypeError, match="needs the string to draw"):
            web().text(4.9, 52.4)

    def test_web_get_layer_returns_the_description(self):
        """The contract's read-back: a layer's own `LayerSpec`, by id."""
        web = _facade("web")
        drawn = web().text(4.9, 52.4, "Amsterdam", name="label")
        assert drawn.get_layer("label").kind == "text", drawn.get_layer("label")

    def test_web_get_layer_refuses_an_unknown_id(self):
        """The message names the layers that are there."""
        web = _facade("web")
        with pytest.raises(KeyError, match="no layer 'nope' on this map"):
            web().get_layer("nope")

    def test_web_colorbar_draws_the_colour_key(self):
        """The contract's name for a colour key, which this tier draws through `legend`."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        web = _facade("web")
        squares = gpd.GeoDataFrame(
            {"pop": [1, 9]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
        drawn = web().choropleth(squares, column="pop", name="area")
        assert sorted(drawn.colorbar("area", label="People")._panels) == ["legend"], (
            drawn._panels
        )

    def test_web_colorbar_without_an_id_keys_the_last_classified_layer(self):
        """A caller who drew one thing should not have to name it."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        web = _facade("web")
        squares = gpd.GeoDataFrame(
            {"pop": [1, 9]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
        drawn = web().choropleth(squares, column="pop", name="area").colorbar()
        assert sorted(drawn._panels) == ["legend"], drawn._panels

    def test_web_colorbar_refuses_an_unknown_layer(self):
        """A key for a layer nobody drew is a caller's mistake, not an empty panel."""
        web = _facade("web")
        with pytest.raises(KeyError, match="no layer 'nope' on this map"):
            web().colorbar("nope")

    def test_web_colorbar_draws_nothing_when_it_is_not_wanted(self):
        """`visible=False` is a caller passing a flag through, not an error."""
        web = _facade("web")
        assert web().colorbar(visible=False)._panels == {}, "no key was asked for"


class TestTheDispatcherPassesUnderDeprecationErrors:
    """Nothing the package calls itself may go through a deprecated spelling."""

    @pytest.mark.parametrize("backend", SEAMED)
    def test_quickmap_uses_the_canonical_names(self, backend, tmp_path):
        """`quickmap` on a seamed backend raises no deprecation warning of its own.

        Args:
            backend: The tier under test.
            tmp_path: Unused; keeps the signature uniform with the other cases.
        """
        # The engine, not the tier: a tier imports without its engine (#290), so a missing extra would only
        # surface when the drawing started.
        pytest.importorskip({"web": "maplibre", "3d": "pyvista"}[backend])
        from pyramids.dataset import Dataset

        from digitalearth import quickmap

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawn = quickmap(
                Dataset.read_file("examples/data/acc4000.tif"), backend=backend
            )
        # Only this package's own warnings: PyVista and NumPy warn about things neither the caller nor this
        # package can act on, and failing on those would make the gate about the installed versions.
        ours = [
            str(record.message)
            for record in caught
            if issubclass(record.category, DeprecationWarning)
            and "digitalearth" in record.filename
        ]
        assert ours == [], f"quickmap reached a deprecated spelling: {ours}"
        assert drawn is not None, "quickmap must draw"
        if hasattr(drawn, "close"):
            drawn.close()
