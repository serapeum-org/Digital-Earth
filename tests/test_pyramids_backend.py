"""The ``pyramids.plot_backends`` plugin end: registration, the contract shape, and the translation.

digitalearth is the renderer pyramids reaches for a non-matplotlib ``plot(backend=…)``. Before this
module it was reached by *name* — pyramids imported ``digitalearth`` and called ``quickmap``
directly — which forwarded the facade's vocabulary verbatim. The facade's first band then arrived
off by one, a ``title`` raised from the tier's raster builder (and from PyVista's parser on the 3-D
tier), and an animation had no route at all. These tests pin the registration (so discovery finds
the backends rather than importing us by name) and every translation and refusal that sits between
the two vocabularies.

Nothing here needs a drawing engine: ``quickmap`` is replaced where the delegation itself is under
test, so the whole file runs in the ``dev`` env. The tiers' own rendering is covered in
``tests/interactive`` and ``tests/web``.
"""

import importlib.metadata as metadata
import inspect
import subprocess
import sys

import numpy as np
import pytest

import digitalearth
from digitalearth.pyramids_backend import (
    ANIMATION_BACKENDS,
    BAND_ORIGIN_SHIFT,
    _animatable,
    _animation_kwargs,
    _basemap_asked,
    _classified,
    _Forwarded,
    _framed,
    _netcdf_stack,
    _netcdf_variable,
    interactive,
    three_d,
    web,
)

#: The group pyramids resolves a backend name through, and the three names we answer to.
GROUP = "pyramids.plot_backends"

#: Each registered name against the callable it must load, and the module path it is declared as.
REGISTERED = (
    ("interactive", interactive, "digitalearth.pyramids_backend:interactive"),
    ("3d", three_d, "digitalearth.pyramids_backend:three_d"),
    ("web", web, "digitalearth.pyramids_backend:web"),
)


class _FakeScene:
    """A stand-in for a tier scene, recording the titles set on it."""

    def __init__(self):
        """Start with no title recorded."""
        self.titles: list[str] = []

    def set_title(self, heading, **kwargs):
        """Record a title the adapter sets, as the web and 3-D scenes do.

        Args:
            heading: The title text.
            **kwargs: The tiers' further styling arguments, unused here.

        Returns:
            _FakeScene: self, as the real verbs return.
        """
        self.titles.append(heading)
        return self


class TestRegistration:
    """The entry points pyramids discovers."""

    def test_all_three_backends_are_registered(self):
        """The installed distribution advertises exactly the three non-matplotlib backend names.

        Test scenario:
            Discovery reads the installed metadata, not the source tree, so this is what decides
            whether pyramids finds us at all. ``matplotlib`` must not appear: pyramids short-circuits
            its default to cleopatra and never resolves it here, so registering it would claim a name
            we are never asked for.
        """
        found = sorted(entry.name for entry in metadata.entry_points(group=GROUP))
        assert found == ["3d", "interactive", "web"], (
            f"the registered backends should be the three non-matplotlib tiers, got {found}"
        )

    @pytest.mark.parametrize("name, expected, declared", REGISTERED)
    def test_each_entry_point_loads_to_its_callable(self, name, expected, declared):
        """Each name loads to the callable it is declared as.

        Args:
            name: The backend name a caller passes.
            expected: The callable the name must resolve to.
            declared: The ``module:attr`` string the metadata must carry.

        Test scenario:
            Discovery skips an entry point whose target cannot be loaded, or is not callable, with a
            warning and no error — so a typo in the declaration would cost the backend silently.
            This reads the installed metadata and loads it, the way discovery does.
        """
        entries = {entry.name: entry for entry in metadata.entry_points(group=GROUP)}
        entry = entries[name]
        assert entry.value == declared, f"{name} is declared as {entry.value!r}"
        loaded = entry.load()
        assert loaded is expected, f"{name} loaded {loaded!r}, not {expected!r}"
        assert callable(loaded), (
            f"{name} must load a callable, got {type(loaded).__name__}"
        )


class TestContractShape:
    """The render call shape the seam dispatches through."""

    @pytest.mark.parametrize("name, fn, _declared", REGISTERED)
    def test_each_callable_accepts_the_seam_call_shape(self, name, fn, _declared):
        """``fn(obj, *, backend, **kwargs)`` binds, and ``**kwargs`` is there to receive forwards.

        Args:
            name: The backend name, for the message.
            fn: The render callable.
            _declared: Unused here; the parametrisation is shared.

        Test scenario:
            This is exactly what upstream's conformance kit checks, asserted locally so the contract
            is pinned in the ``dev`` env too — the kit itself is not in any released pyramids yet.
        """
        signature = inspect.signature(fn)
        signature.bind("a-pyramids-object", backend=name)
        kinds = [p.kind for p in signature.parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD in kinds, (
            f"{name} needs **kwargs to receive the facade's forwarded arguments"
        )

    @pytest.mark.parametrize("name, fn, _declared", REGISTERED)
    def test_upstream_conformance_kit_passes(self, name, fn, _declared):
        """Upstream's own kit reports no problems against each callable.

        Args:
            name: The backend name, for the message.
            fn: The render callable.
            _declared: Unused here; the parametrisation is shared.

        Test scenario:
            The kit is the contract's executable form, so running it is stronger than mirroring its
            checks. It ships in pyramids-core but is not in a release yet, so this skips until the
            installed pyramids carries it — at which point it starts guarding automatically.
        """
        kit = pytest.importorskip(
            "pyramids.base.plot_backend_conformance",
            reason="the plot-backend conformance kit is not in a released pyramids-core yet",
        )
        assert kit.check_plot_backend(fn) == [], f"{name} departs from the contract"


class TestBandOrigin:
    """The 0-based band the facade forwards against the 1-based band this package takes."""

    @pytest.mark.parametrize("forwarded, expected", [(0, 1), (1, 2), (11, 12)])
    def test_the_facades_band_is_shifted_to_this_packages_origin(
        self, forwarded, expected
    ):
        """A 0-based forwarded band becomes the 1-based band ``quickmap`` reads.

        Args:
            forwarded: The 0-based index pyramids forwards.
            expected: The 1-based index digitalearth takes.

        Test scenario:
            The defect reported in #501: forwarded unchanged, ``band=0`` reached the reader as index
            ``-1`` and raised ``band -1 is out of range for a 1-band dataset``. The shift is the
            whole reason this adapter exists rather than a bare ``quickmap`` call.
        """
        plan = _Forwarded.from_facade("web", {"band": forwarded})
        assert plan.draw["band"] == expected, (
            f"0-based band {forwarded} should read as {expected}, got {plan.draw['band']}"
        )

    def test_the_shift_is_one_band(self):
        """The documented offset is one, so the two conventions differ by exactly one band."""
        assert BAND_ORIGIN_SHIFT == 1, (
            f"expected a one-band shift, got {BAND_ORIGIN_SHIFT}"
        )

    @pytest.mark.parametrize("band", [-1, -5])
    def test_a_negative_band_is_refused(self, band):
        """A negative index is no 0-based band, and is refused rather than wrapping.

        Args:
            band: A negative index.

        Test scenario:
            Shifted blindly, ``band=-1`` would become ``0`` and then be refused far away by the
            reader, in the reader's words. Refusing here names the convention instead.
        """
        forwarded = {"band": band}
        with pytest.raises(ValueError, match="0-based band index"):
            _Forwarded.from_facade("web", forwarded)

    @pytest.mark.parametrize("band, expected", [(0, 1), (3, 4)])
    def test_a_numpy_integer_is_a_band_index_too(self, band, expected):
        """A NumPy integer shifts like a Python one, and arrives as a plain ``int``.

        Args:
            band: The 0-based index, passed as ``np.int64``.
            expected: The 1-based index it should become.

        Test scenario:
            This is a NumPy-native stack: a band read off an array's shape or an index comes back as
            ``np.int64``, and refusing those made the seam unusable with the very arrays it draws
            (round-1 M4). The result is converted because a builder may put it in a message, where
            ``np.int64(1)`` reads as ``np.int64(1)``.
        """
        plan = _Forwarded.from_facade("web", {"band": np.int64(band)})
        assert plan.draw["band"] == expected, (
            f"expected {expected}, got {plan.draw['band']}"
        )
        assert type(plan.draw["band"]) is int, (
            f"expected a plain int, got {type(plan.draw['band']).__name__}"
        )

    def test_a_numpy_interval_is_a_frame_interval_too(self):
        """A NumPy number is accepted as a frame interval and converted to a rate.

        Test scenario:
            The same NumPy-native reasoning as the band index: an interval computed from an array of
            timestamps is a NumPy scalar.
        """
        plan = _Forwarded.from_facade(
            "web", {"animate": True, "interval": np.int64(200)}
        )
        assert plan.fps == 5.0, f"200 ms a frame is 5 fps, got {plan.fps}"

    @pytest.mark.parametrize("band", [True, "0", 1.5, None])
    def test_a_band_that_is_not_an_integer_index_is_refused(self, band):
        """A non-integer band is refused, ``True`` included.

        Args:
            band: A value that is not a 0-based integer index.

        Test scenario:
            ``bool`` is an ``int`` subclass, so ``band=True`` would otherwise shift to ``2`` and
            draw the second band for a caller who clearly meant something else.
        """
        forwarded = {"band": band}
        with pytest.raises(ValueError, match="not a 0-based band index"):
            _Forwarded.from_facade("3d", forwarded)


class TestVocabulary:
    """The names that differ between the facade and this package."""

    def test_the_mesh_selector_is_renamed_to_this_packages_spelling(self):
        """``variable_name`` (the UgridDataset facade) becomes ``variable`` (``quickmap``).

        Test scenario:
            Forwarded unchanged it reached cleopatra, which refused it by listing its own accepted
            keywords — a message naming ``figsize`` to someone who asked for a mesh variable.
        """
        plan = _Forwarded.from_facade("interactive", {"variable_name": "depth"})
        assert plan.draw == {"variable": "depth"}, (
            f"the mesh selector should be renamed, got {plan.draw}"
        )

    def test_the_agnostic_keywords_pass_through_untouched(self):
        """``title`` / ``basemap`` / ``colorbar`` / ``kind`` / ``cmap`` are forwarded verbatim.

        Test scenario:
            #501 verified these already pass cleanly, so the adapter must not start reinterpreting
            them; only the three genuinely-different names are translated.
        """
        forwarded = {
            "title": "DEM",
            "basemap": True,
            "colorbar": True,
            "kind": "contourf",
            "cmap": "viridis",
        }
        plan = _Forwarded.from_facade("web", dict(forwarded))
        assert plan.draw == forwarded, (
            f"these should pass through unchanged, got {plan.draw}"
        )


class TestUnserved:
    """Keywords digitalearth cannot act on are refused in its own words."""

    @pytest.mark.parametrize(
        "name, value, names_alternative",
        [
            ("exclude_value", 0, "nodata"),
            ("rgb_options", {"bands": [1, 2, 3]}, "Map.rgb_composite"),
            ("selectors", {"time": 0}, "variable="),
            ("axes", ("y", "x"), "variable="),
        ],
    )
    def test_a_keyword_this_seam_cannot_serve_is_refused_by_name(
        self, name, value, names_alternative
    ):
        """Each unservable keyword is named, with what to do instead.

        Args:
            name: The forwarded keyword.
            value: A plausible value for it.
            names_alternative: Text the message must carry, pointing somewhere real.

        Test scenario:
            pyramids drops defaulted arguments before forwarding, so anything arriving here was set
            deliberately. Left to the tier each of these is refused too, but for the builder it
            reached — ``RasterMixin.field() got an unexpected keyword argument 'rgb_options'`` on
            web, ``'rgb_options' is not an option Image takes on the bokeh backend`` on interactive,
            and PyVista's own ``invalid keyword argument for `_common_arg_parser``` on 3-D. Refusing
            here names the pyramids-side keyword and where the capability lives, which is what
            ``names_alternative`` pins (round-1 H5, N1).
        """
        forwarded = {name: value}
        with pytest.raises(ValueError, match=name) as raised:
            _Forwarded.from_facade("web", forwarded)
        assert names_alternative in str(raised.value), (
            f"the refusal should point at {names_alternative!r}: {raised.value}"
        )

    @pytest.mark.parametrize(
        "name, value",
        [("bands", [0, 1]), ("animation_axis_values", [1, 2]), ("interval", 200)],
    )
    def test_an_animation_keyword_on_a_still_plot_is_refused(self, name, value):
        """An animation-only keyword is refused on a still plot rather than ignored.

        Args:
            name: The animation-only keyword.
            value: A plausible value for it.

        Test scenario:
            ``bands`` and ``interval`` were *accepted* and silently dropped by the old path, so
            ``ds.plot(backend="web", bands=[0, 1])`` drew one band and said nothing.
        """
        forwarded = {name: value}
        with pytest.raises(ValueError, match="only applies to an animation"):
            _Forwarded.from_facade("web", forwarded)


class TestAnimation:
    """The animation calls the facades make, and the one tier that cannot serve them."""

    def test_the_frame_interval_becomes_a_rate(self):
        """``interval`` (milliseconds per frame) becomes the ``fps`` every tier verb takes.

        Test scenario:
            The facades describe animation speed as an interval and digitalearth's verbs take a
            rate, so forwarding the number unchanged would play a 200 ms animation at 200 fps.
        """
        plan = _Forwarded.from_facade("interactive", {"animate": True, "interval": 250})
        assert plan.fps == 4.0, f"250 ms a frame is 4 fps, got {plan.fps}"

    @pytest.mark.parametrize("interval", [0, -5, "fast", True])
    def test_an_unusable_interval_is_refused(self, interval):
        """A non-positive or non-numeric interval is refused, not divided by.

        Args:
            interval: A value that is not a usable frame interval.
        """
        forwarded = {"animate": True, "interval": interval}
        with pytest.raises(ValueError, match="frame interval in milliseconds"):
            _Forwarded.from_facade("interactive", forwarded)

    def test_the_animating_backends_are_the_two_with_a_data_animation_verb(self):
        """Only the tiers with a verb the seam can drive are listed as animating.

        Test scenario:
            The 3-D tier's ``record`` needs a caller-supplied ``update`` callable and its ``orbit``
            sweeps the camera over a static scene; neither can be built from forwarded plot
            keywords, so it is deliberately absent rather than served a still picture.
        """
        assert sorted(ANIMATION_BACKENDS) == ["interactive", "web"], (
            f"expected the two tiers with a data-animation verb, got {sorted(ANIMATION_BACKENDS)}"
        )

    def test_a_three_d_animation_is_refused_naming_the_verbs_to_use_instead(self):
        """``animate(backend="3d")`` is refused by name, pointing at the real 3-D verbs.

        Test scenario:
            The alternative to a by-name refusal is drawing a still scene for an ``animate`` call,
            or spinning the camera and calling that the animation — both answer a different question
            than the one asked.
        """
        with pytest.raises(ValueError, match="Scene3D.record"):
            three_d("a-dataset", backend="3d", animate=True)

    def test_a_web_animation_without_a_destination_is_refused(self):
        """The web tier writes its animation to a file, so a missing ``path`` is named.

        Test scenario:
            ``WebMap.save_animation`` takes a path and has no in-memory form, and the facade forwards
            no destination of its own — so without ``path=`` there is nowhere for the animation to
            go. Saying so beats a ``TypeError`` about a missing positional argument.
        """
        with pytest.raises(ValueError, match="needs a path"):
            web("a-dataset", backend="web", animate=True)


class TestDelegation:
    """What actually reaches ``quickmap`` for a still plot."""

    @pytest.mark.parametrize("name, fn, _declared", REGISTERED)
    def test_the_backend_name_and_translated_band_reach_quickmap(
        self, name, fn, _declared, monkeypatch
    ):
        """A still plot calls ``quickmap`` once, with the backend name and the shifted band.

        Args:
            name: The backend name the seam passes.
            fn: The render callable.
            _declared: Unused here; the parametrisation is shared.
            monkeypatch: pytest's patcher.

        Test scenario:
            The adapter's still-plot job is this delegation, asserted directly rather than inferred
            from a drawn figure, which would need three engines. ``quickmap`` is looked up on the
            package at call time, so replacing the attribute is enough.
        """
        seen = {}

        def fake_quickmap(data, **kwargs):
            seen["data"] = data
            seen["kwargs"] = kwargs
            return _FakeScene()

        monkeypatch.setattr(digitalearth, "quickmap", fake_quickmap)
        fn("a-dataset", backend=name, band=0)
        assert seen["data"] == "a-dataset", (
            "the object must be passed through untouched"
        )
        assert seen["kwargs"] == {"backend": name, "band": 1}, (
            f"quickmap received {seen['kwargs']}"
        )

    @pytest.mark.parametrize("name, fn", [("web", web), ("3d", three_d)])
    def test_a_title_is_set_on_the_scene_where_the_builder_refuses_it(
        self, name, fn, monkeypatch
    ):
        """On web and 3-D, ``title`` is applied after the build, not handed to the builder.

        Args:
            name: The backend name.
            fn: The render callable.
            monkeypatch: pytest's patcher.

        Test scenario:
            Forwarded to the builder, ``title`` raised ``RasterMixin.field() got an unexpected
            keyword argument 'title'`` on the web tier and leaked PyVista's ``"title" is an invalid
            keyword argument for `_common_arg_parser``` on the 3-D one. Both tiers own ``set_title``,
            so the title is set on the finished scene instead (round-1 H2).
        """
        scene = _FakeScene()
        seen = {}

        def fake_quickmap(data, **kwargs):
            seen["kwargs"] = kwargs
            return scene

        monkeypatch.setattr(digitalearth, "quickmap", fake_quickmap)
        fn("a-dataset", backend=name, title="DEM")
        assert "title" not in seen["kwargs"], (
            f"the builder must not be offered the title: {seen['kwargs']}"
        )
        assert scene.titles == ["DEM"], (
            f"the scene should carry the title, got {scene.titles}"
        )

    def test_the_holoviz_tier_keeps_the_title_as_a_builder_keyword(self, monkeypatch):
        """On the interactive tier ``title`` stays with the builder, which is its working route.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            That tier has no ``set_title``, and its builder does accept ``title`` — executed on all
            three tiers. Routing it per tier rather than uniformly is therefore deliberate, and this
            pins the half that differs.
        """
        scene = _FakeScene()
        seen = {}

        def fake_quickmap(data, **kwargs):
            seen["kwargs"] = kwargs
            return scene

        monkeypatch.setattr(digitalearth, "quickmap", fake_quickmap)
        interactive("a-dataset", backend="interactive", title="DEM")
        assert seen["kwargs"].get("title") == "DEM", (
            f"the builder should receive the title here: {seen['kwargs']}"
        )
        assert scene.titles == [], f"set_title must not also be called: {scene.titles}"

    def test_the_default_backend_name_matches_the_registered_name(self):
        """Each callable's default ``backend`` is the name it is registered under.

        Test scenario:
            The seam always passes ``backend=`` explicitly, but the default is what makes each
            callable usable directly (and what a reader checks it against), so a mismatch between
            the default and the registered name would be a quiet lie.
        """
        defaults = {
            fn.__name__: inspect.signature(fn).parameters["backend"].default
            for _name, fn, _declared in REGISTERED
        }
        assert defaults == {
            "interactive": "interactive",
            "three_d": "3d",
            "web": "web",
        }, f"the defaults should match the registered names, got {defaults}"


class TestImportWeight:
    """Discovery imports this module, so it must not pull a drawing stack in."""

    def test_importing_the_module_pulls_in_no_drawing_stack(self):
        """Importing the backend module imports no engine, and not even pyramids.

        Test scenario:
            Discovery imports every registered entry point. A heavy import here would load a
            renderer into a bare ``import pyramids`` — the cost the upstream contract explicitly
            asks plugins to avoid. Measured in a fresh interpreter, because this one has already
            imported most of the package.
        """
        probe = (
            "import sys\n"
            "import digitalearth.pyramids_backend\n"
            "heavy = sorted(m for m in sys.modules\n"
            "    if m.split('.')[0] in {'pyvista','holoviews','geoviews','bokeh','panel',\n"
            "                           'maplibre','matplotlib','cleopatra','pyramids'})\n"
            "print(','.join(heavy))\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(__import__("pathlib").Path(digitalearth.__file__).parents[2]),
        )
        pulled = [name for name in done.stdout.strip().split(",") if name]
        assert pulled == [], f"importing the backend module pulled in {pulled}"


class TestWhatTheSeamNeedsToDraw:
    """The object shapes each path needs, refused in terms of the call that was made."""

    class _Single:
        """Stands in for a single raster: nothing to step through."""

    def test_animating_something_with_no_steps_is_refused_by_name(self):
        """An animation of a single raster says so, instead of failing inside the stack builder.

        Test scenario:
            ``Dataset.animate(backend=…)`` forwards a single ``Dataset``, and both tiers' temporal
            builders step through a collection's members — so it reached ``timecube`` and raised
            ``AttributeError: 'Dataset' object has no attribute 'datasets'`` (round-1 H3).
        """
        single = self._Single()
        with pytest.raises(ValueError, match="steps through a stack"):
            interactive(single, backend="interactive", animate=True)

    def test_the_refusal_names_both_shapes_this_seam_animates(self):
        """The refusal names a collection and a NetCDF, the two things it can step through.

        Test scenario:
            A caller is not left guessing: the message states the two accepted shapes rather than
            describing what their object is not (round-2 H3).
        """
        single = self._Single()
        with pytest.raises(ValueError) as raised:
            web(single, backend="web", animate=True, path="out.gif")
        message = str(raised.value)
        assert "DatasetCollection" in message, f"no collection route offered: {message}"
        assert "NetCDF" in message, f"no NetCDF route offered: {message}"


class TestFrameKeywordsWithNoRoute:
    """A frame keyword is refused when the scene has no verb for it, rather than silently skipped."""

    class _Bare:
        """A scene with neither a basemap verb nor a title verb."""

    def test_a_basemap_on_a_tier_with_no_basemap_verb_is_refused(self):
        """``basemap`` is named when the built scene has no verb to apply it with.

        Test scenario:
            Both tiers that animate do own ``basemap``, so this is the defensive arm — but a scene
            that loses the verb upstream would otherwise drop a requested basemap in silence, which
            is the failure mode this module exists to avoid.
        """
        plan = _Forwarded.from_facade("web", {"animate": True, "basemap": True})
        scene = self._Bare()
        with pytest.raises(ValueError, match="this tier has no basemap"):
            _framed(scene, plan, {})

    def test_a_title_on_a_tier_with_no_title_route_is_refused(self):
        """``title`` is named when the scene has neither a constructor argument nor ``set_title``.

        Test scenario:
            The HoloViz tier reaches the title through its builder and the other two through
            ``set_title``; a scene with neither would otherwise discard it without a word.
        """
        plan = _Forwarded.from_facade("web", {"animate": True, "title": "Flood"})
        scene = self._Bare()
        with pytest.raises(
            ValueError, match="neither a title argument nor a set_title verb"
        ):
            _framed(scene, plan, {})


class TestWhatCountsAsAStack:
    """Deciding what can be stepped through, by type rather than by attribute."""

    def test_a_vector_layer_with_a_datasets_column_is_still_drawable(self, monkeypatch):
        """A ``FeatureCollection`` carrying a column named ``datasets`` draws as the vector it is.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            ``FeatureCollection`` subclasses ``GeoDataFrame``, and pandas resolves a column name as
            an attribute — so an attribute test for ``datasets`` reported a catalogue or join result
            as a multi-member stack and refused to draw it, advising a ``FeatureCollection.animate``
            facade pyramids does not expose (round-2 H1).
        """
        from pyramids.feature import FeatureCollection

        layer = FeatureCollection.read_file("tests/data/points.geojson")
        layer["datasets"] = 1
        assert hasattr(layer, "datasets"), (
            "the column must resolve as an attribute for this test"
        )
        scene = _FakeScene()
        monkeypatch.setattr(digitalearth, "quickmap", lambda data, **kwargs: scene)
        assert web(layer, backend="web") is scene, (
            "a vector layer must still reach the builder"
        )

    def test_a_collection_is_refused_for_a_still_plot_by_type(self, monkeypatch):
        """A real ``DatasetCollection`` is still refused on the still path, now by type.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            The refusal itself is right — ``quickmap`` draws one field — and must survive the move
            from an attribute test to a type test.
        """
        from pyramids.dataset.collection import DatasetCollection

        stack = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 2)
        monkeypatch.setattr(
            digitalearth, "quickmap", lambda data, **kwargs: _FakeScene()
        )
        with pytest.raises(ValueError, match="draws one field"):
            web(stack, backend="web")

    def test_animating_something_that_is_no_stack_names_what_is_needed(self):
        """A plain object is refused with the two shapes this seam animates.

        Test scenario:
            The refusal previously named pyramids' internal ``Container`` class and told a caller
            their data "carries no steps" (round-2 H3). It now names what to pass instead.
        """

        class Neither:
            """Neither a collection nor a NetCDF."""

        neither = Neither()
        with pytest.raises(ValueError, match="steps through a stack"):
            interactive(neither, backend="interactive", animate=True)

    def test_a_named_axis_that_contradicts_an_explicit_kdim_is_refused(self):
        """``animate="time"`` beside ``kdim="level"`` is refused rather than silently overwritten.

        Test scenario:
            Both name the axis to step along. Overwriting one with the other would animate a
            dimension the caller did not ask for, having been told nothing (round-2 M4). Asserted
            on the keyword split, which needs no engine.
        """

        def timecube(collection, *, kdim="time", band=1):
            """Stands in for a temporal builder that takes a kdim."""

        plan = _Forwarded.from_facade(
            "interactive", {"animate": "time", "kdim": "level"}
        )
        with pytest.raises(ValueError, match="name different axes"):
            _animation_kwargs(plan, timecube)


class TestShapesTheFacadesSend:
    """Argument *shapes* a facade may forward that the tiers cannot act on as they arrive."""

    def test_a_basemap_mapping_is_unwrapped_to_its_provider(self):
        """A ``{"provider": name}`` mapping becomes the name the tiers take.

        Test scenario:
            The facades type ``basemap`` as ``bool | str | dict | Basemap | None``. A dict reached
            the web tier and came back as ``AttributeError: 'dict' object has no attribute
            'replace'``, while the 3-D and HoloViz tiers accepted it and drew no basemap at all
            (round-2 M3).
        """
        assert _basemap_asked({"provider": "CartoLight"}, "web") == "CartoLight", (
            "a mapping carrying a provider name should be unwrapped to it"
        )

    def test_a_basemap_mapping_with_no_provider_is_refused(self):
        """A mapping naming no provider is refused, saying what the tiers take instead.

        Test scenario:
            Silently dropping it would leave a caller believing a basemap was drawn — the failure
            this module exists to prevent — and passing it on produced an ``AttributeError`` from
            inside the tier.
        """
        with pytest.raises(ValueError, match="is not a shape"):
            _basemap_asked({"zoom": 3}, "web")

    @pytest.mark.parametrize("value", [True, "CartoLight", None, False])
    def test_the_shapes_the_tiers_already_take_pass_through(self, value):
        """``True``, a provider name and the unset values are handed on untouched.

        Args:
            value: A shape the tiers already accept.

        Test scenario:
            The translation must not start reinterpreting the shapes that already worked.
        """
        assert _basemap_asked(value, "web") is value, (
            f"{value!r} should pass through unchanged"
        )

    @pytest.mark.parametrize("name", ["selectors", "axes"])
    def test_the_netcdf_slice_refusal_names_the_band_origin(self, name):
        """The refusal that advises ``band=`` says which origin it means.

        Args:
            name: The refused read-time keyword.

        Test scenario:
            A caller following the advice passes ``band=`` as an extra keyword, and this seam shifts
            whatever it is handed — so advising the keyword without naming its origin reintroduced
            #501's own off-by-one, one layer up (round-2 M7).
        """
        forwarded = {name: {"time": 0}}
        with pytest.raises(ValueError, match="0-based band="):
            _Forwarded.from_facade("web", forwarded)


class TestLabelValuedColumn:
    """``column`` naming a label column, the ordinary reason the facade forwards it."""

    def test_a_label_column_is_classified_on_the_web_tier(self):
        """The web tier is told the scale is categorical, so a label column draws.

        Test scenario:
            Forwarded alone, ``column`` reached the classifier as a number:
            ``could not convert string to float: 'Baden-Württemberg'``. Executed on the web tier,
            naming the scale categorical draws it — and the facade forwards no ``scheme``, so the
            caller has no way to say so (round-2 M10).
        """
        layer = {"shapeName": np.array(["a", "b"])}
        draw = {"column": "shapeName"}
        _classified(layer, "web", draw)
        assert draw["scheme"] == "categorical", (
            f"the scale should be classified: {draw}"
        )

    def test_a_label_column_is_refused_on_the_tier_with_no_categorical_fill(self):
        """The 3-D tier is told it cannot, rather than failing inside its classifier.

        Test scenario:
            Executed on the 3-D tier, ``scheme="categorical"`` does not rescue it either — that tier
            colours by a number — so there is no route and the refusal names the tiers that have one.
        """
        layer = {"shapeName": np.array(["a", "b"])}
        draw = {"column": "shapeName"}
        with pytest.raises(ValueError, match="no categorical fill"):
            _classified(layer, "3d", draw)

    def test_a_numeric_column_is_left_alone(self):
        """A numeric column needs no classification and gets none.

        Test scenario:
            Adding ``scheme="categorical"`` to a numeric choropleth would replace a continuous scale
            with a categorical one, changing the picture the caller asked for.
        """
        layer = {"pop": np.array([1.0, 2.0])}
        draw = {"column": "pop"}
        _classified(layer, "web", draw)
        assert "scheme" not in draw, (
            f"a numeric column should not be classified: {draw}"
        )


class TestChoosingTheNetcdfVariable:
    """Which variable a NetCDF animation slices, and when there is no axis to slice."""

    def test_a_container_with_no_named_variable_uses_its_first(self):
        """With no ``variable`` forwarded, the same one the still path draws is animated.

        Test scenario:
            ``plot`` and ``animate`` must agree about which variable they mean, or the same call
            differing only in ``animate=`` would draw two different fields.
        """
        from pyramids.netcdf import NetCDF

        nc = NetCDF.read_file("examples/data/global/ersstv5.nc")
        plan = _Forwarded.from_facade("interactive", {"animate": True})
        chosen = _netcdf_variable(nc, plan)
        assert chosen._source_var_name == list(nc.variable_names)[0], (
            f"expected the container's first variable, got {chosen._source_var_name!r}"
        )

    def test_a_variable_already_pinned_is_used_as_it_is(self):
        """A variable pinned out of its container is animated directly, not re-selected.

        Test scenario:
            A pinned subset reports no ``variable_names`` of its own because it *is* the variable —
            the same test ``base.sources.extractors`` makes of a subset. Asking it for a variable
            by name would fail.
        """
        from pyramids.netcdf import NetCDF

        nc = NetCDF.read_file("examples/data/global/ersstv5.nc")
        pinned = nc.get_variable("sst")
        plan = _Forwarded.from_facade("interactive", {"animate": True})
        assert _netcdf_variable(pinned, plan) is pinned, (
            "a pinned variable should be used as it is"
        )

    def test_a_single_band_variable_has_no_axis_to_step_along(self):
        """A one-band variable is refused by name rather than animated as a single frame.

        Test scenario:
            ``time_bnds`` in the bundled file carries one band. An animation of it would be a
            one-frame player, which answers a different question than the one asked.
        """
        from pyramids.netcdf import NetCDF

        nc = NetCDF.read_file("examples/data/global/ersstv5.nc")
        plan = _Forwarded.from_facade(
            "interactive", {"animate": True, "variable": "time_bnds"}
        )
        with pytest.raises(ValueError, match="no axis to step along"):
            _netcdf_stack(nc, plan)


class TestTheVectorSeriesTheWebSliderTakes:
    """The second shape ``timeslider`` documents: a vector layer stepped through by time."""

    def test_a_vector_layer_is_animatable_on_the_web_tier(self):
        """A ``FeatureCollection`` is handed to the slider, which documents taking one.

        Test scenario:
            ``timeslider`` takes either a raster stack or a vector series; refusing the vector half
            would make a documented form of the call impossible on the only tier that offers it.
        """
        from pyramids.feature import FeatureCollection

        layer = FeatureCollection.read_file("tests/data/points.geojson")
        plan = _Forwarded.from_facade("web", {"animate": True, "path": "out.gif"})
        stack, labels, consumed = _animatable(layer, plan)
        assert stack is layer, "the vector layer should be passed through to the slider"
        assert labels is None, (
            f"a vector layer carries no step labels of its own: {labels}"
        )
        assert consumed == (), f"nothing is spent choosing it: {consumed}"

    def test_a_vector_layer_is_not_animatable_on_the_holoviz_tier(self):
        """``timecube`` takes a collection only, so the vector form is refused there.

        Test scenario:
            The two tiers' temporal builders differ, and claiming otherwise would send a vector
            layer into ``timecube``, which reads ``collection.datasets``.
        """
        from pyramids.feature import FeatureCollection

        layer = FeatureCollection.read_file("tests/data/points.geojson")
        plan = _Forwarded.from_facade("interactive", {"animate": True})
        with pytest.raises(ValueError, match="steps through a stack"):
            _animatable(layer, plan)


class TestTheRemainingFrameRoutes:
    """The still path's frame handling, and a scene with no verb for a fallback keyword."""

    def test_a_title_that_is_not_a_heading_is_refused_before_the_render(
        self, monkeypatch
    ):
        """A non-string title is refused before ``quickmap`` is called at all.

        Test scenario:
            Set on the finished scene, a bad title cost the whole picture: the 3-D tier built its
            ``Scene3D`` and then raised, while the web tier accepted an int and returned a map —
            two answers to one call (round-2 L5).
        """
        drawn = []
        monkeypatch.setattr(
            digitalearth,
            "quickmap",
            lambda data, **kwargs: drawn.append(True) or _FakeScene(),
        )
        with pytest.raises(ValueError, match="is not a heading"):
            web("a-dataset", backend="web", title=3)
        assert drawn == [], "the refusal must precede the render, not follow it"

    def test_a_still_plots_basemap_mapping_is_unwrapped_for_the_builder(
        self, monkeypatch
    ):
        """The still path normalises a ``basemap`` mapping too, not only the animate path.

        Test scenario:
            The still path hands ``basemap`` to ``quickmap`` rather than to ``_framed``, so a
            mapping reached the tier as a mapping: ``AttributeError: 'dict' object has no attribute
            'replace'`` (round-2 M3).
        """
        seen = {}

        def fake_quickmap(data, **kwargs):
            seen.update(kwargs)
            return _FakeScene()

        monkeypatch.setattr(digitalearth, "quickmap", fake_quickmap)
        web("a-dataset", backend="web", basemap={"provider": "CartoLight"})
        assert seen["basemap"] == "CartoLight", (
            f"the mapping should be unwrapped: {seen}"
        )

    def test_a_fallback_keyword_with_no_scene_verb_is_refused(self):
        """A keyword the builder cannot take and the scene cannot apply is named, not dropped.

        Test scenario:
            Both animating tiers own ``colorbar``, so this guards a scene that loses the verb —
            the same reasoning as the other two unreachable arms of ``_framed`` (round-2 L3).
        """

        class Bare:
            """A scene with no colorbar verb."""

        plan = _Forwarded.from_facade("web", {"animate": True})
        bare = Bare()
        with pytest.raises(
            ValueError, match="neither this render nor the scene takes it"
        ):
            _framed(bare, plan, {}, {"colorbar": True})

    def test_a_label_column_is_left_alone_on_the_holoviz_tier(self):
        """The HoloViz tier draws a label column unaided, so nothing is added for it.

        Test scenario:
            Executed: that tier colours by a label column without a ``scheme``. Supplying one
            anyway would change a picture that already worked.
        """
        layer = {"shapeName": np.array(["a", "b"])}
        draw = {"column": "shapeName"}
        _classified(layer, "interactive", draw)
        assert draw == {"column": "shapeName"}, f"nothing should be added here: {draw}"
