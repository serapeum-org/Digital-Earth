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
    _Forwarded,
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
        """Stands in for a single raster: no members to step through."""

    class _Stack:
        """Stands in for a collection: members to step through."""

        datasets = ("a", "b", "c")

    def test_animating_something_with_no_steps_is_refused_by_name(self):
        """An animation of a single raster says so, instead of failing inside the stack builder.

        Test scenario:
            ``Dataset.animate(backend=…)`` forwards a single ``Dataset``, and both tiers' temporal
            builders step through a collection's members — so it reached ``timecube`` and raised
            ``AttributeError: 'Dataset' object has no attribute 'datasets'`` (round-1 H3).
        """
        single = self._Single()
        with pytest.raises(ValueError, match="carries no steps to animate"):
            interactive(single, backend="interactive", animate=True)

    def test_the_refusal_names_the_matplotlib_route_for_a_single_rasters_bands(self):
        """The refusal points at the backend that does animate one raster's bands.

        Test scenario:
            A caller who wanted a band animation is not stuck: the matplotlib tier's ``animate``
            takes a stack of bands. Naming it is the difference between a refusal and a dead end.
        """
        single = self._Single()
        try:
            web(single, backend="web", animate=True, path="out.gif")
        except ValueError as exc:
            message = str(exc)
        assert "backend='matplotlib'" in message, f"no route offered: {message}"

    def test_a_still_plot_of_a_stack_is_refused_in_terms_of_the_call_made(self):
        """A still plot of a collection is refused without naming an unrelated function.

        Test scenario:
            ``DatasetCollection.plot(backend=…)`` is one of the facades, and ``quickmap`` cannot
            draw a collection — it answered ``TypeError: quickplot cannot draw a DatasetCollection``,
            naming ``quickplot``, which the caller never used (round-1 M5).
        """
        stack = self._Stack()
        with pytest.raises(ValueError, match=r"plot\(backend='web'\) draws one field"):
            web(stack, backend="web")

    def test_the_still_refusal_offers_the_animation_route(self):
        """The refusal names ``animate`` as the way to show every member.

        Test scenario:
            A collection passed to a still plot is usually a caller wanting the series; pointing at
            the animate call answers what they were reaching for.
        """
        stack = self._Stack()
        try:
            interactive(stack, backend="interactive")
        except ValueError as exc:
            message = str(exc)
        assert "animate(backend='interactive')" in message, (
            f"no route offered: {message}"
        )


class TestMeshIsNotDrawnHereYet:
    """Why the ``variable_name`` translation is inert, pinned so it stops being inert loudly."""

    def test_no_plugin_backend_draws_a_mesh_yet(self):
        """None of the three registered backends has a mesh render, so a mesh never reaches them.

        Test scenario:
            ``variable_name`` is forwarded only by the ``UgridDataset`` facades, and ``quickmap``
            refuses a mesh on every backend outside ``MESH_BACKENDS`` before reading any keyword —
            executed on all three tiers through the repo's mesh fixture. That is what makes the
            translation unreachable today (round-1 M1). If a tier gains a triangular render, this
            fails, which is the signal to cover that translation for real.
        """
        from digitalearth.api import MESH_BACKENDS

        registered = {name for name, _fn, _declared in REGISTERED}
        overlap = registered & set(MESH_BACKENDS)
        assert overlap == set(), (
            f"{sorted(overlap)} now draws a mesh, so variable_name is reachable and needs a test "
            f"that exercises it through that tier"
        )
