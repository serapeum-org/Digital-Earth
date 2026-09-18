"""Tests for T9.1 — quickplot/quickmap one-call entry points and module functions."""

import pytest

import digitalearth
from digitalearth import api as qp
from digitalearth.static import Map


def test_quickplot_returns_finished_map(dataset):
    """quickplot(dataset) returns a Map with one drawn layer and a colorbar."""
    m = digitalearth.quickplot(dataset, crs=dataset.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
    assert len(m.fig.axes) == 2  # main axes + colorbar


def test_quickmap_kind_dispatch(dataset):
    """quickmap honours an explicit raster kind."""
    m = qp.quickmap(dataset, crs=dataset.epsg, kind="contourf")
    assert m.layers


def test_quickmap_saves_png(dataset, tmp_path):
    """The finished map can be saved to a PNG file."""
    m = qp.quickmap(dataset, crs=dataset.epsg)
    out = tmp_path / "quick.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_quickmap_scatter_features():
    """A FeatureCollection of points is drawn as a scatter map."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.quickmap(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_choropleth_polygons():
    """A polygon FeatureCollection with a column is drawn as a choropleth."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg, column="fid")
    assert m.ax.collections


def _zoned_polygons():
    """Buffered-point polygons carrying a nominal 'zone' column (three unordered classes)."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    fc["zone"] = [["urban", "rural", "park"][i % 3] for i in range(len(fc))]
    return fc


def test_quickmap_categorical_has_no_spurious_colorbar():
    """A categorical quickmap keys itself with a swatch legend, so no meaningless numeric colorbar is added.

    The mappable carries integer class codes; a colorbar over them would read -0.5, 0.5, 1.5 next to the swatch
    legend — the one-call api must skip it (M2), leaving a single axes.
    """
    fc = _zoned_polygons()
    m = qp.quickmap(fc, crs=fc.epsg, column="zone", scheme="categorical")
    assert len(m.fig.axes) == 1, (
        "categorical map must not gain a colorbar axes on top of its swatch legend"
    )
    assert m.layers[-1][0].category_legend is not None, (
        "the swatch legend is still the key"
    )


def test_module_choropleth_categorical_has_no_spurious_colorbar():
    """The module-level choropleth() helper (the _finish path) also skips the colorbar for a categorical fill."""
    fc = _zoned_polygons()
    m = qp.choropleth(fc, crs=fc.epsg, column="zone", scheme="categorical")
    assert len(m.fig.axes) == 1, (
        "the _finish path must skip the colorbar for a categorical fill too"
    )


def test_quickmap_graduated_still_gets_its_colorbar():
    """A non-categorical fill must still receive its aggregated colorbar (the M2 guard must not over-fire)."""
    fc = _zoned_polygons()
    m = qp.quickmap(fc, crs=fc.epsg, column="fid", scheme="quantiles", k=3)
    assert len(m.fig.axes) == 2, "a graduated numeric fill still carries a colorbar"


def test_last_layer_is_categorical_on_empty_scene():
    """The categorical predicate is False on a scene with no layers (self-safe when called directly)."""
    assert qp._last_layer_is_categorical(Map(crs=4326)) is False, (
        "an empty scene has no categorical layer"
    )


def test_quickmap_rejects_unsupported_type():
    """quickmap raises on an input type it cannot draw."""
    with pytest.raises(TypeError, match="cannot draw"):
        qp.quickmap("not data")


def test_quickmap_rejects_empty_features():
    """quickmap raises a clear error on an empty FeatureCollection (not silently treated as polygons)."""
    from pyramids.feature import FeatureCollection

    empty = FeatureCollection.read_file("tests/data/points.geojson").iloc[0:0]
    with pytest.raises(ValueError, match="empty FeatureCollection"):
        qp.quickmap(empty, crs=4326)


def test_module_function_contourf(dataset):
    """The module-level contourf builds a finished Map via the contourf kind."""
    m = qp.contourf(dataset, crs=dataset.epsg)
    assert m.layers


def test_module_function_grid_cells(dataset):
    """The module-level grid_cells builds a finished Map with a polygon layer."""
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers


def test_quickmap_with_domain(dataset):
    """quickmap(domain=...) sets the axes extent from the named region."""
    m = qp.quickmap(dataset, crs=3857, domain="europe")
    assert m.ax.get_xlim()[1] > 1e6  # reprojected Europe bbox in metres


@pytest.mark.parametrize(
    ("basemap", "expected_args"),
    [(True, ()), ("CartoDark", ("CartoDark",))],
)
def test_interactive_basemap_branches_reach_the_tier(
    dataset, mocker, basemap, expected_args
):
    """``basemap=True`` asks for the tier's default; a named source is forwarded unchanged.

    Args:
        dataset: The raster to draw.
        mocker: Patches the tile builder so no engine or network is needed.
        basemap: The spelling under test.
        expected_args: What the tile builder must be called with.

    Test scenario:
        The two spellings take different branches — ``True`` means "the backend's default", a name means that
        source — and only one was covered. Reducing a named preset such as ``"Planet.NICFI"`` to the default
        would silently draw the wrong map, which is exactly the kind of failure that leaves no trace.
    """
    interactive = pytest.importorskip("digitalearth.interactive")
    tiles = mocker.patch.object(interactive.InteractiveMap, "tiles")
    mocker.patch.object(interactive.InteractiveMap, "image")
    qp.quickmap(dataset, backend="interactive", basemap=basemap)
    assert tiles.call_args.args == expected_args, (
        f"basemap={basemap!r} must call tiles{expected_args}, got {tiles.call_args!r}"
    )


@pytest.mark.parametrize(
    ("basemap", "expected_args"),
    [(True, ()), ("cartodark", ("cartodark",))],
)
def test_web_basemap_branches_reach_the_tier(dataset, mocker, basemap, expected_args):
    """``basemap=True`` asks for the web tier's default; a named provider is forwarded unchanged.

    Args:
        dataset: The raster to draw.
        mocker: Patches the basemap and raster builders so no engine or network is needed.
        basemap: The spelling under test.
        expected_args: What ``WebMap.basemap`` must be called with.

    Test scenario:
        The web dispatcher's two spellings used to share one line, so running either counted as covering
        both. They are different calls: ``True`` means "whatever the tier defaults to", a name means that
        provider, and collapsing a named provider to the default draws a different map with nothing to show
        for it. This is the web counterpart of the interactive check above.
    """
    web = pytest.importorskip("digitalearth.web")
    tiles = mocker.patch.object(web.WebMap, "basemap")
    mocker.patch.object(web.WebMap, "field")
    qp.quickplot(dataset, backend="web", basemap=basemap)
    assert tiles.call_args.args == expected_args, (
        f"basemap={basemap!r} must call basemap{expected_args}, got {tiles.call_args!r}"
    )


def test_quickmap_decorations_best_effort(dataset):
    """coastlines/basemap flags run without raising even if data/tiles are unreachable."""
    m = qp.quickmap(dataset, crs=3857, coastlines=True, basemap=True)
    assert m.layers  # the raster layer is drawn regardless of decoration availability


def test_quickmap_shapes_without_column_skips_colorbar():
    """A polygon FeatureCollection with no column draws outlines and skips the colorbar gracefully."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg)  # no column -> shapes (outline only)
    assert m.ax.collections


def test_module_choropleth(dataset):
    """The module-level choropleth colours polygons by a column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.choropleth(fc, column="fid", crs=fc.epsg)
    assert m.ax.collections


def test_module_scatter():
    """The module-level scatter draws a point FeatureCollection."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.scatter(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_colorbar_false(dataset):
    """colorbar=False leaves the figure with a single axes (no colorbar axes)."""
    m = qp.quickmap(dataset, crs=dataset.epsg, colorbar=False)
    assert len(m.fig.axes) == 1


def test_quickmap_warns_on_decoration_failures(dataset, mocker):
    """When coastlines/basemap/colorbar fail for an expected reason, quickmap warns and still returns."""
    mocker.patch.object(Map, "coastlines", side_effect=ConnectionError("no net"))
    mocker.patch.object(Map, "basemap", side_effect=ConnectionError("no tiles"))
    mocker.patch.object(Map, "colorbar", side_effect=ValueError("bad mappable"))
    m = qp.quickmap(
        dataset, crs=dataset.epsg, coastlines=True, basemap=True, colorbar=True
    )
    assert m.layers


def test_module_grid_cells_warns_on_colorbar_failure(dataset, mocker):
    """grid_cells still returns a map when its colorbar fails for an expected reason."""
    mocker.patch.object(Map, "colorbar", side_effect=ValueError("bad mappable"))
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers


class _FakeScene:
    """Minimal scene stand-in for _finish: a .layers list and a recording/optionally-raising .colorbar()."""

    def __init__(self, layers, raises=False):
        self.layers = list(layers)
        self._raises = raises
        self.colorbar_calls = 0

    def colorbar(self):
        """Record the call (and optionally raise to mimic an outline-only/unmappable layer)."""
        self.colorbar_calls += 1
        if self._raises:
            raise ValueError("nothing mappable to colorbar")


class TestFinish:
    """Tests for api._finish (PA-5)."""

    def test_adds_colorbar_when_requested_and_layers_present(self):
        """_finish draws a colorbar when colorbar=True and a layer exists.

        Test scenario:
            A scene with one layer and colorbar=True gets exactly one colorbar() call.
        """
        scene = _FakeScene(layers=["layer"])
        out = qp._finish(scene, colorbar=True)
        assert scene.colorbar_calls == 1, (
            f"expected one colorbar call, got {scene.colorbar_calls}"
        )
        assert out is scene, "the same scene must be returned"

    def test_skips_colorbar_when_disabled(self):
        """_finish never draws a colorbar when colorbar=False.

        Test scenario:
            Even with layers present, colorbar=False suppresses the colorbar() call.
        """
        scene = _FakeScene(layers=["layer"])
        out = qp._finish(scene, colorbar=False)
        assert scene.colorbar_calls == 0, "colorbar must not be drawn when disabled"
        assert out is scene, "the same scene must be returned"

    def test_skips_colorbar_when_no_layers(self):
        """_finish skips the colorbar when there are no layers, even if requested.

        Test scenario:
            An empty scene with colorbar=True draws nothing (no layer to map).
        """
        scene = _FakeScene(layers=[])
        out = qp._finish(scene, colorbar=True)
        assert scene.colorbar_calls == 0, "colorbar must not be drawn without layers"
        assert out is scene, "the same scene must be returned"

    def test_swallows_colorbar_exception(self):
        """_finish swallows an exception from colorbar() (outline-only/unmappable layer).

        Test scenario:
            colorbar() raising must not propagate; the scene is still returned.
        """
        scene = _FakeScene(layers=["layer"], raises=True)
        out = qp._finish(scene, colorbar=True)
        assert scene.colorbar_calls == 1, "colorbar() should have been attempted once"
        assert out is scene, "the scene must be returned despite the swallowed error"


class TestBackendCapabilityRefusal:
    """C11 — ``quickmap`` refuses a parameter the chosen backend cannot honour, instead of dropping it.

    The four backends are not the same map. Only the matplotlib one has a fixed extent to set, only the 2-D
    ones have a display CRS or a coastline layer, and the web tier keys itself with a legend rather than a
    colorbar. ``quickmap`` used to take every keyword for every backend and quietly discard the ones that did
    not apply, so ``quickmap(ds, backend="web", domain="europe")`` returned a world map with no hint that the
    domain had gone nowhere. Each such parameter is now checked against
    :data:`digitalearth.api.BACKEND_CAPABILITIES` and refused by name.

    These tests deliberately do not build a scene where they can avoid it: the refusal happens before the
    backend is imported, which is what lets them run without the ``3d``/``web``/``interactive`` extras.
    """

    @pytest.mark.parametrize(
        ("backend", "parameter", "value"),
        [
            ("3d", "domain", "europe"),
            ("3d", "coastlines", True),
            ("3d", "basemap", True),
            ("3d", "kind", "contourf"),
            ("web", "domain", "europe"),
            ("web", "coastlines", True),
            ("web", "kind", "contourf"),
            ("interactive", "domain", "europe"),
        ],
    )
    def test_an_unsupported_parameter_names_itself_and_the_backend(
        self, dataset, backend, parameter, value
    ):
        """Every unsupported (backend, parameter) pair raises a ValueError naming both.

        Args:
            dataset: The raster to draw (never actually drawn — the refusal comes first).
            backend: The backend that cannot honour the parameter.
            parameter: The parameter it cannot honour.
            value: A value that genuinely requests something.

        Test scenario:
            A caller who gets a plain map back has no way to tell a dropped argument from one that had no
            visible effect. The message has to carry both halves so they know which one to change.
        """
        with pytest.raises(ValueError) as raised:
            qp.quickmap(dataset, backend=backend, **{parameter: value})
        message = str(raised.value)
        assert f"{parameter}= is not supported" in message, (
            f"the error must name the parameter, got {message!r}"
        )
        assert f"backend={backend!r}" in message, (
            f"the error must name the backend, got {message!r}"
        )

    @pytest.mark.parametrize(
        ("backend", "parameter"),
        [
            ("3d", "domain"),
            ("3d", "coastlines"),
            ("3d", "basemap"),
            ("3d", "kind"),
            ("web", "coastlines"),
            ("web", "kind"),
            ("web", "colorbar"),
        ],
    )
    def test_a_parameter_that_asks_for_nothing_is_not_a_dropped_request(
        self, dataset, backend, parameter, mocker
    ):
        """``domain=None`` / ``coastlines=False`` pass anywhere: nothing was requested, so nothing was lost.

        Args:
            dataset: The raster to draw.
            backend: The backend that cannot honour the parameter.
            parameter: The parameter, passed at its inert value.
            mocker: Stubs the backend builder so no optional extra is needed.

        Test scenario:
            The rule is about *dropped requests*, not about the presence of a keyword. Refusing an explicit
            ``coastlines=False`` on the web tier would be pedantry — there were no coastlines to lose — and
            would break callers who pass one kwargs dict through to whichever backend they picked.

            ``colorbar=False`` is the same shape and was refused anyway until review L9. It is no longer a
            special case at all: since #254 the web tier honours ``colorbar=`` in both directions, building
            its key through ``WebMap.legend``, so neither value is a dropped request there.
        """
        builder = mocker.patch.object(
            qp, "_quickmap_3d" if backend == "3d" else "_quickmap_web"
        )
        inert = {
            "domain": None,
            "coastlines": False,
            "basemap": False,
            "kind": "auto",
            "colorbar": False,
        }[parameter]
        qp.quickmap(dataset, backend=backend, **{parameter: inert})
        assert builder.called, (
            "an inert value must not stop the call reaching the backend"
        )

    def test_an_unset_parameter_is_never_refused(self, dataset, mocker):
        """A backend is not refused a parameter its caller never named.

        Args:
            dataset: The raster to draw.
            mocker: Stubs the 3-D builder so pyvista is not needed.

        Test scenario:
            ``crs`` and ``colorbar`` carry real defaults (``3857`` and ``True``), so the default value cannot
            double as "not passed". Without a separate sentinel, the plainest call there is —
            ``quickmap(ds, backend="3d")`` — would be refused for a CRS nobody asked for. ``colorbar``
            still needs the sentinel after L9: its inert value is ``False``, and its default is ``True``.
        """
        builder = mocker.patch.object(qp, "_quickmap_3d")
        qp.quickmap(dataset, backend="3d")
        assert builder.called, "a bare backend='3d' call must not be refused"

    def test_the_refusal_precedes_building_anything(self, dataset, mocker):
        """Nothing is constructed before the check, so a refused call leaks no figure or plotter.

        Args:
            dataset: The raster to draw.
            mocker: Watches the 3-D backend builder.

        Test scenario:
            A ``Scene3D`` opens a VTK render window in its constructor and a ``Map`` opens a matplotlib
            figure. Checking first is what keeps every rejected call free of a resource nobody will close.
        """
        three_d = mocker.patch.object(qp, "_quickmap_3d")
        with pytest.raises(ValueError, match="domain="):
            qp.quickmap(dataset, backend="3d", domain="europe")
        assert not three_d.called, "a refused call must not reach the backend builder"

    def test_a_crs_is_forwarded_to_the_3d_backend(self, dataset, mocker):
        """The 3-D tier has a display CRS, so ``crs=`` reaches its builder instead of being refused (#291).

        Args:
            dataset: The raster to draw.
            mocker: Replaces the 3-D backend builder with a recorder.
        """
        three_d = mocker.patch.object(qp, "_quickmap_3d")
        qp.quickmap(dataset, backend="3d", crs=4326)
        assert three_d.call_args.kwargs["crs"] == 4326, three_d.call_args

    def test_every_backend_has_a_capability_entry(self):
        """The dispatcher and the capability table name the same backends.

        Test scenario:
            The table is what the unknown-backend check now reads, so a backend added to one and not the
            other would either be unreachable or escape the parameter check entirely.
        """
        assert set(qp.BACKEND_CAPABILITIES) == {
            "matplotlib",
            "interactive",
            "3d",
            "web",
        }, f"unexpected backend set: {sorted(qp.BACKEND_CAPABILITIES)}"

    def test_an_unknown_backend_still_names_the_real_ones(self, dataset):
        """An unrecognised backend raises before the capability lookup, listing the four that exist.

        Args:
            dataset: The raster to draw.

        Test scenario:
            Routing the unknown-backend check through the capability table must not turn a clear error into
            a ``KeyError`` from inside the refusal helper.
        """
        with pytest.raises(ValueError, match="unknown backend"):
            qp.quickmap(dataset, backend="opengl")

    def test_the_not_passed_sentinel_reads_as_unset(self):
        """The "not passed" sentinel spells itself ``<unset>`` wherever a default is rendered.

        Test scenario:
            ``help(quickmap)``, the rendered signature and every IDE hint print a default with ``repr``.
            Without a spelling of its own the sentinel shows as ``<digitalearth.api._Unset object at
            0x...>``, which reads as an implementation leak rather than as "leave this one alone".
        """
        import inspect

        rendered = str(inspect.signature(qp.quickmap))
        assert repr(qp._UNSET) == "<unset>", (
            f"the sentinel must spell itself <unset>, got {qp._UNSET!r}"
        )
        assert "<unset>" in rendered, (
            f"the rendered signature must show <unset> for a sentinel default, got {rendered!r}"
        )

    def test_matplotlib_honours_all_four(self, dataset):
        """The default backend takes every checked parameter, exactly as it did before.

        Args:
            dataset: The raster to draw.

        Test scenario:
            C11 is about the tiers that *cannot* honour a parameter. The static tier can honour all four, so
            the guard must be invisible to it — this is the regression that would catch an over-fire.
        """
        m = qp.quickmap(
            dataset, crs=3857, domain="europe", coastlines=False, colorbar=False
        )
        assert len(m.fig.axes) == 1, (
            "colorbar=False must still suppress the colorbar axes"
        )
        assert m.layers, "the data layer must still be drawn"

    def test_web_receives_the_crs_it_was_given(self, dataset, mocker):
        """``crs=`` is forwarded to the web tier rather than dropped — it has one, and validates it.

        Args:
            dataset: The raster to draw.
            mocker: Stubs ``_quickmap_web`` so the ``web`` extra is not needed.

        Test scenario:
            The web tier carries its own ``crs`` and refuses anything but 4326, explaining why inline data
            can only be placed in lon/lat. Forwarding lets that tier answer; dropping the argument here, as
            the old code did, was exactly the silent failure C11 exists to remove.
        """
        forward = mocker.patch.object(qp, "_quickmap_web")
        qp.quickmap(dataset, backend="web", crs=4326)
        assert forward.call_args.kwargs["crs"] == 4326, (
            f"crs must reach the web builder, got {forward.call_args!r}"
        )


class TestTheRefusalNamesWhatTheCallerWrote:
    """Round-2 review L3/L4/L5 and M19 — a refusal must name a keyword the caller can actually act on."""

    def test_an_array_domain_is_refused_by_name_not_by_numpy(self):
        """A bbox handed over as an ndarray gets this module's message, not numpy's ambiguity error.

        Test scenario:
            ``_INERT`` was consulted with ``==``, so ``value == None`` on an array returned an *array* and
            the ``if`` over it raised "The truth value of an array with more than one element is
            ambiguous" — from inside the guard whose whole job is to name the offending parameter (L3).
        """
        import numpy as np

        with pytest.raises(
            ValueError, match=r"domain= is not supported by backend='web'"
        ):
            qp._reject_unsupported(
                "web", domain=np.array([0.0, 0.0, 1.0, 1.0]), crs=qp._UNSET
            )

    @pytest.mark.parametrize(
        ("backend", "kwargs"),
        [("3d", {"basemap": 0}), ("web", {"coastlines": 0})],
    )
    def test_a_numeric_zero_is_not_the_inert_false(self, backend, kwargs):
        """``0 == False`` is ``True`` in Python, but ``basemap=0`` is a value the caller typed.

        Args:
            backend: The tier that cannot honour the keyword.
            kwargs: The keyword, written as the number zero.

        Test scenario:
            The inert check means *identity* with the "asks for nothing" value, not equality with it, or a
            caller who passes a number through a shared kwargs dict is told nothing was dropped when it
            was (L4).
        """
        name = next(iter(kwargs))
        with pytest.raises(ValueError, match=rf"{name}= is not supported"):
            qp._reject_unsupported(backend, crs=qp._UNSET, **kwargs)

    @pytest.mark.parametrize("backend", ["web", "3d"])
    def test_a_module_wrapper_names_itself_not_the_kind_it_injected(
        self, dataset, backend
    ):
        """``imshow(ds, backend="web")`` must not tell the caller to drop a keyword they never wrote.

        Args:
            dataset: The raster to draw.
            backend: A tier with no renderer selector.

        Test scenario:
            The wrapper *is* the ``kind``: it injects ``kind="imshow"`` itself. The generic refusal then
            said "drop the argument", naming a parameter that does not appear in the caller's source (L5).
        """
        with pytest.raises(ValueError) as excinfo:
            qp.imshow(dataset, backend=backend)
        message = str(excinfo.value)
        assert message.startswith("imshow()"), message
        assert f"backend={backend!r}" in message, message
        assert "kind=" not in message, (
            f"the message must not name the injected keyword: {message}"
        )

    def test_a_column_on_point_input_is_refused_by_name(self):
        """``quickmap(points, column=...)`` names the keyword instead of leaking cleopatra's error.

        Test scenario:
            The polygon branch pops ``column`` and draws a choropleth; the point branch forwarded it into
            ``Map.scatter``'s ``**opts``, where cleopatra answered with its own accepted-keyword list and
            never mentioned ``column`` (M19). ``Map.scatter`` has no fill column — it sizes markers by
            ``size_column`` — so the honest answer is a refusal that says so.
        """
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        with pytest.raises(ValueError) as excinfo:
            qp.quickmap(fc, crs=fc.epsg, column="fid")
        message = str(excinfo.value)
        assert "column='fid'" in message, message
        assert "size_column=" in message, message
