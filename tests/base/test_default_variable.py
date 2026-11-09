"""Which variable a NetCDF container draws when the call names none.

pyramids-core 0.75 changed `NetCDF.variable_names` from the CF classification's sorted list to the store's own
declaration order. The set of names did not change and nothing warns, but the *first* name did — and three
readers here take the first one. On a store that declares a CF bounds array before its data, taking the
declared first draws the bounds array: `ersstv5.nc` declares `["time_bnds", "sst"]`, so `quickmap(nc)` drew
`sst` under 0.73 and `time_bnds` under 0.75.

A bounds array has no spatial plane, so it is not a thing a map can draw. These pin that the default is the
first variable that *can* be drawn, on whichever order the store happens to declare.
"""

import pytest
from pyramids.netcdf import NetCDF

#: The repository's own container, which declares its bounds array before its data.
CONTAINER = "examples/data/global/ersstv5.nc"


@pytest.fixture(scope="module")
def container():
    """The ERSST container, read once for the module.

    Returns:
        pyramids.netcdf.NetCDF: A container declaring `time_bnds` before `sst`.
    """
    return NetCDF.read_file(CONTAINER)


class TestTheDeclaredOrderPremise:
    """The fixture really does declare a non-plottable array first, or these tests prove nothing."""

    def test_the_store_declares_its_bounds_array_before_its_data(self, container):
        """`variable_names` reports the bounds array first, which is what makes the default wrong.

        Test scenario:
            The tests below are only meaningful while the store reports a non-drawable array *before* its
            data, so that is what is asserted — rather than the exact list, which would also fail on an
            upstream improvement. If a future pyramids drops CF non-data arrays from `variable_names`, the
            bounds array simply stops being reported and this says so, naming what it saw.
        """
        names = list(container.variable_names)

        assert "sst" in names, (
            f"the fixture must still carry its data variable, got {names}"
        )
        assert names and names[0] != "sst", (
            f"these tests need a non-drawable name reported before the data; variable_names is {names}. "
            "If pyramids now filters CF non-data arrays, the default-variable rule is no longer load-bearing "
            "for this fixture and this module needs a store that still exercises it."
        )
        assert sorted(names) != names, (
            f"the declared order must differ from the sorted one, or 0.75's reordering changed nothing "
            f"here and these tests would pass under either rule; got {names}"
        )


class TestTheDefaultIsDrawable:
    """All three readers skip a variable with no spatial plane and agree on the same answer."""

    def test_the_still_path_draws_the_data_not_the_bounds_array(self, container):
        """The extractor picks `sst`, not the `time_bnds` the store declares first.

        Test scenario:
            This is the variable `quickmap(nc)` and `plot(backend=...)` render with no `variable=` given.
        """
        from digitalearth.base.sources.extractors import _netcdf_variable

        chosen = _netcdf_variable(container, None, pinned=False)

        assert chosen == "sst", f"expected the drawable variable, got {chosen!r}"

    def test_the_animate_path_draws_the_same_variable(self, container):
        """The animated path picks the same variable as the still path.

        Test scenario:
            `plot` and `animate` differing only in `animate=` must not draw two different fields, which is
            why both read one shared answer rather than each taking `variable_names[0]`.
        """
        from digitalearth.pyramids_backend import _Forwarded, _netcdf_variable

        plan = _Forwarded.from_facade("interactive", {"animate": True})
        chosen = _netcdf_variable(container, plan)

        assert chosen._source_var_name == "sst", (
            f"expected the animated path to agree with the still path, got {chosen._source_var_name!r}"
        )

    def test_the_colour_key_is_titled_with_the_drawn_variable(self, container):
        """The colour key names `sst`, so the label agrees with what was drawn.

        Test scenario:
            The label is read off the raster before any drawer runs, so it is a third independent reader of
            the same question; a disagreement here titles the key with a field the figure does not show.
        """
        from digitalearth.base.sources.bandname import band_field_name

        assert band_field_name(container, 1) == "sst", (
            f"expected the colour key to name the drawn variable, got {band_field_name(container, 1)!r}"
        )


class _Info:
    """A stand-in for pyramids' ``VariableInfo``, carrying only what drawability is decided from."""

    def __init__(self, dimensions, attributes=None):
        """Record the variable's dimensions and CF attributes.

        Args:
            dimensions: The variable's dimension names.
            attributes: Its CF attributes, or ``None`` for none.
        """
        self.dimensions = dimensions
        self.attributes = attributes or {}


def _container(names, variables):
    """Build a stand-in container declaring `names` and describing `variables`.

    Args:
        names: What ``variable_names`` reports, in declaration order.
        variables: The ``metadata.variables`` mapping.

    Returns:
        type: A class exposing the container surface drawability is read from.
    """
    metadata = type("Metadata", (), {"variables": variables})
    return type("Container", (), {"variable_names": names, "metadata": metadata()})


#: A geographic pair, marked the way CF marks them.
LAT = _Info(["lat"], {"axis": "Y"})
LON = _Info(["lon"], {"axis": "X"})


class TestDrawableVariableNames:
    """Which variables `drawable_variable_names` considers drawable, and which it skips."""

    def test_a_pinned_subset_has_no_drawable_names(self):
        """A container declaring no variables reports none drawable, rather than consulting its metadata.

        Test scenario:
            A subset already pinned to one variable reports no `variable_names` because it *is* the
            variable. `default_variable` answers `None` for that shape, but `drawable_variable_names` is
            public and can be asked directly, so its own empty case is pinned here.
        """
        from digitalearth.base.sources.variable import drawable_variable_names

        pinned = _container(
            [], {"lat": LAT, "lon": LON, "sst": _Info(["lat", "lon"])}
        )()

        assert drawable_variable_names(pinned) == [], (
            f"a container declaring no variables has none drawable, got {drawable_variable_names(pinned)}"
        )

    def test_a_name_its_metadata_does_not_describe_is_skipped(self):
        """A declared name absent from `metadata.variables` cannot be classified, so it is passed over.

        Test scenario:
            A container may declare a name its metadata genuinely does not carry, so there are no
            dimensions to classify it by. Skipping it leaves the next drawable variable as the answer,
            rather than treating the undescribed one as drawable. This is *not* the grouped-store case —
            a bare name against path-keyed metadata is matched, and `TestAGroupedStore` covers it.
        """
        from digitalearth.base.sources.variable import (
            default_variable,
            drawable_variable_names,
        )

        container = _container(
            ["undescribed", "sst"],
            {"lat": LAT, "lon": LON, "sst": _Info(["time", "lat", "lon"])},
        )()

        assert drawable_variable_names(container) == ["sst"], (
            f"the undescribed name must be skipped, got {drawable_variable_names(container)}"
        )
        assert default_variable(container) == "sst", (
            "and the answer is the next drawable one"
        )

    def test_a_store_whose_variables_are_all_flat_reports_none_drawable(self):
        """With no variable carrying both horizontal axes, nothing is drawable and the first is used.

        Test scenario:
            Every declared variable spans time alone, so none has a plane. `drawable_variable_names` is
            empty and `default_variable` falls back to the declared first rather than returning nothing —
            a container with variables always answers with one of them.
        """
        from digitalearth.base.sources.variable import (
            default_variable,
            drawable_variable_names,
        )

        container = _container(
            ["time_bnds", "climatology_bounds"],
            {
                "lat": LAT,
                "lon": LON,
                "time_bnds": _Info(["time", "nbnds"]),
                "climatology_bounds": _Info(["time", "nv"]),
            },
        )()

        assert drawable_variable_names(container) == [], (
            f"a store with no planar variable has none drawable, got {drawable_variable_names(container)}"
        )
        assert default_variable(container) == "time_bnds", (
            "with nothing drawable the declared first is still the answer"
        )

    def test_a_projected_store_is_classified_by_its_standard_names(self):
        """A store naming projected axes by `standard_name` rather than `axis` is still classified.

        Test scenario:
            Not every store declares `axis`. A projected grid naming its coordinates
            `projection_x_coordinate` / `projection_y_coordinate` carries the same plane, so the variable
            spanning them is drawable.
        """
        from digitalearth.base.sources.variable import default_variable

        container = _container(
            ["bounds", "temp"],
            {
                "x": _Info(["x"], {"standard_name": "projection_x_coordinate"}),
                "y": _Info(["y"], {"standard_name": "projection_y_coordinate"}),
                "bounds": _Info(["time", "nv"]),
                "temp": _Info(["time", "y", "x"]),
            },
        )()

        assert default_variable(container) == "temp", (
            f"a projected store's planar variable must be drawable, got {default_variable(container)!r}"
        )


class TestAGroupedStore:
    """A sub-group view reports bare variable names while its metadata is keyed by full store path.

    pyramids documents that shape, and it is the one the first version of this module got wrong: dimension
    names and coordinate keys were stripped of their group path but the variable lookup was not, so every
    lookup missed, nothing classified, and the answer fell back to the declared first — the bounds array the
    module exists to skip.
    """

    def test_a_bare_name_still_finds_its_path_keyed_metadata(self):
        """Classification survives the two sides naming the same variable differently.

        Test scenario:
            `variable_names` reports `["time_bnds", "sst"]` while `metadata.variables` is keyed
            `forecast/time_bnds` / `forecast/sst`. The drawable answer must still be `sst`, not the fallback.
        """
        from digitalearth.base.sources.variable import (
            default_variable,
            drawable_variable_names,
        )

        container = _container(
            ["time_bnds", "sst"],
            {
                "forecast/lat": _Info(["/forecast/lat"], {"axis": "Y"}),
                "forecast/lon": _Info(["/forecast/lon"], {"axis": "X"}),
                "forecast/time_bnds": _Info(["/forecast/time", "/forecast/nbnds"]),
                "forecast/sst": _Info(
                    ["/forecast/time", "/forecast/lat", "/forecast/lon"]
                ),
            },
        )()

        assert drawable_variable_names(container) == ["sst"], (
            f"a bare name must match its path-keyed metadata, got {drawable_variable_names(container)}"
        )
        assert default_variable(container) == "sst", (
            f"expected the drawable variable, got {default_variable(container)!r}"
        )

    def test_a_path_qualified_name_finds_bare_metadata_too(self):
        """The same normalisation works when it is the names that carry the path.

        Test scenario:
            The mismatch can run the other way — `variable_names` reporting `forecast/sst` against metadata
            keyed `sst`. Normalising both sides rather than one covers both.
        """
        from digitalearth.base.sources.variable import default_variable

        container = _container(
            ["forecast/time_bnds", "forecast/sst"],
            {
                "lat": LAT,
                "lon": LON,
                "time_bnds": _Info(["time", "nbnds"]),
                "sst": _Info(["time", "lat", "lon"]),
            },
        )()

        assert default_variable(container) == "forecast/sst", (
            f"the answer must keep the name the container reported, got {default_variable(container)!r}"
        )


class TestAPlaneNeedsBothAxes:
    """A map plane is one X axis and one Y axis, not any two horizontal names."""

    def test_a_variable_over_two_x_axes_is_not_drawable(self):
        """A staggered store's two X coordinates do not make a plane between them.

        Test scenario:
            A staggered (Arakawa-C / ROMS-style) store declares more than one X coordinate — `xi_rho` and
            `xi_u` both carry `axis: X`. A variable spanning both of them spans no Y at all, so it has no
            plane; counting "any two horizontal names" called it drawable and would have drawn it.
        """
        from digitalearth.base.sources.variable import (
            default_variable,
            drawable_variable_names,
        )

        container = _container(
            ["u_flux", "temp"],
            {
                "xi_rho": _Info(["xi_rho"], {"axis": "X"}),
                "xi_u": _Info(["xi_u"], {"axis": "X"}),
                "eta_rho": _Info(["eta_rho"], {"axis": "Y"}),
                "u_flux": _Info(["xi_rho", "xi_u"]),
                "temp": _Info(["eta_rho", "xi_rho"]),
            },
        )()

        assert drawable_variable_names(container) == ["temp"], (
            f"only the variable spanning an X and a Y has a plane, got {drawable_variable_names(container)}"
        )
        assert default_variable(container) == "temp", (
            f"expected the variable with a real plane, got {default_variable(container)!r}"
        )

    def test_a_variable_over_the_same_axis_twice_is_not_drawable(self):
        """Spanning one axis twice is not a plane either.

        Test scenario:
            A degenerate `(lat, lat)` variable intersects the horizontal names twice over, which a count of
            two satisfied. Requiring one on each axis refuses it.
        """
        from digitalearth.base.sources.variable import drawable_variable_names

        container = _container(
            ["square", "field"],
            {
                "lat": LAT,
                "lon": LON,
                "square": _Info(["lat", "lat"]),
                "field": _Info(["lat", "lon"]),
            },
        )()

        assert drawable_variable_names(container) == ["field"], (
            f"a (lat, lat) variable has no plane, got {drawable_variable_names(container)}"
        )


class TestAVariablesSequence:
    """`metadata.variables` as a sequence of infos rather than a mapping."""

    def test_a_sequence_of_infos_classifies_like_a_mapping(self):
        """A non-mapping `variables` is read by iterating it, not by calling `.items()`.

        Test scenario:
            `base/sources/extractors.py` already tolerates both shapes
            (`variables.values() if isinstance(variables, Mapping) else variables`), so a container handing
            back a sequence is a shape this package expects. Calling `.items()` on it raised
            `AttributeError: 'list' object has no attribute 'items'` out of all three readers, where the
            previous one-liner could not fail at all.
        """
        from digitalearth.base.sources.variable import default_variable

        class _Named:
            """An info that knows its own name, as pyramids' VariableInfo does."""

            def __init__(self, name, dimensions, attributes=None):
                """Record the name, dimensions and CF attributes.

                Args:
                    name: The variable's name.
                    dimensions: Its dimension names.
                    attributes: Its CF attributes, or ``None`` for none.
                """
                self.name = name
                self.full_name = name
                self.dimensions = dimensions
                self.attributes = attributes or {}

        metadata = type(
            "Metadata",
            (),
            {
                "variables": [
                    _Named("lat", ["lat"], {"axis": "Y"}),
                    _Named("lon", ["lon"], {"axis": "X"}),
                    _Named("time_bnds", ["time", "nbnds"]),
                    _Named("sst", ["time", "lat", "lon"]),
                ]
            },
        )
        container = type(
            "Container",
            (),
            {"variable_names": ["time_bnds", "sst"], "metadata": metadata()},
        )()

        assert default_variable(container) == "sst", (
            f"a sequence-shaped variables mapping must classify too, got {default_variable(container)!r}"
        )


class TestACurvilinearStore:
    """A store whose horizontal coordinates are 2-D auxiliary arrays rather than dimensions.

    A ROMS-style curvilinear grid names its coordinates `lat_rho` / `lon_rho`, each spanning `(eta, xi)`, and
    its data variables span `(time, eta, xi)`. Nothing is called `lat` or `lon`, so classifying by the
    coordinate's own *name* found no horizontal dimension and the answer fell back to the first declared
    name — a bounds array, silently, which is the defect this module exists to prevent.
    """

    def test_a_2d_coordinate_lends_its_own_dimensions(self):
        """The data variable over the curvilinear grid is drawable; the bounds array is not.

        Test scenario:
            `lat_rho` and `lon_rho` each span `(eta_rho, xi_rho)`, so those are the dimensions a data
            variable references. Reading a horizontal coordinate's dimensions rather than its name is what
            makes `temp` classify while `time_bnds` does not.
        """
        from digitalearth.base.sources.variable import (
            default_variable,
            drawable_variable_names,
        )

        container = _container(
            ["time_bnds", "temp"],
            {
                "lat_rho": _Info(["eta_rho", "xi_rho"], {"standard_name": "latitude"}),
                "lon_rho": _Info(["eta_rho", "xi_rho"], {"standard_name": "longitude"}),
                "time_bnds": _Info(["time", "nbnds"]),
                "temp": _Info(["time", "eta_rho", "xi_rho"]),
            },
        )()

        assert drawable_variable_names(container) == ["temp"], (
            f"a curvilinear data variable must classify, got {drawable_variable_names(container)}"
        )
        assert default_variable(container) == "temp", (
            f"expected the data variable, not the bounds array, got {default_variable(container)!r}"
        )

    def test_a_coordinate_named_differently_from_its_dimension(self):
        """A 1-D coordinate whose name differs from its dimension still contributes the dimension.

        Test scenario:
            A store may call the coordinate `latitude` while the dimension it spans is `y`. Data variables
            reference `y`, not `latitude`, so the dimension is the name that has to be matched.
        """
        from digitalearth.base.sources.variable import default_variable

        container = _container(
            ["bounds", "elevation"],
            {
                "latitude": _Info(["y"], {"standard_name": "latitude"}),
                "longitude": _Info(["x"], {"standard_name": "longitude"}),
                "bounds": _Info(["time", "nv"]),
                "elevation": _Info(["y", "x"]),
            },
        )()

        assert default_variable(container) == "elevation", (
            f"the dimension, not the coordinate's name, is what data variables reference; "
            f"got {default_variable(container)!r}"
        )
