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
            Under 0.75 the order is the store's. If a future pyramids drops CF non-data arrays from
            `variable_names`, or this fixture is replaced, this test fails and says so rather than letting
            the three tests below pass for the wrong reason.
        """
        names = list(container.variable_names)

        assert names == ["time_bnds", "sst"], (
            f"the premise of this module changed: variable_names is {names}"
        )
        assert sorted(names) != names, (
            "the declared order must differ from the sorted one to be a real test"
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

    def test_a_name_its_metadata_does_not_describe_is_skipped(self):
        """A declared name absent from `metadata.variables` cannot be classified, so it is passed over.

        Test scenario:
            A container may declare a name its metadata does not carry — a grouped store whose
            `variable_names` reports a path the mapping keys differently. Skipping it leaves the next
            drawable variable as the answer, rather than treating the undescribed one as drawable.
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
