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
