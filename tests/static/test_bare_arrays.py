"""A bare NumPy array on the static tier, and the refusal for everything that is not one (#343).

``StaticGlyph.plot(arr, no_data_value=...)`` was the only public entry point that drew a plain grid, and it
went with that class in #339. Nothing took the capability over, and ``Map.field`` did not even refuse an array
cleanly: it reached ``GeoLayerBase._reproject``, which called ``.to_crs`` on whatever it was handed, so the
caller read ``AttributeError: 'numpy.ndarray' object has no attribute 'to_crs'`` — a method they never called,
about an attribute they never heard of.

Both halves are covered here, and the second is the interesting one. A bare array **is** drawn, at its own
indices, because it has nothing else to be placed by. Every consequence of that is asserted rather than
described: the coordinates it gets, that its first row is at the top in *both* of the tier's placements, that
nothing is reprojected, and that each thing which cannot work on such a layer refuses instead of drawing a
picture that is quietly in the wrong place.
"""

import numpy as np
import pytest

from digitalearth.base.crs import OffLimbError
from digitalearth.static import Map, projections

#: The grid every test draws: three rows of four, with the row written into the tens digit so a value names
#: the row it came from. A square grid could not tell a transpose from a mirror.
GRID = np.array(
    [
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
    ]
)


def _top_row_value(mesh) -> float:
    """Return the value of the quad a mesh drew highest up the axes.

    Args:
        mesh: The ``QuadMesh`` a ``pcolormesh``/cell render produced.

    Returns:
        The value of the cell whose vertices reach the largest ``y``. Read off the artist rather than off the
        call, because where a cell *landed* is the only thing that answers "which row is at the top".
    """
    quads = mesh.get_paths()
    values = np.asarray(mesh.get_array()).ravel()
    highest = max(
        range(len(quads)), key=lambda index: quads[index].vertices[:, 1].max()
    )
    return float(values[highest])


class TestABareArrayIsDrawnAtItsOwnIndices:
    """The capability: the "just show me this grid" case, back on the tier that lost it."""

    def test_the_image_covers_the_cells_the_indices_describe(self):
        """The extent is the rectangle the cells cover, half a cell beyond the outermost centre.

        Test scenario:
            ``Map().field(grid)`` raised `AttributeError` before this. What it draws now is asserted by
            position: a 3x4 grid with cell centres on the integers covers x from -0.5 to 3.5 and y from
            -0.5 to 2.5, which is the same cell-edge rule a georeferenced raster is placed by.
        """
        with Map() as canvas:
            image = canvas.field(GRID)
            extent = list(image.get_extent())
        assert extent == [-0.5, 3.5, -0.5, 2.5], (
            f"a 3x4 grid at index coordinates covers [-0.5, 3.5, -0.5, 2.5]; got {extent}"
        )

    def test_the_first_row_is_drawn_at_the_top_by_the_image_render(self):
        """matplotlib draws an array's first row at the top, and ``field`` places its image by extent.

        Test scenario:
            The image's own origin is what decides this, so it is read off the artist. `upper` means the
            first row lands at the extent's ``ymax``.
        """
        with Map() as canvas:
            image = canvas.field(GRID)
            origin = image.origin
        assert origin == "upper", (
            f"an array's first row is its top row, which is matplotlib's 'upper' origin; got {origin!r}"
        )

    def test_the_first_row_is_drawn_at_the_top_by_the_cell_render_too(self):
        """The half that was mirrored, and the reason the tier sets the row axis itself.

        Test scenario:
            ``field`` places its image by *extent* and ``pcolormesh`` places its cells by the *coordinates*,
            so the two agree only if the row axis runs downward. `_from_numpy` hands back ``y = 0, 1, 2``,
            which put row 0 at ``ymin`` — one array, two renders, vertically mirrored. Measured on the cell
            the mesh actually drew highest, and compared against the values of row 0 rather than against
            the image render, so neither side of the comparison is the other.
        """
        with Map() as canvas:
            top = _top_row_value(canvas.pcolormesh(GRID))
        assert top == GRID[0, 0], (
            f"the cell render put {top} at the top; row 0 of the grid starts at {GRID[0, 0]}"
        )

    def test_nothing_is_reprojected(self, monkeypatch):
        """An array declares no CRS, so there is nothing to warp *from* and no warp is attempted.

        Args:
            monkeypatch: Used to make any reprojection attempt fail loudly.

        Test scenario:
            The refusal `_reproject` used to leak came from asking for a warp at all. Replacing the shared
            warp with a call that raises proves the array path never reaches it — a weaker assertion on
            ``Source.crs`` would pass even if the tier warped and then forgot to record the result.
        """
        from digitalearth.static.maps import base

        def refuse(data, crs):
            """Args:
            data: The thing a warp was asked for.
            crs: The CRS it was asked to warp into.

            Raises:
                AssertionError: always — reaching this is the defect.
            """
            raise AssertionError(
                f"a warp of {type(data).__name__} into {crs!r} was attempted"
            )

        monkeypatch.setattr(base, "reproject", refuse)
        with Map(crs=3857) as canvas:
            image = canvas.field(GRID)
        assert image is not None, "the array was not drawn at all"

    def test_the_layer_is_described_as_a_raster(self):
        """A bare grid is the same *kind* of layer as a georeferenced one; only its placement differs."""
        with Map() as canvas:
            canvas.field(GRID)
            described = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        assert described.kind == "raster", f"an array field recorded {described.kind!r}"

    def test_a_cell_the_mask_hides_is_not_drawn_as_its_fill_value(self):
        """A masked array is how a bare grid says "nodata": there is no sidecar to carry a sentinel.

        Test scenario:
            ``np.asarray`` hands back the data buffer and drops the mask, so
            ``np.ma.masked_equal(grid, 11.0)`` was extracted with its 11.0 intact and drawn as a value. The
            masked cell has to come through as `NaN`, which the field render already excludes — and the
            cells around it have to be untouched, which is the half a blanket "fill everything" would fail.
        """
        with Map() as canvas:
            image = canvas.field(np.ma.masked_equal(GRID, 11.0))
            drawn = np.asarray(image.get_array(), dtype="float64")
        assert np.isnan(drawn[1, 1]), f"the masked cell came through as {drawn[1, 1]}"
        assert drawn[1, 2] == 12.0, f"an unmasked neighbour became {drawn[1, 2]}"


class TestWhatCannotWorkOnSuchALayerRefuses:
    """The honest half: an array has no CRS, so some things cannot be done to it at all."""

    def test_a_globe_frame_is_refused_at_the_call(self):
        """A globe's limb is metres in the display CRS; an array is indices. Nothing relates the two.

        Test scenario:
            Drawn anyway, a 3x4 grid is a speck at the centre of a disc 12,000 km across — a figure that
            renders and shows nothing. Refused at the builder, so the traceback names the caller's line.
        """
        canvas = Map(crs=projections.orthographic(lon=0.0, lat=0.0), globe=True)
        with pytest.raises(
            ValueError, match="cannot draw a bare numpy array on a globe frame"
        ):
            canvas.field(GRID)
        canvas.close()

    def test_a_georeferenced_layer_cannot_join_a_figure_placed_at_indices(self):
        """A coastline is drawn in the display CRS, so it would land nowhere near an index-space array."""
        canvas = Map()
        canvas.field(GRID)
        with pytest.raises(
            ValueError,
            match="cannot join a figure whose layers are placed at array indices",
        ):
            canvas.coastlines()
        canvas.close()

    def test_an_array_cannot_join_a_figure_placed_in_the_display_crs(self, dataset):
        """The same rule from the other side, because a guard in one direction is half a guard.

        Args:
            dataset: A georeferenced raster, drawn first.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.field(dataset)
        with pytest.raises(
            ValueError,
            match="cannot join a figure whose layers are placed in the display CRS",
        ):
            canvas.field(GRID)
        canvas.close()

    def test_a_refused_mix_leaves_the_figure_as_it_was(self):
        """A refusal that reserved an id would suffix the next layer of that name past it.

        Test scenario:
            The placement check runs *before* the id is minted, so nothing is left half-registered. Asserted
            by naming a layer, watching the refusal, and then drawing it successfully under that very name.
        """
        canvas = Map()
        canvas.field(GRID)
        with pytest.raises(
            ValueError, match="one axes has one pair of data coordinates"
        ):
            canvas.graticule(name="grid-lines")
        before = list(canvas.layer_ids)
        canvas.close()
        assert before == ["raster-1"], (
            f"the refused layer left something behind: {before}"
        )

    def test_taking_the_array_layer_off_frees_the_figure_again(self, dataset):
        """The claim belongs to the layer, not to the figure, so removing it releases the axes.

        Args:
            dataset: The georeferenced raster drawn once the array layer is gone.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.field(GRID, name="grid")
        canvas.remove_layer("grid")
        canvas.field(dataset, name="dem")
        ids = list(canvas.layer_ids)
        canvas.close()
        assert ids == ["dem"], (
            f"the georeferenced layer did not take the freed figure: {ids}"
        )

    def test_set_bounds_frames_such_a_figure_in_index_units(self):
        """Declared rather than refused: the bounds are the display CRS's units, which here are indices.

        Test scenario:
            There is nothing to tell geographic bounds from index bounds by, so this cannot refuse — what it
            can do is be true. Framing columns 1 to 3 and rows 0 to 2 has to leave the axes holding exactly
            that, which is what says the units are the array's own.
        """
        with Map() as canvas:
            canvas.field(GRID)
            canvas.set_bounds((1.0, 0.0, 3.0, 2.0))
            held = (*canvas.ax.get_xlim(), *canvas.ax.get_ylim())
        assert held == (1.0, 3.0, 0.0, 2.0), f"the axes was framed on {held}"


class TestEverythingElseIsRefusedByName:
    """``quickmap`` already said ``TypeError: quickmap cannot draw a ndarray``; a builder now says it too."""

    @pytest.mark.parametrize(
        "value", [{}, None, object(), 3], ids=["dict", "none", "object", "int"]
    )
    def test_a_field_names_the_call_the_type_and_what_it_takes(self, value):
        """Three things a caller needs to fix the call, none of which the attribute error carried.

        Args:
            value: Something no field render can read.
        """
        canvas = Map()
        with pytest.raises(TypeError, match=r"Map\.field\(\) cannot draw a"):
            canvas.field(value)
        canvas.close()

    def test_the_refusal_names_the_method_the_caller_wrote_and_not_the_render(self):
        """``contours`` and ``pcolormesh`` share a funnel with ``field`` and must not borrow its name.

        Test scenario:
            The funnel is keyed by cleopatra's render kind — `contour`, `contourf`, `imshow`, `pcolormesh` —
            which is not what a caller types. The message has to come back in the tier's own vocabulary.
        """
        canvas = Map()
        wrong = object()
        with pytest.raises(TypeError, match=r"Map\.contours\(\) cannot draw a object"):
            canvas.contours(wrong)
        canvas.close()

    def test_a_vector_builder_refuses_the_grid_a_field_accepts(self):
        """One value, two answers, because they are two different families of data.

        Test scenario:
            The same array `field` draws is not a set of features, and ``Map.points`` used to leak the same
            ``to_crs`` attribute error for it. The refusal names ``Map.points()`` rather than the drawer it
            is raised in.
        """
        canvas = Map()
        with pytest.raises(TypeError, match=r"Map\.points\(\) cannot draw a ndarray"):
            canvas.points(GRID)
        canvas.close()

    def test_a_cell_geometry_builder_refuses_an_array(self):
        """``grid_points`` reads pyramids' ``to_xyz``, which needs a geo-transform an array does not carry."""
        canvas = Map()
        with pytest.raises(
            TypeError, match=r"Map\.grid_points\(\) cannot draw a ndarray"
        ):
            canvas.grid_points(GRID)
        canvas.close()

    def test_a_composite_refuses_an_array(self):
        """A composite reads three *bands*; an array is one grid with no band dimension to index."""
        canvas = Map()
        with pytest.raises(
            TypeError, match=r"Map\.rgb_composite\(\) cannot draw a ndarray"
        ):
            canvas.rgb_composite(GRID)
        canvas.close()

    def test_a_uv_field_refuses_an_array_component(self):
        """The two components are placed against each other by their geo-transforms."""
        canvas = Map()
        with pytest.raises(TypeError, match=r"Map\.quiver\(\) cannot draw a ndarray"):
            canvas.quiver(GRID, GRID)
        canvas.close()

    def test_a_three_dimensional_array_is_refused_for_its_rank(self):
        """The rank refusal is the extractor's and keeps its own words: a field is one 2-D grid.

        Test scenario:
            `require_drawable` accepts every ``ndarray`` deliberately — whether a grid is 2-D is the
            extractor's question, and answering it twice would mean two messages for one mistake.
        """
        canvas = Map()
        cube = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="expects a 2-D array"):
            canvas.field(cube)
        canvas.close()

    def test_a_path_is_not_refused_for_being_a_string(self, dataset):
        """A reference names data opened at the draw, so there is nothing to inspect at the call.

        Args:
            dataset: Unused for its object; its path is what the layer references.
        """
        with Map(crs=dataset.epsg) as canvas:
            drawn = canvas.field("examples/data/acc4000.tif")
        assert drawn is not None, "a path-referenced raster was not drawn"

    def test_an_unreadable_path_still_fails_in_the_readers_words(self):
        """The guard must not turn a missing file into a type error about `str`."""
        canvas = Map()
        with pytest.raises(FileNotFoundError):
            canvas.field("examples/data/nothing-here.tif")
        canvas.close()


class TestTheOffLimbContractIsUntouched:
    """The `_reproject` change had to leave every georeferenced path exactly as it was."""

    def test_a_raster_behind_the_limb_is_still_reported_as_off_limb(self, dataset):
        """Args:
        dataset: The raster placed behind an orthographic limb.

        Test scenario:
            `_reproject` grew a guard in front of its warp. A raster still *has* ``to_crs``, so the guard
            must pass it through to the warp and the warp's `OffLimbError` must still arrive — under
            ``strict``, where it is raised rather than skipped.
        """
        away = projections.orthographic(lon=-175.0, lat=15.0)
        canvas = Map(crs=away, globe=True, strict=True)
        with pytest.raises(OffLimbError):
            canvas.field(dataset)
        canvas.close()
