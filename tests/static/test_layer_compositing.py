"""Layers compose on one axes: what the axes holds matches what ``figure_spec`` describes (#313).

Every other module under ``tests/static/`` asks whether *a* layer drew — this one asks whether the layers
drew **together**. They did not: a cleopatra glyph clears every glyph's artists off the axes it draws on
unless it is told to compose, so a map that described two field layers kept only the second's image. The
description said two, the axes held one, and nothing in the suite compared them.

That comparison is the property here. A scene's description and its artists are deliberately separate
structures (see :mod:`digitalearth.static.scene`), so they can only be held together by a test that reads
both: for each combination below, every layer :attr:`~digitalearth.static.scene.Scene.figure_spec` names
must still have its artists attached to :attr:`~digitalearth.static.scene.Scene.ax`.

The redraw half matters just as much and pulls the other way. ``Renderer`` rebuilds a layer by drawing it
again, so composing must not turn a rebuild into a second copy stacked on the first — the very thing the
clearing exists to prevent. Both directions are pinned here.
"""

from dataclasses import replace as with_fields

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection

from digitalearth.base.spec import Symbology
from digitalearth.static import Map


def attached_per_layer(canvas):
    """Return, for each layer the figure describes, how many of its artists are still on the axes.

    The one measurement this module is about. It reads both structures — the description through
    ``figure_spec`` and the artists through the renderer's record — and checks each recorded artist against
    the axes' own children by identity, so an artist that was drawn and then removed by a later layer counts
    as zero rather than as one.

    Args:
        canvas: The :class:`~digitalearth.static.map.Map` to measure.

    Returns:
        dict[str, int]: layer id -> the number of its artists still attached. A described layer that drew
        nothing at all (skipped off-limb, or wiped) maps to ``0``, which is what makes a disagreement
        between the drawing and its description visible.
    """
    children = canvas.ax.get_children()
    drawn = canvas._renderer.drawn
    counts = {}
    for layer_id in canvas.figure_spec.layers.ids:
        recorded = drawn[layer_id].artists if layer_id in drawn else ()
        counts[layer_id] = sum(
            1 for artist in recorded if any(artist is child for child in children)
        )
    return counts


def restyled(figure, layer_id, **props):
    """Return `figure` with one layer's symbology changed, which is what makes the renderer redraw it.

    Args:
        figure: The figure the axes currently shows.
        layer_id: The layer to restyle.
        **props: Symbology props to set on it (e.g. ``cmap="magma"``).

    Returns:
        FigureSpec: a copy of `figure` carrying the restyled layer.
    """
    layer = figure.layers.get(layer_id)
    changed = with_fields(
        layer, symbology=Symbology(props={**dict(layer.symbology.props), **props})
    )
    return with_fields(figure, layers=figure.layers.replace(changed))


@pytest.fixture
def flat_grid():
    """A backdrop raster and a matching ``(u, v)`` pair on one increasing-y EPSG:4326 grid.

    The three share a grid so the arrows land on top of the field rather than beside it, which is what makes
    "the raster survived the vector layer" a question about compositing rather than about extents.

    Returns:
        tuple[Dataset, Dataset, Dataset]: the backdrop field, and the u and v components.
    """
    rows, cols = 6, 8
    geo_ref = GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0), epsg=4326)
    u = np.ones((rows, cols), dtype="float32")
    v = np.linspace(-1.0, 1.0, rows, dtype="float32")[:, None] * np.ones(
        (1, cols), "float32"
    )
    backdrop = np.linspace(0.0, 1.0, rows * cols, dtype="float32").reshape(rows, cols)
    return (
        Dataset.from_array(arr=backdrop, geo_ref=geo_ref),
        Dataset.from_array(arr=u, geo_ref=geo_ref),
        Dataset.from_array(arr=v, geo_ref=geo_ref),
    )


@pytest.fixture
def point_features():
    """The committed point fixture as a pyramids FeatureCollection.

    Returns:
        FeatureCollection: the ``tests/data/points.geojson`` points.
    """
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def two_fields(dataset):
    """Yield a map carrying two raster fields drawn on one axes, closed on the way out.

    Args:
        dataset: The raster fixture both layers draw.

    Yields:
        Map: the map, whose layers are ``raster-1`` and ``raster-2``.
    """
    canvas = Map(crs=dataset.epsg)
    canvas.imshow(dataset, cmap="viridis")
    canvas.imshow(dataset, cmap="cividis")
    yield canvas
    canvas.close()


class TestLayersCompose:
    """Each layer a figure describes keeps its artists when the next one is drawn."""

    def test_two_rasters_leave_two_images(self, two_fields):
        """A second field render draws over the first rather than in place of it.

        Args:
            two_fields: A map with two raster layers on one axes.

        Test scenario:
            The reported symptom: two ``imshow`` calls, two described layers, one image on the axes.
        """
        assert len(two_fields.ax.images) == 2, (
            f"expected both fields on the axes, got {len(two_fields.ax.images)}"
        )

    def test_two_rasters_agree_with_what_the_figure_describes(self, two_fields):
        """Both described layers still own an artist on the axes.

        Args:
            two_fields: A map with two raster layers on one axes.

        Test scenario:
            The artist count alone would pass on a figure that described three layers and drew two; this
            compares the drawing against the description, layer by layer.
        """
        assert attached_per_layer(two_fields) == {"raster-1": 1, "raster-2": 1}

    def test_contours_over_a_raster_keep_the_image_below(self, dataset):
        """Two different field renders compose: the image stays, the isolines join it.

        Args:
            dataset: The raster fixture.

        Test scenario:
            ``contour`` goes through the same glyph as ``imshow`` but leaves a collection rather than an
            image, so this is the case where the wipe was invisible to any per-artist-type count.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.imshow(dataset, cmap="viridis")
            canvas.contour(dataset, cmap="plasma")
            images, contours = len(canvas.ax.images), len(canvas.ax.collections)
            described = attached_per_layer(canvas)
        assert (images, contours) == (1, 1), f"{images} images, {contours} collections"
        assert described == {"raster-1": 1, "contours-1": 1}

    def test_a_vector_field_over_a_raster_keeps_the_image_below(self, flat_grid):
        """Arrows drawn over a field leave the field where it is.

        Args:
            flat_grid: The backdrop raster and the ``(u, v)`` pair it shares a grid with.

        Test scenario:
            ``quiver`` renders through cleopatra's ``VectorGlyph``, the *other* glyph that clears on render,
            so the raster path's fix has to reach it too.
        """
        backdrop, u_dataset, v_dataset = flat_grid
        with Map(crs=4326) as canvas:
            canvas.imshow(backdrop, cmap="viridis")
            canvas.quiver(u_dataset, v_dataset)
            images = len(canvas.ax.images)
            described = attached_per_layer(canvas)
        assert images == 1, f"the backdrop was taken off the axes ({images} images)"
        assert described == {"raster-1": 1, "vectors-1": 1}

    def test_a_point_layer_over_a_raster_keeps_the_image_below(
        self, dataset, point_features
    ):
        """A scatter layer composes over a field, and is not handed a keyword its glyph cannot take.

        Args:
            dataset: The raster fixture drawn underneath.
            point_features: The committed point fixture drawn on top.

        Test scenario:
            ``ScatterGlyph`` never clears, so it needs no ``compose`` — and its ``plot`` has no such
            parameter, so forwarding one would reach matplotlib as an unknown property rather than be
            refused. Drawing it over a raster exercises both halves.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.imshow(dataset, cmap="viridis")
            canvas.scatter(point_features)
            images = len(canvas.ax.images)
            described = attached_per_layer(canvas)
        assert images == 1, f"the field was taken off the axes ({images} images)"
        assert described == {"raster-1": 1, "points-1": 1}

    def test_a_scene_has_not_drawn_until_it_has(self, dataset):
        """The flag that drives all of this is off until this scene has rendered something.

        Args:
            dataset: The raster fixture the first layer draws.

        Test scenario:
            Composing is the default from the *second* render onwards only; this pins which render is the
            first one, and it is per scene rather than per axes.
        """
        with Map(crs=dataset.epsg) as canvas:
            before = canvas._drew_on_axes
            canvas.imshow(dataset)
            after = canvas._drew_on_axes
        assert before is False, "a scene has drawn nothing before its first layer"
        assert after is True, "one render is enough to have something to compose over"

    def test_the_first_render_on_a_borrowed_axes_still_replaces_what_is_there(
        self, dataset
    ):
        """A scene handed an axes something else was drawn on replaces it, as it always has.

        Args:
            dataset: The raster fixture the map draws.

        Test scenario:
            The other half of the contract, and the reason composing is decided per scene rather than per
            axes: a caller who reuses one axes for a second map — or who drew a cleopatra glyph on it
            themselves — gets the new render in place of the old, not the two of them stacked. Only this
            scene's *own* earlier layers are composed over.
        """
        fig, ax = plt.subplots()
        ArrayGlyph(np.arange(9.0).reshape(3, 3), ax=ax, fig=fig).plot(
            add_colorbar=False
        )
        stale = len(ax.images)
        canvas = Map(crs=dataset.epsg, ax=ax, fig=fig)
        canvas.imshow(dataset)
        replaced = len(ax.images)
        plt.close(fig)
        assert stale == 1, "the borrowed axes should start with the caller's own image"
        assert replaced == 1, f"the caller's image was left behind ({replaced} images)"


class TestRedrawnLayer:
    """Rebuilding one layer replaces that layer's artists and leaves its neighbours alone."""

    def test_a_redrawn_layer_leaves_one_copy_of_itself(self, two_fields):
        """Drawing a layer a second time replaces its artists rather than stacking a copy.

        Args:
            two_fields: A map with two raster layers on one axes.

        Test scenario:
            ``Renderer`` rebuilds a layer by drawing it again, which is exactly the call composing narrows.
            If the rebuild's own clear were narrowed *away* the axes would end up with three images.
        """
        figure = two_fields.figure_spec
        two_fields._renderer.apply(figure, restyled(figure, "raster-2", cmap="magma"))
        images = len(two_fields.ax.images)
        assert images == 2, f"the redraw stacked a copy ({images} images for 2 layers)"

    def test_a_redrawn_layer_does_not_wipe_the_one_below(self, two_fields):
        """The layer that was not rebuilt keeps the artist it already had.

        Args:
            two_fields: A map with two raster layers on one axes.

        Test scenario:
            Counting images alone cannot tell "both survived" from "the lower one was wiped and the upper
            one duplicated", so this holds the untouched layer's artist by identity.
        """
        figure = two_fields.figure_spec
        below = two_fields._renderer.drawn["raster-1"].artist
        two_fields._renderer.apply(figure, restyled(figure, "raster-2", cmap="magma"))
        assert any(image is below for image in two_fields.ax.images), (
            "rebuilding the upper layer took the lower one off the axes"
        )

    def test_a_redrawn_figure_still_agrees_with_its_description(self, two_fields):
        """After a rebuild, both described layers own an artist again.

        Args:
            two_fields: A map with two raster layers on one axes.
        """
        figure = two_fields.figure_spec
        two_fields._renderer.apply(figure, restyled(figure, "raster-2", cmap="magma"))
        assert attached_per_layer(two_fields) == {"raster-1": 1, "raster-2": 1}
