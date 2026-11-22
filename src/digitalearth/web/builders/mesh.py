"""MeshMixin — the web tier's unstructured-mesh builder (#547).

``trimesh`` draws a UGRID ``pyramids.netcdf.ugrid.mesh.Mesh2d`` — connectivity plus node coordinates — as a
deck.gl ``PolygonLayer``: every face is fan-cut into triangles by pyramids (``Mesh2d.fan_triangles``), and each
triangle becomes one polygon on a GPU overlay. It is the web sibling of the static ``tripcolor`` and the
interactive ``trimesh`` renders, and like them it consumes the mesh pyramids hands over rather than
re-triangulating anything itself.

**The tier is EPSG:4326-only, so the mesh is reprojected first.** A mesh that declares a CRS of its own has its
node coordinates warped to lon/lat through pyramids (``pyramids.base.crs.reproject_coordinates``, the same
helper the raster-framing path uses) before the triangles are built; a mesh with no declared CRS is drawn in
its own coordinates, read as lon/lat — the bargain the static tier strikes for an unreferenced mesh too.

**A bare ``Mesh2d`` carries geometry, not a value table**, so the faces take one flat colour rather than being
shaded by a per-element variable — exactly as the interactive tier draws a UGRID mesh. The colouring kwargs are
deck.gl RGBA, matching ``deck_scatter`` / ``deck_polygons``.

The layer is described as values (the triangle rings and the two colours) and rebuilt by :func:`draw_trimesh`,
so it owns a ``LayerSpec`` and a queue slot like every other web layer: it is in ``layer_ids``, and is
removable, hideable and reorderable by id. numpy and maplibre are imported lazily; importing the tier needs
neither.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.base.spec.bounds import same_crs
from digitalearth.web.builders.bigdata import DECK_TYPE_KEY
from digitalearth.web.core.base import DISPLAY_CRS, _require_layer_api

#: deck.gl RGBA (0-255 per channel) a mesh's faces are filled with when the caller names no colour. A translucent
#: blue, matching the vector builders' default accent; its alpha lets an overlapping face read through.
MESH_FILL_COLOR = (51, 136, 255, 140)

#: deck.gl RGBA the triangle edges are stroked with, so the mesh's connectivity is visible over the fill.
MESH_LINE_COLOR = (255, 255, 255, 255)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.core.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_trimesh(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the deck.gl ``PolygonLayer`` a described mesh layer asks for.

    A mesh draws from no source in the figure: its triangle rings and their colours are values the builder
    reprojected and recorded, so the layer rebuilds from its description alone — which is why ``_data`` is
    unused and why a mesh layer survives :meth:`~digitalearth.web.core.base.WebMapBase.figure_spec`'s
    ``to_dict``/``from_dict`` round trip.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws from the description alone.
        _data: Unused — a mesh has no feature source in the figure to place; its geometry is on the layer.
        layer: The layer's description, whose props carry the triangle rings and the two deck.gl colours.

    Returns:
        A :class:`~digitalearth.web.render.renderer.DrawnLayer` on the deck route, carrying the deck.gl JSON
        ``PolygonLayer`` the page composes into its one deck overlay.

    Raises:
        ValueError: when the description records none of the values the mesh is built from — its triangle
            rings or its fill/line colours — naming the layer, its kind and what is missing.
    """
    from digitalearth.web.render.renderer import DECK_ROUTE, DrawnLayer, required_props

    _require_layer_api()
    props = required_props(layer, "polygons", "fill_color", "line_color")
    return DrawnLayer(
        source_id=None,
        source_spec=None,
        layer={
            DECK_TYPE_KEY: "PolygonLayer",
            "id": layer.id,
            # One row per triangle; the accessor reads the ring off it, as the point-cloud layer reads its
            # position. The rings are plain lists of [lon, lat] pairs, so the layer is pure JSON.
            "data": [{"polygon": ring} for ring in props["polygons"]],
            "getPolygon": "@@=polygon",
            "getFillColor": list(props["fill_color"]),
            "getLineColor": list(props["line_color"]),
            "filled": True,
            "stroked": True,
            "lineWidthMinPixels": 1,
            "pickable": True,
        },
        route=DECK_ROUTE,
    )


class MeshMixin(_MixinBase):
    """Unstructured-mesh builder for :class:`~digitalearth.web.map.WebMap`.

    A capability mixin of :class:`~digitalearth.web.map.WebMap`: it is only ever composed into that map class,
    never instantiated or subclassed on its own. Its methods reach the layer registry, the display CRS and the
    render/save lifecycle — and the sibling mixins' methods — through ``self``, and only the composition
    supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.web.core.base.WebMapBase`, the state ``WebMap``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``WebMap`` MRO
    exactly what it was before the annotation.

    See Also:
        digitalearth.web.map.WebMap: the composition that supplies the state these methods use.
        digitalearth.web.builders.bigdata.BigDataMixin.deck_polygons: the GPU polygon builder this one's
            output is drawn beside, on the same deck.gl overlay.
    """

    def _mesh_rings(self, mesh: Any) -> list:
        """Reproject a mesh's nodes to EPSG:4326 and return its fan triangles as closed lon/lat rings.

        The connectivity is pyramids' (``Mesh2d.fan_triangles`` cuts every face — triangle, quad or larger —
        into triangles with pure numpy); this only warps the node coordinates to the tier's display CRS and
        reads the triangles off them. A triangle any of whose nodes warped to a non-finite coordinate — the far
        side of a globe, a point outside the projection's domain — is dropped, which is how a partly off-view
        mesh still draws the part that is on it.

        Args:
            mesh: A pyramids ``Mesh2d`` (``node_x`` / ``node_y`` / ``fan_triangles``, and an optional ``crs``).

        Returns:
            One ``[[lon, lat], [lon, lat], [lon, lat]]`` ring per drawable triangle, in lon/lat degrees. Empty
            when no triangle keeps all three nodes on the EPSG:4326 view.
        """
        import numpy as np
        from pyramids.base.crs import reproject_coordinates

        x = np.asarray(mesh.node_x, dtype="float64")
        y = np.asarray(mesh.node_y, dtype="float64")
        held = getattr(mesh, "crs", None)
        # A mesh with no declared CRS is drawn in its own coordinates, read as lon/lat — the same bargain the
        # static tier strikes for an unreferenced mesh. One that declares another CRS is warped through
        # pyramids, because MapLibre and deck.gl place coordinates as lon/lat degrees and reproject nothing.
        if held is not None and not same_crs(held, DISPLAY_CRS):
            xs, ys = reproject_coordinates(
                list(x), list(y), from_crs=held, to_crs=DISPLAY_CRS
            )
            x, y = np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")
        rings = []
        for triangle in np.asarray(mesh.fan_triangles):
            corners = [(float(x[node]), float(y[node])) for node in triangle]
            if all(math.isfinite(value) for corner in corners for value in corner):
                rings.append([[lon, lat] for lon, lat in corners])
        return rings

    def trimesh(
        self,
        mesh: Any,
        *,
        fill_color: Sequence[int] = MESH_FILL_COLOR,
        line_color: Sequence[int] = MESH_LINE_COLOR,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a UGRID mesh's triangulated faces as a deck.gl ``PolygonLayer`` (#547).

        The web sibling of the static ``tripcolor`` and the interactive ``trimesh`` renders. The connectivity
        is the mesh's own — pyramids fan-cuts every face into triangles — so nothing is re-triangulated here;
        the nodes are reprojected to the tier's only CRS (EPSG:4326) through pyramids first (see
        :meth:`_mesh_rings`), and each triangle becomes one polygon on the page's one GPU deck overlay.

        A bare ``Mesh2d`` carries geometry, not a per-element value table, so the faces take one flat
        ``fill_color`` rather than being shaded by a node or face variable — exactly as the interactive tier
        draws a UGRID mesh. To shade a mesh by a variable, draw the ``MeshDataset`` on the static tier, whose
        ``tripcolor`` colours each face through cleopatra.

        Args:
            mesh: A pyramids ``Mesh2d`` — anything exposing ``node_x`` / ``node_y`` / ``fan_triangles``, and
                optionally a ``crs`` to reproject from. This is what
                :func:`~digitalearth.api._mesh_of` hands over when ``quickmap`` routes a ``MeshDataset`` here.
            fill_color: deck.gl RGBA (0-255 per channel) the faces are filled with; defaults to
                :data:`MESH_FILL_COLOR`, a translucent blue.
            line_color: deck.gl RGBA the triangle edges are stroked with; defaults to
                :data:`MESH_LINE_COLOR`, so the connectivity reads over the fill.
            name: What the layer is addressed by — ``set_visible``, ``move_layer``, ``replace_layer`` and
                ``remove_layer`` all take it. ``None`` generates ``trimesh-1``, ``trimesh-2``, … as every
                other unnamed layer is numbered.
            visible: Whether the layer starts drawn. As for the other deck kinds, the live layer-switcher
                toggle does not reach a deck.gl overlay; use ``set_visible`` from Python to change it after
                the fact.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            AttributeError: when ``mesh`` is not a UGRID mesh — a plain array or a ``FeatureCollection`` has no
                ``fan_triangles`` to read, and is turned away rather than drawn as an empty overlay.
            OffLimbError: when the map is ``strict`` and no triangle of the mesh keeps all three nodes on the
                EPSG:4326 view; without ``strict`` the layer is skipped with a warning instead.

        Examples:
            - Draw a two-face mesh and read back the GPU layer it built (needs the ``web`` extra, so the
              block is skipped without it):
                ```python
                >>> import numpy as np                                            # doctest: +SKIP
                >>> from pyramids.netcdf.ugrid.connectivity import Connectivity    # doctest: +SKIP
                >>> from pyramids.netcdf.ugrid.mesh import Mesh2d                   # doctest: +SKIP
                >>> from digitalearth.web import WebMap                            # doctest: +SKIP
                >>> mesh = Mesh2d(                                                 # doctest: +SKIP
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
                ...     face_node_connectivity=Connectivity(
                ...         data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]]),
                ...         fill_value=-1,
                ...         cf_role="face_node_connectivity",
                ...         original_start_index=0,
                ...     ),
                ... )
                >>> m = WebMap().trimesh(mesh, name="bed")                         # doctest: +SKIP
                >>> m.layers[0]["@@type"], m.layer_ids                             # doctest: +SKIP
                ('PolygonLayer', ['bed'])

                ```

        See Also:
            digitalearth.web.builders.bigdata.BigDataMixin.deck_polygons: the GPU polygon builder a mesh is
                drawn beside, on the same deck overlay.
        """
        _require_layer_api()
        rings = self._mesh_rings(mesh)
        if not rings:
            self._skipped(
                "trimesh",
                "no triangle of the mesh keeps all three nodes on the EPSG:4326 view, so there is nothing "
                "to draw",
            )
            return self
        # Frame the map on the mesh, in lon/lat — the rings already are — so a bare trimesh() opens on its
        # data rather than on the whole world, like every other builder's choke point does (_note_bounds).
        lons = [lon for ring in rings for lon, _ in ring]
        lats = [lat for ring in rings for _, lat in ring]
        self._note_lonlat_bounds((min(lons), min(lats), max(lons), max(lats)))
        layer_id = self._layer_id("trimesh", name)
        self._index_layer(
            layer_id,
            name,
            kind="mesh",
            visible=visible,
            # No figure source: the triangle rings are the geometry, recorded as values so the layer is drawn
            # from its description alone and a stored figure round-trips.
            symbology=Symbology(
                props={
                    "polygons": rings,
                    "fill_color": list(fill_color),
                    "line_color": list(line_color),
                }
            ),
        )
        # A deck.gl overlay is not a MapLibre style layer, so it is not a click target and does not claim the
        # default popup()/tooltip() binding (as deck_scatter/deck_polygons do not).
        return self
