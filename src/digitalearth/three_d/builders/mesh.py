"""MeshMixin — the 3-D tier's unstructured-mesh builder (#546).

``trimesh`` draws a UGRID ``pyramids.netcdf.ugrid.mesh.Mesh2d`` — connectivity plus node coordinates — as one
PyVista ``PolyData`` surface: every face is fan-cut into triangles by pyramids (``Mesh2d.fan_triangles``), and
the triangles become the surface's cells. It is the 3-D sibling of the static ``tripcolor``, the interactive
``trimesh`` and the web ``trimesh`` renders, and like them it consumes the connectivity pyramids hands over
rather than re-triangulating anything itself.

**A ``MeshDataset`` is shaded by its plotted variable** (#557), the 3-D sibling of static ``tripcolor``: the
seam reads the per-node or per-face values off the shared ``MeshSource`` and hands them to ``trimesh`` as
``values``/``location``, which classifies them (one ``scheme``/``k``/``cmap`` paints the same classes as every
tier) and draws the surface unlit so each face's colour reads flat on a strict GL. A **bare ``Mesh2d``**, with
no value table, still takes one flat ``color`` — the geometry-only render the interactive and web tiers also
draw. The surface lies on the ground plane (``z=0``) by default; pass ``node_z`` to lift each node and read the
mesh as 3-D relief.

**CRS follows the tier's rule, not the web tier's.** The 3-D tier draws in whatever display CRS the scene
carries — it is not EPSG:4326-only — so a mesh that declares a CRS sets the scene's when it has none yet, and
is reprojected through pyramids (``pyramids.base.crs.reproject_coordinates``) only when the scene already draws
in a different one. A mesh with no declared CRS is drawn in its own coordinates, the same bargain every tier
strikes for an unreferenced mesh.

The layer is described from its source (the ``Mesh2d`` itself) and rebuilt by :func:`draw_trimesh`, so it owns
a ``LayerSpec`` and a slot in the figure like every other 3-D layer: it is in ``layer_ids``, and is removable,
hideable and re-describable by id. numpy is imported lazily and PyVista only inside the drawer; importing the
tier needs neither.

**The ``Mesh2d`` is held as an in-memory object source, not recorded as values**, so the layer redraws in this
process (``Scene3D.from_figure(scene.figure_spec)``) but a mesh scene cannot be ``figure_spec.to_dict()``-d or
``save_bundle``-d — the same object-source limitation an in-memory-sourced ``terrain`` has. This diverges on
purpose from the web tier, whose :class:`~digitalearth.web.builders.mesh.MeshMixin` records the triangle rings
as plain values precisely so its layer survives a JSON round trip; the 3-D tier keeps the connectivity on the
``Mesh2d`` and triangulates at draw time instead.
"""

from typing import TYPE_CHECKING, Any

from digitalearth.three_d.core.layer import drawing_props

#: The flat colour a mesh's faces take when the caller names none. A soft steel blue; a bare ``Mesh2d`` has no
#: value table to shade by, so this is the honest default the interactive and web UGRID paths also use.
MESH_FILL_COLOR = "lightsteelblue"

#: The colour the triangle edges are drawn in, so the mesh's connectivity reads over the fill.
MESH_EDGE_COLOR = "gray"

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.core.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class MeshMixin(_MixinBase):
    """Adds the unstructured-mesh builder :meth:`trimesh` to a :class:`~digitalearth.three_d.scene3d.Scene3D`.

    A capability mixin of :class:`~digitalearth.three_d.scene3d.Scene3D`: it is only ever composed into that
    scene class, never instantiated or subclassed on its own. Its one method reaches the plotter, the layer
    registry and the display CRS — and the sibling mixins' methods — through ``self``, and only the
    composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.three_d.core.base.Scene3DBase`, the state
    ``Scene3D`` inherits. At runtime that base is plain ``object``, so composing this mixin leaves the
    ``Scene3D`` MRO exactly what it was before the annotation.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state this method uses.
        digitalearth.three_d.builders.flat.FlatMixin.polygons: the flat-fill builder this surface is drawn
            beside on the ground plane.
    """

    def trimesh(
        self,
        mesh: Any,
        *,
        values: Any = None,
        location: str = "node",
        cmap: Any = "viridis",
        scheme: Any | None = None,
        k: int = 5,
        relief: Any = None,
        node_z: Any = None,
        source_crs: Any = None,
        color: Any = MESH_FILL_COLOR,
        opacity: float | None = None,
        show_edges: bool = True,
        edge_color: Any = MESH_EDGE_COLOR,
        name: Any = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw a UGRID mesh's triangulated faces as one PyVista surface (#546).

        The 3-D sibling of the static ``tripcolor`` and the interactive and web ``trimesh`` renders. The
        connectivity is the mesh's own — pyramids fan-cuts every face (triangle, quad or larger) into
        triangles — so nothing is re-triangulated here; the triangles become the cells of a single
        ``pyvista.PolyData``. A mesh that declares a CRS is placed in the scene's display CRS first (see
        :func:`draw_trimesh`).

        Pass ``values`` to shade the surface by a node- or face-located variable (#557) — the 3-D sibling of
        static ``tripcolor``; ``quickmap`` fills them in from the ``MeshDataset``. With no ``values`` the
        surface takes one flat ``color``, the geometry-only render the interactive and web tiers draw.

        Args:
            mesh: A pyramids ``Mesh2d`` — anything exposing ``node_x`` / ``node_y`` / ``fan_triangles``, and
                optionally a ``crs`` to reproject from. This is what
                :func:`~digitalearth.api._mesh_of` hands over when ``quickmap`` routes a ``MeshDataset`` here.
            values: The per-node or per-face values to shade the surface by; ``None`` (default) draws it in
                the flat ``color``. ``quickmap`` passes the plotted variable's values here.
            location: ``"node"`` (the default) or ``"face"`` — whether ``values`` sit on the nodes (smooth
                shading) or on the fan triangles (flat per-face shading), aligned with the geometry's
                drop/re-index of off-view faces.
            cmap: Colormap the ``values`` are shaded through; ignored when ``values`` is ``None``.
            scheme: How ``values`` are classified — ``None`` (default) for a continuous ramp, ``"categorical"``
                or a ``classify`` scheme name — so one ``scheme``/``k``/``cmap`` paints the same classes as on
                every other tier.
            k: Number of classes for a graduated ``scheme``.
            relief: Lift a **node-located** surface by its own value so coloured data reads as 3-D relief —
                ``True`` raises each node to its value, a number exaggerates by that factor. ``None`` (default)
                leaves the surface flat (or at an explicit ``node_z``); a face-located mesh is refused and
                pointed at ``node_z``. ``quickmap`` passes a caller's ``relief=`` through.
            source_crs: The CRS the mesh's coordinates are in, so the surface reprojects into the scene's
                display CRS. ``quickmap`` fills it from the ``MeshDataset`` (whose wrapper, not its bare
                ``Mesh2d``, carries the CRS); a direct caller with a CRS-tagged ``Mesh2d`` can leave it
                ``None`` and the mesh's own ``crs`` is used.
            node_z: The elevation each node is lifted to, read as 3-D relief. ``None`` (default) lays the
                surface flat on the ground plane (``z=0``); a scalar lifts every node to one height; a
                per-node array (one value per node, in mesh order) gives the surface its own relief.
            color: The flat colour the faces are filled with — any PyVista colour (a name or an RGB) —
                defaulting to :data:`MESH_FILL_COLOR`.
            opacity: Surface opacity in ``[0, 1]``; ``None`` leaves the engine default.
            show_edges: Whether the triangle edges are drawn, so the mesh's connectivity reads over the fill.
            edge_color: The colour the edges are drawn in; defaults to :data:`MESH_EDGE_COLOR`.
            name: The caller's own name for the layer, used as its id and label; ``None`` (default) numbers
                it by kind (``trimesh-1``, ``trimesh-2``, …), and a name already on the scene is suffixed.
            visible: Whether the layer is drawn when added. ``False`` adds it hidden but described, so
                ``set_visible(id, True)`` later reveals it without a rebuild.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            The registered actor, or ``None`` when the mesh placed no drawable face (see ``strict``).

        Examples:
            - Draw the two-face fixture mesh headless and read back the surface it built (needs the ``3d``
              extra, so the block is skipped without it):
                ```python
                >>> import numpy as np                                            # doctest: +SKIP
                >>> from pyramids.netcdf.ugrid.connectivity import Connectivity    # doctest: +SKIP
                >>> from pyramids.netcdf.ugrid.mesh import Mesh2d                   # doctest: +SKIP
                >>> from digitalearth.three_d import Scene3D                       # doctest: +SKIP
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
                >>> scene = Scene3D(off_screen=True)                              # doctest: +SKIP
                >>> _ = scene.trimesh(mesh, name="bed")                           # doctest: +SKIP
                >>> scene.layers[0][0].n_cells, scene.layer_ids                   # doctest: +SKIP
                (3, ['bed'])
                >>> scene.close()                                                 # doctest: +SKIP

                ```

        See Also:
            digitalearth.three_d.builders.terrain.TerrainMixin.terrain: the gridded-relief builder a mesh
                surface is the unstructured counterpart of.
        """
        if relief is not None and node_z is None:
            import numpy as np

            # "Lift by the value" (#557): raise each node to its own value so a coloured surface reads as 3-D
            # relief. It needs a node-located value per node; a face value belongs to a triangle, not a node,
            # so a face-located mesh is pointed at the explicit `node_z` instead.
            if values is None or location != "node":
                raise ValueError(
                    "relief= lifts each node by its own value, so it needs a node-located values-carrying "
                    "mesh; pass node_z= for an explicit relief, or for a face-located one"
                )
            node_z = np.asarray(values, dtype="float64") * (
                1.0 if relief is True else float(relief)
            )
        return self._add_described_layer(
            kind="mesh",
            data=mesh,
            name=name,
            visible=visible,
            encodings=_mesh_color_encoding(values, scheme, k, cmap),
            values=values,
            location=location,
            cmap=cmap,
            scheme=scheme,
            k=k,
            node_z=node_z,
            source_crs=source_crs,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            **kwargs,
        )


def _mesh_color_encoding(values: Any, scheme: Any, k: int, cmap: Any) -> Any:
    """Return the colour encoding a value-shaded mesh publishes, or ``None`` for a flat one (#557).

    The mesh's colour comes from a value per node or per face, so a key over it describes that variable —
    recorded on the layer (order 24) rather than rebuilt from whatever was drawn last, the same way the point
    cloud and choropleth publish theirs. A mesh with no ``values`` colours by nothing and publishes no key.

    Args:
        values: The per-node or per-face value array the mesh is shaded by, or ``None`` for a flat fill.
        scheme: How the values are classified, or ``None`` for a continuous ramp.
        k: How many classes a graduated ``scheme`` cuts.
        cmap: The colormap the ramp / classes come from.

    Returns:
        ``{"color": Encoding}`` naming the ``value`` scalar the drawer binds on the surface and carrying the
        scale the values are cut into, or ``None`` when the mesh shades by nothing.
    """
    if values is None:
        return None
    import numpy as np

    from digitalearth.base.spec import Encoding
    from digitalearth.three_d.builders.vector import VALUE
    from digitalearth.three_d.legends.legend import color_scale

    column = np.asarray(values)
    return {
        "color": Encoding.by_field(
            "color", VALUE, scale=color_scale(column, scheme=scheme, k=k, cmap=cmap)
        )
    }


def _elevation(node_z: Any, count: int) -> Any:
    """Return the z coordinate for every node, from the builder's ``node_z`` argument.

    Args:
        node_z: ``None`` for a flat surface, a scalar lifting every node to one height, or a per-node array
            (one value per node, in mesh order).
        count: The number of nodes, so a flat or scalar elevation is broadcast to the right length.

    Returns:
        A ``(count,)`` float array of node elevations.

    Raises:
        ValueError: when ``node_z`` is an array whose length is not ``count`` — a relief that does not match
            the mesh is a caller error, caught here naming both lengths rather than failing deep in VTK.
    """
    import numpy as np

    if node_z is None:
        return np.zeros(count, dtype="float64")
    array = np.asarray(node_z, dtype="float64")
    if array.ndim == 0:
        return np.full(count, float(array), dtype="float64")
    if array.shape[0] != count:
        raise ValueError(
            f"node_z has {array.shape[0]} values but the mesh has {count} nodes; pass one elevation per "
            "node (in mesh order) or a single scalar"
        )
    return array


def _mesh_surface(
    scene: Any,
    mesh: Any,
    node_z: Any,
    values: Any = None,
    location: str = "node",
    source_crs: Any = None,
) -> Any:
    """Build the PyVista ``PolyData`` surface a mesh layer draws, placed in the scene's display CRS.

    The connectivity is pyramids' (``Mesh2d.fan_triangles`` cuts every face into triangles with pure numpy);
    this warps the node coordinates into the scene's display CRS when the mesh declares a different one, lifts
    them to ``node_z``, and reads the triangles off them as the surface's cells. A triangle any of whose nodes
    warped to a non-finite coordinate — outside the projection's domain — is dropped, and the surviving nodes
    are re-indexed so the surface's bounds stay finite; that is how a partly off-view mesh still draws the part
    that is on it.

    Args:
        scene: The scene being drawn into. Its ``display_crs`` is adopted from the mesh when the scene has
            none yet **and the mesh places at least one face**, mirroring
            :meth:`~digitalearth.three_d.core.base.Scene3DBase._place` for a warpable input; a mesh that draws
            nothing leaves the scene's CRS untouched.
        mesh: A pyramids ``Mesh2d`` (``node_x`` / ``node_y`` / ``fan_triangles``, and an optional ``crs``).
        node_z: The elevation argument from :meth:`MeshMixin.trimesh`, resolved by :func:`_elevation`.

    Returns:
        A ``pyvista.PolyData`` surface whose cells are the mesh's fan triangles, or ``None`` when no triangle
        keeps all three nodes finite in the display CRS.
    """
    import numpy as np
    import pyvista as pv

    from digitalearth.base.spec.bounds import same_crs

    x = np.asarray(mesh.node_x, dtype="float64")
    y = np.asarray(mesh.node_y, dtype="float64")
    # The mesh's source CRS: a `MeshDataset` carries it on the wrapper, not on the bare `Mesh2d`, so the seam
    # conveys it as `source_crs`; a direct caller with a CRS-tagged `Mesh2d` falls back to `mesh.crs` (#557).
    held = source_crs if source_crs is not None else getattr(mesh, "crs", None)
    # Reproject only when the scene already draws in another CRS — this reads `display_crs`, it does not set
    # it. Adoption (setting the scene's CRS from this mesh) is deferred past the drawable check below, so a
    # mesh that places no face leaves the scene's CRS untouched (review L2).
    if (
        held is not None
        and scene.display_crs is not None
        and not same_crs(held, scene.display_crs)
    ):
        from pyramids.base.crs import reproject_coordinates

        xs, ys = reproject_coordinates(
            list(x), list(y), from_crs=held, to_crs=scene.display_crs
        )
        x, y = np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")
    z = _elevation(node_z, len(x))
    triangles = np.asarray(mesh.fan_triangles, dtype=np.intp)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    drawable = finite[triangles].all(axis=1)
    triangles = triangles[drawable]
    if len(triangles) == 0:
        return None
    # The mesh draws at least one face, so now adopt its CRS as the scene's when the scene has none yet —
    # the first drawn layer carrying a CRS sets it, exactly as `_place` does for a warpable input, but a
    # skipped mesh (returned above) never reaches here and so never sets it.
    if held is not None and scene.display_crs is None:
        scene.display_crs = held
    # Re-index to only the nodes the drawable triangles use, so a dropped (non-finite) node leaves no stray
    # point to stretch the surface's bounds. `searchsorted` maps each old index to its position in `used`.
    used = np.unique(triangles)
    remapped = np.searchsorted(used, triangles)
    points = np.column_stack([x[used], y[used], z[used]])
    # VTK face stream: each cell is `[n_points, *point_ids]`; every cell here is a triangle, so a leading 3.
    faces = np.hstack([np.full((len(remapped), 1), 3, dtype=np.intp), remapped]).ravel()
    surface = pv.PolyData(points, faces)
    if values is not None:
        from digitalearth.three_d.builders.vector import VALUE

        # Carry the shading values through the same drop/re-index as the geometry: a face value sits on each
        # drawn triangle (cell data), a node value on each kept node (point data), so a partly off-view mesh
        # keeps its colours aligned with the part that drew. `draw_trimesh` classifies them in place (#557).
        column = np.asarray(values, dtype="float64")
        if location == "face":
            surface.cell_data[VALUE] = column[drawable]
        else:
            surface.point_data[VALUE] = column[used]
    return surface


def draw_trimesh(scene: Any, data: Any, layer: Any) -> Any:
    """Build the PyVista surface a described mesh layer asks for, and add it to the plotter.

    The 3-D drawer counterpart of :meth:`MeshMixin.trimesh`: it reads the flat-colour and relief keywords off
    the layer's description and the geometry off its source (the ``Mesh2d``), so the layer rebuilds from its
    description plus its source alone — which is what lets a stored figure draw the same surface again.

    Args:
        scene: The scene being drawn into.
        data: The layer's source — the ``Mesh2d`` the builder stored. The surface's connectivity and node
            coordinates come off it.
        layer: The layer's description, whose props carry ``node_z``, ``color``, ``opacity``, ``show_edges``
            and ``edge_color``.

    Returns:
        The ``(surface, actor)`` pair, or ``None`` when the mesh placed no drawable face and the scene is not
        ``strict``.
    """
    from digitalearth.three_d.builders.vector import VALUE
    from digitalearth.three_d.core.base import classified_scalars

    props = drawing_props(layer.symbology.props)
    node_z = props.pop("node_z", None)
    values = props.pop("values", None)
    location = props.pop("location", "node")
    cmap = props.pop("cmap", "viridis")
    scheme = props.pop("scheme", None)
    k = props.pop("k", 5)
    source_crs = props.pop("source_crs", None)
    color = props.pop("color", MESH_FILL_COLOR)
    opacity = props.pop("opacity", None)
    show_edges = props.pop("show_edges", True)
    edge_color = props.pop("edge_color", MESH_EDGE_COLOR)
    surface = _mesh_surface(scene, data, node_z, values, location, source_crs)
    if surface is None:
        scene._skip_empty(
            "trimesh",
            "no triangle of the mesh keeps all three nodes finite in the display CRS, so there is nothing "
            "to draw",
        )
        return None
    extra: dict[str, Any] = {"show_edges": bool(show_edges), "edge_color": edge_color}
    if opacity is not None:
        extra["opacity"] = opacity
    if values is None:
        actor = scene.plotter.add_mesh(surface, color=color, **extra, **props)
        return surface, actor
    # Shaded by the mesh's own variable (#557): `_mesh_surface` carried the raw values through the geometry's
    # drop/re-index; classify them into the scalars PyVista colours by — the same classification every tier
    # cuts its classes with — and draw the surface unlit so the colour reads flat on a strict GL (a lit
    # normal-less surface came back blank on Mesa 26 llvmpipe, #538), the 3-D sibling of static `tripcolor`.
    store = surface.cell_data if location == "face" else surface.point_data
    encoding = layer.symbology.encoding("color")
    style = classified_scalars(
        store[VALUE],
        scheme=scheme,
        k=k,
        cmap=cmap,
        scale=None if encoding is None else encoding.scale,
    )
    store[VALUE] = style.pop("scalars")
    actor = scene.plotter.add_mesh(
        surface, scalars=VALUE, lighting=False, **style, **extra, **props
    )
    return surface, actor
