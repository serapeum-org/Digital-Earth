"""MeshMixin — the 3-D tier's unstructured-mesh builder (#546).

``trimesh`` draws a UGRID ``pyramids.netcdf.ugrid.mesh.Mesh2d`` — connectivity plus node coordinates — as one
PyVista ``PolyData`` surface: every face is fan-cut into triangles by pyramids (``Mesh2d.fan_triangles``), and
the triangles become the surface's cells. It is the 3-D sibling of the static ``tripcolor``, the interactive
``trimesh`` and the web ``trimesh`` renders, and like them it consumes the connectivity pyramids hands over
rather than re-triangulating anything itself.

**A bare ``Mesh2d`` carries geometry, not a per-element value table**, so the surface takes one flat ``color``
rather than being shaded by a node or face variable — exactly as the interactive and web tiers draw a UGRID
mesh. To shade a mesh by a variable, draw the ``MeshDataset`` on the static tier, whose ``tripcolor`` colours
each face through cleopatra. The surface lies on the ground plane (``z=0``) by default; pass ``node_z`` to lift
each node and read the mesh as 3-D relief.

**CRS follows the tier's rule, not the web tier's.** The 3-D tier draws in whatever display CRS the scene
carries — it is not EPSG:4326-only — so a mesh that declares a CRS sets the scene's when it has none yet, and
is reprojected through pyramids (``pyramids.base.crs.reproject_coordinates``) only when the scene already draws
in a different one. A mesh with no declared CRS is drawn in its own coordinates, the same bargain every tier
strikes for an unreferenced mesh.

The layer is described from its source (the ``Mesh2d`` itself) and rebuilt by :func:`draw_trimesh`, so it owns
a ``LayerSpec`` and a slot in the figure like every other 3-D layer: it is in ``layer_ids``, and is removable,
hideable and re-describable by id. numpy is imported lazily and PyVista only inside the drawer; importing the
tier needs neither.
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
        node_z: Any = None,
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

        A bare ``Mesh2d`` carries geometry, not a per-element value table, so the surface takes one flat
        ``color`` rather than being shaded by a node or face variable — exactly as the interactive and web
        tiers draw a UGRID mesh. To shade a mesh by a variable, draw the ``MeshDataset`` on the static tier,
        whose ``tripcolor`` colours each face through cleopatra.

        Args:
            mesh: A pyramids ``Mesh2d`` — anything exposing ``node_x`` / ``node_y`` / ``fan_triangles``, and
                optionally a ``crs`` to reproject from. This is what
                :func:`~digitalearth.api._mesh_of` hands over when ``quickmap`` routes a ``MeshDataset`` here.
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
        return self._add_described_layer(
            kind="mesh",
            data=mesh,
            name=name,
            visible=visible,
            node_z=node_z,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            edge_color=edge_color,
            **kwargs,
        )


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


def _mesh_surface(scene: Any, mesh: Any, node_z: Any) -> Any:
    """Build the PyVista ``PolyData`` surface a mesh layer draws, placed in the scene's display CRS.

    The connectivity is pyramids' (``Mesh2d.fan_triangles`` cuts every face into triangles with pure numpy);
    this warps the node coordinates into the scene's display CRS when the mesh declares a different one, lifts
    them to ``node_z``, and reads the triangles off them as the surface's cells. A triangle any of whose nodes
    warped to a non-finite coordinate — outside the projection's domain — is dropped, and the surviving nodes
    are re-indexed so the surface's bounds stay finite; that is how a partly off-view mesh still draws the part
    that is on it.

    Args:
        scene: The scene being drawn into. Its ``display_crs`` is adopted from the mesh when the scene has
            none yet, mirroring :meth:`~digitalearth.three_d.core.base.Scene3DBase._place` for a warpable input.
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
    held = getattr(mesh, "crs", None)
    if held is not None:
        if scene.display_crs is None:
            # The first layer carrying a CRS sets the scene's, exactly as `_place` does for a warpable input.
            scene.display_crs = held
        elif not same_crs(held, scene.display_crs):
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
    # Re-index to only the nodes the drawable triangles use, so a dropped (non-finite) node leaves no stray
    # point to stretch the surface's bounds. `searchsorted` maps each old index to its position in `used`.
    used = np.unique(triangles)
    remapped = np.searchsorted(used, triangles)
    points = np.column_stack([x[used], y[used], z[used]])
    # VTK face stream: each cell is `[n_points, *point_ids]`; every cell here is a triangle, so a leading 3.
    faces = np.hstack([np.full((len(remapped), 1), 3, dtype=np.intp), remapped]).ravel()
    return pv.PolyData(points, faces)


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
    props = drawing_props(layer.symbology.props)
    node_z = props.pop("node_z", None)
    color = props.pop("color", MESH_FILL_COLOR)
    opacity = props.pop("opacity", None)
    show_edges = props.pop("show_edges", True)
    edge_color = props.pop("edge_color", MESH_EDGE_COLOR)
    surface = _mesh_surface(scene, data, node_z)
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
    actor = scene.plotter.add_mesh(surface, color=color, **extra, **props)
    return surface, actor
