"""Scene bundles — save a 3-D scene to a versioned zip and open it back (#495).

forge3d saves and reloads whole scenes. This tier already has the hard half — a :class:`FigureSpec` that
serializes to a dict and redraws from one — so a bundle is a small, versioned container around that: a zip
holding one ``manifest.json`` with the figure description and the scene's drawn decoration (title styling, the
coordinate frame, the cartographic furniture, the lighting). The camera and vertical exaggeration already
travel inside the figure's description, so the manifest does not store them separately.

A bundle stores a **description**, not the data: a layer drawn from a path or URL is re-opened from it on load,
which is why :meth:`~digitalearth.three_d.core.base.Scene3DBase.save_bundle` refuses a scene whose layers were
built from an in-memory array or object — there is nothing a description can point back at. Only the standard
library is used here, so writing or reading a bundle needs no renderer.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

__all__ = ["BUNDLE_VERSION", "MANIFEST_NAME", "read_manifest", "write_bundle"]

#: The bundle format version written into every manifest and checked on read, so a future reader can refuse —
#: or migrate — a bundle it does not understand rather than failing on a missing field.
BUNDLE_VERSION: int = 1

#: The single JSON member every bundle zip carries.
MANIFEST_NAME: str = "manifest.json"


def write_bundle(
    path: Any, *, figure_dict: dict[str, Any], decoration: dict[str, Any]
) -> Path:
    """Write a scene bundle zip holding the figure description and the scene's decoration.

    Args:
        path: The output file.
        figure_dict: The scene's ``figure_spec.to_dict()`` — the description every layer redraws from.
        decoration: The scene's drawn decoration state (title styling, axes, furniture, lighting).

    Returns:
        pathlib.Path: the file that was written.
    """
    manifest = {
        "format_version": BUNDLE_VERSION,
        "figure": figure_dict,
        "decoration": decoration,
    }
    out = Path(str(path))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    return out


def read_manifest(path: Any) -> dict[str, Any]:
    """Read and version-check a scene bundle's manifest.

    Args:
        path: The bundle file to read.

    Returns:
        The manifest dict, with its ``figure`` and ``decoration`` members.

    Raises:
        ValueError: when the file carries no manifest, or a format version this release cannot read.
    """
    with zipfile.ZipFile(str(path)) as bundle:
        if MANIFEST_NAME not in bundle.namelist():
            raise ValueError(
                f"{path} is not a digitalearth scene bundle: it has no {MANIFEST_NAME}"
            )
        manifest: dict[str, Any] = json.loads(bundle.read(MANIFEST_NAME))
    version = manifest.get("format_version")
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"the scene bundle at {path} is format version {version!r}, which this digitalearth cannot read "
            f"(it writes and reads version {BUNDLE_VERSION})"
        )
    return manifest
