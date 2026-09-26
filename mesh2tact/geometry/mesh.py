"""Mesh loading and optional primitives for geometric image rendering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    units: str | None = "mm",
    center: bool = True,
) -> trimesh.Trimesh:
    """Load an STL/OBJ/PLY/STEP-ish mesh and return it in metres.

    Parameters
    ----------
    path:
        File to load.  Anything ``trimesh`` can read (``.stl``, ``.obj``,
        ``.ply``, ``.glb`` ...).
    scale:
        Extra multiplicative scale applied after unit conversion.
    units:
        Units the file is authored in.  Most CAD-exported STL files are in
        millimetres, which is the default.  Pass ``None`` to skip conversion.
    center:
        Translate the mesh so its centroid sits at the origin.  The solver
        treats the object's local origin as its pivot, so centring makes
        rotation behave the way a user expects when dragging.
    """
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a single triangular mesh")

    factor = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}.get(units or "m", 1.0)
    mesh.apply_scale(factor * scale)

    if center:
        mesh.apply_translation(-mesh.centroid)

    mesh.merge_vertices()
    # trimesh 4 removed Trimesh.remove_degenerate_faces(); nondegenerate_faces()
    # is the supported replacement and exists in late 3.x as well.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.fix_normals()
    return mesh


def primitive(kind: str = "sphere", size: float = 0.004, **kwargs) -> trimesh.Trimesh:
    """Small analytic shapes, handy for testing without a CAD file."""
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=size)
    if kind == "box":
        return trimesh.creation.box(extents=(size, size, size))
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=size, height=kwargs.get("height", 2 * size))
    if kind == "cone":
        return trimesh.creation.cone(radius=size, height=kwargs.get("height", 2 * size))
    raise ValueError(f"unknown primitive {kind!r}")


