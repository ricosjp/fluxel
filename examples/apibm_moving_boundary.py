#!/usr/bin/env python3
"""Rigid moving-boundary demo for ApibmSession using the basket OBJ.

Writes a VTU (plus APIBM debug artifacts) after each phase under
``examples/output/apibm/moving_boundary/``.
"""

from __future__ import annotations

import pathlib
import sys

from fluxel import (
    Axis,
    BoundingBox,
    CfdAxisProjectedMesh,
    FluxelManager,
    quaternion_from_axis_angle,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(__file__).resolve().parent
BASKET_OBJ = ROOT / "data" / "obj" / "basket.obj"
OUT_DIR = EXAMPLES / "output" / "apibm" / "moving_boundary"

sys.path.insert(0, str(EXAMPLES))
from export_apibm import export_apibm_mesh  # noqa: E402


def save_phase(mesh: CfdAxisProjectedMesh, name: str) -> pathlib.Path:
    """
    Export one phase as VTU plus debug NPZ / polylines.

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh snapshot to write.
    name : str
        Stem used for the VTU filename under ``OUT_DIR``.

    Returns
    -------
    pathlib.Path
        Path of the written VTU file.
    """
    path = OUT_DIR / f"{name}.vtu"
    export_apibm_mesh(mesh, path)
    return path


def main() -> None:
    """
    Run the basket moving-boundary demo and write VTU snapshots.

    Phases are identity, ``+1`` translation, ``+4`` translation, remesh at
    ``+4``, then a 45-degree rotation about y at the remeshed pose.
    """
    if not BASKET_OBJ.exists():
        raise SystemExit(f"Missing OBJ: {BASKET_OBJ}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bbox = BoundingBox([-8.0, -4.0, -4.0], [16.0, 4.0, 4.0])
    manager = FluxelManager(bbox, base_res=[24, 8, 8], n_leaf_refinement=2)
    session = manager.create_axis_projected_session(
        str(BASKET_OBJ), target_level=2
    )

    mesh = session.mesh
    n0 = int(mesh.ap.is_immersed_face.sum())
    print(
        f"initial cells={mesh.n_cells} faces={len(mesh.internal_faces_owner)} "
        f"immersed={n0}"
    )
    save_phase(mesh, "00_initial")

    # Small translation: IB stays in the refined band → immersed faces remain.
    mesh = session.update_ib(
        translation=[1.0, 0.0, 0.0],
        warn_outside_refinement=True,
    )
    n1 = int(mesh.ap.is_immersed_face.sum())
    print(
        f"after update_ib(+1.0) translation={session.translation} immersed={n1}"
    )
    save_phase(mesh, "01_update_ib_plus1")

    # Larger translation: IB leaves the original refined band.
    mesh = session.update_ib(
        translation=[4.0, 0.0, 0.0],
        warn_outside_refinement=True,
    )
    n2 = int(mesh.ap.is_immersed_face.sum())
    print(
        f"after update_ib(+4.0) translation={session.translation} immersed={n2}"
    )
    save_phase(mesh, "02_update_ib_plus4")

    # Remesh at the same pose so AMR / IB are rebuilt together.
    mesh = session.remesh(
        translation=[4.0, 0.0, 0.0],
        warn_outside_refinement=False,
    )
    n3 = int(mesh.ap.is_immersed_face.sum())
    print(
        f"after remesh(+4.0) cells={mesh.n_cells} "
        f"faces={len(mesh.internal_faces_owner)} immersed={n3}"
    )
    save_phase(mesh, "03_remesh_plus4")

    # Rotate 45 deg about y at the remeshed pose (topology fixed).
    mesh = session.update_ib(
        translation=[4.0, 0.0, 0.0],
        rotation_quaternion=quaternion_from_axis_angle(
            Axis.Y, 45.0, degrees=True
        ),
        warn_outside_refinement=True,
    )
    n4 = int(mesh.ap.is_immersed_face.sum())
    print(
        f"after update_ib(+4.0, rot y 45deg) translation={session.translation} "
        f"rotation={session.rotation_quaternion} immersed={n4}"
    )
    save_phase(mesh, "04_update_ib_rot_y_45")

    print(f"\nVTU outputs written under {OUT_DIR}")
    print(
        "Open the .vtu files in ParaView; "
        "cell arrays x/y/z_bnd_type mark IB sides."
    )


if __name__ == "__main__":
    main()
