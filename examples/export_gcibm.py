"""
Fluxel Export Script for ParaView

This script generates an AMR mesh using the `fluxel` Python module
and exports it to VTK (.vtu) using PyVista for visualization in ParaView.

Configuration is read from a YAML file.
"""

import argparse
import pathlib
import time

import numpy as np
import pyvista as pv
import yaml
from pydantic import BaseModel, Field, field_validator

from fluxel import BoundingBox, CfdGhostCellMesh, FluxelManager


class BoundingBoxConfig(BaseModel, frozen=True):
    """Axis-aligned domain bounding box (min / max corners)."""

    min: list[float, float, float]
    max: list[float, float, float]


class GcibmExportConfig(BaseModel, frozen=True):
    """Validated GCIBM export settings loaded from YAML."""

    input_stl: pathlib.Path
    output_vtu: pathlib.Path = Field(
        default_factory=lambda: pathlib.Path("gcibm_result.vtu")
    )
    target_level: int = 3
    n_leaf_refinement: int = 3
    base_resolution: list[int, int, int]
    bounding_box: BoundingBoxConfig = Field(default_factory=BoundingBoxConfig)
    fluid_seed_point: list[float, float, float]

    @field_validator("input_stl", mode="after")
    @classmethod
    def _validate_input_stl(cls, v: pathlib.Path) -> pathlib.Path:
        if v.suffix != ".stl":
            raise ValueError(f"input_stl must be an STL file: {v}")
        return v

    @field_validator("base_resolution", mode="after")
    @classmethod
    def _validate_base_resolution(
        cls, v: list[int, int, int]
    ) -> list[int, int, int]:
        for n in v:
            if n < 1:
                msg = "base_resolution entries must be positive integers"
                raise ValueError(msg)
        return v


def load_gcibm_config(path: pathlib.Path) -> GcibmExportConfig:
    """
    Load GCIBM export settings from a YAML file and validate with Pydantic.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    GcibmExportConfig
        Parsed and validated configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return GcibmExportConfig.model_validate(raw)


def mesh_to_unstructured_grid(
    cell_centers: np.ndarray, cell_sizes: np.ndarray
) -> pv.UnstructuredGrid:
    """
    Converts cell centers and sizes into a UnstructuredGrid of Hexahedrons.

    Parameters
    ----------
    cell_centers : np.ndarray
        The (N, 3) array of cell center coordinates.
    cell_sizes : np.ndarray
        The (N, 3) array of cell sizes.

    Returns
    -------
    pv.UnstructuredGrid
        The generated UnstructuredGrid.
    """
    n_cells = len(cell_centers)

    hx = cell_sizes[:, 0] / 2.0
    hy = cell_sizes[:, 1] / 2.0
    hz = cell_sizes[:, 2] / 2.0
    cx = cell_centers[:, 0]
    cy = cell_centers[:, 1]
    cz = cell_centers[:, 2]

    # VTK Hexahedron node ordering
    # 0: -x, -y, -z
    # 1: +x, -y, -z
    # 2: +x, +y, -z
    # 3: -x, +y, -z
    # 4: -x, -y, +z
    # 5: +x, -y, +z
    # 6: +x, +y, +z
    # 7: -x, +y, +z
    pts = np.empty((n_cells, 8, 3), dtype=np.float64)

    pts[:, 0, 0] = cx - hx
    pts[:, 0, 1] = cy - hy
    pts[:, 0, 2] = cz - hz
    pts[:, 1, 0] = cx + hx
    pts[:, 1, 1] = cy - hy
    pts[:, 1, 2] = cz - hz
    pts[:, 2, 0] = cx + hx
    pts[:, 2, 1] = cy + hy
    pts[:, 2, 2] = cz - hz
    pts[:, 3, 0] = cx - hx
    pts[:, 3, 1] = cy + hy
    pts[:, 3, 2] = cz - hz

    pts[:, 4, 0] = cx - hx
    pts[:, 4, 1] = cy - hy
    pts[:, 4, 2] = cz + hz
    pts[:, 5, 0] = cx + hx
    pts[:, 5, 1] = cy - hy
    pts[:, 5, 2] = cz + hz
    pts[:, 6, 0] = cx + hx
    pts[:, 6, 1] = cy + hy
    pts[:, 6, 2] = cz + hz
    pts[:, 7, 0] = cx - hx
    pts[:, 7, 1] = cy + hy
    pts[:, 7, 2] = cz + hz

    points = pts.reshape(-1, 3)

    # Connectivity array for PyVista: [n_points, p0, p1, p2, p3, p4, p5, p6, p7]
    cells = np.empty((n_cells, 9), dtype=np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = np.arange(n_cells * 8).reshape(n_cells, 8)

    cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    return grid


def export_gcibm_debug_data(
    mesh: CfdGhostCellMesh, output_prefix: str = "gcibm_debug"
) -> None:
    """
    Exports GCIBM debug arrays and `N_ghosts` polylines:
    C_i -> B_i -> I_i for each ghost i.

    - C_i: ghost cell center (from `cell_centers[gc_cell_ids[i]]`)
    - B_i: boundary intercept (`gc_bnd_intercepts[i]`)
    - I_i: image point (`gc_image_points[i]`)

    Parameters
    ----------
    mesh : CfdGhostCellMesh
        Mesh object returned by `build_ghost_cell_mesh`.
    output_prefix : str
        File prefix for debug outputs.
    """
    # Keep all raw debug arrays in a single compressed artifact.
    np.savez_compressed(
        f"{output_prefix}.npz",
        gc_is_fluid=np.asarray(mesh.gc_is_fluid),
        gc_cell_ids=np.asarray(mesh.gc_cell_ids),
        gc_bnd_anchor_ids=np.asarray(mesh.gc_bnd_anchor_ids),
        gc_bnd_intercepts=np.asarray(mesh.gc_bnd_intercepts),
        gc_image_points=np.asarray(mesh.gc_image_points),
        gc_interp_stencil_indices=np.asarray(mesh.gc_interp_stencil_indices),
        gc_interp_stencil_weights=np.asarray(mesh.gc_interp_stencil_weights),
    )
    print(f"Saved debug arrays to {output_prefix}.npz")

    gc_cell_ids = np.asarray(mesh.gc_cell_ids)
    gc_bnd_anchor_ids = np.asarray(mesh.gc_bnd_anchor_ids)
    centers = np.asarray(mesh.cell_centers)[gc_cell_ids]
    intercepts = np.asarray(mesh.gc_bnd_intercepts)
    image_points = np.asarray(mesh.gc_image_points)

    if not (len(centers) == len(intercepts) == len(image_points)):
        raise ValueError(
            "Ghost arrays must have the same length: "
            "len(gc_cell_ids) == len(gc_bnd_intercepts) == len(gc_image_points)"
        )

    n_ghosts = len(gc_cell_ids)
    if n_ghosts == 0:
        print("No ghost cells found; skipping polyline export.")
        return

    # Points are laid out as [C0, B0, I0, C1, B1, I1, ...].
    points = np.empty((n_ghosts * 3, 3), dtype=np.float64)
    points[0::3] = centers
    points[1::3] = intercepts
    points[2::3] = image_points

    # VTK polyline connectivity: [3, p0, p1, p2] repeated for each polyline.
    lines = np.empty((n_ghosts, 4), dtype=np.int64)
    lines[:, 0] = 3
    base = (np.arange(n_ghosts, dtype=np.int64) * 3).reshape(-1, 1)
    lines[:, 1:] = base + np.array([0, 1, 2], dtype=np.int64)[None, :]

    # Build PolyData with line cells only (avoid implicit vertex cells).
    pd = pv.PolyData(points, lines=lines.ravel())
    pd.cell_data["gc_cell_id"] = gc_cell_ids
    pd.cell_data["gc_bnd_anchor_id"] = gc_bnd_anchor_ids
    pd.save(f"{output_prefix}.vtp")
    print(f"Saved ghost polylines to {output_prefix}.vtp")


def export_gcibm_mesh(
    mesh: CfdGhostCellMesh, output_path: pathlib.Path
) -> None:
    """
    Exports a GCIBM mesh to a VTU file.
    Also exports debug arrays and ghost polylines as a separate VTK file.

    Parameters
    ----------
    mesh : CfdGhostCellMesh
        Mesh object returned by `build_ghost_cell_mesh`.
    output_path : pathlib.Path
        Path to the output VTU file.
    """
    # 1. セルメッシュの作成
    grid = mesh_to_unstructured_grid(mesh.cell_centers, mesh.cell_sizes)

    # セルデータの割り当て
    is_ghost_cell = np.zeros(len(mesh.cell_centers), dtype=bool)
    is_ghost_cell[mesh.gc_cell_ids] = True
    grid.cell_data["cell_sizes"] = mesh.cell_sizes
    grid.cell_data["cell_volume"] = np.prod(mesh.cell_sizes, axis=1)
    grid.cell_data["is_fluid_cell"] = mesh.gc_is_fluid
    grid.cell_data["is_ghost_cell"] = is_ghost_cell

    # 保存
    grid.save(output_path)
    print(f"Saved GCIBM mesh to {output_path}")

    # 2. GCIBM の 可視化 (デバッグ用)
    parent_dir = output_path.parent
    output_prefix = output_path.stem
    export_gcibm_debug_data(
        mesh, output_prefix=f"{parent_dir}/{output_prefix}_debug"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a GCIBM mesh from an STL file and export it for ParaView."
            "All run parameters are read from a YAML configuration file."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to YAML configuration file (see configs/DrivAer.yaml).",
    )

    args = parser.parse_args()
    cfg = load_gcibm_config(args.config)

    print("=== Fluxel One-Stop GCIBM Mesh Generator ===")

    # 1. ドメインの設定
    bbox = BoundingBox(
        min=cfg.bounding_box.min,
        max=cfg.bounding_box.max,
    )
    base_res = cfg.base_resolution
    target_level = cfg.target_level
    n_leaf_refinement = cfg.n_leaf_refinement
    stl_path = cfg.input_stl
    output_path = cfg.output_vtu
    fluid_seed = cfg.fluid_seed_point

    print(f"Config file: {args.config.resolve()}")
    print(f"Domain Bounds: Min {bbox.min} Max {bbox.max}")
    print(f"Base Resolution: {base_res}")
    print(f"Target Level: {target_level}")
    print(f"Number of Leaf Refinements: {n_leaf_refinement}")
    print(f"Input STL File: {stl_path}")
    print(f"Output VTU File: {output_path}")
    print(f"Fluid seed point: {fluid_seed}")

    # 2. FluxelManager の初期化
    manager = FluxelManager(
        bbox, base_res=base_res, n_leaf_refinement=n_leaf_refinement
    )
    if not stl_path.exists():
        raise SystemExit(f"Input STL file does not exist: {stl_path}")

    # 3. メッシュの生成
    print("\n Starting mesh generation...")
    t0 = time.time()
    mesh = manager.build_ghost_cell_mesh(
        str(stl_path), target_level=target_level, fluid_seed_point=fluid_seed
    )
    t1 = time.time()
    print(f"Mesh generation completed in {t1 - t0:.2f} seconds")

    # 4. メッシュサマリーの出力
    num_cells = len(mesh.cell_centers)
    num_x_faces = len(mesh.x_faces_owner)
    num_y_faces = len(mesh.y_faces_owner)
    num_z_faces = len(mesh.z_faces_owner)
    num_x_bnd_minus = len(mesh.x_bnd_minus_owner)
    num_x_bnd_plus = len(mesh.x_bnd_plus_owner)
    num_y_bnd_minus = len(mesh.y_bnd_minus_owner)
    num_y_bnd_plus = len(mesh.y_bnd_plus_owner)
    num_z_bnd_minus = len(mesh.z_bnd_minus_owner)
    num_z_bnd_plus = len(mesh.z_bnd_plus_owner)
    num_ghosts = len(mesh.gc_cell_ids)
    print("\n=== Mesh Summary ===")
    print(f"Total Cells: {num_cells}")
    print(f"Total X Faces: {num_x_faces}")
    print(f"Total Y Faces: {num_y_faces}")
    print(f"Total Z Faces: {num_z_faces}")
    print(f"Total X Boundary Minus: {num_x_bnd_minus}")
    print(f"Total X Boundary Plus: {num_x_bnd_plus}")
    print(f"Total Y Boundary Minus: {num_y_bnd_minus}")
    print(f"Total Y Boundary Plus: {num_y_bnd_plus}")
    print(f"Total Z Boundary Minus: {num_z_bnd_minus}")
    print(f"Total Z Boundary Plus: {num_z_bnd_plus}")
    print(f"Total Ghost Cells: {num_ghosts}")
    print("====================")

    # 5. メッシュの保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_gcibm_mesh(mesh, output_path)
