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
from pydantic import BaseModel, Field, field_validator, model_validator

from fluxel import (
    BoundingBox,
    CfdGhostCellMesh,
    Cylindrical,
    Float3,
    FluxelManager,
    Int3,
)


class BoundingBoxConfig(BaseModel, frozen=True):
    """Axis-aligned domain bounding box (min / max corners)."""

    min: Float3
    max: Float3


class CylindricalDomainConfig(BaseModel, frozen=True):
    """Annular sector or full turn about world Z. Angles are radians."""

    origin: Float3
    r_min: float
    r_max: float
    theta_start: float
    theta_extent: float
    z_min: float
    z_max: float


class GcibmExportConfig(BaseModel, frozen=True):
    """Validated GCIBM export settings loaded from YAML."""

    input_mesh: pathlib.Path
    output_vtu: pathlib.Path = Field(
        default_factory=lambda: pathlib.Path("gcibm_result.vtu")
    )
    target_level: int = 3
    n_leaf_refinement: int = 3
    max_cells: int | None = Field(default=None, gt=0, strict=True)
    base_resolution: Int3
    bounding_box: BoundingBoxConfig | None = None
    cylindrical: CylindricalDomainConfig | None = None
    fluid_seed_point: Float3

    @model_validator(mode="after")
    def _one_domain(self) -> "GcibmExportConfig":
        if (self.bounding_box is None) == (self.cylindrical is None):
            raise ValueError(
                "specify exactly one of bounding_box or cylindrical"
            )
        return self

    @field_validator("input_mesh", mode="after")
    @classmethod
    def _validate_input_mesh(cls, v: pathlib.Path) -> pathlib.Path:
        match v.suffix:
            case ".stl":
                return v
            case ".obj":
                return v
            case _:
                raise ValueError(f"input_mesh must be an STL or OBJ file: {v}")

    @field_validator("base_resolution", mode="after")
    @classmethod
    def _validate_base_resolution(cls, v: Int3) -> Int3:
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


def mesh_to_unstructured_grid(cell_corners: np.ndarray) -> pv.UnstructuredGrid:
    """
    Build hexahedra from VTK corner order.

    Parameters
    ----------
    cell_corners : np.ndarray
        World corners of shape (N, 8, 3). Cartesian corners match the former
        center-and-size boxes. Cylindrical corners are parameter corners.

    Returns
    -------
    pv.UnstructuredGrid
        The generated UnstructuredGrid.
    """
    corners = np.asarray(cell_corners, dtype=np.float64)
    n_cells = corners.shape[0]
    points = corners.reshape(-1, 3)

    # Connectivity array for PyVista: [n_points, p0, p1, p2, p3, p4, p5, p6, p7]
    cells = np.empty((n_cells, 9), dtype=np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = np.arange(n_cells * 8).reshape(n_cells, 8)

    cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    cleaned_grid = grid.clean(tolerance=1e-6)
    return cleaned_grid


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
        gc_is_fluid=mesh.gc_is_fluid,
        gc_cell_ids=mesh.gc_cell_ids,
        gc_bnd_anchor_ids=mesh.gc_bnd_anchor_ids,
        gc_bnd_patch_ids=mesh.gc_bnd_patch_ids,
        gc_bnd_intercepts=mesh.gc_bnd_intercepts,
        gc_image_points=mesh.gc_image_points,
        gc_interp_stencil_indices=mesh.gc_interp_stencil_indices,
        gc_interp_stencil_weights=mesh.gc_interp_stencil_weights,
    )
    print(f"Saved debug arrays to {output_prefix}.npz")

    centers = mesh.cell_centers[mesh.gc_cell_ids]
    intercepts = mesh.gc_bnd_intercepts
    image_points = mesh.gc_image_points

    if not (len(centers) == len(intercepts) == len(image_points)):
        raise ValueError(
            "Ghost arrays must have the same length: "
            "len(gc_cell_ids) == len(gc_bnd_intercepts) == len(gc_image_points)"
        )

    n_ghosts = len(mesh.gc_cell_ids)
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
    pd.cell_data["gc_cell_id"] = mesh.gc_cell_ids
    pd.cell_data["gc_bnd_anchor_id"] = mesh.gc_bnd_anchor_ids
    pd.cell_data["gc_bnd_patch_id"] = mesh.gc_bnd_patch_ids
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
    grid = mesh_to_unstructured_grid(mesh.cell_corners)

    # セルデータの割り当て
    is_ghost_cell = np.zeros(len(mesh.cell_centers), dtype=bool)
    is_ghost_cell[mesh.gc_cell_ids] = True
    grid.cell_data["cell_sizes"] = mesh.cell_sizes
    grid.cell_data["cell_volume"] = mesh.cell_volumes
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
            "Generate a GCIBM mesh from STL / OBJ and export it as VTU."
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
    base_res = cfg.base_resolution
    target_level = cfg.target_level
    n_leaf_refinement = cfg.n_leaf_refinement
    mesh_path = cfg.input_mesh
    output_path = cfg.output_vtu
    fluid_seed = cfg.fluid_seed_point

    print(f"Config file: {args.config.resolve()}")
    if cfg.cylindrical is None:
        assert cfg.bounding_box is not None
        print(
            "Domain Bounds: "
            f"Min {cfg.bounding_box.min} Max {cfg.bounding_box.max}"
        )
    else:
        print(
            "Cylindrical domain: "
            f"r [{cfg.cylindrical.r_min}, {cfg.cylindrical.r_max}], "
            f"theta [{cfg.cylindrical.theta_start}, "
            f"{cfg.cylindrical.theta_extent}]"
        )
    print(f"Base Resolution: {base_res}")
    print(f"Target Level: {target_level}")
    print(f"Number of Leaf Refinements: {n_leaf_refinement}")
    print(
        "Maximum cells: "
        f"{cfg.max_cells if cfg.max_cells is not None else 'unlimited'}"
    )
    print(f"Input Mesh File: {mesh_path}")
    print(f"Output VTU File: {output_path}")
    print(f"Fluid seed point: {fluid_seed}")

    # 2. FluxelManager の初期化
    if cfg.cylindrical is None:
        assert cfg.bounding_box is not None
        manager = FluxelManager(
            BoundingBox(min=cfg.bounding_box.min, max=cfg.bounding_box.max),
            base_res=base_res,
            n_leaf_refinement=n_leaf_refinement,
            max_cells=cfg.max_cells,
        )
    else:
        domain = cfg.cylindrical
        manager = FluxelManager(
            Cylindrical(
                domain.origin,
                domain.r_min,
                domain.r_max,
                domain.theta_start,
                domain.theta_extent,
                domain.z_min,
                domain.z_max,
            ),
            base_res=base_res,
            n_leaf_refinement=n_leaf_refinement,
            max_cells=cfg.max_cells,
        )
    if not mesh_path.exists():
        raise SystemExit(f"Input mesh file does not exist: {mesh_path}")

    # 3. メッシュの生成
    print("\n Starting mesh generation...")
    t0 = time.time()
    mesh = manager.build_ghost_cell_mesh(
        str(mesh_path), target_level=target_level, fluid_seed_point=fluid_seed
    )
    t1 = time.time()
    print(f"Mesh generation completed in {t1 - t0:.2f} seconds")

    # 4. メッシュサマリーの出力
    num_cells = len(mesh.cell_centers)
    axis = mesh.internal_faces_axis
    num_x_faces = int(np.count_nonzero(axis == 0))
    num_y_faces = int(np.count_nonzero(axis == 1))
    num_z_faces = int(np.count_nonzero(axis == 2))
    bnd_dir = mesh.domain_bnd_faces_dir
    num_x_bnd_minus = int(np.count_nonzero(bnd_dir == 0))
    num_x_bnd_plus = int(np.count_nonzero(bnd_dir == 1))
    num_y_bnd_minus = int(np.count_nonzero(bnd_dir == 2))
    num_y_bnd_plus = int(np.count_nonzero(bnd_dir == 3))
    num_z_bnd_minus = int(np.count_nonzero(bnd_dir == 4))
    num_z_bnd_plus = int(np.count_nonzero(bnd_dir == 5))
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
    print(f"Patch Names: {list(mesh.patch_name_to_id.keys())}")
    print("====================")

    # 5. メッシュの保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_gcibm_mesh(mesh, output_path)
