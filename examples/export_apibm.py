"""
Fluxel Export Script for ParaView

This script generates an AMR mesh using the `fluxel` Python module
and exports it to VTK (.vtu) using PyVista for visualization in ParaView.

Configuration is read from a YAML file.
"""

import argparse
import pathlib
import time
from typing import Literal

import numpy as np
import pyvista as pv
import yaml
from pydantic import BaseModel, Field, field_validator

from fluxel import BoundingBox, CfdAxisProjectedMesh, FluxelManager


class BoundingBoxConfig(BaseModel, frozen=True):
    """Axis-aligned domain bounding box (min / max corners)."""

    min: list[float, float, float]
    max: list[float, float, float]


class ApibmExportConfig(BaseModel, frozen=True):
    """Validated APIBM export settings loaded from YAML."""

    input_stl: pathlib.Path
    output_vtu: pathlib.Path = Field(
        default_factory=lambda: pathlib.Path("apibm_result.vtu")
    )
    target_level: int = 3
    n_leaf_refinement: int = 3
    base_resolution: list[int, int, int]
    bounding_box: BoundingBoxConfig = Field(default_factory=BoundingBoxConfig)

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


def load_apibm_config(path: pathlib.Path) -> ApibmExportConfig:
    """
    Load APIBM export settings from a YAML file and validate with Pydantic.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    ApibmExportConfig
        Parsed and validated configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ApibmExportConfig.model_validate(raw)


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


def export_axis_projected_polylines(
    axis: Literal["x", "y", "z"],
    cell_centers: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    has_bnd: np.ndarray,
    dist_owner_to_bnd: np.ndarray,
    dist_neighbour_to_bnd: np.ndarray,
    owner_far_cell_id: np.ndarray,
    neighbour_far_cell_id: np.ndarray,
    owner_weights: np.ndarray,
    neighbour_weights: np.ndarray,
    owner_bnd_anchor_id: np.ndarray,
    neighbour_bnd_anchor_id: np.ndarray,
    output_prefix: str,
) -> None:
    """
    Exports axis-projected polylines:
    Far -> owner/neighbour center -> boundary intercept per face.

    Each interior face with a boundary hit produces **two** VTK polylines:
    first the owner-side stencil (Far_o, C_o, B),
    then the neighbour-side stencil (Far_n, C_n, B).
    """
    has_bnd = np.asarray(has_bnd, dtype=bool)
    n_hit = int(np.sum(has_bnd))
    owner_bnd_anchor_id = np.asarray(owner_bnd_anchor_id)
    neighbour_bnd_anchor_id = np.asarray(neighbour_bnd_anchor_id)
    if n_hit != len(owner_bnd_anchor_id) or n_hit != len(
        neighbour_bnd_anchor_id
    ):
        raise ValueError(
            f"owner_bnd_anchor_id and neighbour_bnd_anchor_id length \
            ({len(owner_bnd_anchor_id)} and {len(neighbour_bnd_anchor_id)}) \
            must equal {n_hit}"
        )
    if n_hit != len(dist_owner_to_bnd) or n_hit != len(dist_neighbour_to_bnd):
        raise ValueError(
            "dist_* and has_bnd selections must have consistent lengths"
        )
    direction = np.array([0.0, 0.0, 0.0])
    match axis:
        case "x":
            direction[0] = 1.0
        case "y":
            direction[1] = 1.0
        case "z":
            direction[2] = 1.0
        case _:
            raise ValueError(f"Invalid axis: {axis}")

    x_o_far = cell_centers[owner_far_cell_id]
    x_o = cell_centers[owner[has_bnd]]
    x_o_bnd = x_o + dist_owner_to_bnd[:, None] * direction[None, :]

    x_n_far = cell_centers[neighbour_far_cell_id]
    x_n = cell_centers[neighbour[has_bnd]]
    x_n_bnd = x_n - dist_neighbour_to_bnd[:, None] * direction[None, :]

    n_owner_faces_has_bnd = len(x_o)
    n_neighbour_faces_has_bnd = len(x_n)
    if n_owner_faces_has_bnd != n_neighbour_faces_has_bnd:
        raise ValueError("owner and neighbour filtered face counts must match")
    n_faces_has_bnd = n_owner_faces_has_bnd + n_neighbour_faces_has_bnd

    x_far = np.concatenate([x_o_far, x_n_far], axis=0)
    x = np.concatenate([x_o, x_n], axis=0)
    x_bnd = np.concatenate([x_o_bnd, x_n_bnd], axis=0)

    sides = np.concatenate(
        [np.zeros(n_owner_faces_has_bnd), np.ones(n_neighbour_faces_has_bnd)],
        axis=0,
    )
    weights = np.concatenate([owner_weights, neighbour_weights], axis=0)
    anchor_per_polyline = np.concatenate(
        [owner_bnd_anchor_id, neighbour_bnd_anchor_id], axis=0
    )
    # [F0, C0, B0, F1, C1, B1, ...]
    points = np.empty((n_faces_has_bnd * 3, 3), dtype=np.float64)
    points[0::3] = x_far
    points[1::3] = x
    points[2::3] = x_bnd

    # VTK polyline connectivity: [3, p0, p1, p2] repeated for each polyline.
    lines = np.empty((n_faces_has_bnd, 4), dtype=np.int64)
    lines[:, 0] = 3
    base = (np.arange(n_faces_has_bnd, dtype=np.int64) * 3).reshape(-1, 1)
    lines[:, 1:] = base + np.array([0, 1, 2], dtype=np.int64)

    pd = pv.PolyData(points, lines=lines.ravel())
    pd.cell_data["side"] = sides
    pd.cell_data["weights"] = weights
    pd.cell_data["bnd_anchor_id"] = anchor_per_polyline
    pd.save(f"{output_prefix}_polylines_{axis}.vtp")
    print(f"Saved {axis} polylines to {output_prefix}_polylines_{axis}.vtp")


def export_apibm_debug_data(
    mesh: CfdAxisProjectedMesh, output_prefix: str = "apibm_debug"
) -> None:
    """
    Exports APIBM debug arrays and `N_faces` polylines:

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh object returned by `build_axis_projected_mesh`.
    output_prefix : str
        File prefix for debug outputs.
    """
    np.savez_compressed(
        f"{output_prefix}.npz",
        ap_x_has_bnd=np.asarray(mesh.ap_x_has_bnd),
        ap_x_dist_owner_to_bnd=np.asarray(mesh.ap_x_dist_owner_to_bnd),
        ap_x_dist_neighbour_to_bnd=np.asarray(mesh.ap_x_dist_neighbour_to_bnd),
        ap_x_owner_far_cell_id=np.asarray(mesh.ap_x_owner_far_cell_id),
        ap_x_neighbour_far_cell_id=np.asarray(mesh.ap_x_neighbour_far_cell_id),
        ap_x_owner_weights=np.asarray(mesh.ap_x_owner_weights),
        ap_x_neighbour_weights=np.asarray(mesh.ap_x_neighbour_weights),
        ap_x_owner_bnd_anchor_id=np.asarray(mesh.ap_x_owner_bnd_anchor_id),
        ap_x_neighbour_bnd_anchor_id=np.asarray(
            mesh.ap_x_neighbour_bnd_anchor_id
        ),
        ap_y_has_bnd=np.asarray(mesh.ap_y_has_bnd),
        ap_y_dist_owner_to_bnd=np.asarray(mesh.ap_y_dist_owner_to_bnd),
        ap_y_dist_neighbour_to_bnd=np.asarray(mesh.ap_y_dist_neighbour_to_bnd),
        ap_y_owner_far_cell_id=np.asarray(mesh.ap_y_owner_far_cell_id),
        ap_y_neighbour_far_cell_id=np.asarray(mesh.ap_y_neighbour_far_cell_id),
        ap_y_owner_weights=np.asarray(mesh.ap_y_owner_weights),
        ap_y_neighbour_weights=np.asarray(mesh.ap_y_neighbour_weights),
        ap_y_owner_bnd_anchor_id=np.asarray(mesh.ap_y_owner_bnd_anchor_id),
        ap_y_neighbour_bnd_anchor_id=np.asarray(
            mesh.ap_y_neighbour_bnd_anchor_id
        ),
        ap_z_has_bnd=np.asarray(mesh.ap_z_has_bnd),
        ap_z_dist_owner_to_bnd=np.asarray(mesh.ap_z_dist_owner_to_bnd),
        ap_z_dist_neighbour_to_bnd=np.asarray(mesh.ap_z_dist_neighbour_to_bnd),
        ap_z_owner_far_cell_id=np.asarray(mesh.ap_z_owner_far_cell_id),
        ap_z_neighbour_far_cell_id=np.asarray(mesh.ap_z_neighbour_far_cell_id),
        ap_z_owner_weights=np.asarray(mesh.ap_z_owner_weights),
        ap_z_neighbour_weights=np.asarray(mesh.ap_z_neighbour_weights),
        ap_z_owner_bnd_anchor_id=np.asarray(mesh.ap_z_owner_bnd_anchor_id),
        ap_z_neighbour_bnd_anchor_id=np.asarray(
            mesh.ap_z_neighbour_bnd_anchor_id
        ),
    )
    print(f"Saved debug arrays to {output_prefix}.npz")
    export_axis_projected_polylines(
        axis="x",
        cell_centers=mesh.cell_centers,
        owner=mesh.x_faces_owner,
        neighbour=mesh.x_faces_neighbour,
        has_bnd=mesh.ap_x_has_bnd,
        dist_owner_to_bnd=mesh.ap_x_dist_owner_to_bnd,
        dist_neighbour_to_bnd=mesh.ap_x_dist_neighbour_to_bnd,
        owner_far_cell_id=mesh.ap_x_owner_far_cell_id,
        neighbour_far_cell_id=mesh.ap_x_neighbour_far_cell_id,
        owner_weights=mesh.ap_x_owner_weights,
        neighbour_weights=mesh.ap_x_neighbour_weights,
        owner_bnd_anchor_id=mesh.ap_x_owner_bnd_anchor_id,
        neighbour_bnd_anchor_id=mesh.ap_x_neighbour_bnd_anchor_id,
        output_prefix=output_prefix,
    )
    export_axis_projected_polylines(
        axis="y",
        cell_centers=mesh.cell_centers,
        owner=mesh.y_faces_owner,
        neighbour=mesh.y_faces_neighbour,
        has_bnd=mesh.ap_y_has_bnd,
        dist_owner_to_bnd=mesh.ap_y_dist_owner_to_bnd,
        dist_neighbour_to_bnd=mesh.ap_y_dist_neighbour_to_bnd,
        owner_far_cell_id=mesh.ap_y_owner_far_cell_id,
        neighbour_far_cell_id=mesh.ap_y_neighbour_far_cell_id,
        owner_weights=mesh.ap_y_owner_weights,
        neighbour_weights=mesh.ap_y_neighbour_weights,
        owner_bnd_anchor_id=mesh.ap_y_owner_bnd_anchor_id,
        neighbour_bnd_anchor_id=mesh.ap_y_neighbour_bnd_anchor_id,
        output_prefix=output_prefix,
    )
    export_axis_projected_polylines(
        axis="z",
        cell_centers=mesh.cell_centers,
        owner=mesh.z_faces_owner,
        neighbour=mesh.z_faces_neighbour,
        has_bnd=mesh.ap_z_has_bnd,
        dist_owner_to_bnd=mesh.ap_z_dist_owner_to_bnd,
        dist_neighbour_to_bnd=mesh.ap_z_dist_neighbour_to_bnd,
        owner_far_cell_id=mesh.ap_z_owner_far_cell_id,
        neighbour_far_cell_id=mesh.ap_z_neighbour_far_cell_id,
        owner_weights=mesh.ap_z_owner_weights,
        neighbour_weights=mesh.ap_z_neighbour_weights,
        owner_bnd_anchor_id=mesh.ap_z_owner_bnd_anchor_id,
        neighbour_bnd_anchor_id=mesh.ap_z_neighbour_bnd_anchor_id,
        output_prefix=output_prefix,
    )


def export_apibm_mesh(
    mesh: CfdAxisProjectedMesh, output_path: pathlib.Path
) -> None:
    """
    Exports an APIBM mesh to a VTU file.
    Also exports debug arrays as a separate NPZ file.

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh object returned by `build_axis_projected_mesh`.
    output_path : pathlib.Path
        Path to the output VTU file.
    """
    # 1. セルメッシュの作成
    grid = mesh_to_unstructured_grid(mesh.cell_centers, mesh.cell_sizes)

    # セルデータの割り当て
    n_cells = len(mesh.cell_centers)
    x_bnd_type = np.zeros(n_cells, dtype=int)
    x_bnd_type[mesh.ap_x_owner_far_cell_id] = -2
    x_bnd_type[mesh.ap_x_neighbour_far_cell_id] = 2
    x_bnd_type[mesh.x_faces_owner[mesh.ap_x_has_bnd]] = -1
    x_bnd_type[mesh.x_faces_neighbour[mesh.ap_x_has_bnd]] = 1

    y_bnd_type = np.zeros(n_cells, dtype=int)
    y_bnd_type[mesh.ap_y_owner_far_cell_id] = -2
    y_bnd_type[mesh.ap_y_neighbour_far_cell_id] = 2
    y_bnd_type[mesh.y_faces_owner[mesh.ap_y_has_bnd]] = -1
    y_bnd_type[mesh.y_faces_neighbour[mesh.ap_y_has_bnd]] = 1
    z_bnd_type = np.zeros(n_cells, dtype=int)
    z_bnd_type[mesh.ap_z_owner_far_cell_id] = -2
    z_bnd_type[mesh.ap_z_neighbour_far_cell_id] = 2
    z_bnd_type[mesh.z_faces_owner[mesh.ap_z_has_bnd]] = -1
    z_bnd_type[mesh.z_faces_neighbour[mesh.ap_z_has_bnd]] = 1

    grid.cell_data["cell_sizes"] = mesh.cell_sizes
    grid.cell_data["cell_volume"] = np.prod(mesh.cell_sizes, axis=1)
    grid.cell_data["x_bnd_type"] = x_bnd_type
    grid.cell_data["y_bnd_type"] = y_bnd_type
    grid.cell_data["z_bnd_type"] = z_bnd_type

    # 保存
    grid.save(output_path)
    print(f"Saved APIBM mesh to {output_path}")

    # 2. APIBM の 可視化 (デバッグ用)
    parent_dir = output_path.parent
    output_prefix = output_path.stem
    export_apibm_debug_data(
        mesh, output_prefix=f"{parent_dir}/{output_prefix}_debug"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a APIBM mesh from an STL file and export it for ParaView."
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
    cfg = load_apibm_config(args.config)

    print("=== Fluxel One-Stop APIBM Mesh Generator ===")

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

    print(f"Config file: {args.config.resolve()}")
    print(f"Domain Bounds: Min {bbox.min} Max {bbox.max}")
    print(f"Base Resolution: {base_res}")
    print(f"Target Level: {target_level}")
    print(f"Number of Leaf Refinements: {n_leaf_refinement}")
    print(f"Input STL File: {stl_path}")
    print(f"Output VTU File: {output_path}")

    # 2. FluxelManager の初期化
    manager = FluxelManager(
        bbox, base_res=base_res, n_leaf_refinement=n_leaf_refinement
    )
    if not stl_path.exists():
        raise SystemExit(f"Input STL file does not exist: {stl_path}")

    # 3. メッシュの生成
    print("\n Starting mesh generation...")
    t0 = time.time()
    mesh = manager.build_axis_projected_mesh(
        str(stl_path), target_level=target_level
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
    print("====================")

    # 5. メッシュの保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_apibm_mesh(mesh, output_path)
