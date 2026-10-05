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
from pydantic import BaseModel, Field, field_validator, model_validator

from fluxel import (
    Axis,
    BoundingBox,
    CfdAxisProjectedMesh,
    CoordinateType,
    Cylindrical,
    Direction,
    Float3,
    FluxelManager,
    Int3,
)


class BoundingBoxConfig(BaseModel, frozen=True):
    """Axis-aligned domain bounding box (min / max corners)."""

    min: Float3
    max: Float3


class RefinementRegionConfig(BaseModel, frozen=True):
    name: str | None = None
    """
    name : str | None
        Optional label for this local refinement region.
    """
    min: Float3
    """
    min : Float3
        Lower corner of the axis-aligned refinement box.
    """
    max: Float3
    """
    max : Float3
        Upper corner of the axis-aligned refinement box.
    """
    level: int
    """
    level : int
        Target octree refinement level for cells intersecting this box.
    """

    @field_validator("level")
    @classmethod
    def validate_level(cls, value: int) -> int:
        if value < 0:
            raise ValueError("refinement region level must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "RefinementRegionConfig":
        if (
            self.min[0] >= self.max[0]
            or self.min[2] >= self.max[2]
            or self.min[1] == self.max[1]
        ):
            raise ValueError(
                "refinement region requires ordered first and third components "
                "and a nonzero middle span"
            )
        return self


class CylindricalDomainConfig(BaseModel, frozen=True):
    """Annular sector or full turn about world Z. Angles are radians."""

    origin: Float3
    r_min: float
    r_max: float
    theta_start: float
    theta_extent: float
    z_min: float
    z_max: float

class ApibmExportConfig(BaseModel, frozen=True):
    """Validated APIBM export settings loaded from YAML."""

    input_mesh: pathlib.Path
    output_vtu: pathlib.Path = Field(
        default_factory=lambda: pathlib.Path("apibm_result.vtu")
    )
    target_level: int = 3
    n_leaf_refinement: int = 3
    max_cells: int | None = Field(default=None, gt=0, strict=True)
    base_resolution: Int3
    bounding_box: BoundingBoxConfig | None = None
    cylindrical: CylindricalDomainConfig | None = None
    refinement_regions: list[RefinementRegionConfig] = Field(
        default_factory=list
    )

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

    @model_validator(mode="after")
    def _one_domain(self) -> "ApibmExportConfig":
        if (self.bounding_box is None) == (self.cylindrical is None):
            raise ValueError("specify exactly one of bounding_box or cylindrical")
        return self


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


def _basis_path(
    start: np.ndarray, end: np.ndarray, tangent: np.ndarray, curved: bool
) -> np.ndarray:
    """
    Sample the search path from a cell sample to its intersection.

    Straight axes are the segment along ``e_r``, ``e_z``, or a world axis.
    A cylindrical θ path is the circular arc whose length is ``r|Δθ|``.

    Parameters
    ----------
    start, end : numpy.ndarray
        World sample and world intersection, shape (3,).
    tangent : numpy.ndarray
        Unit path tangent at the intersection, shape (3,).
    curved : bool
        True for a cylindrical θ arc.

    Returns
    -------
    numpy.ndarray
        Path samples of shape (n, 3). The first point is ``start``
        and the last point is ``end``.
    """
    if not curved:
        return np.vstack([start, end])
    delta = start[:2] - end[:2]
    chord2 = float(delta @ delta)
    # tangent = sign * e_θ, so this perpendicular is sign * e_r.
    radial = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    denom = 2.0 * float(delta @ radial)
    if chord2 < 1e-24 or abs(denom) < 1e-14:
        return np.vstack([start, end])
    signed_radius = -chord2 / denom
    center = end[:2] - radial * signed_radius
    radius = abs(signed_radius)
    start_angle = np.atan2(start[1] - center[1], start[0] - center[0])
    end_angle = np.atan2(end[1] - center[1], end[0] - center[0])
    delta_angle = end_angle - start_angle
    sweep = np.atan2(np.sin(delta_angle), np.cos(delta_angle))
    count = max(2, int(np.ceil(abs(sweep) / (np.pi / 32.0))) + 1)
    angles = start_angle + sweep * np.linspace(0.0, 1.0, count)
    path = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(count, start[2]),
        ]
    )
    path[0] = start
    path[-1] = end
    return path


def export_axis_projected_polylines(
    axis: Literal["x", "y", "z"],
    cell_centers: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    is_immersed_face: np.ndarray,
    owner_bnd_point: np.ndarray,
    neighbour_bnd_point: np.ndarray,
    owner_bnd_tangent: np.ndarray,
    neighbour_bnd_tangent: np.ndarray,
    owner_near_boundary: np.ndarray,
    neighbour_near_boundary: np.ndarray,
    owner_bnd_anchor_id: np.ndarray,
    neighbour_bnd_anchor_id: np.ndarray,
    owner_bnd_patch_id: np.ndarray,
    neighbour_bnd_patch_id: np.ndarray,
    coordinate_type: int,
    output_prefix: str,
) -> None:
    """
    Export one polyline per side from the cell sample to the intersection.

    Cartesian segments follow world X, Y, or Z. Cylindrical r and z segments
    follow ``e_r`` and ``e_z``. Cylindrical θ is the arc at the caster radius.
    Owner polylines precede neighbour polylines. A hit through the sample has
    zero length. ``near_boundary`` marks a cell-center Dirichlet candidate.

    Parameters
    ----------
    axis : {"x", "y", "z"}
        Logical face axis. On a cylinder these are r, θ, and z.
    cell_centers : numpy.ndarray
        World sample points, shape (n_cells, 3).
    owner, neighbour : numpy.ndarray
        Cell indices of the faces on this axis.
    is_immersed_face : numpy.ndarray
        Mask of faces on this axis whose both sides hit.
    owner_bnd_point, neighbour_bnd_point : numpy.ndarray
        World intersections, shape (n_immersed, 3).
    owner_bnd_tangent, neighbour_bnd_tangent : numpy.ndarray
        Unit path tangents at those intersections.
    owner_near_boundary, neighbour_near_boundary : numpy.ndarray
        Gibou near-boundary flags.
    owner_bnd_anchor_id, neighbour_bnd_anchor_id : numpy.ndarray
        Hit triangle ids.
    owner_bnd_patch_id, neighbour_bnd_patch_id : numpy.ndarray
        Hit patch ids.
    coordinate_type : int
        0 for Cartesian and 1 for cylindrical.
    output_prefix : str
        File prefix for the VTP.
    """
    n_hit = int(np.sum(is_immersed_face))
    if n_hit == 0:
        print(f"Skip {axis} polylines (no immersed faces on this axis)")
        return
    if n_hit != len(owner_bnd_point) or n_hit != len(neighbour_bnd_point):
        raise ValueError(
            "boundary points and is_immersed_face must have consistent lengths"
        )
    curved = coordinate_type == CoordinateType.Cylindrical and axis == "y"
    starts = np.concatenate(
        [
            cell_centers[owner[is_immersed_face]],
            cell_centers[neighbour[is_immersed_face]],
        ],
        axis=0,
    )
    ends = np.concatenate([owner_bnd_point, neighbour_bnd_point], axis=0)
    tangents = np.concatenate(
        [owner_bnd_tangent, neighbour_bnd_tangent], axis=0
    )
    paths = [
        _basis_path(start, end, tangent, curved)
        for start, end, tangent in zip(starts, ends, tangents, strict=True)
    ]
    points = np.vstack(paths)
    connectivity = []
    offset = 0
    for path in paths:
        count = len(path)
        connectivity.append(count)
        connectivity.extend(range(offset, offset + count))
        offset += count
    n_owner = int(np.sum(is_immersed_face))
    sides = np.concatenate([np.zeros(n_owner), np.ones(n_owner)])
    near_boundary = np.concatenate(
        [owner_near_boundary, neighbour_near_boundary]
    )
    anchor_per_polyline = np.concatenate(
        [owner_bnd_anchor_id, neighbour_bnd_anchor_id]
    )
    patch_per_polyline = np.concatenate(
        [owner_bnd_patch_id, neighbour_bnd_patch_id]
    )
    pd = pv.PolyData(points, lines=np.asarray(connectivity, dtype=np.int64))
    pd.cell_data["side"] = sides
    pd.cell_data["near_boundary"] = near_boundary
    pd.cell_data["bnd_anchor_id"] = anchor_per_polyline
    pd.cell_data["bnd_patch_id"] = patch_per_polyline
    pd.save(f"{output_prefix}_polylines_{axis}.vtp")
    print(f"Saved {axis} polylines to {output_prefix}_polylines_{axis}.vtp")


def _axis_internal_mesh_slice(
    mesh: CfdAxisProjectedMesh, axis: Axis
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return per-axis interior-face arrays (same order as the Rust builder)."""
    mask_for_n_faces = mesh.internal_faces_axis == axis.value
    mask_for_n_immersed_faces = (
        mesh.internal_faces_axis[mesh.ap.is_immersed_face] == axis.value
    )
    return (
        mesh.internal_faces_owner[mask_for_n_faces],
        mesh.internal_faces_neighbour[mask_for_n_faces],
        mesh.ap.is_immersed_face[mask_for_n_faces],
        mesh.ap.owner_bnd_point[mask_for_n_immersed_faces],
        mesh.ap.neighbour_bnd_point[mask_for_n_immersed_faces],
        mesh.ap.owner_bnd_tangent[mask_for_n_immersed_faces],
        mesh.ap.neighbour_bnd_tangent[mask_for_n_immersed_faces],
        mesh.ap.owner_near_boundary[mask_for_n_immersed_faces],
        mesh.ap.neighbour_near_boundary[mask_for_n_immersed_faces],
        mesh.ap.owner_bnd_anchor_id[mask_for_n_immersed_faces],
        mesh.ap.neighbour_bnd_anchor_id[mask_for_n_immersed_faces],
        mesh.ap.owner_bnd_patch_id[mask_for_n_immersed_faces],
        mesh.ap.neighbour_bnd_patch_id[mask_for_n_immersed_faces],
    )


def export_apibm_debug_data(
    mesh: CfdAxisProjectedMesh, output_prefix: str = "apibm_debug"
) -> None:
    """
    Export APIBM debug arrays and two polylines per immersed face.

    The NPZ stores physical distances alongside boolean constraint candidates
    named ``ap_owner_near_boundary`` and ``ap_neighbour_near_boundary``.
    Each axis's VTP follows that side's search path to the world intersection.

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh object returned by `build_axis_projected_mesh`.
    output_prefix : str
        File prefix for debug outputs.
    """
    np.savez_compressed(
        f"{output_prefix}.npz",
        internal_faces_owner=mesh.internal_faces_owner,
        internal_faces_neighbour=mesh.internal_faces_neighbour,
        internal_faces_axis=mesh.internal_faces_axis,
        domain_bnd_faces_owner=mesh.domain_bnd_faces_owner,
        domain_bnd_faces_dir=mesh.domain_bnd_faces_dir,
        ap_is_immersed_face=mesh.ap.is_immersed_face,
        ap_dist_owner_to_bnd=mesh.ap.dist_owner_to_bnd,
        ap_dist_neighbour_to_bnd=mesh.ap.dist_neighbour_to_bnd,
        ap_owner_near_boundary=mesh.ap.owner_near_boundary,
        ap_neighbour_near_boundary=mesh.ap.neighbour_near_boundary,
        ap_owner_bnd_anchor_id=mesh.ap.owner_bnd_anchor_id,
        ap_neighbour_bnd_anchor_id=mesh.ap.neighbour_bnd_anchor_id,
        ap_owner_bnd_patch_id=mesh.ap.owner_bnd_patch_id,
        ap_neighbour_bnd_patch_id=mesh.ap.neighbour_bnd_patch_id,
    )
    print(f"Saved debug arrays to {output_prefix}.npz")
    for axis_name, axis in (("x", Axis.X), ("y", Axis.Y), ("z", Axis.Z)):
        (
            owner,
            neighbour,
            is_immersed_face,
            owner_point,
            neighbour_point,
            owner_tangent,
            neighbour_tangent,
            owner_near_boundary,
            neighbour_near_boundary,
            oba,
            nba,
            obp,
            nbp,
        ) = _axis_internal_mesh_slice(mesh, axis)
        export_axis_projected_polylines(
            axis=axis_name,
            cell_centers=mesh.cell_centers,
            owner=owner,
            neighbour=neighbour,
            is_immersed_face=is_immersed_face,
            owner_bnd_point=owner_point,
            neighbour_bnd_point=neighbour_point,
            owner_bnd_tangent=owner_tangent,
            neighbour_bnd_tangent=neighbour_tangent,
            owner_near_boundary=owner_near_boundary,
            neighbour_near_boundary=neighbour_near_boundary,
            owner_bnd_anchor_id=oba,
            neighbour_bnd_anchor_id=nba,
            owner_bnd_patch_id=obp,
            neighbour_bnd_patch_id=nbp,
            coordinate_type=mesh.coordinate_type,
            output_prefix=output_prefix,
        )


def bnd_type_for_axis(mesh: CfdAxisProjectedMesh, axis: Axis) -> np.ndarray:
    """
    Mark cells that own or neighbour an immersed face on ``axis``.

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh whose immersed faces are classified.
    axis : Axis
        Face-normal axis to inspect.

    Returns
    -------
    numpy.ndarray
        Integer array of length ``n_cells``. ``-1`` marks owner cells of
        immersed faces, ``1`` marks neighbour cells, and ``0`` is unmarked.
    """
    mask_for_n_faces = (
        mesh.internal_faces_axis == axis.value
    ) & mesh.ap.is_immersed_face
    bt = np.zeros(mesh.n_cells, dtype=int)
    ow_m = mesh.internal_faces_owner[mask_for_n_faces]
    nb_m = mesh.internal_faces_neighbour[mask_for_n_faces]
    bt[ow_m] = -1
    bt[nb_m] = 1
    return bt


def export_apibm_mesh(
    mesh: CfdAxisProjectedMesh, output_path: pathlib.Path
) -> None:
    """
    Exports an APIBM mesh to a VTU file.
    Also exports debug arrays as a separate NPZ file and per-axis VTP files
    containing boundary-intersection segments and near-boundary flags.

    Parameters
    ----------
    mesh : CfdAxisProjectedMesh
        Mesh object returned by `build_axis_projected_mesh`.
    output_path : pathlib.Path
        Path to the output VTU file.
    """
    # 1. セルメッシュの作成
    grid = mesh_to_unstructured_grid(mesh.cell_corners)

    # セルデータの割り当て
    x_bnd_type = bnd_type_for_axis(mesh, Axis.X)
    y_bnd_type = bnd_type_for_axis(mesh, Axis.Y)
    z_bnd_type = bnd_type_for_axis(mesh, Axis.Z)

    grid.cell_data["cell_sizes"] = mesh.cell_sizes
    grid.cell_data["cell_volume"] = mesh.cell_volumes
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
            "Generate a APIBM mesh from STL / OBJ and export it as VTU."
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
    base_res = cfg.base_resolution
    target_level = cfg.target_level
    n_leaf_refinement = cfg.n_leaf_refinement
    mesh_path = cfg.input_mesh
    output_path = cfg.output_vtu
    refinement_regions = [
        (region.min, region.max, region.level)
        for region in cfg.refinement_regions
    ]

    print(f"Config file: {args.config.resolve()}")
    if cfg.cylindrical is None:
        assert cfg.bounding_box is not None
        print(
            f"Domain Bounds: Min {cfg.bounding_box.min} Max {cfg.bounding_box.max}"
        )
    else:
        print(
            "Cylindrical domain: "
            f"r [{cfg.cylindrical.r_min}, {cfg.cylindrical.r_max}], "
            f"theta [{cfg.cylindrical.theta_start}, {cfg.cylindrical.theta_extent}]"
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
    print(f"Refinement regions: {refinement_regions}")

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
    mesh = manager.build_axis_projected_mesh(
        str(mesh_path), target_level, refinement_regions
    )
    t1 = time.time()
    print(f"Mesh generation completed in {t1 - t0:.2f} seconds")

    # 4. メッシュサマリーの出力
    if mesh.n_cells != len(mesh.cell_centers):
        raise ValueError("n_cells does not match the number of cell centers")
    num_cells = mesh.n_cells
    ax = mesh.internal_faces_axis
    num_x_faces = int(np.count_nonzero(ax == Axis.X))
    num_y_faces = int(np.count_nonzero(ax == Axis.Y))
    num_z_faces = int(np.count_nonzero(ax == Axis.Z))
    bnd_dir = mesh.domain_bnd_faces_dir
    num_x_bnd_minus = int(np.count_nonzero(bnd_dir == Direction.XMinus))
    num_x_bnd_plus = int(np.count_nonzero(bnd_dir == Direction.XPlus))
    num_y_bnd_minus = int(np.count_nonzero(bnd_dir == Direction.YMinus))
    num_y_bnd_plus = int(np.count_nonzero(bnd_dir == Direction.YPlus))
    num_z_bnd_minus = int(np.count_nonzero(bnd_dir == Direction.ZMinus))
    num_z_bnd_plus = int(np.count_nonzero(bnd_dir == Direction.ZPlus))
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
    print(f"Patch Names: {list(mesh.patch_name_to_id.keys())}")
    print(f"Patch IDs: {list(mesh.patch_name_to_id.values())}")
    print("====================")

    # 5. メッシュの保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_apibm_mesh(mesh, output_path)
