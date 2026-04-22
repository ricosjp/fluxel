//! GCIBM mesh construction from forest topology and ghost-cell IBM data.

use crate::mesh::CfdGhostCellMesh;
use fluxel_core::{Axis, CoordinateType, Direction, Forest};
use fluxel_geometry::Geometry;
use fluxel_ibm::mesh::IBMMesh;
use fluxel_ibm::types::GhostCellData;
use rayon::prelude::*;

/// Per-cell topology for [`build_ghost_cell_mesh`], merged in global cell-index order.
#[derive(Default)]
struct GhostCellPart {
    center: [f64; 3],
    size: [f64; 3],
    gc_is_fluid: bool,
    internal_faces_owner: Vec<usize>,
    internal_faces_neighbour: Vec<usize>,
    internal_faces_axis: Vec<Axis>,
    domain_bnd_faces_owner: Vec<usize>,
    domain_bnd_faces_dir: Vec<Direction>,
}

fn extract_ghost_cell_part(
    forest: &Forest,
    geom: &Geometry,
    global_id: usize,
    gc_is_fluid: &[bool],
) -> GhostCellPart {
    let key = forest.keys()[global_id];
    let logical = key.to_logical();
    let (center, size) = geom.cell_bounds(&logical);

    let mut part = GhostCellPart {
        center,
        size,
        gc_is_fluid: gc_is_fluid[global_id],
        ..Default::default()
    };

    for axis in Axis::ALL {
        let (dir_minus, dir_plus) = axis.split_into_directions();
        let minus_nbrs = forest.face_neighbour_global_ids(global_id, dir_minus);
        if minus_nbrs.is_empty() {
            part.domain_bnd_faces_owner.push(global_id);
            part.domain_bnd_faces_dir.push(dir_minus);
        }

        let plus_nbrs = forest.face_neighbour_global_ids(global_id, dir_plus);
        if plus_nbrs.is_empty() {
            part.domain_bnd_faces_owner.push(global_id);
            part.domain_bnd_faces_dir.push(dir_plus);
        } else {
            for &n_id in &plus_nbrs {
                part.internal_faces_owner.push(global_id);
                part.internal_faces_neighbour.push(n_id);
                part.internal_faces_axis.push(axis);
            }
        }
    }

    part
}

/// Forest トポロジーとゴーストセル IBM データから GCIBM 用メッシュを構築する。
///
/// セル単位で [`rayon`] により並列化する。結合順はセルインデックス昇順で、逐次版と同一の配列内容になる。
pub fn build_ghost_cell_mesh(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    gc_data: GhostCellData,
) -> CfdGhostCellMesh {
    let n_cells = forest.num_cells();

    let parts: Vec<GhostCellPart> = (0..n_cells)
        .into_par_iter()
        .map(|global_id| extract_ghost_cell_part(forest, geom, global_id, &gc_data.gc_is_fluid))
        .collect();

    let mut mesh = CfdGhostCellMesh {
        n_cells,
        coordinate_type: CoordinateType::Cartesian,
        patch_names: ibm_mesh.patch_names.clone(),
        ..Default::default()
    };

    mesh.cell_centers.reserve(n_cells);
    mesh.cell_sizes.reserve(n_cells);
    mesh.gc_is_fluid.reserve(n_cells);

    for p in parts {
        mesh.cell_centers.push(p.center);
        mesh.cell_sizes.push(p.size);
        mesh.gc_is_fluid.push(p.gc_is_fluid);
        mesh.internal_faces_owner.extend(p.internal_faces_owner);
        mesh.internal_faces_neighbour
            .extend(p.internal_faces_neighbour);
        mesh.internal_faces_axis.extend(p.internal_faces_axis);
        mesh.domain_bnd_faces_owner.extend(p.domain_bnd_faces_owner);
        mesh.domain_bnd_faces_dir.extend(p.domain_bnd_faces_dir);
    }

    mesh.gc_cell_ids = gc_data.gc_cell_ids;
    mesh.gc_bnd_anchor_ids = gc_data.gc_bnd_anchor_ids;
    mesh.gc_bnd_patch_ids = gc_data.gc_bnd_patch_ids;
    mesh.gc_bnd_intercepts = gc_data.gc_bnd_intercepts;
    mesh.gc_image_points = gc_data.gc_image_points;
    mesh.gc_interp_stencil_indices = gc_data.gc_interp_stencil_indices;
    mesh.gc_interp_stencil_weights = gc_data.gc_interp_stencil_weights;

    mesh
}
