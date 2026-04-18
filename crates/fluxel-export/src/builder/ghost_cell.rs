//! GCIBM mesh construction from forest topology and ghost-cell IBM data.

use crate::mesh::CfdGhostCellMesh;
use fluxel_core::neighbour::Direction;
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use fluxel_ibm::types::GhostCellData;
use rayon::prelude::*;

/// Per-cell payload for [`build_ghost_cell_mesh`] (parallel map, then merged in index order).
struct GhostCellPart {
    center: [f64; 3],
    size: [f64; 3],
    is_fluid: bool,
    x_minus_is_domain_bnd: bool,
    x_plus_is_domain_bnd: bool,
    x_plus_nbrs: Vec<usize>,
    y_minus_is_domain_bnd: bool,
    y_plus_is_domain_bnd: bool,
    y_plus_nbrs: Vec<usize>,
    z_minus_is_domain_bnd: bool,
    z_plus_is_domain_bnd: bool,
    z_plus_nbrs: Vec<usize>,
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

    let x_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::XMinus);
    let x_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::XPlus);
    let y_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::YMinus);
    let y_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::YPlus);
    let z_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::ZMinus);
    let z_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::ZPlus);

    GhostCellPart {
        center,
        size,
        is_fluid: gc_is_fluid[global_id],
        x_minus_is_domain_bnd: x_minus_nbrs.is_empty(),
        x_plus_is_domain_bnd: x_plus_nbrs.is_empty(),
        x_plus_nbrs,
        y_minus_is_domain_bnd: y_minus_nbrs.is_empty(),
        y_plus_is_domain_bnd: y_plus_nbrs.is_empty(),
        y_plus_nbrs,
        z_minus_is_domain_bnd: z_minus_nbrs.is_empty(),
        z_plus_is_domain_bnd: z_plus_nbrs.is_empty(),
        z_plus_nbrs,
    }
}

/// Builds a [`CfdGhostCellMesh`] from a [`Forest`], [`Geometry`], per-cell IBM classification,
/// and precomputed ghost-cell data from the IBM stage.
pub fn build_ghost_cell_mesh(
    forest: &Forest,
    geom: &Geometry,
    gc_data: GhostCellData,
) -> CfdGhostCellMesh {
    let n = forest.num_cells();
    let parts: Vec<GhostCellPart> = (0..n)
        .into_par_iter()
        .map(|global_id| extract_ghost_cell_part(forest, geom, global_id, &gc_data.gc_is_fluid))
        .collect();

    let mut mesh = CfdGhostCellMesh::default();

    for global_id in 0..n {
        let p = &parts[global_id];
        mesh.cell_centers.push(p.center);
        mesh.cell_sizes.push(p.size);
        mesh.gc_is_fluid.push(p.is_fluid);

        if p.x_minus_is_domain_bnd {
            mesh.x_bnd_minus_owner.push(global_id);
        }
        if p.x_plus_is_domain_bnd {
            mesh.x_bnd_plus_owner.push(global_id);
        } else {
            p.x_plus_nbrs.iter().copied().for_each(|nbr_id| {
                mesh.x_faces_owner.push(global_id);
                mesh.x_faces_neighbour.push(nbr_id);
            });
        }

        if p.y_minus_is_domain_bnd {
            mesh.y_bnd_minus_owner.push(global_id);
        }
        if p.y_plus_is_domain_bnd {
            mesh.y_bnd_plus_owner.push(global_id);
        } else {
            p.y_plus_nbrs.iter().copied().for_each(|nbr_id| {
                mesh.y_faces_owner.push(global_id);
                mesh.y_faces_neighbour.push(nbr_id);
            });
        }

        if p.z_minus_is_domain_bnd {
            mesh.z_bnd_minus_owner.push(global_id);
        }
        if p.z_plus_is_domain_bnd {
            mesh.z_bnd_plus_owner.push(global_id);
        } else {
            p.z_plus_nbrs.iter().copied().for_each(|nbr_id| {
                mesh.z_faces_owner.push(global_id);
                mesh.z_faces_neighbour.push(nbr_id);
            });
        }
    }

    mesh.gc_cell_ids = gc_data.gc_cell_ids;
    mesh.gc_bnd_anchor_ids = gc_data.gc_bnd_anchor_ids;
    mesh.gc_bnd_intercepts = gc_data.gc_bnd_intercepts;
    mesh.gc_image_points = gc_data.gc_image_points;
    mesh.gc_interp_stencil_indices = gc_data.gc_interp_stencil_indices;
    mesh.gc_interp_stencil_weights = gc_data.gc_interp_stencil_weights;

    mesh
}
