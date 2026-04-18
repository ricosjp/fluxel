//! APIBM (axis-projected) mesh construction.

use crate::mesh::CfdAxisProjectedMesh;
use fluxel_core::neighbour::Direction;
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use fluxel_ibm::solver::{resolve_apibm_face, ApibmIntersection};
use fluxel_ibm::types::CellType;
use fluxel_ibm::IBMMesh;
use rayon::prelude::*;

/// Per-cell payload for [`build_axis_projected_mesh`].
struct AxisProjectedPart {
    center: [f64; 3],
    size: [f64; 3],
    x_minus_is_domain_bnd: bool,
    x_plus_is_domain_bnd: bool,
    x_faces: Vec<(usize, Option<ApibmIntersection>)>,
    y_minus_is_domain_bnd: bool,
    y_plus_is_domain_bnd: bool,
    y_faces: Vec<(usize, Option<ApibmIntersection>)>,
    z_minus_is_domain_bnd: bool,
    z_plus_is_domain_bnd: bool,
    z_faces: Vec<(usize, Option<ApibmIntersection>)>,
}

fn extract_axis_projected_part(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
    global_id: usize,
) -> AxisProjectedPart {
    let key = forest.keys()[global_id];
    let logical = key.to_logical();
    let (center, size) = geom.cell_bounds(&logical);

    let x_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::XMinus);
    let x_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::XPlus);
    let y_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::YMinus);
    let y_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::YPlus);
    let z_minus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::ZMinus);
    let z_plus_nbrs = forest.face_neighbour_global_ids(global_id, Direction::ZPlus);

    let x_faces = if x_plus_nbrs.is_empty() {
        Vec::new()
    } else {
        x_plus_nbrs
            .iter()
            .map(|&nbr_id| {
                let needs_raycast = cell_types[global_id] == CellType::Intersect
                    || cell_types[nbr_id] == CellType::Intersect;
                let ap_res = if needs_raycast {
                    resolve_apibm_face(forest, geom, ibm_mesh, global_id, nbr_id, 0)
                } else {
                    None
                };
                (nbr_id, ap_res)
            })
            .collect()
    };

    let y_faces = if y_plus_nbrs.is_empty() {
        Vec::new()
    } else {
        y_plus_nbrs
            .iter()
            .map(|&nbr_id| {
                let needs_raycast = cell_types[global_id] == CellType::Intersect
                    || cell_types[nbr_id] == CellType::Intersect;
                let ap_res = if needs_raycast {
                    resolve_apibm_face(forest, geom, ibm_mesh, global_id, nbr_id, 1)
                } else {
                    None
                };
                (nbr_id, ap_res)
            })
            .collect()
    };

    let z_faces = if z_plus_nbrs.is_empty() {
        Vec::new()
    } else {
        z_plus_nbrs
            .iter()
            .map(|&nbr_id| {
                let needs_raycast = cell_types[global_id] == CellType::Intersect
                    || cell_types[nbr_id] == CellType::Intersect;
                let ap_res = if needs_raycast {
                    resolve_apibm_face(forest, geom, ibm_mesh, global_id, nbr_id, 2)
                } else {
                    None
                };
                (nbr_id, ap_res)
            })
            .collect()
    };

    AxisProjectedPart {
        center,
        size,
        x_minus_is_domain_bnd: x_minus_nbrs.is_empty(),
        x_plus_is_domain_bnd: x_plus_nbrs.is_empty(),
        x_faces,
        y_minus_is_domain_bnd: y_minus_nbrs.is_empty(),
        y_plus_is_domain_bnd: y_plus_nbrs.is_empty(),
        y_faces,
        z_minus_is_domain_bnd: z_minus_nbrs.is_empty(),
        z_plus_is_domain_bnd: z_plus_nbrs.is_empty(),
        z_faces,
    }
}

/// Builds a [`CfdAxisProjectedMesh`] from a [`Forest`], [`Geometry`], and precomputed axis-projected
/// data from the IBM stage.
pub fn build_axis_projected_mesh(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
) -> CfdAxisProjectedMesh {
    let n = forest.num_cells();
    let parts: Vec<AxisProjectedPart> = (0..n)
        .into_par_iter()
        .map(|global_id| extract_axis_projected_part(forest, geom, ibm_mesh, cell_types, global_id))
        .collect();

    let mut mesh = CfdAxisProjectedMesh::default();

    for global_id in 0..n {
        let p = &parts[global_id];
        mesh.cell_centers.push(p.center);
        mesh.cell_sizes.push(p.size);

        if p.x_minus_is_domain_bnd {
            mesh.x_bnd_minus_owner.push(global_id);
        }
        if p.x_plus_is_domain_bnd {
            mesh.x_bnd_plus_owner.push(global_id);
        } else {
            p.x_faces.iter().for_each(|&(nbr_id, ref ap_res)| {
                mesh.x_faces_owner.push(global_id);
                mesh.x_faces_neighbour.push(nbr_id);
                if let Some(ap) = ap_res {
                    mesh.ap_x_has_bnd.push(true);
                    mesh.ap_x_dist_owner_to_bnd.push(ap.dist_owner_to_bnd);
                    mesh.ap_x_dist_neighbour_to_bnd
                        .push(ap.dist_neighbour_to_bnd);
                    mesh.ap_x_owner_far_cell_id.push(ap.owner_far_cell_id);
                    mesh.ap_x_neighbour_far_cell_id
                        .push(ap.neighbour_far_cell_id);
                    mesh.ap_x_owner_weights.push(ap.owner_weights);
                    mesh.ap_x_neighbour_weights.push(ap.neighbour_weights);
                    mesh.ap_x_owner_bnd_anchor_id.push(ap.owner_bnd_anchor_id);
                    mesh.ap_x_neighbour_bnd_anchor_id
                        .push(ap.neighbour_bnd_anchor_id);
                } else {
                    mesh.ap_x_has_bnd.push(false);
                }
            });
        }

        if p.y_minus_is_domain_bnd {
            mesh.y_bnd_minus_owner.push(global_id);
        }
        if p.y_plus_is_domain_bnd {
            mesh.y_bnd_plus_owner.push(global_id);
        } else {
            p.y_faces.iter().for_each(|&(nbr_id, ref ap_res)| {
                mesh.y_faces_owner.push(global_id);
                mesh.y_faces_neighbour.push(nbr_id);
                if let Some(ap) = ap_res {
                    mesh.ap_y_has_bnd.push(true);
                    mesh.ap_y_dist_owner_to_bnd.push(ap.dist_owner_to_bnd);
                    mesh.ap_y_dist_neighbour_to_bnd
                        .push(ap.dist_neighbour_to_bnd);
                    mesh.ap_y_owner_far_cell_id.push(ap.owner_far_cell_id);
                    mesh.ap_y_neighbour_far_cell_id
                        .push(ap.neighbour_far_cell_id);
                    mesh.ap_y_owner_weights.push(ap.owner_weights);
                    mesh.ap_y_neighbour_weights.push(ap.neighbour_weights);
                    mesh.ap_y_owner_bnd_anchor_id.push(ap.owner_bnd_anchor_id);
                    mesh.ap_y_neighbour_bnd_anchor_id
                        .push(ap.neighbour_bnd_anchor_id);
                } else {
                    mesh.ap_y_has_bnd.push(false);
                }
            });
        }

        if p.z_minus_is_domain_bnd {
            mesh.z_bnd_minus_owner.push(global_id);
        }
        if p.z_plus_is_domain_bnd {
            mesh.z_bnd_plus_owner.push(global_id);
        } else {
            p.z_faces.iter().for_each(|&(nbr_id, ref ap_res)| {
                mesh.z_faces_owner.push(global_id);
                mesh.z_faces_neighbour.push(nbr_id);
                if let Some(ap) = ap_res {
                    mesh.ap_z_has_bnd.push(true);
                    mesh.ap_z_dist_owner_to_bnd.push(ap.dist_owner_to_bnd);
                    mesh.ap_z_dist_neighbour_to_bnd
                        .push(ap.dist_neighbour_to_bnd);
                    mesh.ap_z_owner_far_cell_id.push(ap.owner_far_cell_id);
                    mesh.ap_z_neighbour_far_cell_id
                        .push(ap.neighbour_far_cell_id);
                    mesh.ap_z_owner_weights.push(ap.owner_weights);
                    mesh.ap_z_neighbour_weights.push(ap.neighbour_weights);
                    mesh.ap_z_owner_bnd_anchor_id.push(ap.owner_bnd_anchor_id);
                    mesh.ap_z_neighbour_bnd_anchor_id
                        .push(ap.neighbour_bnd_anchor_id);
                } else {
                    mesh.ap_z_has_bnd.push(false);
                }
            });
        }
    }

    mesh
}
