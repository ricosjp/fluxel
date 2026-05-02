//! APIBM (axis-projected) mesh construction.

use crate::mesh::CfdAxisProjectedMesh;
use fluxel_core::{Axis, CoordinateType, Direction, Forest};
use fluxel_geometry::Geometry;
use fluxel_ibm::solver::resolve_apibm_face;
use fluxel_ibm::types::CellType;
use fluxel_ibm::IBMMesh;
use rayon::prelude::*;

/// Per-cell payload for [`build_axis_projected_mesh`].
#[derive(Default)]
struct AxisProjectedPart {
    center: [f64; 3],
    size: [f64; 3],
    internal_faces_owner: Vec<usize>,
    internal_faces_neighbour: Vec<usize>,
    internal_faces_axis: Vec<Axis>,
    domain_bnd_faces_owner: Vec<usize>,
    domain_bnd_faces_dir: Vec<Direction>,

    ap_is_immersed_face: Vec<bool>,
    ap_dist_owner_to_bnd: Vec<f64>,
    ap_dist_neighbour_to_bnd: Vec<f64>,
    ap_owner_weights: Vec<[f64; 2]>,
    ap_neighbour_weights: Vec<[f64; 2]>,
    ap_owner_bnd_anchor_id: Vec<usize>,
    ap_owner_bnd_patch_id: Vec<usize>,
    ap_neighbour_bnd_anchor_id: Vec<usize>,
    ap_neighbour_bnd_patch_id: Vec<usize>,
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

    let mut part = AxisProjectedPart {
        center,
        size,
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
            for &nbr_id in &plus_nbrs {
                part.internal_faces_owner.push(global_id);
                part.internal_faces_neighbour.push(nbr_id);
                part.internal_faces_axis.push(axis);

                let needs_raycast = cell_types[global_id] == CellType::Intersect
                    || cell_types[nbr_id] == CellType::Intersect;
                let ap_res = if needs_raycast {
                    resolve_apibm_face(forest, geom, ibm_mesh, global_id, nbr_id, axis)
                } else {
                    None
                };

                if let Some(ap) = ap_res {
                    part.ap_is_immersed_face.push(true);
                    part.ap_dist_owner_to_bnd.push(ap.dist_owner_to_bnd);
                    part.ap_dist_neighbour_to_bnd.push(ap.dist_neighbour_to_bnd);
                    part.ap_owner_weights.push(ap.owner_weights);
                    part.ap_neighbour_weights.push(ap.neighbour_weights);
                    part.ap_owner_bnd_anchor_id.push(ap.owner_bnd_anchor_id);
                    part.ap_owner_bnd_patch_id.push(ap.owner_bnd_patch_id);
                    part.ap_neighbour_bnd_anchor_id
                        .push(ap.neighbour_bnd_anchor_id);
                    part.ap_neighbour_bnd_patch_id
                        .push(ap.neighbour_bnd_patch_id);
                } else {
                    part.ap_is_immersed_face.push(false);
                }
            }
        }
    }

    part
}

/// Builds a [`CfdAxisProjectedMesh`] from a [`Forest`], [`Geometry`], and precomputed axis-projected
/// data from the IBM stage.
pub fn build_axis_projected_mesh(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
) -> CfdAxisProjectedMesh {
    let n_cells = forest.num_cells();

    let parts: Vec<AxisProjectedPart> = (0..n_cells)
        .into_par_iter()
        .map(|global_id| extract_axis_projected_part(forest, geom, ibm_mesh, cell_types, global_id))
        .collect();

    let mut mesh = CfdAxisProjectedMesh {
        n_cells,
        coordinate_type: CoordinateType::Cartesian,
        patch_names: ibm_mesh.patch_names.clone(),
        ..Default::default()
    };

    mesh.cell_centers.reserve(n_cells);
    mesh.cell_sizes.reserve(n_cells);
    for p in parts {
        mesh.cell_centers.push(p.center);
        mesh.cell_sizes.push(p.size);

        mesh.internal_faces_owner.extend(p.internal_faces_owner);
        mesh.internal_faces_neighbour
            .extend(p.internal_faces_neighbour);
        mesh.internal_faces_axis.extend(p.internal_faces_axis);
        mesh.domain_bnd_faces_owner.extend(p.domain_bnd_faces_owner);
        mesh.domain_bnd_faces_dir.extend(p.domain_bnd_faces_dir);
        mesh.ap_is_immersed_face.extend(p.ap_is_immersed_face);
        mesh.ap_dist_owner_to_bnd.extend(p.ap_dist_owner_to_bnd);
        mesh.ap_dist_neighbour_to_bnd
            .extend(p.ap_dist_neighbour_to_bnd);
        mesh.ap_owner_weights.extend(p.ap_owner_weights);
        mesh.ap_neighbour_weights.extend(p.ap_neighbour_weights);
        mesh.ap_owner_bnd_anchor_id.extend(p.ap_owner_bnd_anchor_id);
        mesh.ap_owner_bnd_patch_id.extend(p.ap_owner_bnd_patch_id);
        mesh.ap_neighbour_bnd_anchor_id
            .extend(p.ap_neighbour_bnd_anchor_id);
        mesh.ap_neighbour_bnd_patch_id
            .extend(p.ap_neighbour_bnd_patch_id);
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_geometry::BoundingBox;
    use fluxel_ibm::mesh::IBMMesh;

    fn two_cell_setup() -> (Forest, Geometry, IBMMesh) {
        let mut forest = Forest::new([2, 1, 1]);
        forest.populate_root_cells();

        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);

        let verts = [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]];
        let indices = [[0u32, 1, 2]];
        let mesh = IBMMesh::from_vertices_indices_and_patches(
            &verts,
            &indices,
            vec!["default".into()],
            vec![0],
        );
        (forest, geom, mesh)
    }

    #[test]
    fn axis_projected_all_fluid_sets_only_immersed_flags() {
        let (forest, geom, ibm_mesh) = two_cell_setup();
        let cell_types = vec![CellType::Fluid, CellType::Fluid];

        let mesh = build_axis_projected_mesh(&forest, &geom, &ibm_mesh, &cell_types);

        assert_eq!(mesh.n_cells, 2);
        assert!(mesh.ap_dist_owner_to_bnd.is_empty());
        assert_eq!(mesh.ap_is_immersed_face.iter().filter(|&&x| x).count(), 0);
        assert!(!mesh.internal_faces_owner.is_empty());
    }
}
