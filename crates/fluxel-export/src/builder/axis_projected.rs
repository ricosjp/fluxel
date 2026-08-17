//! APIBM (axis-projected) mesh construction.

use crate::mesh::{ApIbmFaceData, CfdAxisProjectedMesh};
use fluxel_core::{Axis, CoordinateType, Direction, Forest};
use fluxel_geometry::Geometry;
use fluxel_ibm::solver::{mark_intersecting_cells, resolve_apibm_face};
use fluxel_ibm::types::CellType;
use fluxel_ibm::IBMMesh;
use parry3d_f64::math::Pose;
use rayon::prelude::*;

/// Per-cell topology payload for [`build_axis_projected_mesh`].
#[derive(Default)]
struct AxisProjectedPart {
    center: [f64; 3],
    size: [f64; 3],
    internal_faces_owner: Vec<usize>,
    internal_faces_neighbour: Vec<usize>,
    internal_faces_axis: Vec<Axis>,
    domain_bnd_faces_owner: Vec<usize>,
    domain_bnd_faces_dir: Vec<Direction>,
    ap: ApIbmFaceData,
}

/// Appends one interior-face APIBM record.
///
/// Ray-casts along `axis` when either adjacent cell is [`CellType::Intersect`];
/// otherwise records a non-immersed face.
#[allow(clippy::too_many_arguments)]
fn push_face_ap(
    ap: &mut ApIbmFaceData,
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
    owner: usize,
    neighbour: usize,
    axis: Axis,
    pose: &Pose,
) {
    let needs_raycast =
        cell_types[owner] == CellType::Intersect || cell_types[neighbour] == CellType::Intersect;
    let ap_res = if needs_raycast {
        resolve_apibm_face(forest, geom, ibm_mesh, owner, neighbour, axis, pose)
    } else {
        None
    };

    if let Some(hit) = ap_res {
        ap.is_immersed_face.push(true);
        ap.dist_owner_to_bnd.push(hit.dist_owner_to_bnd);
        ap.dist_neighbour_to_bnd.push(hit.dist_neighbour_to_bnd);
        ap.owner_weights.push(hit.owner_weights);
        ap.neighbour_weights.push(hit.neighbour_weights);
        ap.owner_bnd_anchor_id.push(hit.owner_bnd_anchor_id);
        ap.owner_bnd_patch_id.push(hit.owner_bnd_patch_id);
        ap.neighbour_bnd_anchor_id.push(hit.neighbour_bnd_anchor_id);
        ap.neighbour_bnd_patch_id.push(hit.neighbour_bnd_patch_id);
    } else {
        ap.is_immersed_face.push(false);
    }
}

/// Collects interior / domain faces and APIBM data for one background cell.
///
/// Interior faces are emitted only on the plus side of each axis so that each
/// shared face is recorded once.
fn extract_axis_projected_part(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
    global_id: usize,
    pose: &Pose,
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

                push_face_ap(
                    &mut part.ap,
                    forest,
                    geom,
                    ibm_mesh,
                    cell_types,
                    global_id,
                    nbr_id,
                    axis,
                    pose,
                );
            }
        }
    }

    part
}

/// Fills [`ApIbmFaceData`] for an existing internal-face topology.
#[allow(clippy::too_many_arguments)]
pub fn fill_ap_ibm_face_data(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
    owners: &[usize],
    neighbours: &[usize],
    axes: &[Axis],
    pose: &Pose,
) -> ApIbmFaceData {
    debug_assert_eq!(owners.len(), neighbours.len());
    debug_assert_eq!(owners.len(), axes.len());

    let n_faces = owners.len();
    let face_hits: Vec<Option<fluxel_ibm::solver::ApibmIntersection>> = (0..n_faces)
        .into_par_iter()
        .map(|i| {
            let owner = owners[i];
            let neighbour = neighbours[i];
            let axis = axes[i];
            let needs_raycast = cell_types[owner] == CellType::Intersect
                || cell_types[neighbour] == CellType::Intersect;
            if needs_raycast {
                resolve_apibm_face(forest, geom, ibm_mesh, owner, neighbour, axis, pose)
            } else {
                None
            }
        })
        .collect();

    let mut ap = ApIbmFaceData::default();
    ap.is_immersed_face.reserve(n_faces);
    for hit in face_hits {
        if let Some(hit) = hit {
            ap.is_immersed_face.push(true);
            ap.dist_owner_to_bnd.push(hit.dist_owner_to_bnd);
            ap.dist_neighbour_to_bnd.push(hit.dist_neighbour_to_bnd);
            ap.owner_weights.push(hit.owner_weights);
            ap.neighbour_weights.push(hit.neighbour_weights);
            ap.owner_bnd_anchor_id.push(hit.owner_bnd_anchor_id);
            ap.owner_bnd_patch_id.push(hit.owner_bnd_patch_id);
            ap.neighbour_bnd_anchor_id.push(hit.neighbour_bnd_anchor_id);
            ap.neighbour_bnd_patch_id.push(hit.neighbour_bnd_patch_id);
        } else {
            ap.is_immersed_face.push(false);
        }
    }
    ap
}

/// Rebuilds only the APIBM immersed-boundary payload, keeping mesh topology fixed.
///
/// Reclassifies intersecting cells under `pose`, then replaces [`CfdAxisProjectedMesh::ap`].
pub fn rebuild_axis_projected_ib(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    mesh: &mut CfdAxisProjectedMesh,
    pose: &Pose,
) -> Vec<CellType> {
    let cell_types = mark_intersecting_cells(forest, geom, ibm_mesh, pose);
    mesh.ap = fill_ap_ibm_face_data(
        forest,
        geom,
        ibm_mesh,
        &cell_types,
        &mesh.internal_faces_owner,
        &mesh.internal_faces_neighbour,
        &mesh.internal_faces_axis,
        pose,
    );
    mesh.patch_names = ibm_mesh.patch_names.clone();
    cell_types
}

/// Builds a [`CfdAxisProjectedMesh`] from a [`Forest`], [`Geometry`], and IBM geometry.
///
/// `pose` is the rigid transform applied to the immersed-boundary mesh during classification and
/// ray casting.
pub fn build_axis_projected_mesh(
    forest: &Forest,
    geom: &Geometry,
    ibm_mesh: &IBMMesh,
    cell_types: &[CellType],
    pose: &Pose,
) -> CfdAxisProjectedMesh {
    let n_cells = forest.num_cells();

    let parts: Vec<AxisProjectedPart> = (0..n_cells)
        .into_par_iter()
        .map(|global_id| {
            extract_axis_projected_part(forest, geom, ibm_mesh, cell_types, global_id, pose)
        })
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

        mesh.ap.is_immersed_face.extend(p.ap.is_immersed_face);
        mesh.ap.dist_owner_to_bnd.extend(p.ap.dist_owner_to_bnd);
        mesh.ap
            .dist_neighbour_to_bnd
            .extend(p.ap.dist_neighbour_to_bnd);
        mesh.ap.owner_weights.extend(p.ap.owner_weights);
        mesh.ap.neighbour_weights.extend(p.ap.neighbour_weights);
        mesh.ap
            .owner_bnd_anchor_id
            .extend(p.ap.owner_bnd_anchor_id);
        mesh.ap.owner_bnd_patch_id.extend(p.ap.owner_bnd_patch_id);
        mesh.ap
            .neighbour_bnd_anchor_id
            .extend(p.ap.neighbour_bnd_anchor_id);
        mesh.ap
            .neighbour_bnd_patch_id
            .extend(p.ap.neighbour_bnd_patch_id);
    }

    mesh
}

/// Returns `true` if any intersecting cell sits below `target_level` (under-refined band).
pub fn has_under_refined_intersect_cells(
    forest: &Forest,
    cell_types: &[CellType],
    target_level: u8,
) -> bool {
    forest
        .keys()
        .iter()
        .zip(cell_types.iter())
        .any(|(key, ct)| *ct == CellType::Intersect && key.level() < target_level)
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

    fn plane_between_cells() -> (Forest, Geometry, IBMMesh) {
        let mut forest = Forest::new([2, 1, 1]);
        forest.populate_root_cells();

        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);

        let verts = [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]];
        let indices = [[0u32, 1, 2]];
        let mesh = IBMMesh::from_vertices_indices_and_patches(
            &verts,
            &indices,
            vec!["wall".into()],
            vec![0],
        );
        (forest, geom, mesh)
    }

    #[test]
    fn axis_projected_all_fluid_sets_only_immersed_flags() {
        let (forest, geom, ibm_mesh) = two_cell_setup();
        let cell_types = vec![CellType::Fluid, CellType::Fluid];

        let mesh =
            build_axis_projected_mesh(&forest, &geom, &ibm_mesh, &cell_types, &Pose::identity());

        assert_eq!(mesh.n_cells, 2);
        assert!(mesh.ap.dist_owner_to_bnd.is_empty());
        assert_eq!(mesh.ap.is_immersed_face.iter().filter(|&&x| x).count(), 0);
        assert!(!mesh.internal_faces_owner.is_empty());
    }

    #[test]
    fn rebuild_ib_keeps_topology_and_updates_with_pose() {
        let (forest, geom, ibm_mesh) = plane_between_cells();
        let identity = Pose::identity();
        let cell_types = mark_intersecting_cells(&forest, &geom, &ibm_mesh, &identity);
        let mut mesh = build_axis_projected_mesh(&forest, &geom, &ibm_mesh, &cell_types, &identity);

        let n_faces = mesh.internal_faces_owner.len();
        let immersed_before = mesh.ap.is_immersed_face.iter().filter(|&&x| x).count();
        assert!(immersed_before > 0);

        let far = Pose::translation(10.0, 0.0, 0.0);
        rebuild_axis_projected_ib(&forest, &geom, &ibm_mesh, &mut mesh, &far);

        assert_eq!(mesh.internal_faces_owner.len(), n_faces);
        assert_eq!(mesh.ap.is_immersed_face.len(), n_faces);
        assert_eq!(mesh.ap.is_immersed_face.iter().filter(|&&x| x).count(), 0);
    }
}
