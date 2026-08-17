//! Intersection classification: cell vs surface mesh.

use crate::mesh::IBMMesh;
use crate::types::CellType;
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use parry3d_f64::math::{Pose, Vector};
use parry3d_f64::query::{self, Ray, RayCast};
use rayon::prelude::*;

/// Step 1: exact intersection classification.
///
/// Runs a robust intersection test between every forest cell and the `IBMMesh` (BVH-backed).
/// Cells that intersect the boundary are marked as `Intersect`; all others are initialized as
/// `Solid` and may later be reclassified by flood fill.
///
/// `pose` is the rigid transform applied to the immersed-boundary mesh.
pub fn mark_intersecting_cells(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    pose: &Pose,
) -> Vec<CellType> {
    let mut cell_types = vec![CellType::Solid; forest.num_cells()];

    cell_types
        .par_iter_mut()
        .zip(forest.keys().par_iter())
        .for_each(|(cell_type, key)| {
            let logical = key.to_logical();

            let (center, size) = geom.cell_bounds(&logical);

            let half_extents = Vector::new(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0);
            let cuboid = parry3d_f64::shape::Cuboid::new(half_extents);

            let iso = Pose::translation(center[0], center[1], center[2]);

            if query::intersection_test(&iso, &cuboid, pose, &mesh.bvh).unwrap_or(false) {
                *cell_type = CellType::Intersect;
            }
        });

    cell_types
}

/// Ray-casting parity check (robust majority vote).
///
/// Lightweight alternative to heavy GWN: O(log N) inside test. Uses 13 rays with majority vote
/// to reduce edge-grazing misclassification.
///
/// `pose` is the rigid transform applied to the immersed-boundary mesh.
pub(crate) fn check_inside_parity_robust(
    p: &Vector,
    bvh: &parry3d_f64::shape::TriMesh,
    pose: &Pose,
) -> bool {
    let dirs = [
        Vector::new(1.0, 0.11, 0.05).normalize(),
        Vector::new(-1.0, -0.05, 0.11).normalize(),
        Vector::new(0.05, 1.0, -0.11).normalize(),
        Vector::new(-0.11, -1.0, 0.05).normalize(),
        Vector::new(0.11, 0.05, 1.0).normalize(),
        Vector::new(-0.05, -0.11, -1.0).normalize(),
        Vector::new(0.577, 0.577, 0.577).normalize(),
        Vector::new(-0.577, -0.577, -0.577).normalize(),
        Vector::new(0.577, -0.577, 0.577).normalize(),
        Vector::new(-0.577, 0.577, -0.577).normalize(),
        Vector::new(0.577, 0.577, -0.577).normalize(),
        Vector::new(-0.577, -0.577, 0.577).normalize(),
        Vector::new(-0.577, 0.577, 0.577).normalize(),
    ];

    let inside_count = dirs
        .iter()
        .filter(|dir| {
            let dir = **dir;
            let mut intersections = 0;
            let mut current_p = *p;
            let mut max_dist = 1e6;

            while let Some(toi) = bvh.cast_ray(pose, &Ray::new(current_p, dir), max_dist, false) {
                intersections += 1;
                current_p += dir * (toi + 1e-7);
                max_dist -= toi + 1e-7;
                if max_dist <= 0.0 {
                    break;
                }
            }

            intersections % 2 != 0
        })
        .count();

    inside_count >= 7
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::IBMMesh;
    use fluxel_geometry::BoundingBox;

    #[test]
    fn mark_intersecting_marks_cell_with_surface_patch() {
        let mut forest = Forest::new([2, 1, 1]);
        forest.populate_root_cells();

        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);

        let verts = [[0.05, 0.05, 0.05], [0.2, 0.05, 0.05], [0.1, 0.2, 0.05]];
        let indices = [[0u32, 1, 2]];
        let mesh =
            IBMMesh::from_vertices_indices_and_patches(&verts, &indices, vec!["t".into()], vec![0]);

        let types = mark_intersecting_cells(&forest, &geom, &mesh, &Pose::identity());
        assert!(
            types.contains(&CellType::Intersect),
            "expected at least one intersecting cell"
        );
    }
}
