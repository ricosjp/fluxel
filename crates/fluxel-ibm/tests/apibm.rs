use fluxel_core::Forest;
use fluxel_geometry::{BoundingBox, Geometry, RigidPose};
use fluxel_ibm::*;
use fluxel_mesh::GridContext;
use std::sync::Arc;
fn setup(scale: f64, sheets: &[f64]) -> (GridContext, Boundary) {
    let mut forest = Forest::new([2, 1, 1]).unwrap();
    forest.populate_root_cells();
    let grid = GridContext::new(
        forest,
        Geometry::new(BoundingBox::new([0.0; 3], [scale; 3]).unwrap(), [2, 1, 1]).unwrap(),
    )
    .unwrap();
    let mut vertices = vec![];
    let mut indices = vec![];
    for &x in sheets {
        let offset = vertices.len() as u32;
        vertices.extend([
            [x * scale, -scale, -scale],
            [x * scale, 3.0 * scale, -scale],
            [x * scale, -scale, 3.0 * scale],
        ]);
        indices.push([offset, offset + 1, offset + 2]);
    }
    let surface = BoundarySurface::new(
        &vertices,
        &indices,
        vec!["wall".into()],
        vec![0; sheets.len()],
    )
    .unwrap();
    (grid, Boundary::Surface(Arc::new(surface)))
}
fn compute(grid: &GridContext, boundary: Boundary, pose: RigidPose) -> ApIbmData {
    let state = BoundaryState::new(boundary, pose);
    let mask = classify_intersections(grid, &state).unwrap();
    compute_apibm(grid, &state, &mask).unwrap()
}
#[test]
fn center_endpoint_and_length_units_are_preserved() {
    for scale in [0.001, 1.0, 1000.0] {
        let (grid, boundary) = setup(scale, &[0.5]);
        let data = compute(
            &grid,
            boundary,
            RigidPose::from_translation([-0.25 * scale, 0.0, 0.0]).unwrap(),
        );
        assert_eq!(data.is_immersed_face(), &[true]);
        assert!(data.dist_owner_to_bnd()[0].abs() < 1e-12 * scale);
        assert!((data.dist_neighbour_to_bnd()[0] - 0.5 * scale).abs() < 1e-12 * scale);
        assert!(data.owner_near_boundary()[0]);
        let point = data.owner_bnd_point()[0];
        assert!((point[0] - 0.25 * scale).abs() < 1e-9 * scale);
        assert!((point[1] - 0.5 * scale).abs() < 1e-9 * scale);
        assert!((point[2] - 0.5 * scale).abs() < 1e-9 * scale);
        assert!(data.owner_bnd_tangent()[0][0] > 0.9);
        assert!(data.neighbour_bnd_tangent()[0][0] < -0.9);
        assert!(data.owner_bnd_normal()[0][0] > 0.9);
        assert!(data.neighbour_bnd_normal()[0][0] > 0.9);
    }
}
#[test]
fn near_wall_flag_does_not_change_physical_distance() {
    let (grid, boundary) = setup(1.0, &[0.5]);
    let data = compute(
        &grid,
        boundary,
        RigidPose::from_translation([-0.249, 0.0, 0.0]).unwrap(),
    );
    assert!((data.dist_owner_to_bnd()[0] - 0.001).abs() < 1e-14);
    assert!(data.owner_near_boundary()[0]);
}
#[test]
fn two_sheets_have_distinct_first_hits() {
    let (grid, boundary) = setup(1.0, &[0.4, 0.6]);
    let data = compute(&grid, boundary, RigidPose::identity());
    assert!((data.dist_owner_to_bnd()[0] - 0.15).abs() < 1e-14);
    assert!((data.dist_neighbour_to_bnd()[0] - 0.15).abs() < 1e-14);
    assert_ne!(data.owner_bnd_anchor_id(), data.neighbour_bnd_anchor_id());
}
#[test]
fn translated_outside_boundary_produces_empty_compressed_payload() {
    let (grid, boundary) = setup(1.0, &[0.5]);
    let data = compute(
        &grid,
        boundary,
        RigidPose::from_translation([10.0, 0.0, 0.0]).unwrap(),
    );
    assert_eq!(data.is_immersed_face(), &[false]);
    assert!(data.dist_owner_to_bnd().is_empty());
}
