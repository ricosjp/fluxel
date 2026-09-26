use fluxel_core::Forest;
use fluxel_geometry::{BoundingBox, Geometry, RigidPose};
use fluxel_ibm::*;
use fluxel_mesh::GridContext;
use std::sync::Arc;
fn grid() -> GridContext {
    let mut forest = Forest::new([4; 3]).unwrap();
    forest.populate_root_cells();
    GridContext::new(
        forest,
        Geometry::new(BoundingBox::new([0.0; 3], [1.0; 3]).unwrap(), [4; 3]).unwrap(),
    )
    .unwrap()
}
#[test]
fn masks_reject_other_grids_and_boundary_revisions() {
    let grid = grid();
    let boundary = BoundaryState::new(Boundary::None, RigidPose::identity());
    let mask = classify_intersections(&grid, &boundary).unwrap();
    let other = boundary.with_pose(RigidPose::identity());
    assert!(compute_apibm(&grid, &other, &mask).is_err());
    assert!(compute_gcibm(&grid, &other, &mask, [0.1; 3]).is_err());
}
#[test]
fn ghost_geometry_preserves_classification_and_normalized_weights() {
    let grid = grid();
    let surface = BoundarySurface::new(
        &[
            [0.6, -1.0, -1.0],
            [0.6, 2.0, -1.0],
            [0.6, 2.0, 2.0],
            [0.6, -1.0, 2.0],
        ],
        &[[0, 1, 2], [0, 2, 3]],
        vec!["wall".into()],
        vec![0, 0],
    )
    .unwrap();
    let boundary = BoundaryState::new(Boundary::Surface(Arc::new(surface)), RigidPose::identity());
    let mask = classify_intersections(&grid, &boundary).unwrap();
    let (data, diagnostics) = compute_gcibm(&grid, &boundary, &mask, [0.1; 3]).unwrap();
    assert_eq!(data.gc_is_fluid().len(), 64);
    assert!(!data.gc_cell_ids().is_empty());
    assert!(diagnostics.interpolation_fallbacks <= data.gc_cell_ids().len());
    for weights in data.gc_interp_stencil_weights() {
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert!(compute_gcibm(&grid, &boundary, &mask, [0.6, 0.5, 0.5]).is_err());
    assert!(compute_gcibm(&grid, &boundary, &mask, [f64::NAN, 0.0, 0.0]).is_err());
}
#[test]
fn invalid_surfaces_return_errors() {
    assert!(BoundarySurface::new(&[], &[], vec![], vec![]).is_err());
    let vertices = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    assert!(BoundarySurface::new(&vertices, &[[0, 1, 3]], vec!["a".into()], vec![0]).is_err());
    assert!(BoundarySurface::new(&vertices, &[[0, 1, 2]], vec!["a".into()], vec![1]).is_err());
    assert!(BoundarySurface::new(&vertices, &[[0, 0, 0]], vec!["a".into()], vec![0]).is_err());
}
