use fluxel_engine::*;
use fluxel_geometry::BoundingBox;
use fluxel_ibm::{Boundary, BoundarySurface};
use std::sync::Arc;

fn config(base: [u32; 3], limit: usize) -> MeshBuildConfig {
    MeshBuildConfig::new(
        BoundingBox::new([0.0; 3], [1.0; 3]).unwrap(),
        base,
        0,
        BuildLimits { max_cells: limit },
    )
    .unwrap()
}
fn plane() -> Boundary {
    Boundary::Surface(Arc::new(
        BoundarySurface::new(
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
        .unwrap(),
    ))
}
fn plan(level: u8) -> RefinementPlan {
    RefinementPlan::new(level, vec![]).unwrap()
}
#[test]
fn no_boundary_is_all_fluid_and_has_no_synthetic_patch() {
    let output = build_gcibm(
        &config([2; 3], usize::MAX),
        Boundary::None,
        plan(1),
        [0.1; 3],
    )
    .unwrap();
    assert_eq!(output.mesh.background().n_cells(), 8);
    assert!(output.mesh.payload().gc_is_fluid().iter().all(|&v| v));
    assert!(output.mesh.payload().gc_cell_ids().is_empty());
    assert!(output.mesh.patches().names().is_empty());
    assert!(build_gcibm(
        &config([1; 3], usize::MAX),
        Boundary::None,
        plan(0),
        [1.0; 3]
    )
    .is_err());
}
#[test]
fn region_refinement_uses_strict_volume_overlap() {
    let region =
        RefinementRegion::new(BoundingBox::new([0.0; 3], [0.5, 1.0, 1.0]).unwrap(), 1).unwrap();
    let output = build_apibm(
        &config([2, 1, 1], usize::MAX),
        Boundary::None,
        RefinementPlan::new(0, vec![region]).unwrap(),
    )
    .unwrap();
    assert_eq!(output.mesh.background().n_cells(), 9);
}
#[test]
fn ap_and_gc_share_the_same_topology_definition() {
    let config = config([2; 3], usize::MAX);
    let ap = build_apibm(&config, plane(), plan(1)).unwrap();
    let gc = build_gcibm(&config, plane(), plan(1), [0.1; 3]).unwrap();
    assert_eq!(
        ap.mesh.background().geometry().centers(),
        gc.mesh.background().geometry().centers()
    );
    assert_eq!(
        ap.mesh.background().topology().internal_owner(),
        gc.mesh.background().topology().internal_owner()
    );
    assert_eq!(
        ap.mesh.background().topology().internal_neighbour(),
        gc.mesh.background().topology().internal_neighbour()
    );
}
#[test]
fn updates_share_background_and_old_snapshots_survive_remesh() {
    let mut session = ApibmSession::new(config([2; 3], usize::MAX), plane(), plan(1)).unwrap();
    let before = session.mesh().clone();
    assert!(!before.payload().dist_owner_to_bnd().is_empty());
    let moved = session
        .update_ib(PoseUpdate {
            translation: Some([10.0, 0.0, 0.0]),
            ..Default::default()
        })
        .unwrap()
        .clone();
    assert!(Arc::ptr_eq(before.background(), moved.background()));
    assert!(moved.payload().dist_owner_to_bnd().is_empty());
    assert!(!before.payload().dist_owner_to_bnd().is_empty());
    session.remesh(RemeshRequest::default()).unwrap();
    assert_ne!(before.background().id(), session.mesh().background().id());
    assert_eq!(before.background().n_cells(), moved.background().n_cells());
}
#[test]
fn discarded_failed_foreign_and_stale_candidates_leave_state_unchanged() {
    let mut session = ApibmSession::new(config([2; 3], 64), plane(), plan(0)).unwrap();
    let before = session.mesh().clone();
    let update = session
        .prepare_update(PoseUpdate {
            translation: Some([0.1, 0.0, 0.0]),
            ..Default::default()
        })
        .unwrap();
    drop(update);
    assert_eq!(session.pose().translation(), [0.0; 3]);
    assert!(session
        .prepare_remesh(RemeshRequest {
            target_level: Some(4),
            ..Default::default()
        })
        .is_err());
    assert!(Arc::ptr_eq(
        before.background(),
        session.mesh().background()
    ));
    let mut other = ApibmSession::new(config([2; 3], 64), plane(), plan(0)).unwrap();
    let foreign = session.prepare_update(PoseUpdate::default()).unwrap();
    assert!(other.commit(foreign).is_err());
    let stale = session.prepare_update(PoseUpdate::default()).unwrap();
    session.update_ib(PoseUpdate::default()).unwrap();
    assert!(session.commit(stale).is_err());
}
#[test]
fn clearing_regions_differs_from_omitting_them() {
    let region =
        RefinementRegion::new(BoundingBox::new([0.0; 3], [0.5, 1.0, 1.0]).unwrap(), 1).unwrap();
    let mut session = ApibmSession::new(
        config([2, 1, 1], usize::MAX),
        Boundary::None,
        RefinementPlan::new(0, vec![region]).unwrap(),
    )
    .unwrap();
    assert_eq!(session.mesh().background().n_cells(), 9);
    session.remesh(RemeshRequest::default()).unwrap();
    assert_eq!(session.mesh().background().n_cells(), 9);
    session
        .remesh(RemeshRequest {
            regions: Some(vec![]),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(session.mesh().background().n_cells(), 2);
}
#[test]
fn manual_and_automatic_regions_match_and_reject_invalid_inputs() {
    let mut grid = ManualGrid::new(
        BoundingBox::new([0.0; 3], [1.0; 3]).unwrap(),
        [2, 1, 1],
        BuildLimits::default(),
    )
    .unwrap();
    grid.refine_region(
        RefinementRegion::new(BoundingBox::new([0.0; 3], [0.5, 1.0, 1.0]).unwrap(), 1).unwrap(),
    )
    .unwrap();
    assert_eq!(grid.num_cells(), 9);
    assert!(grid.refine_by_flags(&[true]).is_err());
    assert!(grid.uniform_refinement(33).is_err());
    assert_eq!(grid.num_cells(), 9);
}
