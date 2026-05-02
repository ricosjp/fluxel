//! Integration: classify → flood fill → ghost-cell geometry (paths requiring full solver coupling).

use fluxel_core::Forest;
use fluxel_geometry::{BoundingBox, Geometry};
use fluxel_ibm::solver::{compute_ghost_cell_geometry, flood_fill_inside_outside, mark_intersecting_cells};
use fluxel_ibm::IBMMesh;

fn unit_two_cell_setup() -> (Forest, Geometry, IBMMesh) {
    let mut forest = Forest::new([2, 1, 1]);
    forest.populate_root_cells();

    let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let geom = Geometry::new(bbox, [2, 1, 1]);

    let verts = [[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]];
    let indices = [[0u32, 1, 2]];
    let mesh = IBMMesh::from_vertices_indices_and_patches(
        &verts,
        &indices,
        vec!["far".into()],
        vec![0],
    );
    (forest, geom, mesh)
}

#[test]
fn ghost_geometry_after_flood_produces_consistent_gc_is_fluid() {
    let (forest, geom, ibm_mesh) = unit_two_cell_setup();

    let mut cell_types = mark_intersecting_cells(&forest, &geom, &ibm_mesh);
    flood_fill_inside_outside(&forest, &geom, &mut cell_types, [0.75, 0.5, 0.5]).unwrap();

    let gc = compute_ghost_cell_geometry(
        &forest,
        &geom,
        &ibm_mesh,
        &cell_types,
        [0.75, 0.5, 0.5],
    );

    assert_eq!(gc.gc_is_fluid.len(), 2);
    assert!(gc.gc_is_fluid.iter().any(|&f| f));
}
