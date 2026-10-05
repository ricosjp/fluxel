use fluxel_core::{Axis, Forest};
use fluxel_geometry::{CylindricalGeometry, RigidPose};
use fluxel_ibm::*;
use fluxel_mesh::GridContext;
use std::sync::Arc;

fn annulus() -> (GridContext, CylindricalGeometry) {
    let mut forest = Forest::with_periodic_axes([2, 4, 1], [false, true, false]).unwrap();
    forest.populate_root_cells();
    let geometry = CylindricalGeometry::full_turn([0.0; 3], 1.0, 3.0, 0.0, 0.0, 1.0).unwrap();
    let grid = GridContext::from_cylindrical(forest, geometry).unwrap();
    (grid, geometry)
}

fn cutting_plane() -> Boundary {
    let scale = 2.0 * std::f64::consts::SQRT_2;
    let vertices = [
        [scale, 0.0, -1.0],
        [0.0, scale, -1.0],
        [0.0, scale, 2.0],
        [scale, 0.0, 2.0],
    ];
    let surface = BoundarySurface::new(
        &vertices,
        &[[0, 1, 2], [0, 2, 3]],
        vec!["wall".into()],
        vec![0, 0],
    )
    .unwrap();
    Boundary::Surface(Arc::new(surface))
}

#[test]
fn radial_face_uses_the_cylindrical_ray() {
    let (grid, _) = annulus();
    let state = BoundaryState::new(cutting_plane(), RigidPose::identity());
    let mask = classify_intersections(&grid, &state).unwrap();
    let data = compute_apibm(&grid, &state, &mask).unwrap();
    let topology = grid.background().topology();
    let face = topology
        .internal_axis()
        .iter()
        .zip(topology.internal_owner())
        .zip(topology.internal_neighbour())
        .position(|((axis, owner), neighbour)| {
            *axis == Axis::X && owner.index() == 0 && neighbour.index() == 1
        })
        .expect("radial face");
    assert!(data.is_immersed_face()[face]);
    let slot = data_index(&data, face);
    assert!((data.dist_owner_to_bnd()[slot] - 0.5).abs() < 1e-8);
    assert!((data.dist_neighbour_to_bnd()[slot] - 0.5).abs() < 1e-8);
    let unit = std::f64::consts::FRAC_1_SQRT_2;
    let point = data.owner_bnd_point()[slot];
    assert!((point[0] - 2.0 * unit).abs() < 1e-8);
    assert!((point[1] - 2.0 * unit).abs() < 1e-8);
    assert!((point[2] - 0.5).abs() < 1e-8);
    assert!((data.owner_bnd_tangent()[slot][0] - unit).abs() < 1e-8);
    assert!((data.neighbour_bnd_tangent()[slot][0] + unit).abs() < 1e-8);
    let normal = data.owner_bnd_normal()[slot];
    assert!((normal[0] - unit).abs() < 1e-8 && (normal[1] - unit).abs() < 1e-8);
}

#[test]
fn angular_seam_uses_the_short_arc() {
    let (grid, _) = annulus();
    let vertices = [
        [1.0, 0.0, -1.0],
        [4.0, 0.0, -1.0],
        [4.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
    ];
    let surface = BoundarySurface::new(
        &vertices,
        &[[0, 1, 2], [0, 2, 3]],
        vec!["seam".into()],
        vec![0, 0],
    )
    .unwrap();
    let state = BoundaryState::new(Boundary::Surface(Arc::new(surface)), RigidPose::identity());
    let mask = classify_intersections(&grid, &state).unwrap();
    let data = compute_apibm(&grid, &state, &mask).unwrap();
    let topology = grid.background().topology();
    let face = topology
        .internal_axis()
        .iter()
        .zip(topology.internal_owner())
        .zip(topology.internal_neighbour())
        .position(|((axis, owner), neighbour)| {
            *axis == Axis::Y && owner.index() == 6 && neighbour.index() == 0
        })
        .expect("seam face");
    let expected = 1.5 * std::f64::consts::FRAC_PI_4;
    let slot = data_index(&data, face);
    assert!((data.dist_owner_to_bnd()[slot] - expected).abs() < 1e-8);
    assert!((data.dist_neighbour_to_bnd()[slot] - expected).abs() < 1e-8);
    let point = data.owner_bnd_point()[slot];
    assert!((point[0] - 1.5).abs() < 1e-8);
    assert!(point[1].abs() < 1e-8);
    assert!((point[2] - 0.5).abs() < 1e-8);
    assert!(data.owner_bnd_tangent()[slot][1] > 0.9);
    assert!(data.neighbour_bnd_tangent()[slot][1] < -0.9);
    assert!(data.owner_bnd_normal()[slot][1] < -0.9);
}

#[test]
fn ghost_cells_accept_a_fluid_seed_inside_the_annulus() {
    let (grid, geometry) = annulus();
    let state = BoundaryState::new(Boundary::None, RigidPose::identity());
    let mask = classify_intersections(&grid, &state).unwrap();
    let seed = geometry.world_from_param(1.5, 0.1, 0.5);
    let (data, _) = compute_gcibm(&grid, &state, &mask, seed).unwrap();
    assert!(data.gc_is_fluid().iter().all(|fluid| *fluid));
    let outside = geometry.world_from_param(0.2, 0.1, 0.5);
    assert!(compute_gcibm(&grid, &state, &mask, outside).is_err());
}

#[test]
fn ghost_stencil_offsets_in_the_local_frame() {
    let (grid, geometry) = annulus();
    let center = geometry.world_from_param(1.5, std::f64::consts::FRAC_PI_4, 0.5);
    let state = BoundaryState::new(closed_box(center, 0.2), RigidPose::identity());
    let mask = classify_intersections(&grid, &state).unwrap();
    let seed = geometry.world_from_param(1.5, 1.25 * std::f64::consts::PI, 0.5);
    let (data, _) = compute_gcibm(&grid, &state, &mask, seed).unwrap();
    assert!(!data.gc_is_fluid()[0]);
    assert!(!data.gc_cell_ids().is_empty());
}

fn closed_box(center: [f64; 3], half: f64) -> Boundary {
    let [x, y, z] = center;
    let vertices = [
        [x - half, y - half, z - half],
        [x + half, y - half, z - half],
        [x + half, y + half, z - half],
        [x - half, y + half, z - half],
        [x - half, y - half, z + half],
        [x + half, y - half, z + half],
        [x + half, y + half, z + half],
        [x - half, y + half, z + half],
    ];
    let indices = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let surface =
        BoundarySurface::new(&vertices, &indices, vec!["box".into()], vec![0; 12]).unwrap();
    Boundary::Surface(Arc::new(surface))
}

fn data_index(data: &ApIbmData, face: usize) -> usize {
    data.is_immersed_face()
        .iter()
        .take(face + 1)
        .filter(|immersed| **immersed)
        .count()
        - 1
}
