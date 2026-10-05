use crate::{BackgroundMesh, CellGeometry, CellIndex, MeshError, MeshId, MeshTopology};
use fluxel_core::{Axis, Direction, Forest};
use fluxel_geometry::{ParameterBox, SpatialDomain};
use rayon::prelude::*;

/// Cell and face ordering is part of the Python array contract.
pub(crate) fn build_background(
    forest: &Forest,
    domain: &SpatialDomain,
) -> Result<BackgroundMesh, MeshError> {
    let parts: Result<Vec<_>, MeshError> = (0..forest.num_cells())
        .into_par_iter()
        .map(|id| {
            let measure = domain
                .cell_measure(forest, id)
                .map_err(|_| MeshError::GeometryMismatch)?;
            // Record every plus-side neighbour, and a boundary where either side is empty.
            let mut topology = MeshTopology::default();
            for axis in Axis::ALL {
                let (minus, plus) = axis.split_into_directions();
                if forest.face_neighbour_global_ids(id, minus).is_empty() {
                    push_boundary(domain, &mut topology, id, minus, measure.parameter)?;
                }
                let neighbours = forest.face_neighbour_global_ids(id, plus);
                if neighbours.is_empty() {
                    push_boundary(domain, &mut topology, id, plus, measure.parameter)?;
                }
                for neighbour in neighbours {
                    let neighbour_box = domain
                        .parameter_box(forest, neighbour)
                        .map_err(|_| MeshError::GeometryMismatch)?;
                    let face = overlap_face(measure.parameter, neighbour_box, axis.as_index());
                    let area = domain
                        .face_area_vector(face, 1.0)
                        .map_err(|_| MeshError::GeometryMismatch)?;
                    topology.internal_owner.push(CellIndex(id));
                    topology.internal_neighbour.push(CellIndex(neighbour));
                    topology.internal_axis.push(axis);
                    topology.internal_bounds.push(face);
                    topology.internal_area.push(area);
                    topology
                        .internal_winding
                        .push(winding(forest, id, neighbour, axis));
                }
            }
            Ok((measure, topology))
        })
        .collect();
    let parts = parts?;
    let mut topology = MeshTopology::default();
    let mut geometry = CellGeometry::default();
    geometry.centers.reserve(parts.len());
    geometry.sizes.reserve(parts.len());
    for (measure, part) in parts {
        geometry.centers.push(measure.sample);
        geometry.sizes.push(measure.widths);
        geometry.volumes.push(measure.volume);
        geometry.centroids.push(measure.centroid);
        geometry.corners.push(measure.corners);
        topology.internal_owner.extend(part.internal_owner);
        topology.internal_neighbour.extend(part.internal_neighbour);
        topology.internal_axis.extend(part.internal_axis);
        topology.internal_bounds.extend(part.internal_bounds);
        topology.internal_area.extend(part.internal_area);
        topology.internal_winding.extend(part.internal_winding);
        topology.boundary_owner.extend(part.boundary_owner);
        topology.boundary_direction.extend(part.boundary_direction);
        topology.boundary_bounds.extend(part.boundary_bounds);
        topology.boundary_area.extend(part.boundary_area);
    }
    Ok(BackgroundMesh {
        id: MeshId::fresh(),
        topology,
        geometry,
        coordinate_type: domain.coordinate_type(),
    })
}

fn push_boundary(
    domain: &SpatialDomain,
    topology: &mut MeshTopology,
    owner: usize,
    direction: Direction,
    cell: ParameterBox,
) -> Result<(), MeshError> {
    let face = boundary_face(cell, direction);
    let sign = if direction.sign() > 0 { 1.0 } else { -1.0 };
    let area = domain
        .face_area_vector(face, sign)
        .map_err(|_| MeshError::GeometryMismatch)?;
    topology.boundary_owner.push(CellIndex(owner));
    topology.boundary_direction.push(direction);
    topology.boundary_bounds.push(face);
    topology.boundary_area.push(area);
    Ok(())
}

fn boundary_face(cell: ParameterBox, direction: Direction) -> ParameterBox {
    let axis = direction.axis().as_index();
    let mut face = cell;
    if direction.sign() > 0 {
        face.min[axis] = cell.max[axis];
    } else {
        face.max[axis] = cell.min[axis];
    }
    face
}

fn overlap_face(owner: ParameterBox, neighbour: ParameterBox, axis: usize) -> ParameterBox {
    // Place the face on the owner's plus side. A positive transverse overlap is the
    // partial face; an inverted overlap keeps the owner's full transverse range.
    let mut min = owner.min;
    let mut max = owner.max;
    min[axis] = owner.max[axis];
    max[axis] = owner.max[axis];
    for transverse in 0..3 {
        if transverse == axis {
            continue;
        }
        let start = owner.min[transverse].max(neighbour.min[transverse]);
        let end = owner.max[transverse].min(neighbour.max[transverse]);
        if start < end {
            min[transverse] = start;
            max[transverse] = end;
        }
    }
    ParameterBox { min, max }
}

fn winding(forest: &Forest, owner: usize, neighbour: usize, axis: Axis) -> i8 {
    let periodic = forest.periodic_axes();
    if !periodic[axis.as_index()] {
        return 0;
    }
    let base = forest.base_resolution();
    let owner_tree = tree_component(forest.keys()[owner].tree_id(), base, axis);
    let neighbour_tree = tree_component(forest.keys()[neighbour].tree_id(), base, axis);
    if neighbour_tree < owner_tree {
        1
    } else {
        0
    }
}

fn tree_component(tree_id: u32, base: [u32; 3], axis: Axis) -> u32 {
    let [nx, ny, _] = base;
    match axis {
        Axis::X => tree_id % nx,
        Axis::Y => (tree_id / nx) % ny,
        Axis::Z => tree_id / (nx * ny),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridContext;
    use fluxel_geometry::{BoundingBox, Geometry};
    #[test]
    fn coarse_fine_faces_are_unique_and_cover_the_interface() {
        let mut forest = Forest::new([2, 1, 1]).unwrap();
        forest.populate_root_cells();
        forest.refine_by_flags(&[false, true]).unwrap();
        let geometry = Geometry::new(
            BoundingBox::new([0.0; 3], [2.0, 1.0, 1.0]).unwrap(),
            [2, 1, 1],
        )
        .unwrap();
        let grid = GridContext::new(forest, geometry).unwrap();
        let topology = grid.background().topology();
        let pairs: std::collections::HashSet<_> = topology
            .internal_owner()
            .iter()
            .zip(topology.internal_neighbour())
            .map(|(a, b)| (a.index(), b.index()))
            .collect();
        assert_eq!(pairs.len(), topology.n_internal_faces());
        assert_eq!(pairs.iter().filter(|(a, _)| *a == 0).count(), 4);
        assert_eq!(grid.background().n_cells(), 9);
        let cell = grid.background().geometry();
        let size = cell.sizes()[0];
        assert!((cell.volumes()[0] - size[0] * size[1] * size[2]).abs() < 1e-12);
        assert!(topology
            .internal_winding()
            .iter()
            .all(|winding| *winding == 0));
    }
    #[test]
    fn full_turn_connects_the_angular_seam_once() {
        use fluxel_core::Axis;
        use fluxel_geometry::{CoordinateType, CylindricalGeometry};
        let mut forest = Forest::with_periodic_axes([1, 4, 1], [false, true, false]).unwrap();
        forest.populate_root_cells();
        let geometry = CylindricalGeometry::full_turn([0.0; 3], 1.0, 2.0, 0.0, 0.0, 1.0).unwrap();
        let grid = GridContext::from_cylindrical(forest, geometry).unwrap();
        let topology = grid.background().topology();
        assert_eq!(
            grid.background().coordinate_type(),
            CoordinateType::Cylindrical
        );
        assert_eq!(topology.n_internal_faces(), 4);
        assert!(topology
            .boundary_direction()
            .iter()
            .all(|direction| direction.axis() != Axis::Y));
        assert_eq!(topology.boundary_owner().len(), 16);
        let center = grid.background().geometry().centers()[0];
        let radius = center[0].hypot(center[1]);
        assert!((radius - 1.5).abs() < 1e-12);
        assert!((center[2] - 0.5).abs() < 1e-12);
        let cells = grid.background().geometry();
        let volume: f64 = cells.volumes().iter().sum();
        assert!((volume - 3.0 * std::f64::consts::PI).abs() < 1e-9);
        let sample = cells.centers()[0];
        let centroid = cells.centroids()[0];
        assert!((centroid[0] - sample[0]).hypot(centroid[1] - sample[1]) > 1e-3);
        let seam = topology
            .internal_winding()
            .iter()
            .position(|winding| *winding == 1)
            .unwrap();
        let area = topology.internal_area()[seam];
        assert!(area[1] > 0.9 && area[0].abs() < 1e-9 && area[2].abs() < 1e-9);
        let mut sector_forest =
            Forest::with_periodic_axes([1, 4, 1], [false, true, false]).unwrap();
        sector_forest.populate_root_cells();
        let sector = CylindricalGeometry::new([0.0; 3], 1.0, 2.0, 0.0, 1.0, 0.0, 1.0).unwrap();
        let sector_grid = GridContext::from_cylindrical(sector_forest, sector).unwrap();
        assert!(sector_grid
            .background()
            .topology()
            .internal_winding()
            .iter()
            .any(|winding| *winding == 1));
    }
    #[test]
    fn cartesian_periodic_x_wraps_outer_faces() {
        let mut forest = Forest::with_periodic_axes([2, 1, 1], [true, false, false]).unwrap();
        forest.populate_root_cells();
        let geometry = Geometry::new(
            BoundingBox::new([0.0; 3], [2.0, 1.0, 1.0]).unwrap(),
            [2, 1, 1],
        )
        .unwrap();
        let grid = GridContext::new(forest, geometry).unwrap();
        let topology = grid.background().topology();
        let x_boundaries = topology
            .boundary_direction()
            .iter()
            .filter(|direction| matches!(direction, Direction::XMinus | Direction::XPlus))
            .count();
        assert_eq!(x_boundaries, 0);
        assert_eq!(
            topology
                .internal_winding()
                .iter()
                .filter(|winding| **winding == 1)
                .count(),
            1
        );
    }
    #[test]
    fn coarse_fine_seam_keeps_partial_faces_and_winding() {
        use fluxel_geometry::CylindricalGeometry;
        let mut forest = Forest::with_periodic_axes([1, 4, 1], [false, true, false]).unwrap();
        forest.populate_root_cells();
        forest
            .refine_by_flags(&[false, false, false, true])
            .unwrap();
        let geometry = CylindricalGeometry::full_turn([0.0; 3], 1.0, 2.0, 0.0, 0.0, 1.0).unwrap();
        let grid = GridContext::from_cylindrical(forest, geometry).unwrap();
        assert_eq!(grid.background().n_cells(), 11);
        let seams: Vec<_> = grid
            .background()
            .topology()
            .internal_winding()
            .iter()
            .enumerate()
            .filter(|(_, winding)| **winding == 1)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(seams.len(), 4);
        for index in seams {
            let bounds = grid.background().topology().internal_bounds()[index];
            assert!((bounds.min[1] - std::f64::consts::TAU).abs() < 1e-12);
            assert!((bounds.max[1] - bounds.min[1]).abs() < 1e-12);
            assert!((bounds.max[0] - bounds.min[0] - 0.5).abs() < 1e-12);
            assert!((bounds.max[2] - bounds.min[2] - 0.5).abs() < 1e-12);
        }
    }
    #[test]
    fn unbalanced_cylinder_is_rejected_until_balanced() {
        use fluxel_geometry::CylindricalGeometry;
        let mut forest = Forest::with_periodic_axes([1, 4, 1], [false, true, false]).unwrap();
        forest.populate_root_cells();
        forest
            .refine_by_flags(&[false, false, false, true])
            .unwrap();
        let mut again = vec![false; forest.num_cells()];
        let child = forest
            .keys()
            .iter()
            .position(|key| key.level() == 1)
            .unwrap();
        again[child] = true;
        forest.refine_by_flags(&again).unwrap();
        let geometry = CylindricalGeometry::full_turn([0.0; 3], 1.0, 2.0, 0.0, 0.0, 1.0).unwrap();
        assert!(GridContext::from_cylindrical(forest.clone(), geometry).is_err());
        let balanced = forest.into_balanced().unwrap();
        assert!(GridContext::from_cylindrical(balanced.into_forest(), geometry).is_ok());
    }
    #[test]
    fn refuses_empty_coverage_and_mismatched_geometry() {
        let geometry =
            Geometry::new(BoundingBox::new([0.0; 3], [1.0; 3]).unwrap(), [1; 3]).unwrap();
        assert!(GridContext::new(Forest::new([1; 3]).unwrap(), geometry.clone()).is_err());
        assert!(GridContext::new(Forest::new([2; 3]).unwrap(), geometry).is_err());
    }
}
