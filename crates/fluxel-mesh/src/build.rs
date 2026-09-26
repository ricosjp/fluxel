use crate::{BackgroundMesh, CellGeometry, CellIndex, MeshId, MeshTopology};
use fluxel_core::{Axis, Forest};
use fluxel_geometry::Geometry;
use rayon::prelude::*;

/// Cell and face ordering is part of the Python array contract.
pub(crate) fn build_background(forest: &Forest, geometry: &Geometry) -> BackgroundMesh {
    let parts: Vec<_> = (0..forest.num_cells())
        .into_par_iter()
        .map(|id| {
            let (center, size) = geometry.cell_bounds(&forest.keys()[id].to_logical());
            let mut topology = MeshTopology::default();
            for axis in Axis::ALL {
                let (minus, plus) = axis.split_into_directions();
                if forest.face_neighbour_global_ids(id, minus).is_empty() {
                    topology.boundary_owner.push(CellIndex(id));
                    topology.boundary_direction.push(minus);
                }
                let neighbours = forest.face_neighbour_global_ids(id, plus);
                if neighbours.is_empty() {
                    topology.boundary_owner.push(CellIndex(id));
                    topology.boundary_direction.push(plus);
                }
                for neighbour in neighbours {
                    topology.internal_owner.push(CellIndex(id));
                    topology.internal_neighbour.push(CellIndex(neighbour));
                    topology.internal_axis.push(axis);
                }
            }
            (center, size, topology)
        })
        .collect();
    let mut topology = MeshTopology::default();
    let mut geometry = CellGeometry::default();
    geometry.centers.reserve(parts.len());
    geometry.sizes.reserve(parts.len());
    for (center, size, part) in parts {
        geometry.centers.push(center);
        geometry.sizes.push(size);
        topology.internal_owner.extend(part.internal_owner);
        topology.internal_neighbour.extend(part.internal_neighbour);
        topology.internal_axis.extend(part.internal_axis);
        topology.boundary_owner.extend(part.boundary_owner);
        topology.boundary_direction.extend(part.boundary_direction);
    }
    BackgroundMesh {
        id: MeshId::fresh(),
        topology,
        geometry,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::GridContext;
    use fluxel_geometry::BoundingBox;
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
    }
    #[test]
    fn refuses_empty_coverage_and_mismatched_geometry() {
        let geometry =
            Geometry::new(BoundingBox::new([0.0; 3], [1.0; 3]).unwrap(), [1; 3]).unwrap();
        assert!(GridContext::new(Forest::new([1; 3]).unwrap(), geometry.clone()).is_err());
        assert!(GridContext::new(Forest::new([2; 3]).unwrap(), geometry).is_err());
    }
}
