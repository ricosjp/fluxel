//! BFS flood fill for fluid/solid labeling.

use super::locate::get_global_id_from_phys;
use crate::types::CellType;
use fluxel_core::{Direction, Forest};
use fluxel_geometry::Geometry;
use std::collections::VecDeque;

/// Step 2: BFS flood fill for inside/outside classification (solid voxelization).
///
/// Uses user-specified seed points as seeds and fills reachable `Solid` cells as `Fluid`.
/// `Intersect` cells act as barriers, so cells inside closed geometry remain `Solid`.
/// This strategy is robust against imperfect or noisy STL surfaces.
pub fn flood_fill_inside_outside(
    forest: &Forest,
    geom: &Geometry,
    cell_types: &mut [CellType],
    fluid_seed_point: [f64; 3],
) -> Result<(), String> {
    let total_cells = forest.num_cells();
    let mut visited = vec![false; total_cells];

    visited
        .iter_mut()
        .zip(cell_types.iter())
        .for_each(|(vis, cell_type)| {
            if *cell_type == CellType::Intersect {
                *vis = true;
            }
        });

    let seed_pt = fluid_seed_point;

    if let Some(seed_id) = get_global_id_from_phys(forest, geom, seed_pt) {
        if cell_types[seed_id] == CellType::Intersect {
            return Err(
                "The fluid seed point `fluid_seed_point` lies on an intersect (wall) cell; specify a seed clearly inside the fluid region.".to_string(),
            );
        }

        let mut queue = VecDeque::new();
        queue.push_back(seed_id);
        visited[seed_id] = true;
        cell_types[seed_id] = CellType::Fluid;

        while let Some(curr_id) = queue.pop_front() {
            for dir in Direction::ALL {
                let nbrs = forest.face_neighbour_global_ids(curr_id, dir);
                for nbr_id in nbrs {
                    if !visited[nbr_id] {
                        visited[nbr_id] = true;
                        cell_types[nbr_id] = CellType::Fluid;
                        queue.push_back(nbr_id);
                    }
                }
            }
        }
    } else {
        return Err(
            "The fluid seed point `fluid_seed_point` lies outside the computational domain (bounding box)."
                .to_string(),
        );
    }
    visited
        .iter()
        .zip(cell_types.iter_mut())
        .for_each(|(vis, cell_type)| {
            if !*vis && *cell_type != CellType::Intersect {
                *cell_type = CellType::Solid;
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_geometry::BoundingBox;

    #[test]
    fn err_when_seed_outside_domain() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells();
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [1, 1, 1]);
        let mut types = vec![CellType::Solid];

        let err = flood_fill_inside_outside(&forest, &geom, &mut types, [2.0, 0.5, 0.5])
            .unwrap_err();
        assert!(err.contains("outside"));
    }

    #[test]
    fn err_when_seed_on_intersect_cell() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells();
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [1, 1, 1]);
        let mut types = vec![CellType::Intersect];

        assert!(flood_fill_inside_outside(&forest, &geom, &mut types, [0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn fills_single_solid_cell_from_interior_seed() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells();
        let bbox = BoundingBox::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [1, 1, 1]);
        let mut types = vec![CellType::Solid];

        flood_fill_inside_outside(&forest, &geom, &mut types, [0.5, 0.5, 0.5]).unwrap();
        assert_eq!(types[0], CellType::Fluid);
    }
}
