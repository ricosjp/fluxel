//! BFS flood fill for fluid/solid labeling.

use super::locate::get_global_id_from_phys;
use crate::types::CellType;
use fluxel_core::neighbour::Direction;
use fluxel_core::Forest;
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

    let dirs = [
        Direction::XMinus,
        Direction::XPlus,
        Direction::YMinus,
        Direction::YPlus,
        Direction::ZMinus,
        Direction::ZPlus,
    ];
    let seed_pt = fluid_seed_point;

    if let Some(seed_id) = get_global_id_from_phys(forest, geom, seed_pt) {
        if cell_types[seed_id] == CellType::Intersect {
            return Err("指定されたシードポイント (fluid_seed_point) が壁面 (Intersectセル) 上にあります。流体領域の内部を明確に指定してください。".to_string());
        }

        let mut queue = VecDeque::new();
        queue.push_back(seed_id);
        visited[seed_id] = true;
        cell_types[seed_id] = CellType::Fluid;

        while let Some(curr_id) = queue.pop_front() {
            for &dir in &dirs {
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
        return Err("指定されたシードポイント (fluid_seed_point) が計算領域 (BoundingBox) の外部にあります。".to_string());
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
