//! Face-adjacent neighbours and multi-block coordinate wrapping.

use crate::forest::Forest;
use fluxel_sfc::{Key, MAX_LEVEL};

/// One of the six axis-aligned face directions between octree cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    XMinus,
    XPlus,
    YMinus,
    YPlus,
    ZMinus,
    ZPlus,
}

impl Forest {
    /// Maps global logical coordinates to `(tree_id, local x, y, z)` when crossing tree bounds.
    ///
    /// Each tree owns a `[0, 2^MAX_LEVEL)` logical cube; coordinates outside wrap to neighbour trees
    /// using Euclidean division by `2^MAX_LEVEL`. Returns [`None`] if the point leaves the global
    /// `[0, nx) × [0, ny) × [0, nz)` tree grid.
    pub fn resolve_coord(
        &self,
        tree_id: u32,
        x: i64,
        y: i64,
        z: i64,
    ) -> Option<(u32, u32, u32, u32)> {
        let [nx, ny, nz] = self.base_res;
        let mut tx = (tree_id % nx) as i64;
        let mut ty = ((tree_id / nx) % ny) as i64;
        let mut tz = (tree_id / (nx * ny)) as i64;

        let max_val = 1i64 << MAX_LEVEL;

        tx += x.div_euclid(max_val);
        ty += y.div_euclid(max_val);
        tz += z.div_euclid(max_val);
        let new_x = x.rem_euclid(max_val);
        let new_y = y.rem_euclid(max_val);
        let new_z = z.rem_euclid(max_val);

        if tx < 0 || tx >= nx as i64 || ty < 0 || ty >= ny as i64 || tz < 0 || tz >= nz as i64 {
            return None;
        }

        let new_tree_id = (tx + ty * (nx as i64) + tz * (nx as i64) * (ny as i64)) as u32;
        Some((new_tree_id, new_x as u32, new_y as u32, new_z as u32))
    }

    /// Returns distinct cells that touch `key`'s face in `dir`, using four point queries.
    ///
    /// Under a 2:1 size rule, a face is covered by either one same-or-larger neighbour or four
    /// half-size cells. Samples are placed at the face quad centres, offset one unit along the
    /// outward normal, then resolved with [`Self::find_cell_containing`].
    pub fn neighbours_covering_face(&self, key: Key, dir: Direction) -> Vec<Key> {
        let logical = key.to_logical();
        let size = logical.size() as i64;

        let quarter = size / 4;
        let three_quarter = 3 * size / 4;

        let cx = logical.x as i64;
        let cy = logical.y as i64;
        let cz = logical.z as i64;

        let mut points = Vec::with_capacity(4);

        match dir {
            Direction::XMinus => {
                let px = cx - 1;
                points.push((px, cy + quarter, cz + quarter));
                points.push((px, cy + three_quarter, cz + quarter));
                points.push((px, cy + quarter, cz + three_quarter));
                points.push((px, cy + three_quarter, cz + three_quarter));
            }
            Direction::XPlus => {
                let px = cx + size;
                points.push((px, cy + quarter, cz + quarter));
                points.push((px, cy + three_quarter, cz + quarter));
                points.push((px, cy + quarter, cz + three_quarter));
                points.push((px, cy + three_quarter, cz + three_quarter));
            }
            Direction::YMinus => {
                let py = cy - 1;
                points.push((cx + quarter, py, cz + quarter));
                points.push((cx + three_quarter, py, cz + quarter));
                points.push((cx + quarter, py, cz + three_quarter));
                points.push((cx + three_quarter, py, cz + three_quarter));
            }
            Direction::YPlus => {
                let py = cy + size;
                points.push((cx + quarter, py, cz + quarter));
                points.push((cx + three_quarter, py, cz + quarter));
                points.push((cx + quarter, py, cz + three_quarter));
                points.push((cx + three_quarter, py, cz + three_quarter));
            }
            Direction::ZMinus => {
                let pz = cz - 1;
                points.push((cx + quarter, cy + quarter, pz));
                points.push((cx + three_quarter, cy + quarter, pz));
                points.push((cx + quarter, cy + three_quarter, pz));
                points.push((cx + three_quarter, cy + three_quarter, pz));
            }
            Direction::ZPlus => {
                let pz = cz + size;
                points.push((cx + quarter, cy + quarter, pz));
                points.push((cx + three_quarter, cy + quarter, pz));
                points.push((cx + quarter, cy + three_quarter, pz));
                points.push((cx + three_quarter, cy + three_quarter, pz));
            }
        }

        let mut neighbours = Vec::new();
        points
            .into_iter()
            .filter_map(|(px, py, pz)| {
                self.resolve_coord(logical.tree_id, px, py, pz)
                    .and_then(|(t_id, x, y, z)| self.find_cell_containing(t_id, x, y, z))
            })
            .for_each(|neighbour_key| {
                if !neighbours.contains(&neighbour_key) {
                    neighbours.push(neighbour_key);
                }
            });

        neighbours
    }

    /// Returns global forest indices of cells face-adjacent to `cell_idx` in `dir`.
    ///
    /// Uses [`Self::neighbours_covering_face`] and maps each neighbour [`Key`] to its index in
    /// the sorted key array.
    pub fn face_neighbour_global_ids(&self, cell_idx: usize, dir: Direction) -> Vec<usize> {
        let key = self.keys()[cell_idx];
        let nbr_keys = self.neighbours_covering_face(key, dir);
        nbr_keys
            .into_iter()
            .filter_map(|nk| self.keys().binary_search(&nk).ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_coord_cross_tree() {
        // Two trees along +X
        let forest = Forest::new([2, 1, 1]);

        let logical_max = (1i64 << MAX_LEVEL) - 1;

        // Cross tree 0's X+ face → tree 1 at x = 0
        let res1 = forest.resolve_coord(0, logical_max + 1, 0, 0).unwrap();
        assert_eq!(res1, (1, 0, 0, 0));

        // Cross tree 1's X- face → tree 0 at x = logical_max
        let res2 = forest.resolve_coord(1, -1, 0, 0).unwrap();
        assert_eq!(res2, (0, logical_max as u32, 0, 0));

        // Cross tree 1's X+ face → outside the domain
        let res3 = forest.resolve_coord(1, logical_max + 1, 0, 0);
        assert!(res3.is_none());
    }

    #[test]
    fn test_face_neighbours_different_levels() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells();

        // 1. Refine the whole domain to L=1 (8 cells)
        forest.refine_by_flags(&[true]);

        // 2. Refine one cell on the +X side (child_1) to L=2 — in Morton order it is index 1
        let mut refine_flags = vec![false; 8];
        refine_flags[1] = true;
        forest.refine_by_flags(&refine_flags);

        // keys has 15 cells (7 at L=1 and 8 at L=2)
        assert_eq!(forest.num_cells(), 15);

        // Neighbours across the X+ face of child_0 (index 0, L=1, front-lower-left)
        let child_0_key = forest.keys()[0];
        let neighbours = forest.neighbours_covering_face(child_0_key, Direction::XPlus);

        // child_1 was refined to L=2, so child_0's X+ face should touch four L=2 cells
        assert_eq!(neighbours.len(), 4);
        neighbours
            .into_iter()
            .for_each(|n| assert_eq!(n.level(), 2));
    }
}
