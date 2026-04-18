//! 2:1 balance (26-neighbour stencil): refine coarse cells until adjacent level gaps are ≤ 1.

use crate::forest::Forest;
use rayon::prelude::*;

/// Iterates the 26 probe offsets around a cell of edge length `size` (logical units).
///
/// Uses `{-1, size/2, size}` per axis and skips the interior sample `(size/2, size/2, size/2)`.
/// Order matches nested `for dx { for dy { for dz { ... }}}` with `dz` innermost.
fn iter_26_probe_triples(size: i64) -> impl Iterator<Item = (i64, i64, i64)> {
    let half = size / 2;
    let o = [-1i64, half, size];
    (0..27).filter_map(move |k| {
        let dx = o[k / 9];
        let dy = o[(k / 3) % 3];
        let dz = o[k % 3];
        (dx != half || dy != half || dz != half).then_some((dx, dy, dz))
    })
}

/// Collects forest key indices that must be refined so that this cell satisfies the 2:1 rule.
fn balance_refine_indices_for_cell(forest: &Forest, i: usize) -> Vec<usize> {
    let key = forest.keys()[i];
    let logical = key.to_logical();
    let l = logical.level;

    if l < 2 {
        return Vec::new();
    }

    let size = logical.size() as i64;
    let cx = logical.x as i64;
    let cy = logical.y as i64;
    let cz = logical.z as i64;

    iter_26_probe_triples(size)
        .filter_map(|(dx, dy, dz)| {
            let px = cx + dx;
            let py = cy + dy;
            let pz = cz + dz;
            let (t_id, rx, ry, rz) = forest.resolve_coord(logical.tree_id, px, py, pz)?;
            let neighbor = forest.find_cell_containing(t_id, rx, ry, rz)?;
            if l < neighbor.level() + 2 {
                return None;
            }
            forest.keys.binary_search(&neighbor).ok()
        })
        .collect()
}

impl Forest {
    /// Enforces 2:1 level balance against the 26-neighbour stencil.
    ///
    /// When a cell at level `L` touches a neighbour at most `L-2`, the coarser neighbour is
    /// marked for [`Forest::refine_by_flags`]. Repeats until no flags are set.
    pub fn enforce_2_to_1_balance(&mut self) {
        loop {
            let mut flags = vec![false; self.keys.len()];
            let neighbor_sets: Vec<Vec<usize>> = (0..self.keys.len())
                .into_par_iter()
                .map(|i| balance_refine_indices_for_cell(self, i))
                .collect();

            let mut changed = false;
            for indices in neighbor_sets {
                for idx in indices {
                    if !flags[idx] {
                        flags[idx] = true;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }

            self.refine_by_flags(&flags);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2_to_1_balance() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells();

        // L=0 -> refine the whole domain to L=1 (8 cells)
        forest.refine_by_flags(&[true]);

        // Refine only the cell that contains the origin to L=2 (not always keys[0] in SFC order)
        let mut flags2 = vec![false; forest.num_cells()];
        let i2 = forest
            .keys()
            .binary_search(&forest.find_cell_containing(0, 0, 0, 0).unwrap())
            .unwrap();
        flags2[i2] = true;
        forest.refine_by_flags(&flags2);

        // Refine the first 8 keys to L=3 so the forest has a strong level imbalance to fix
        let mut flags3 = vec![false; forest.num_cells()];
        flags3.iter_mut().take(8).for_each(|f| *f = true);
        forest.refine_by_flags(&flags3);

        // Before enforcement, some 26-probe neighbour pair should differ by at least 2 levels
        let max_level_diff_before = compute_max_level_diff(&forest);
        assert!(max_level_diff_before >= 2);

        // Apply 2:1 balance
        forest.enforce_2_to_1_balance();

        // Adjacent level difference (26-probe sampling) must be at most 1
        let max_level_diff_after = compute_max_level_diff(&forest);
        assert!(max_level_diff_after <= 1);
    }

    /// Maximum absolute level difference found between a cell and a 26-probe neighbour.
    fn compute_max_level_diff(forest: &Forest) -> u8 {
        let mut max_diff = 0;

        for i in 0..forest.keys().len() {
            let key = forest.keys()[i];
            let logical = key.to_logical();
            let size = logical.size() as i64;
            let l = logical.level;
            let cx = logical.x as i64;
            let cy = logical.y as i64;
            let cz = logical.z as i64;

            for (dx, dy, dz) in iter_26_probe_triples(size) {
                let px = cx + dx;
                let py = cy + dy;
                let pz = cz + dz;

                if let Some((t_id, rx, ry, rz)) = forest.resolve_coord(logical.tree_id, px, py, pz)
                {
                    if let Some(neighbor) = forest.find_cell_containing(t_id, rx, ry, rz) {
                        let nl = neighbor.level();
                        let diff = l.abs_diff(nl);
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
            }
        }
        max_diff
    }
}
