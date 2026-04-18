//! Point location: binary search on the sorted key array.

use crate::forest::Forest;
use fluxel_sfc::{Key, MAX_LEVEL};

impl Forest {
    /// Finds the cell that contains the logical point `(tree_id, x, y, z)` in `O(log N)`.
    ///
    /// Uses a synthetic key at [`MAX_LEVEL`] as the search needle; the predecessor in sort
    /// order is usually the finest cell covering the point, then verified with an axis-aligned
    /// box test (sparse forests may have gaps, so geometric checks are required).
    pub fn find_cell_containing(&self, tree_id: u32, x: u32, y: u32, z: u32) -> Option<Key> {
        let morton = fluxel_sfc::morton::encode(x, y, z);

        // Hypothetical finest leaf at this Morton location.
        let search_key = Key::new(tree_id, morton, MAX_LEVEL);

        match self.keys.binary_search(&search_key) {
            Ok(idx) => Some(self.keys[idx]),
            Err(idx) => {
                // Insertion index `idx`: keys[..idx] < search_key < keys[idx..].
                // Predecessor `idx - 1` is the usual candidate for the containing cell.
                if idx == 0 {
                    return None;
                }

                let candidate = self.keys[idx - 1];
                if candidate.tree_id() != tree_id {
                    return None;
                }

                let logical = candidate.to_logical();
                let size = logical.size();

                let cx = logical.x as u64;
                let cy = logical.y as u64;
                let cz = logical.z as u64;
                let px = x as u64;
                let py = y as u64;
                let pz = z as u64;

                if px >= cx
                    && px < cx + size
                    && py >= cy
                    && py < cy + size
                    && pz >= cz
                    && pz < cz + size
                {
                    Some(candidate)
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_cell_containing() {
        let mut forest = Forest::new([1, 1, 1]);

        // Manually create the L=0 root cell (covering the entire domain) and mimic splitting it into four.
        // For convenience, here add only one L=1 child cell without removing the parent.
        // Normally, Refine keeps cell consistency, but here we manually construct the cells for Query testing.

        let root = Key::root(0);
        let child_keys = root.children();

        forest.keys = vec![child_keys[0], child_keys[1], child_keys[7]];
        forest.keys.sort();

        // Search for a point inside child_keys[0]
        let found = forest.find_cell_containing(0, 100, 100, 100).unwrap();
        assert_eq!(found, child_keys[0]);

        // Search for a point inside child_keys[7] (coordinates > 2^31)
        let mid = 1u32 << 31;
        let found2 = forest
            .find_cell_containing(0, mid + 100, mid + 100, mid + 100)
            .unwrap();
        assert_eq!(found2, child_keys[7]);

        // Sparse gap: no cell owns this region.
        let not_found = forest.find_cell_containing(0, 100, mid + 100, 100);
        assert!(not_found.is_none());
    }
}
