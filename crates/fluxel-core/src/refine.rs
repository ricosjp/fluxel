//! Bulk refine and coarsen in linear time over the sorted key array.

use crate::forest::Forest;
use fluxel_sfc::MAX_LEVEL;

impl Forest {
    /// Refines selected cells: for each `true` flag, replaces the key with its eight children.
    ///
    /// Skips refinement when `key.level() == MAX_LEVEL`. Replacing a parent by its children in
    /// SFC order keeps the array sorted (`O(N)` pass).
    pub fn refine_by_flags(&mut self, flags: &[bool]) {
        assert_eq!(
            self.keys.len(),
            flags.len(),
            "Flags length must match the number of cells"
        );

        let mut new_keys = Vec::with_capacity(self.keys.len());

        for (key, &should_refine) in self.keys.iter().zip(flags.iter()) {
            if should_refine && key.level() < MAX_LEVEL {
                new_keys.extend_from_slice(&key.children());
            } else {
                new_keys.push(*key);
            }
        }

        self.keys = new_keys;
    }

    /// Coarsens where eight consecutive siblings are present, flagged, and contiguous in sort order.
    ///
    /// Requirements for a merge at index `i`:
    /// - `flags[i..i+8]` are all `true`,
    /// - `keys[i..i+8]` equal `key.parent().unwrap().children()`,
    /// - the cell is not the root (`level > 0`).
    ///
    /// Those eight keys are replaced by their parent (`O(N)` single pass).
    pub fn coarsen_by_flags(&mut self, flags: &[bool]) {
        assert_eq!(
            self.keys.len(),
            flags.len(),
            "Flags length must match the number of cells"
        );

        let mut new_keys = Vec::with_capacity(self.keys.len());
        let mut i = 0;

        while i < self.keys.len() {
            let key = self.keys[i];

            if flags[i] && key.level() > 0 && i + 8 <= self.keys.len() {
                let parent = key.parent().unwrap();
                let expected_children = parent.children();

                let can_coarsen =
                    (0..8).all(|j| self.keys[i + j] == expected_children[j] && flags[i + j]);

                if can_coarsen {
                    new_keys.push(parent);
                    i += 8;
                    continue;
                }
            }

            new_keys.push(key);
            i += 1;
        }

        self.keys = new_keys;
    }

    /// Uniformly refines all cells by the specified number of times.
    pub fn uniform_refinement(&mut self, n_times: u8) {
        (0..n_times).for_each(|_| {
            let refine_flags = vec![true; self.num_cells()];
            self.refine_by_flags(&refine_flags);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_and_coarsen() {
        let mut forest = Forest::new([1, 1, 1]);
        forest.populate_root_cells(); // Single L=0 cell

        assert_eq!(forest.num_cells(), 1);

        // --- Refine ---
        // Split the L=0 cell into L=1 (8 cells)
        forest.refine_by_flags(&[true]);
        assert_eq!(forest.num_cells(), 8);

        // Refine only the first cell to L=2
        // First cell becomes 8 cells; the other 7 stay → 8 + 7 = 15 cells
        let mut refine_flags = vec![false; 8];
        refine_flags[0] = true;
        forest.refine_by_flags(&refine_flags);
        assert_eq!(forest.num_cells(), 15);

        // First 8 keys should be L=2; the rest L=1
        (0..8).for_each(|i| assert_eq!(forest.keys()[i].level(), 2));
        (8..15).for_each(|i| assert_eq!(forest.keys()[i].level(), 1));

        // --- Coarsen ---
        // Coarsen the first 8 L=2 cells back to one L=1 cell
        // Set all flags true on purpose (trailing L=1 cells without full sibling sets should not coarsen)
        let coarsen_flags = vec![true; 15];
        forest.coarsen_by_flags(&coarsen_flags);

        // First 8 merge to 1; the other 7 unchanged → 1 + 7 = 8 cells
        // (The 7 trailing L=1 cells cannot merge because their original L=1 sibling is still subdivided)
        assert_eq!(forest.num_cells(), 8);
        assert_eq!(forest.keys()[0].level(), 1);

        // Coarsen with all flags true again: eight L=1 siblings are complete → back to L=0
        let coarsen_flags_2 = vec![true; 8];
        forest.coarsen_by_flags(&coarsen_flags_2);

        assert_eq!(forest.num_cells(), 1);
        assert_eq!(forest.keys()[0].level(), 0);
    }
}
