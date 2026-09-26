//! Bulk refine and coarsen in linear time over the sorted key array.

use crate::{Forest, ForestError};
use fluxel_sfc::MAX_LEVEL;

impl Forest {
    /// Refines selected cells: for each `true` flag, replaces the key with its eight children.
    ///
    /// Skips refinement when `key.level() == MAX_LEVEL`. Replacing a parent by its children in
    /// SFC order keeps the array sorted (`O(N)` pass).
    /// Flags refer to current key order. Does not automatically restore balance.
    /// Returns InvalidFlags for length mismatch or CellLimit for overflow/limit excess,
    /// before mutating the keys. Successful refinement changes subsequent indices.
    pub fn refine_by_flags(&mut self, flags: &[bool]) -> Result<(), ForestError> {
        if self.keys.len() != flags.len() {
            return Err(ForestError::InvalidFlags);
        }

        let additions = self
            .keys
            .iter()
            .zip(flags)
            .filter(|(key, flag)| **flag && key.level() < MAX_LEVEL)
            .count();
        let count = additions
            .checked_mul(7)
            .and_then(|n| n.checked_add(self.keys.len()))
            .filter(|&n| n <= self.max_cells)
            .ok_or(ForestError::CellLimit {
                limit: self.max_cells,
            })?;
        let mut new_keys = Vec::with_capacity(count);

        for (key, &should_refine) in self.keys.iter().zip(flags.iter()) {
            if should_refine && key.level() < MAX_LEVEL {
                new_keys.extend_from_slice(&key.children());
            } else {
                new_keys.push(*key);
            }
        }

        self.keys = new_keys;
        Ok(())
    }

    /// Coarsens where eight consecutive siblings are present, flagged, and contiguous in sort order.
    ///
    /// Requirements for a merge at index `i`:
    /// - `flags[i..i+8]` are all `true`,
    /// - `keys[i..i+8]` equal `key.parent().unwrap().children()`,
    /// - the cell is not the root (`level > 0`).
    ///
    /// Those eight keys are replaced by their parent (`O(N)` single pass).
    /// Returns InvalidFlags for length mismatch without mutation. Incomplete sibling
    /// groups are ignored, not errors. May invalidate balance and changes cell indices.
    pub fn coarsen_by_flags(&mut self, flags: &[bool]) -> Result<(), ForestError> {
        if self.keys.len() != flags.len() {
            return Err(ForestError::InvalidFlags);
        }

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
        Ok(())
    }

    /// Uniformly refines all cells by the specified number of times.
    /// Zero is a no-op. Preserves existing balance without establishing it.
    /// Prechecks final level and count; LevelLimit or CellLimit leaves keys unchanged.
    pub fn uniform_refinement(&mut self, n_times: u8) -> Result<(), ForestError> {
        let max_level = self.keys.iter().map(|k| k.level()).max().unwrap_or(0);
        if n_times > MAX_LEVEL - max_level {
            return Err(ForestError::LevelLimit);
        }
        (0..n_times)
            .try_fold(self.num_cells(), |n, _| n.checked_mul(8))
            .filter(|&n| n <= self.max_cells)
            .ok_or(ForestError::CellLimit {
                limit: self.max_cells,
            })?;
        for _ in 0..n_times {
            self.refine_by_flags(&vec![true; self.num_cells()])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refine_and_coarsen() {
        let mut forest = Forest::new([1, 1, 1]).unwrap();
        forest.populate_root_cells(); // Single L=0 cell

        assert_eq!(forest.num_cells(), 1);

        // --- Refine ---
        // Split the L=0 cell into L=1 (8 cells)
        forest.refine_by_flags(&[true]).unwrap();
        assert_eq!(forest.num_cells(), 8);

        // Refine only the first cell to L=2
        // First cell becomes 8 cells; the other 7 stay → 8 + 7 = 15 cells
        let mut refine_flags = vec![false; 8];
        refine_flags[0] = true;
        forest.refine_by_flags(&refine_flags).unwrap();
        assert_eq!(forest.num_cells(), 15);

        // First 8 keys should be L=2; the rest L=1
        (0..8).for_each(|i| assert_eq!(forest.keys()[i].level(), 2));
        (8..15).for_each(|i| assert_eq!(forest.keys()[i].level(), 1));

        // --- Coarsen ---
        // Coarsen the first 8 L=2 cells back to one L=1 cell
        // Set all flags true on purpose (trailing L=1 cells without full sibling sets should not coarsen)
        let coarsen_flags = vec![true; 15];
        forest.coarsen_by_flags(&coarsen_flags).unwrap();

        // First 8 merge to 1; the other 7 unchanged → 1 + 7 = 8 cells
        // (The 7 trailing L=1 cells cannot merge because their original L=1 sibling is still subdivided)
        assert_eq!(forest.num_cells(), 8);
        assert_eq!(forest.keys()[0].level(), 1);

        // Coarsen with all flags true again: eight L=1 siblings are complete → back to L=0
        let coarsen_flags_2 = vec![true; 8];
        forest.coarsen_by_flags(&coarsen_flags_2).unwrap();

        assert_eq!(forest.num_cells(), 1);
        assert_eq!(forest.keys()[0].level(), 0);
    }
}
