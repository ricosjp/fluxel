//! Packed cell keys: a [`Key`] is a single `u128` that stores `tree_id`, Morton code, and `level`,
//! so lexicographic order matches space-filling-curve order in the octree.

use crate::morton;
use crate::MAX_LEVEL;

/// Human-readable cell in logical coordinates: the expanded form of a [`Key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalCell {
    /// Identifier of the root tree / forest this cell belongs to.
    pub tree_id: u32,
    /// Logical X coordinate in `0 ..= u32::MAX` (cell corner or origin, depending on convention).
    pub x: u32,
    /// Logical Y coordinate in `0 ..= u32::MAX`.
    pub y: u32,
    /// Logical Z coordinate in `0 ..= u32::MAX`.
    pub z: u32,
    /// Refinement depth: `0` is the coarsest root cell; larger values are finer octants.
    pub level: u8,
}

impl LogicalCell {
    /// Constructs a logical cell and asserts that `(x, y, z)` is aligned to the cell size at `level`.
    ///
    /// At level `L`, each coordinate must be a multiple of `2^(MAX_LEVEL - L)` (equivalently, the
    /// least significant `(MAX_LEVEL - L)` bits are zero). For `L == 0` the only `u32` multiple of
    /// `2^MAX_LEVEL` is `0`, so the root cell is anchored at the origin.
    pub fn new(tree_id: u32, x: u32, y: u32, z: u32, level: u8) -> Self {
        assert!(level <= MAX_LEVEL, "Level exceeds maximum of 32");

        // At level L, coordinates must be multiples of 2^(MAX_LEVEL-L): mask the low (MAX_LEVEL-L) bits.
        let mask = if level == 0 {
            u32::MAX
        } else {
            (1u32 << (MAX_LEVEL - level)) - 1
        };

        assert!(
            (x & mask) == 0,
            "x is not aligned to cell size at this level"
        );
        assert!(
            (y & mask) == 0,
            "y is not aligned to cell size at this level"
        );
        assert!(
            (z & mask) == 0,
            "z is not aligned to cell size at this level"
        );

        Self {
            tree_id,
            x,
            y,
            z,
            level,
        }
    }

    /// Root cell for `tree_id`: origin `(0, 0, 0)` at level `0`.
    pub fn root(tree_id: u32) -> Self {
        Self::new(tree_id, 0, 0, 0, 0)
    }

    /// Edge length of a cell at this level in logical space (number of finest voxels along an axis).
    ///
    /// At level `0` this is `2^MAX_LEVEL`, so the result is `u64` to avoid overflow.
    pub fn size(&self) -> u64 {
        1u64 << (MAX_LEVEL - self.level)
    }
}

/// Packed cell identifier: canonical storage and ordering key for octree cells.
///
/// Bit layout (low to high):
///
/// `[ tree_id (26) | morton_code (96) | level (6) ]`
///
/// [`Ord`] on `Key` matches a full space-filling-curve order: compare the raw `u128` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(u128);

impl Key {
    const TREE_BITS: u8 = 26;
    const MORTON_BITS: u8 = 96;
    const LEVEL_BITS: u8 = 6;

    const LEVEL_MASK: u128 = (1 << Self::LEVEL_BITS) - 1;
    // Use `1u128` so shifts stay in `u128` and do not overflow narrower integer types.
    const MORTON_MASK: u128 = ((1u128 << Self::MORTON_BITS) - 1) << Self::LEVEL_BITS;
    const TREE_MASK: u128 =
        ((1u128 << Self::TREE_BITS) - 1) << (Self::LEVEL_BITS + Self::MORTON_BITS);

    /// Packs `tree_id`, `morton`, and `level` into a [`Key`].
    ///
    /// # Panics
    ///
    /// Debug builds: `debug_assert!` if `tree_id`, `morton`, or `level` exceed their bit widths.
    #[inline(always)]
    pub fn new(tree_id: u32, morton: u128, level: u8) -> Self {
        debug_assert!(tree_id < (1 << Self::TREE_BITS), "tree_id exceeds 26 bits");
        debug_assert!(
            morton < (1 << Self::MORTON_BITS),
            "morton_code exceeds 96 bits"
        );
        debug_assert!(level <= MAX_LEVEL, "level exceeds 32");

        let packed = ((tree_id as u128) << (Self::LEVEL_BITS + Self::MORTON_BITS))
            | (morton << Self::LEVEL_BITS)
            | (level as u128);
        Key(packed)
    }

    /// Returns the 26-bit tree identifier field.
    #[inline(always)]
    pub fn tree_id(&self) -> u32 {
        ((self.0 & Self::TREE_MASK) >> (Self::LEVEL_BITS + Self::MORTON_BITS)) as u32
    }

    /// Returns the 96-bit Morton code (interleaved `x`, `y`, `z` from [`crate::morton::encode`]).
    #[inline(always)]
    pub fn morton(&self) -> u128 {
        (self.0 & Self::MORTON_MASK) >> Self::LEVEL_BITS
    }

    /// Returns the 6-bit refinement level (same meaning as [`LogicalCell::level`]).
    #[inline(always)]
    pub fn level(&self) -> u8 {
        (self.0 & Self::LEVEL_MASK) as u8
    }

    /// Builds a [`Key`] from a [`LogicalCell`] by encoding `(x, y, z)` with [`crate::morton::encode`].
    pub fn from_logical(cell: &LogicalCell) -> Self {
        let morton = morton::encode(cell.x, cell.y, cell.z);
        Self::new(cell.tree_id, morton, cell.level)
    }

    /// Decodes this key to a [`LogicalCell`] (decodes the Morton field with [`crate::morton::decode`]).
    pub fn to_logical(&self) -> LogicalCell {
        let (x, y, z) = morton::decode(self.morton());
        LogicalCell {
            tree_id: self.tree_id(),
            x,
            y,
            z,
            level: self.level(),
        }
    }

    /// Root key for `tree_id`: zero Morton code and level `0`.
    pub fn root(tree_id: u32) -> Self {
        Self::new(tree_id, 0, 0)
    }

    /// Parent cell in Morton space using only bit operations on the packed key, in O(1).
    ///
    /// Returns `None` if this key is already at level `0`.
    pub fn parent(&self) -> Option<Self> {
        let level = self.level();
        if level == 0 {
            return None;
        }
        let parent_level = level - 1;

        // Parent Morton code: clear the lowest child-selector bits for this subdivision step.
        let shift = 3 * (MAX_LEVEL - parent_level) as u32;
        let mask = u128::MAX << shift;
        let parent_morton = self.morton() & mask;

        Some(Self::new(self.tree_id(), parent_morton, parent_level))
    }

    /// The eight child cells in Morton space using only bit operations on the packed key, in O(1).
    ///
    /// # Panics
    ///
    /// If `self.level() == MAX_LEVEL`, subdivision is impossible and this function panics.
    pub fn children(&self) -> [Self; 8] {
        let level = self.level();
        assert!(level < MAX_LEVEL, "Cannot subdivide beyond MAX_LEVEL");
        let child_level = level + 1;

        let shift = 3 * (MAX_LEVEL - child_level) as u32;
        let base_morton = self.morton();
        let tree_id = self.tree_id();

        [
            Self::new(tree_id, base_morton | (0 << shift), child_level),
            Self::new(tree_id, base_morton | (1 << shift), child_level),
            Self::new(tree_id, base_morton | (2 << shift), child_level),
            Self::new(tree_id, base_morton | (3 << shift), child_level),
            Self::new(tree_id, base_morton | (4 << shift), child_level),
            Self::new(tree_id, base_morton | (5 << shift), child_level),
            Self::new(tree_id, base_morton | (6 << shift), child_level),
            Self::new(tree_id, base_morton | (7 << shift), child_level),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_packing() {
        let expected_tree = 12345;
        let expected_morton = 0x1234_5678_9ABC_DEF0_1234_5678;
        let expected_level = 15;

        let key = Key::new(expected_tree, expected_morton, expected_level);

        assert_eq!(key.tree_id(), expected_tree);
        assert_eq!(key.morton(), expected_morton);
        assert_eq!(key.level(), expected_level);
    }

    #[test]
    fn test_key_sorting() {
        // Parent cell (L=1)
        let parent = Key::from_logical(&LogicalCell::new(0, 0, 0, 0, 1));
        // Child 0 (L=2, same spatial origin as parent)
        let child_0 = Key::from_logical(&LogicalCell::new(0, 0, 0, 0, 2));
        // Child 1 (L=2, offset in +x)
        let child_1 = Key::from_logical(&LogicalCell::new(0, 1 << 30, 0, 0, 2));

        let mut keys = vec![child_1, parent, child_0];
        keys.sort();

        // Same Morton prefix: shallower level sorts first; otherwise Morton order in space.
        assert_eq!(keys, vec![parent, child_0, child_1]);
    }

    #[test]
    fn test_algebraic_parent_children() {
        let parent_logical = LogicalCell::new(0, 1 << 31, 1 << 31, 0, 1);
        let parent_key = Key::from_logical(&parent_logical);

        let children_keys = parent_key.children();

        // Eight-way split: check first and last child
        let child_0_logical = children_keys[0].to_logical();
        assert_eq!(child_0_logical.level, 2);
        assert_eq!(child_0_logical.x, 1 << 31);
        assert_eq!(child_0_logical.y, 1 << 31);
        assert_eq!(child_0_logical.z, 0);

        let child_7_logical = children_keys[7].to_logical();
        assert_eq!(child_7_logical.level, 2);
        assert_eq!(child_7_logical.x, (1 << 31) + (1 << 30));
        assert_eq!(child_7_logical.y, (1 << 31) + (1 << 30));
        assert_eq!(child_7_logical.z, 1 << 30);

        assert_eq!(children_keys[7].parent().unwrap(), parent_key);
    }
}
