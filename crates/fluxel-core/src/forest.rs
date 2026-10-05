//! [`Forest`]: sparse multi-tree storage backed by a sorted key list.

use crate::{validate_resolution, ForestError};
use fluxel_sfc::{Key, MAX_LEVEL};

/// Multi-block sparse forest: no explicit octree links, only a sorted [`Key`] vector.
///
/// [`Key`] ordering follows the space-filling curve, which is the canonical ordering for
/// searches and bulk updates.
#[derive(Debug, Clone)]
pub struct Forest {
    /// Cell keys in strict SFC (Morton) order.
    pub(crate) keys: Vec<Key>,
    /// Base grid resolution: number of root trees along each axis `[Nx, Ny, Nz]`.
    pub(crate) base_res: [u32; 3],
    pub(crate) max_cells: usize,
    /// Logical axes that wrap to the opposite side of the root grid.
    pub(crate) periodic: [bool; 3],
}

impl Forest {
    /// Creates an empty forest with the given base resolution (number of root trees per axis).
    /// Errors on invalid root resolution. Populate roots explicitly before building a mesh.
    pub fn new(base_res: [u32; 3]) -> Result<Self, ForestError> {
        Self::with_cell_limit(base_res, usize::MAX)
    }

    /// Create empty storage with a maximum leaf-cell count; does not populate roots.
    /// Rejects zero resolution components, more than 2^26 roots, or roots above the limit.
    /// The limit is checked before growth, not a cap on all allocations or byte usage.
    pub fn with_cell_limit(base_res: [u32; 3], max_cells: usize) -> Result<Self, ForestError> {
        let roots = validate_resolution(base_res)?;
        if roots > max_cells {
            return Err(ForestError::CellLimit { limit: max_cells });
        }
        Self::with_limit_and_period(base_res, max_cells, [false; 3])
    }

    /// Create an empty forest whose selected logical axes wrap across the root grid.
    ///
    /// `periodic` is `[x, y, z]`. A true component identifies the two outer faces of
    /// that axis. Root-count and cell-limit checks match [`Self::with_cell_limit`].
    pub fn with_periodic_axes(
        base_res: [u32; 3],
        periodic: [bool; 3],
    ) -> Result<Self, ForestError> {
        Self::with_limit_and_period(base_res, usize::MAX, periodic)
    }

    /// Periodic forest with the same leaf-cell limit as [`Self::with_cell_limit`].
    pub fn with_periodic_cell_limit(
        base_res: [u32; 3],
        max_cells: usize,
        periodic: [bool; 3],
    ) -> Result<Self, ForestError> {
        Self::with_limit_and_period(base_res, max_cells, periodic)
    }

    fn with_limit_and_period(
        base_res: [u32; 3],
        max_cells: usize,
        periodic: [bool; 3],
    ) -> Result<Self, ForestError> {
        let roots = validate_resolution(base_res)?;
        if roots > max_cells {
            return Err(ForestError::CellLimit { limit: max_cells });
        }
        Ok(Self {
            keys: Vec::new(),
            base_res,
            max_cells,
            periodic,
        })
    }

    /// Logical axes that wrap, in X/Y/Z order.
    pub fn periodic_axes(&self) -> [bool; 3] {
        self.periodic
    }

    /// Root-tree counts in X/Y/Z order.
    pub fn base_resolution(&self) -> [u32; 3] {
        self.base_res
    }

    /// Checks sorted, nonoverlapping, complete SFC coverage without allocating.
    /// Returns IncompleteCoverage for gaps, overlaps, bad ordering, or absent roots; never mutates.
    pub fn validate_coverage(&self) -> Result<(), ForestError> {
        let mut tree = 0u32;
        let mut position = 0u128;
        let width = 1u128 << (3 * MAX_LEVEL as u32);
        for key in &self.keys {
            if key.tree_id() != tree || key.morton() != position {
                return Err(ForestError::IncompleteCoverage);
            }
            let span = 1u128 << (3 * (MAX_LEVEL - key.level()) as u32);
            if !position.is_multiple_of(span) {
                return Err(ForestError::IncompleteCoverage);
            }
            position += span;
            if position == width {
                tree += 1;
                position = 0;
            }
        }
        if position != 0 || tree as usize != validate_resolution(self.base_res)? {
            return Err(ForestError::IncompleteCoverage);
        }
        Ok(())
    }

    /// Fills the forest with one level-0 root cell per tree in a regular `[nx × ny × nz]` grid.
    ///
    /// Keys are emitted in `tree_id` order, which matches SFC order for this layout.
    /// Replaces all existing refinement; root count was validated at construction.
    pub fn populate_root_cells(&mut self) {
        self.keys.clear();
        let [nx, ny, nz] = self.base_res;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let tree_id = x + y * nx + z * nx * ny;
                    self.keys.push(Key::root(tree_id));
                }
            }
        }
    }

    /// Returns the number of leaf cells (keys) currently stored.
    #[inline(always)]
    pub fn num_cells(&self) -> usize {
        self.keys.len()
    }

    /// Returns the sorted key slice.
    #[inline(always)]
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }
}
