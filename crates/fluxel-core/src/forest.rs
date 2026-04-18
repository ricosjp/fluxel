//! [`Forest`]: sparse multi-tree storage backed by a sorted key list.

use fluxel_sfc::Key;

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
}

impl Forest {
    /// Creates an empty forest with the given base resolution (number of root trees per axis).
    pub fn new(base_res: [u32; 3]) -> Self {
        Self {
            keys: Vec::new(),
            base_res,
        }
    }

    /// Fills the forest with one level-0 root cell per tree in a regular `[nx × ny × nz]` grid.
    ///
    /// Keys are emitted in `tree_id` order, which matches SFC order for this layout.
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
