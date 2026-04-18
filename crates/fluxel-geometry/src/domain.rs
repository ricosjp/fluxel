//! Axis-aligned bounding boxes and mapping from logical octree cells to physical space.

use fluxel_sfc::{LogicalCell, MAX_LEVEL};

/// Physical axis-aligned bounding box of a region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Minimum corner `(x, y, z)`.
    pub min: [f64; 3],
    /// Maximum corner `(x, y, z)` (strictly greater than `min` per axis).
    pub max: [f64; 3],
}

impl BoundingBox {
    /// Constructs a box with `min < max` on every axis.
    ///
    /// # Panics
    ///
    /// If any axis has `min >= max`.
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        assert!(
            min[0] < max[0] && min[1] < max[1] && min[2] < max[2],
            "Invalid BoundingBox bounds"
        );
        Self { min, max }
    }
}

/// Maps the global logical domain (`0..2^MAX_LEVEL` per tree) to physical coordinates.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// World-space AABB covering all trees.
    pub global_bbox: BoundingBox,
    /// Number of root trees along each axis `[Nx, Ny, Nz]`.
    pub base_res: [u32; 3],
    /// Physical edge length of one tree’s logical cube `[dx, dy, dz]`.
    pub tree_size: [f64; 3],
}

impl Geometry {
    /// Builds geometry from the world AABB and base tree grid resolution.
    ///
    /// Each tree occupies an equal sub-box of `global_bbox`; `tree_size` is that box’s extent.
    pub fn new(global_bbox: BoundingBox, base_res: [u32; 3]) -> Self {
        let tree_dx = (global_bbox.max[0] - global_bbox.min[0]) / base_res[0] as f64;
        let tree_dy = (global_bbox.max[1] - global_bbox.min[1]) / base_res[1] as f64;
        let tree_dz = (global_bbox.max[2] - global_bbox.min[2]) / base_res[2] as f64;

        Self {
            global_bbox,
            base_res,
            tree_size: [tree_dx, tree_dy, tree_dz],
        }
    }

    /// Returns the physical **center** and **edge lengths** of `cell`’s AABB.
    ///
    /// The logical cell is interpreted as a box from `(x,y,z)` with edge length `cell.size()`
    /// in normalized logical units of width `2^MAX_LEVEL` per tree axis.
    ///
    /// # Returns
    ///
    /// `(center, size)` where `center` and `size` are `[x, y, z]` in world units.
    pub fn cell_bounds(&self, cell: &LogicalCell) -> ([f64; 3], [f64; 3]) {
        let [nx, ny, _] = self.base_res;
        let t_id = cell.tree_id;

        // Tree index in the 3D root grid
        let tx = t_id % nx;
        let ty = (t_id / nx) % ny;
        let tz = t_id / (nx * ny);

        // Physical origin (min corner) of this tree’s box
        let tree_min_x = self.global_bbox.min[0] + (tx as f64) * self.tree_size[0];
        let tree_min_y = self.global_bbox.min[1] + (ty as f64) * self.tree_size[1];
        let tree_min_z = self.global_bbox.min[2] + (tz as f64) * self.tree_size[2];

        // Logical cell edge length and full logical width per axis (2^MAX_LEVEL)
        let l_size = cell.size() as f64;
        let max_logical = (1u64 << MAX_LEVEL) as f64;

        // Fraction of the tree’s logical extent occupied by this cell (0.0 ..= 1.0)
        let ratio = l_size / max_logical;

        // Physical cell size
        let cell_dx = self.tree_size[0] * ratio;
        let cell_dy = self.tree_size[1] * ratio;
        let cell_dz = self.tree_size[2] * ratio;

        // Logical centre (corner + half edge in logical integer space)
        let l_cx = cell.x as f64 + l_size / 2.0;
        let l_cy = cell.y as f64 + l_size / 2.0;
        let l_cz = cell.z as f64 + l_size / 2.0;

        // Map logical centre to world coordinates
        let cx = tree_min_x + self.tree_size[0] * (l_cx / max_logical);
        let cy = tree_min_y + self.tree_size[1] * (l_cy / max_logical);
        let cz = tree_min_z + self.tree_size[2] * (l_cz / max_logical);

        ([cx, cy, cz], [cell_dx, cell_dy, cell_dz])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_mapping() {
        // World box 2×2×2 centred at the origin
        let bbox = BoundingBox::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        // Split into 2×2×2 = 8 trees along X, Y, Z
        let geom = Geometry::new(bbox, [2, 2, 2]);

        // Each tree should be 1×1×1 in physical space
        assert_eq!(geom.tree_size, [1.0, 1.0, 1.0]);

        // Tree 0 (front-lower-left): L=0 cell is the whole tree
        let root_0 = LogicalCell::new(0, 0, 0, 0, 0);
        let (c0, s0) = geom.cell_bounds(&root_0);
        assert_eq!(s0, [1.0, 1.0, 1.0]);
        assert_eq!(c0, [-0.5, -0.5, -0.5]);

        // Tree 7 (back-upper-right): L=0 cell
        let root_7 = LogicalCell::new(7, 0, 0, 0, 0);
        let (c7, s7) = geom.cell_bounds(&root_7);
        assert_eq!(s7, [1.0, 1.0, 1.0]);
        assert_eq!(c7, [0.5, 0.5, 0.5]);

        // Tree 0: L=1 cell (one octant of the tree)
        let child_logical_max = 1u32 << 31;
        // Inside tree 0, the L=1 cell at the back-upper-right octant
        let l1_cell = LogicalCell::new(
            0,
            child_logical_max,
            child_logical_max,
            child_logical_max,
            1,
        );
        let (cl1, sl1) = geom.cell_bounds(&l1_cell);

        // Half the tree size (0.5)
        assert_eq!(sl1, [0.5, 0.5, 0.5]);
        // Tree 0 spans [-1, 0]; back-upper-right octant centre is [-0.25, -0.25, -0.25]
        assert_eq!(cl1, [-0.25, -0.25, -0.25]);
    }
}
