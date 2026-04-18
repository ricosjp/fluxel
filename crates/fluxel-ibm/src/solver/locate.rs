//! Map world-space points to forest global cell indices.

use fluxel_core::Forest;
use fluxel_geometry::Geometry;

/// Resolves physical coordinates `[x, y, z]` to the finest cell `global_id` for the current
/// [`Forest`] and [`Geometry`].
pub fn get_global_id_from_phys(forest: &Forest, geom: &Geometry, p: [f64; 3]) -> Option<usize> {
    let dx = p[0] - geom.global_bbox.min[0];
    let dy = p[1] - geom.global_bbox.min[1];
    let dz = p[2] - geom.global_bbox.min[2];

    if dx < 0.0 || dy < 0.0 || dz < 0.0 {
        return None;
    }

    let tx = (dx / geom.tree_size[0]) as u32;
    let ty = (dy / geom.tree_size[1]) as u32;
    let tz = (dz / geom.tree_size[2]) as u32;

    if tx >= geom.base_res[0] || ty >= geom.base_res[1] || tz >= geom.base_res[2] {
        return None;
    }

    let tree_id = tx + ty * geom.base_res[0] + tz * geom.base_res[0] * geom.base_res[1];
    let local_x = dx % geom.tree_size[0];
    let local_y = dy % geom.tree_size[1];
    let local_z = dz % geom.tree_size[2];

    let max_logical = (1u64 << 32) as f64;
    let lx = ((local_x / geom.tree_size[0]) * max_logical) as u32;
    let ly = ((local_y / geom.tree_size[1]) * max_logical) as u32;
    let lz = ((local_z / geom.tree_size[2]) * max_logical) as u32;

    if let Some(target_key) = forest.find_cell_containing(tree_id, lx, ly, lz) {
        if let Ok(global_id) = forest.keys().binary_search(&target_key) {
            return Some(global_id);
        }
    }
    None
}
