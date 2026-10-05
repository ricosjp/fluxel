//! Cylindrical point location: a world point to logical octree coordinates.
//!
//! This is the counterpart of [`crate::cartesian::locate`]. The returned
//! coordinates still go through the forest search in [`crate::SpatialDomain::locate`].

use super::domain::CylindricalGeometry;
use fluxel_sfc::MAX_LEVEL;

impl CylindricalGeometry {
    /// Logical octree coordinates of a world point, or None when it is outside.
    pub fn logical_point(
        &self,
        world: [f64; 3],
        base_res: [u32; 3],
    ) -> Option<(u32, u32, u32, u32)> {
        let [radius, theta, z] = self.param_from_world(world)?;
        let [nr, nt, nz] = base_res;
        if nr == 0 || nt == 0 || nz == 0 {
            return None;
        }
        let u = (radius - self.r_min()) / (self.r_max() - self.r_min());
        let v = (theta - self.theta_start()) / self.theta_extent();
        let w = (z - self.z_min()) / (self.z_max() - self.z_min());
        let tree = |fraction: f64, count: u32| -> Option<u32> {
            if !(0.0..1.0).contains(&fraction) {
                return None;
            }
            let index = (fraction * f64::from(count)) as u32;
            Some(index.min(count - 1))
        };
        let tx = tree(u, nr)?;
        let ty = tree(v, nt)?;
        let tz = tree(w, nz)?;
        let local = |fraction: f64, count: u32, tree_index: u32| {
            let within = fraction * f64::from(count) - f64::from(tree_index);
            let scaled = within * (1u64 << MAX_LEVEL) as f64;
            scaled.clamp(0.0, u32::MAX as f64) as u32
        };
        let tree_id = tx + ty * nr + tz * nr * nt;
        Some((
            tree_id,
            local(u, nr, tx),
            local(v, nt, ty),
            local(w, nz, tz),
        ))
    }
}
