//! Cylindrical domain: annular sector, parameter mapping, and cell intervals.
//!
//! This is the counterpart of [`crate::cartesian::domain`]. Point location lives
//! in [`super::locate`]. Cell and face measures live in [`super::metrics`].

use crate::GeometryError;
use fluxel_sfc::{LogicalCell, MAX_LEVEL};

/// Validated annular sector about an axis parallel to world Z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylindricalGeometry {
    origin: [f64; 3],
    r_min: f64,
    r_max: f64,
    theta_start: f64,
    theta_extent: f64,
    z_min: f64,
    z_max: f64,
    periodic_theta: bool,
}

/// Half-open parameter box `[min, max)` in `(r, θ, z)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterInterval {
    /// Inclusive lower corner `(r, θ, z)`.
    pub min: [f64; 3],
    /// Exclusive upper corner `(r, θ, z)`.
    pub max: [f64; 3],
}

impl CylindricalGeometry {
    /// Build a sector of positive inner radius that is shorter than a full turn.
    ///
    /// `theta_start` and `theta_extent` are radians. `r` and `z` use the same
    /// length units as world coordinates. The axis is parallel to world Z and
    /// passes through `origin`.
    ///
    /// # Errors
    /// Rejects nonfinite values, `r_min <= 0`, reversed or empty extents, and
    /// `theta_extent >= 2π`. A full turn needs periodic face connectivity, which
    /// this sector does not provide.
    pub fn new(
        origin: [f64; 3],
        r_min: f64,
        r_max: f64,
        theta_start: f64,
        theta_extent: f64,
        z_min: f64,
        z_max: f64,
    ) -> Result<Self, GeometryError> {
        let values = [
            origin[0],
            origin[1],
            origin[2],
            r_min,
            r_max,
            theta_start,
            theta_extent,
            z_min,
            z_max,
        ];
        if values.iter().any(|value| !value.is_finite())
            || r_min <= 0.0
            || r_min >= r_max
            || theta_extent <= 0.0
            || theta_extent >= std::f64::consts::TAU
            || z_min >= z_max
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        Ok(Self {
            origin,
            r_min,
            r_max,
            theta_start,
            theta_extent,
            z_min,
            z_max,
            periodic_theta: false,
        })
    }

    /// Sector, or a full turn when `theta_extent` is within `1e-9` of `2π`.
    ///
    /// # Errors
    /// Same rejections as [`Self::new`] or [`Self::full_turn`].
    pub fn from_extent(
        origin: [f64; 3],
        r_min: f64,
        r_max: f64,
        theta_start: f64,
        theta_extent: f64,
        z_min: f64,
        z_max: f64,
    ) -> Result<Self, GeometryError> {
        if (theta_extent - std::f64::consts::TAU).abs() <= 1e-9 {
            Self::full_turn(origin, r_min, r_max, theta_start, z_min, z_max)
        } else {
            Self::new(
                origin,
                r_min,
                r_max,
                theta_start,
                theta_extent,
                z_min,
                z_max,
            )
        }
    }

    /// Build a full turn about Z. The angular seam is one periodic face, not two boundaries.
    ///
    /// `theta_start` is the seam angle. Callers must pair this domain with a forest whose
    /// second logical axis is periodic and whose root angle is at most `π/2`.
    ///
    /// # Errors
    /// Rejects nonfinite values, `r_min <= 0`, and reversed or empty radial or axial extents.
    pub fn full_turn(
        origin: [f64; 3],
        r_min: f64,
        r_max: f64,
        theta_start: f64,
        z_min: f64,
        z_max: f64,
    ) -> Result<Self, GeometryError> {
        let values = [
            origin[0],
            origin[1],
            origin[2],
            r_min,
            r_max,
            theta_start,
            z_min,
            z_max,
        ];
        if values.iter().any(|value| !value.is_finite())
            || r_min <= 0.0
            || r_min >= r_max
            || z_min >= z_max
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        Ok(Self {
            origin,
            r_min,
            r_max,
            theta_start,
            theta_extent: std::f64::consts::TAU,
            z_min,
            z_max,
            periodic_theta: true,
        })
    }

    /// True when the angular ends are the same ray and must be connected periodically.
    pub fn is_full_turn(&self) -> bool {
        self.periodic_theta
    }

    /// World origin of the cylindrical axis.
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Inclusive inner radius.
    pub fn r_min(&self) -> f64 {
        self.r_min
    }

    /// Exclusive outer radius.
    pub fn r_max(&self) -> f64 {
        self.r_max
    }

    /// Inclusive lower axial bound, relative to the origin.
    pub fn z_min(&self) -> f64 {
        self.z_min
    }

    /// Exclusive upper axial bound, relative to the origin.
    pub fn z_max(&self) -> f64 {
        self.z_max
    }

    /// Inclusive start angle in radians.
    pub fn theta_start(&self) -> f64 {
        self.theta_start
    }

    /// Positive angular width in radians. A full turn is `2π` and is periodic.
    pub fn theta_extent(&self) -> f64 {
        self.theta_extent
    }

    /// Map `(r, θ, z)` to world `(x, y, z)`.
    pub fn world_from_param(&self, radius: f64, theta: f64, z: f64) -> [f64; 3] {
        [
            self.origin[0] + radius * theta.cos(),
            self.origin[1] + radius * theta.sin(),
            self.origin[2] + z,
        ]
    }

    /// Map a world point into the half-open sector, or return None when it is outside.
    ///
    /// `r` and `z` include the lower endpoint and exclude the upper endpoint.
    /// `θ` is lifted into `[theta_start, theta_start + theta_extent)`.
    pub fn param_from_world(&self, world: [f64; 3]) -> Option<[f64; 3]> {
        let dx = world[0] - self.origin[0];
        let dy = world[1] - self.origin[1];
        let radius = dx.hypot(dy);
        let z = world[2] - self.origin[2];
        if !(radius.is_finite() && z.is_finite())
            || radius < self.r_min
            || radius >= self.r_max
            || z < self.z_min
            || z >= self.z_max
        {
            return None;
        }
        let theta = lift_angle(dy.atan2(dx), self.theta_start, self.theta_extent)?;
        Some([radius, theta, z])
    }

    /// Right-handed orthonormal frame `(e_r, e_θ, e_z)` at `theta`.
    ///
    /// The frame does not depend on radius. Callers use it only for `r > 0`.
    pub fn local_frame(&self, theta: f64) -> [[f64; 3]; 3] {
        let _ = self;
        [
            [theta.cos(), theta.sin(), 0.0],
            [-theta.sin(), theta.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Parameter interval of one logical cell.
    ///
    /// Roots divide `(r, θ, z)` equally. `base_res` is the root count along those
    /// axes. The cell must use the same logical alignment as a Cartesian cell.
    ///
    /// # Errors
    /// Rejects an invalid root count, a root angle above `π/2`, or a tree id
    /// outside the root grid.
    pub fn cell_interval(
        &self,
        cell: &LogicalCell,
        base_res: [u32; 3],
    ) -> Result<ParameterInterval, GeometryError> {
        // Root box of this tree, then the child fraction inside that root.
        let root = self.root_extent(base_res)?;
        let [nx, ny, nz] = base_res;
        let n_trees = u64::from(nx) * u64::from(ny) * u64::from(nz);
        if u64::from(cell.tree_id) >= n_trees {
            return Err(GeometryError::InvalidResolution);
        }
        let tx = cell.tree_id % nx;
        let ty = (cell.tree_id / nx) % ny;
        let tz = cell.tree_id / (nx * ny);
        let tree_min = [
            self.r_min + f64::from(tx) * root[0],
            self.theta_start + f64::from(ty) * root[1],
            self.z_min + f64::from(tz) * root[2],
        ];
        let max_logical = (1u64 << MAX_LEVEL) as f64;
        let ratio = cell.size() as f64 / max_logical;
        let min = [
            tree_min[0] + root[0] * (cell.x as f64 / max_logical),
            tree_min[1] + root[1] * (cell.y as f64 / max_logical),
            tree_min[2] + root[2] * (cell.z as f64 / max_logical),
        ];
        let max = [
            min[0] + root[0] * ratio,
            min[1] + root[1] * ratio,
            min[2] + root[2] * ratio,
        ];
        Ok(ParameterInterval { min, max })
    }

    /// World AABB of a parameter interval, as center and full side lengths.
    ///
    /// The box includes angular extrema inside the interval, not only the four corners.
    pub fn world_aabb(&self, interval: ParameterInterval) -> ([f64; 3], [f64; 3]) {
        // Corners are not enough: the box must include the extreme angles inside the sector.
        let mut points = Vec::new();
        for radius in [interval.min[0], interval.max[0]] {
            for theta in extremal_angles(interval.min[1], interval.max[1]) {
                for z in [interval.min[2], interval.max[2]] {
                    points.push(self.world_from_param(radius, theta, z));
                }
            }
        }
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for point in points {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        let center = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        (center, size)
    }

    /// Largest world extent of the sector, used as the Gibou length scale.
    pub fn domain_length(&self) -> f64 {
        (2.0 * self.r_max).max(self.z_max - self.z_min)
    }
    fn root_extent(&self, base_res: [u32; 3]) -> Result<[f64; 3], GeometryError> {
        fluxel_core::validate_resolution(base_res).map_err(|_| GeometryError::InvalidResolution)?;
        let extent = [
            (self.r_max - self.r_min) / f64::from(base_res[0]),
            self.theta_extent / f64::from(base_res[1]),
            (self.z_max - self.z_min) / f64::from(base_res[2]),
        ];
        if extent[1] > std::f64::consts::FRAC_PI_2 {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        Ok(extent)
    }
}

fn lift_angle(theta: f64, start: f64, extent: f64) -> Option<f64> {
    let shifted = (theta - start).rem_euclid(std::f64::consts::TAU);
    if shifted >= extent {
        None
    } else {
        Some(start + shifted)
    }
}

fn extremal_angles(start: f64, end: f64) -> Vec<f64> {
    let mut angles = vec![start, end];
    let mut candidate = (start / std::f64::consts::FRAC_PI_2).ceil() * std::f64::consts::FRAC_PI_2;
    while candidate < end {
        if candidate > start {
            angles.push(candidate);
        }
        candidate += std::f64::consts::FRAC_PI_2;
    }
    angles
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_sfc::LogicalCell;

    fn sector() -> CylindricalGeometry {
        CylindricalGeometry::new(
            [1.0, 2.0, 3.0],
            1.0,
            2.0,
            -0.25,
            std::f64::consts::FRAC_PI_2,
            4.0,
            7.0,
        )
        .unwrap()
    }

    #[test]
    fn rejects_axis_and_full_turn() {
        assert!(CylindricalGeometry::new([0.0; 3], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0).is_err());
        assert!(
            CylindricalGeometry::new([0.0; 3], 0.5, 1.0, 0.0, std::f64::consts::TAU, 0.0, 1.0)
                .is_err()
        );
    }

    #[test]
    fn round_trips_interior_points_and_excludes_the_far_boundary() {
        let domain = sector();
        let theta = domain.theta_start() + 0.4;
        let world = domain.world_from_param(1.5, theta, 5.0);
        let parameter = domain.param_from_world(world).unwrap();
        assert!((parameter[0] - 1.5).abs() < 1e-12);
        assert!((parameter[1] - theta).abs() < 1e-12);
        assert!((parameter[2] - 5.0).abs() < 1e-12);
        assert!(domain
            .param_from_world(domain.world_from_param(domain.r_min(), theta, 5.0))
            .is_some());
        assert!(domain
            .param_from_world(domain.world_from_param(domain.r_max(), theta, 5.0))
            .is_none());
        let end = domain.theta_start() + domain.theta_extent();
        assert!(domain
            .param_from_world(domain.world_from_param(1.5, end, 5.0))
            .is_none());
    }

    #[test]
    fn lifts_angles_across_the_atan2_branch() {
        let domain =
            CylindricalGeometry::new([0.0; 3], 1.0, 2.0, std::f64::consts::PI, 0.5, 0.0, 1.0)
                .unwrap();
        let theta = std::f64::consts::PI + 0.1;
        let world = domain.world_from_param(1.2, theta, 0.3);
        let parameter = domain.param_from_world(world).unwrap();
        assert!((parameter[1] - theta).abs() < 1e-12);
    }
    #[test]
    fn root_angle_above_a_right_angle_is_rejected() {
        let domain =
            CylindricalGeometry::new([0.0; 3], 1.0, 2.0, 0.0, std::f64::consts::PI, 0.0, 1.0)
                .unwrap();
        assert!(domain
            .cell_interval(&LogicalCell::root(0), [1, 1, 1])
            .is_err());
        let interval = domain
            .cell_interval(&LogicalCell::root(1), [1, 2, 1])
            .unwrap();
        assert!((interval.min[1] - 0.5 * std::f64::consts::PI).abs() < 1e-12);
        assert!((interval.max[1] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn full_turn_lifts_every_angle_onto_the_seam() {
        let domain =
            CylindricalGeometry::full_turn([1.0, 2.0, 3.0], 1.0, 3.0, 0.2, -1.0, 4.0).unwrap();
        assert!(domain.is_full_turn());
        assert!((domain.theta_extent() - std::f64::consts::TAU).abs() < 1e-15);
        let world = domain.world_from_param(2.0, 0.2 + std::f64::consts::TAU - 1e-9, 0.0);
        let parameter = domain.param_from_world(world).unwrap();
        assert!((parameter[0] - 2.0).abs() < 1e-9);
        assert!(
            (parameter[1] - 0.2).abs() < 1e-6
                || (parameter[1] - (0.2 + std::f64::consts::TAU)).abs() < 1e-6
        );
        assert!(CylindricalGeometry::full_turn([0.0; 3], 0.0, 1.0, 0.0, 0.0, 1.0).is_err());
    }
}
