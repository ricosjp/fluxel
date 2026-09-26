//! Backend-independent validated rigid transforms. Quaternions use [w, x, y, z].
use crate::GeometryError;
#[derive(Debug, Clone, Copy, PartialEq)]
/// Finite translation and normalized quaternion in w/x/y/z order.
/// Transforms rotate about the original coordinate origin, then translate in world
/// length units. Values are absolute rather than incremental.
pub struct RigidPose {
    translation: [f64; 3],
    quaternion: [f64; 4],
}
impl Default for RigidPose {
    fn default() -> Self {
        Self::identity()
    }
}
impl RigidPose {
    /// Zero translation and identity rotation.
    pub const fn identity() -> Self {
        Self {
            translation: [0.0; 3],
            quaternion: [1.0, 0.0, 0.0, 0.0],
        }
    }
    /// Validate finite components and normalize a nonzero quaternion using scaled arithmetic.
    /// Returns GeometryError::InvalidPose for nonfinite components or a zero quaternion.
    /// Unlike the Python compatibility adapter, Rust does not map zero to identity.
    pub fn new(translation: [f64; 3], quaternion: [f64; 4]) -> Result<Self, GeometryError> {
        if !translation
            .iter()
            .chain(quaternion.iter())
            .all(|v| v.is_finite())
        {
            return Err(GeometryError::InvalidPose);
        }
        let scale = quaternion.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if scale == 0.0 {
            return Err(GeometryError::InvalidPose);
        }
        let scaled = quaternion.map(|v| v / scale);
        let norm = scaled.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok(Self {
            translation,
            quaternion: scaled.map(|v| v / norm),
        })
    }
    /// Use an identity rotation; reject any nonfinite translation component.
    pub fn from_translation(translation: [f64; 3]) -> Result<Self, GeometryError> {
        Self::new(translation, [1.0, 0.0, 0.0, 0.0])
    }
    /// Absolute physical displacement in X/Y/Z order.
    pub fn translation(&self) -> [f64; 3] {
        self.translation
    }
    /// Unit quaternion in w/x/y/z order.
    pub fn quaternion(&self) -> [f64; 4] {
        self.quaternion
    }
}
/// A zero axis maps to identity for the Python axis-angle helper's historical contract.
/// Normalize the finite axis and return a w/x/y/z quaternion for `angle` in radians.
/// Exactly zero axes yield identity; arbitrarily small nonzero axes are normalized.
/// Returns GeometryError::InvalidPose for nonfinite axis components or angle.
pub fn quaternion_from_axis_angle(axis: [f64; 3], angle: f64) -> Result<[f64; 4], GeometryError> {
    if !axis.iter().all(|x| x.is_finite()) || !angle.is_finite() {
        return Err(GeometryError::InvalidPose);
    }
    let scale = axis.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok([1.0, 0.0, 0.0, 0.0]);
    }
    let scaled = axis.map(|x| x / scale);
    let norm = scaled.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (sin, cos) = (angle * 0.5).sin_cos();
    Ok([
        cos,
        scaled[0] / norm * sin,
        scaled[1] / norm * sin,
        scaled[2] / norm * sin,
    ])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_poses() {
        assert!(RigidPose::new([0.0; 3], [0.0; 4]).is_err());
        assert!(RigidPose::from_translation([f64::NAN, 0.0, 0.0]).is_err());
    }
    #[test]
    fn normalizes_large_quaternions() {
        let pose = RigidPose::new([0.0; 3], [f64::MAX; 4]).unwrap();
        assert_eq!(pose.quaternion(), [0.5; 4]);
    }
    #[test]
    fn axis_angle_conventions() {
        assert_eq!(
            quaternion_from_axis_angle([0.0; 3], 1.0).unwrap(),
            [1.0, 0.0, 0.0, 0.0]
        );
        let q = quaternion_from_axis_angle([0.0, 2.0, 0.0], std::f64::consts::FRAC_PI_2).unwrap();
        assert!((q[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
        assert!((q[2] - q[0]).abs() < 1e-15);
    }
}
