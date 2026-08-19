//! Helpers for building rigid poses used by IBM queries.

use parry3d_f64::math::{Pose, Rotation, Vector};

/// Builds a unit quaternion `[w, x, y, z]` from an axis-angle rotation.
///
/// `axis` is normalized. A near-zero axis falls back to the identity quaternion
/// `[1, 0, 0, 0]`. `angle` is in radians.
pub fn quaternion_from_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let v = Vector::new(axis[0], axis[1], axis[2]);
    let Some(unit) = v.try_normalize() else {
        return [1.0, 0.0, 0.0, 0.0];
    };
    let rot = Rotation::from_axis_angle(unit, angle);
    [rot.w, rot.x, rot.y, rot.z]
}

/// Builds a rigid [`Pose`] from translation and a unit quaternion in `[w, x, y, z]` order.
///
/// The quaternion is normalized. A near-zero length falls back to the identity rotation.
pub fn pose_from_translation_quaternion(
    translation: [f64; 3],
    rotation_quaternion: [f64; 4],
) -> Pose {
    let [w, x, y, z] = rotation_quaternion;
    let mut quat = Rotation::from_xyzw(x, y, z, w);
    let len_sq = quat.length_squared();
    if len_sq > f64::EPSILON {
        quat /= len_sq.sqrt();
    } else {
        quat = Rotation::IDENTITY;
    }
    Pose::from_parts(
        Vector::new(translation[0], translation[1], translation[2]),
        quat,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_quaternion_gives_identity_rotation() {
        let pose = pose_from_translation_quaternion([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(pose.translation, Vector::new(1.0, 2.0, 3.0));
        assert_eq!(pose.rotation, Rotation::IDENTITY);
    }

    #[test]
    fn zero_axis_gives_identity_quaternion() {
        let q = quaternion_from_axis_angle([0.0, 0.0, 0.0], 1.0);
        assert_eq!(q, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn y_axis_45deg_matches_half_angle_formula() {
        let angle = std::f64::consts::FRAC_PI_4;
        let q = quaternion_from_axis_angle([0.0, 1.0, 0.0], angle);
        let half = angle / 2.0;
        assert!((q[0] - half.cos()).abs() < 1e-12);
        assert!(q[1].abs() < 1e-12);
        assert!((q[2] - half.sin()).abs() < 1e-12);
        assert!(q[3].abs() < 1e-12);
    }

    #[test]
    fn unnormalized_axis_matches_unit_axis() {
        let angle = 1.2;
        let q_unit = quaternion_from_axis_angle([0.0, 1.0, 0.0], angle);
        let q_scaled = quaternion_from_axis_angle([0.0, 2.0, 0.0], angle);
        for (a, b) in q_unit.iter().zip(q_scaled.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
