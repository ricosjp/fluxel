//! Helpers for building rigid poses used by IBM queries.

use parry3d_f64::math::{Pose, Rotation, Vector};

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
}
