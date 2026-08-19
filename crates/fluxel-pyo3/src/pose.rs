//! Python-facing rigid-pose helpers.

use pyo3::prelude::*;

/// Builds a unit quaternion `[w, x, y, z]` from an axis-angle rotation.
///
/// `axis` is a 3-vector (normalized internally). `angle` is in radians.
#[pyfunction]
#[pyo3(name = "_quaternion_from_axis_angle")]
pub fn quaternion_from_axis_angle(axis: [f64; 3], angle: f64) -> [f64; 4] {
    fluxel_ibm::quaternion_from_axis_angle(axis, angle)
}
