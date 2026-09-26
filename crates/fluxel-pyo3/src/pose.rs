use pyo3::prelude::*;
#[pyfunction]
#[pyo3(name = "_quaternion_from_axis_angle")]
/// Return a normalized w/x/y/z quaternion for a finite axis and angle in radians; zero axis is identity.
pub fn quaternion_from_axis_angle(axis: [f64; 3], angle: f64) -> PyResult<[f64; 4]> {
    fluxel_geometry::quaternion_from_axis_angle(axis, angle)
        .map_err(|e| crate::errors::to_python(e.into()))
}
