//! Python input conventions terminate here; Rust receives validated values.
use crate::errors::to_python;
use fluxel_engine::{PoseUpdate, RefinementPlan, RefinementRegion};
use fluxel_geometry::BoundingBox;
use pyo3::prelude::*;
pub(crate) type RegionInput = ([f64; 3], [f64; 3], u8);
/// Validate Python region triples into physical bounds and levels; return Python input errors.
pub(crate) fn regions(values: Vec<RegionInput>) -> PyResult<Vec<RefinementRegion>> {
    values
        .into_iter()
        .map(|(min, max, level)| {
            let bounds = BoundingBox::new(min, max).map_err(|e| to_python(e.into()))?;
            RefinementRegion::new(bounds, level).map_err(to_python)
        })
        .collect()
}
/// Convert a fresh-build request, treating absent regions as an empty list.
pub(crate) fn plan(level: u8, values: Option<Vec<RegionInput>>) -> PyResult<RefinementPlan> {
    RefinementPlan::new(level, regions(values.unwrap_or_default())?).map_err(to_python)
}
/// Apply the Python zero-quaternion compatibility rule, preserving omitted values.
/// Finite quaternions with squared norm <= float64 epsilon become identity. Other
/// normalization and finite checks are deferred to the Rust pose constructor.
pub(crate) fn pose(translation: Option<[f64; 3]>, quaternion: Option<[f64; 4]>) -> PoseUpdate {
    // Historical Python behavior: a finite near-zero quaternion means identity.
    let quaternion = quaternion.map(|q| {
        let norm = q.iter().map(|v| v * v).sum::<f64>();
        if q.iter().all(|v| v.is_finite()) && norm <= f64::EPSILON {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            q
        }
    });
    PoseUpdate {
        translation,
        quaternion,
    }
}
