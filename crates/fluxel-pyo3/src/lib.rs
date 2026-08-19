//! Fluxel Python bindings (PyO3).
//!
//! Exposes the high-performance Rust AMR core and IBM geometry engine as
//! Python-usable classes.

mod apibm_session;
mod cfd_mesh;
mod conversion;
mod forest;
mod manager;
mod pipeline;
mod pose;

use pyo3::prelude::*;

use apibm_session::ApibmSession;
use cfd_mesh::{ApIbmFaceData, BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh};
use forest::Forest;
use manager::FluxelManager;
use pose::quaternion_from_axis_angle;

/// Python module entry point.
#[pymodule]
#[pyo3(name = "_fluxel")]
fn fluxel(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<CfdGhostCellMesh>()?;
    m.add_class::<ApIbmFaceData>()?;
    m.add_class::<CfdAxisProjectedMesh>()?;
    m.add_class::<ApibmSession>()?;
    m.add_class::<FluxelManager>()?;
    m.add_class::<Forest>()?;
    m.add_function(wrap_pyfunction!(quaternion_from_axis_angle, m)?)?;
    Ok(())
}
