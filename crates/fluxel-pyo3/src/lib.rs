//! Fluxel Python bindings (PyO3).
//!
//! Exposes the high-performance Rust AMR core and IBM geometry engine as
//! Python-usable classes.

mod cfd_mesh;
mod conversion;
mod forest;
mod manager;
mod stl;

use pyo3::prelude::*;

use cfd_mesh::{BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh};
use forest::Forest;
use manager::FluxelManager;

/// Python module entry point.
#[pymodule]
#[pyo3(name = "_fluxel")]
fn fluxel(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<CfdGhostCellMesh>()?;
    m.add_class::<CfdAxisProjectedMesh>()?;
    m.add_class::<FluxelManager>()?;
    m.add_class::<Forest>()?;
    Ok(())
}
