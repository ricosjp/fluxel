//! The sole mapping from typed Rust failures to the existing Python contract.
use fluxel_engine::BuildError;
use fluxel_ibm::IbmError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    PyErr,
};
pub(crate) fn to_python(error: BuildError) -> PyErr {
    match error {
        BuildError::StaleUpdate
        | BuildError::Ibm(IbmError::Query(_) | IbmError::StaleClassification)
        | BuildError::Mesh(_) => PyRuntimeError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}
