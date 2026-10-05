//! Explicit array ownership. Shared exports use immutable bytes and bounded caches.
//! This safe fallback copies at publication; no raw pointer or Rust alias is exposed.
use numpy::{
    ndarray::Dimension, Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3,
    PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

/// Keep writable storage, or copy into immutable bytes so WRITEABLE cannot be re-enabled.
fn finish<T: Element, D: Dimension>(
    py: Python<'_>,
    array: Bound<'_, PyArray<T, D>>,
    readonly: bool,
) -> PyResult<Py<PyArray<T, D>>> {
    if !readonly {
        return Ok(array.unbind());
    }
    // bytes owns immutable storage: unlike setflags(False) on an owning ndarray,
    // consumers cannot enable WRITEABLE again, including through .base.
    let bytes = array.call_method0("tobytes")?;
    let result = py
        .import("numpy")?
        .getattr("frombuffer")?
        .call1((bytes, array.getattr("dtype")?))?
        .call_method1("reshape", (array.shape().to_vec(),))?
        .cast_into::<PyArray<T, D>>()?;
    Ok(result.unbind())
}
/// Copy a slice to a one-dimensional NumPy array; readonly adds immutable byte backing.
pub(crate) fn array1<T: Element + Copy>(
    py: Python<'_>,
    values: &[T],
    readonly: bool,
) -> PyResult<Py<PyArray1<T>>> {
    finish(py, values.to_vec().into_pyarray(py), readonly)
}
/// Copy fixed-width rows to shape (row_count, K), retaining width even for zero rows.
pub(crate) fn array2<T: Element + Copy, const K: usize>(
    py: Python<'_>,
    values: &[[T; K]],
    readonly: bool,
) -> PyResult<Py<PyArray2<T>>> {
    let flat: Vec<T> = values.iter().flatten().copied().collect();
    let array = flat.into_pyarray(py).reshape((values.len(), K))?;
    finish(py, array, readonly)
}
/// Copy rank-3 values to shape (count, A, B), retaining that shape for zero rows.
pub(crate) fn array3<T: Element + Copy, const A: usize, const B: usize>(
    py: Python<'_>,
    values: &[[[T; B]; A]],
    readonly: bool,
) -> PyResult<Py<PyArray3<T>>> {
    let flat: Vec<T> = values.iter().flatten().flatten().copied().collect();
    let array = flat.into_pyarray(py).reshape((values.len(), A, B))?;
    finish(py, array, readonly)
}
/// Share array storage with a fresh ndarray header; mutations to shape/dtype stay local to the view.
pub(crate) fn view<T: Element, D: Dimension>(
    py: Python<'_>,
    array: &Py<PyArray<T, D>>,
) -> PyResult<Py<PyArray<T, D>>> {
    // A fresh ndarray header isolates dtype/shape changes from cached arrays.
    Ok(array
        .bind(py)
        .call_method0("view")?
        .cast_into::<PyArray<T, D>>()?
        .unbind())
}
