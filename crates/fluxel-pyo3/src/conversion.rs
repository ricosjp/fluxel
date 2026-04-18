//! Zero-copy helpers: Rust `Vec` to NumPy arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::Py;

#[inline]
pub(crate) fn vec_to_py1<'py, T: numpy::Element>(py: Python<'py>, vec: Vec<T>) -> Py<PyArray1<T>> {
    vec.into_pyarray(py).into()
}

/// Converts a `Vec<[T; K]>` into a 2D NumPy array of shape `(N, K)`.
#[inline]
pub(crate) fn vec_k_to_py2<'py, T: numpy::Element + Copy, const K: usize>(
    py: Python<'py>,
    vec: Vec<[T; K]>,
) -> Py<PyArray2<T>> {
    let len = vec.len();
    let flat_vec: Vec<T> = vec.into_iter().flatten().collect();
    PyArray1::from_vec(py, flat_vec)
        .reshape((len, K))
        .unwrap()
        .into()
}
