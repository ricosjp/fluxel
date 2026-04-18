//! STL file loading for Python entry points.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) type StlMeshData = (Vec<[f64; 3]>, Vec<[u32; 3]>);

pub(crate) fn load_stl(path: &str) -> PyResult<StlMeshData> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open STL file '{}': {}", path, e)))?;

    let stl = stl_io::read_stl(&mut file)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse STL: {}", e)))?;

    let vertices: Vec<[f64; 3]> = stl
        .vertices
        .into_iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();

    let indices: Vec<[u32; 3]> = stl
        .faces
        .into_iter()
        .map(|f| {
            [
                f.vertices[0] as u32,
                f.vertices[1] as u32,
                f.vertices[2] as u32,
            ]
        })
        .collect();

    Ok((vertices, indices))
}
