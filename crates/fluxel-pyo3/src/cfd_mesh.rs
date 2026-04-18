//! Python-facing CFD mesh SoA types (GCIBM / APIBM).

use crate::conversion::{vec_k_to_py2, vec_to_py1};
use fluxel_export::CfdGhostCellMesh as CoreCfdGhostCellMesh;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::Py;

/// Physical simulation domain bounds.
#[pyclass(get_all)]
pub struct BoundingBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[pymethods]
impl BoundingBox {
    #[new]
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }
}

/// SoA mesh layout for GCIBM.
#[pyclass(get_all)]
pub struct CfdGhostCellMesh {
    pub cell_centers: Py<PyArray2<f64>>,
    pub cell_sizes: Py<PyArray2<f64>>,

    pub x_faces_owner: Py<PyArray1<usize>>,
    pub x_faces_neighbour: Py<PyArray1<usize>>,
    pub x_bnd_minus_owner: Py<PyArray1<usize>>,
    pub x_bnd_plus_owner: Py<PyArray1<usize>>,

    pub y_faces_owner: Py<PyArray1<usize>>,
    pub y_faces_neighbour: Py<PyArray1<usize>>,
    pub y_bnd_minus_owner: Py<PyArray1<usize>>,
    pub y_bnd_plus_owner: Py<PyArray1<usize>>,

    pub z_faces_owner: Py<PyArray1<usize>>,
    pub z_faces_neighbour: Py<PyArray1<usize>>,
    pub z_bnd_minus_owner: Py<PyArray1<usize>>,
    pub z_bnd_plus_owner: Py<PyArray1<usize>>,

    pub gc_is_fluid: Py<PyArray1<bool>>,
    pub gc_cell_ids: Py<PyArray1<usize>>,
    pub gc_bnd_anchor_ids: Py<PyArray1<usize>>,
    pub gc_bnd_intercepts: Py<PyArray2<f64>>,
    pub gc_image_points: Py<PyArray2<f64>>,
    pub gc_interp_stencil_indices: Py<PyArray2<usize>>,
    pub gc_interp_stencil_weights: Py<PyArray2<f64>>,
}

impl CfdGhostCellMesh {
    pub fn from_core(py: Python<'_>, core_mesh: CoreCfdGhostCellMesh) -> Self {
        Self {
            cell_centers: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_centers),
            cell_sizes: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_sizes),

            x_faces_owner: vec_to_py1(py, core_mesh.x_faces_owner),
            x_faces_neighbour: vec_to_py1(py, core_mesh.x_faces_neighbour),
            x_bnd_minus_owner: vec_to_py1(py, core_mesh.x_bnd_minus_owner),
            x_bnd_plus_owner: vec_to_py1(py, core_mesh.x_bnd_plus_owner),

            y_faces_owner: vec_to_py1(py, core_mesh.y_faces_owner),
            y_faces_neighbour: vec_to_py1(py, core_mesh.y_faces_neighbour),
            y_bnd_minus_owner: vec_to_py1(py, core_mesh.y_bnd_minus_owner),
            y_bnd_plus_owner: vec_to_py1(py, core_mesh.y_bnd_plus_owner),

            z_faces_owner: vec_to_py1(py, core_mesh.z_faces_owner),
            z_faces_neighbour: vec_to_py1(py, core_mesh.z_faces_neighbour),
            z_bnd_minus_owner: vec_to_py1(py, core_mesh.z_bnd_minus_owner),
            z_bnd_plus_owner: vec_to_py1(py, core_mesh.z_bnd_plus_owner),

            gc_is_fluid: vec_to_py1(py, core_mesh.gc_is_fluid),
            gc_cell_ids: vec_to_py1(py, core_mesh.gc_cell_ids),
            gc_bnd_anchor_ids: vec_to_py1(py, core_mesh.gc_bnd_anchor_ids),
            gc_bnd_intercepts: vec_k_to_py2::<f64, 3>(py, core_mesh.gc_bnd_intercepts),
            gc_image_points: vec_k_to_py2::<f64, 3>(py, core_mesh.gc_image_points),
            gc_interp_stencil_indices: vec_k_to_py2::<usize, 8>(
                py,
                core_mesh.gc_interp_stencil_indices,
            ),
            gc_interp_stencil_weights: vec_k_to_py2::<f64, 8>(
                py,
                core_mesh.gc_interp_stencil_weights,
            ),
        }
    }
}

/// SoA mesh layout for APIBM (Axis-Projected Immersed Boundary Method).
#[pyclass(get_all)]
pub struct CfdAxisProjectedMesh {
    pub cell_centers: Py<PyArray2<f64>>,
    pub cell_sizes: Py<PyArray2<f64>>,

    pub x_faces_owner: Py<PyArray1<usize>>,
    pub x_faces_neighbour: Py<PyArray1<usize>>,
    pub x_bnd_minus_owner: Py<PyArray1<usize>>,
    pub x_bnd_plus_owner: Py<PyArray1<usize>>,

    pub y_faces_owner: Py<PyArray1<usize>>,
    pub y_faces_neighbour: Py<PyArray1<usize>>,
    pub y_bnd_minus_owner: Py<PyArray1<usize>>,
    pub y_bnd_plus_owner: Py<PyArray1<usize>>,

    pub z_faces_owner: Py<PyArray1<usize>>,
    pub z_faces_neighbour: Py<PyArray1<usize>>,
    pub z_bnd_minus_owner: Py<PyArray1<usize>>,
    pub z_bnd_plus_owner: Py<PyArray1<usize>>,

    pub ap_x_has_bnd: Py<PyArray1<bool>>,
    pub ap_x_dist_owner_to_bnd: Py<PyArray1<f64>>,
    pub ap_x_dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    pub ap_x_owner_far_cell_id: Py<PyArray1<usize>>,
    pub ap_x_neighbour_far_cell_id: Py<PyArray1<usize>>,
    pub ap_x_owner_weights: Py<PyArray2<f64>>,
    pub ap_x_neighbour_weights: Py<PyArray2<f64>>,
    pub ap_x_owner_bnd_anchor_id: Py<PyArray1<usize>>,
    pub ap_x_neighbour_bnd_anchor_id: Py<PyArray1<usize>>,

    pub ap_y_has_bnd: Py<PyArray1<bool>>,
    pub ap_y_dist_owner_to_bnd: Py<PyArray1<f64>>,
    pub ap_y_dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    pub ap_y_owner_far_cell_id: Py<PyArray1<usize>>,
    pub ap_y_neighbour_far_cell_id: Py<PyArray1<usize>>,
    pub ap_y_owner_weights: Py<PyArray2<f64>>,
    pub ap_y_neighbour_weights: Py<PyArray2<f64>>,
    pub ap_y_owner_bnd_anchor_id: Py<PyArray1<usize>>,
    pub ap_y_neighbour_bnd_anchor_id: Py<PyArray1<usize>>,

    pub ap_z_has_bnd: Py<PyArray1<bool>>,
    pub ap_z_dist_owner_to_bnd: Py<PyArray1<f64>>,
    pub ap_z_dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    pub ap_z_owner_far_cell_id: Py<PyArray1<usize>>,
    pub ap_z_neighbour_far_cell_id: Py<PyArray1<usize>>,
    pub ap_z_owner_weights: Py<PyArray2<f64>>,
    pub ap_z_neighbour_weights: Py<PyArray2<f64>>,
    pub ap_z_owner_bnd_anchor_id: Py<PyArray1<usize>>,
    pub ap_z_neighbour_bnd_anchor_id: Py<PyArray1<usize>>,
}

impl CfdAxisProjectedMesh {
    pub fn from_core(py: Python<'_>, core_mesh: fluxel_export::CfdAxisProjectedMesh) -> Self {
        Self {
            cell_centers: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_centers),
            cell_sizes: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_sizes),

            x_faces_owner: vec_to_py1(py, core_mesh.x_faces_owner),
            x_faces_neighbour: vec_to_py1(py, core_mesh.x_faces_neighbour),
            x_bnd_minus_owner: vec_to_py1(py, core_mesh.x_bnd_minus_owner),
            x_bnd_plus_owner: vec_to_py1(py, core_mesh.x_bnd_plus_owner),

            y_faces_owner: vec_to_py1(py, core_mesh.y_faces_owner),
            y_faces_neighbour: vec_to_py1(py, core_mesh.y_faces_neighbour),
            y_bnd_minus_owner: vec_to_py1(py, core_mesh.y_bnd_minus_owner),
            y_bnd_plus_owner: vec_to_py1(py, core_mesh.y_bnd_plus_owner),

            z_faces_owner: vec_to_py1(py, core_mesh.z_faces_owner),
            z_faces_neighbour: vec_to_py1(py, core_mesh.z_faces_neighbour),
            z_bnd_minus_owner: vec_to_py1(py, core_mesh.z_bnd_minus_owner),
            z_bnd_plus_owner: vec_to_py1(py, core_mesh.z_bnd_plus_owner),

            ap_x_has_bnd: vec_to_py1(py, core_mesh.ap_x_has_bnd),
            ap_x_dist_owner_to_bnd: vec_to_py1(py, core_mesh.ap_x_dist_owner_to_bnd),
            ap_x_dist_neighbour_to_bnd: vec_to_py1(py, core_mesh.ap_x_dist_neighbour_to_bnd),
            ap_x_owner_far_cell_id: vec_to_py1(py, core_mesh.ap_x_owner_far_cell_id),
            ap_x_neighbour_far_cell_id: vec_to_py1(py, core_mesh.ap_x_neighbour_far_cell_id),
            ap_x_owner_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_x_owner_weights),
            ap_x_neighbour_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_x_neighbour_weights),
            ap_x_owner_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_x_owner_bnd_anchor_id),
            ap_x_neighbour_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_x_neighbour_bnd_anchor_id),

            ap_y_has_bnd: vec_to_py1(py, core_mesh.ap_y_has_bnd),
            ap_y_dist_owner_to_bnd: vec_to_py1(py, core_mesh.ap_y_dist_owner_to_bnd),
            ap_y_dist_neighbour_to_bnd: vec_to_py1(py, core_mesh.ap_y_dist_neighbour_to_bnd),
            ap_y_owner_far_cell_id: vec_to_py1(py, core_mesh.ap_y_owner_far_cell_id),
            ap_y_neighbour_far_cell_id: vec_to_py1(py, core_mesh.ap_y_neighbour_far_cell_id),
            ap_y_owner_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_y_owner_weights),
            ap_y_neighbour_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_y_neighbour_weights),
            ap_y_owner_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_y_owner_bnd_anchor_id),
            ap_y_neighbour_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_y_neighbour_bnd_anchor_id),

            ap_z_has_bnd: vec_to_py1(py, core_mesh.ap_z_has_bnd),
            ap_z_dist_owner_to_bnd: vec_to_py1(py, core_mesh.ap_z_dist_owner_to_bnd),
            ap_z_dist_neighbour_to_bnd: vec_to_py1(py, core_mesh.ap_z_dist_neighbour_to_bnd),
            ap_z_owner_far_cell_id: vec_to_py1(py, core_mesh.ap_z_owner_far_cell_id),
            ap_z_neighbour_far_cell_id: vec_to_py1(py, core_mesh.ap_z_neighbour_far_cell_id),
            ap_z_owner_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_z_owner_weights),
            ap_z_neighbour_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_z_neighbour_weights),
            ap_z_owner_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_z_owner_bnd_anchor_id),
            ap_z_neighbour_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_z_neighbour_bnd_anchor_id),
        }
    }
}
