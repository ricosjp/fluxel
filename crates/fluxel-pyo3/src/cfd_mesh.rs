//! Python-facing CFD mesh SoA types (GCIBM / APIBM).

use crate::conversion::{vec_k_to_py2, vec_to_py1};
use fluxel_export::CfdAxisProjectedMesh as CoreCfdAxisProjectedMesh;
use fluxel_export::CfdGhostCellMesh as CoreCfdGhostCellMesh;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
    pub n_cells: usize,
    pub coordinate_type: u8,
    pub patch_name_to_id: Py<PyDict>,

    pub cell_centers: Py<PyArray2<f64>>,
    pub cell_sizes: Py<PyArray2<f64>>,

    pub internal_faces_owner: Py<PyArray1<usize>>,
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    pub internal_faces_axis: Py<PyArray1<u8>>,
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,

    pub gc_is_fluid: Py<PyArray1<bool>>,
    pub gc_cell_ids: Py<PyArray1<usize>>,
    pub gc_bnd_anchor_ids: Py<PyArray1<usize>>,
    pub gc_bnd_patch_ids: Py<PyArray1<usize>>,
    pub gc_bnd_intercepts: Py<PyArray2<f64>>,
    pub gc_image_points: Py<PyArray2<f64>>,
    pub gc_interp_stencil_indices: Py<PyArray2<usize>>,
    pub gc_interp_stencil_weights: Py<PyArray2<f64>>,
}

impl CfdGhostCellMesh {
    pub fn from_core(py: Python<'_>, core_mesh: CoreCfdGhostCellMesh) -> Self {
        let patch_name_to_id = PyDict::new(py);
        for (index, patch_name) in core_mesh.patch_names.iter().enumerate() {
            patch_name_to_id
                .set_item(patch_name, index)
                .expect("failed to set patch_name_to_id");
        }

        Self {
            n_cells: core_mesh.n_cells,
            coordinate_type: core_mesh.coordinate_type as u8,
            patch_name_to_id: patch_name_to_id.unbind(),

            cell_centers: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_centers),
            cell_sizes: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_sizes),

            internal_faces_owner: vec_to_py1(py, core_mesh.internal_faces_owner),
            internal_faces_neighbour: vec_to_py1(py, core_mesh.internal_faces_neighbour),
            internal_faces_axis: vec_to_py1(
                py,
                core_mesh
                    .internal_faces_axis
                    .into_iter()
                    .map(|axis| axis as u8)
                    .collect(),
            ),
            domain_bnd_faces_owner: vec_to_py1(py, core_mesh.domain_bnd_faces_owner),
            domain_bnd_faces_dir: vec_to_py1(
                py,
                core_mesh
                    .domain_bnd_faces_dir
                    .into_iter()
                    .map(|dir| dir as u8)
                    .collect(),
            ),

            gc_is_fluid: vec_to_py1(py, core_mesh.gc_is_fluid),
            gc_cell_ids: vec_to_py1(py, core_mesh.gc_cell_ids),
            gc_bnd_anchor_ids: vec_to_py1(py, core_mesh.gc_bnd_anchor_ids),
            gc_bnd_patch_ids: vec_to_py1(py, core_mesh.gc_bnd_patch_ids),
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
    pub n_cells: usize,
    pub coordinate_type: u8,
    pub patch_name_to_id: Py<PyDict>,

    pub cell_centers: Py<PyArray2<f64>>,
    pub cell_sizes: Py<PyArray2<f64>>,

    pub internal_faces_owner: Py<PyArray1<usize>>,
    pub internal_faces_neighbour: Py<PyArray1<usize>>,
    pub internal_faces_axis: Py<PyArray1<u8>>,
    pub domain_bnd_faces_owner: Py<PyArray1<usize>>,
    pub domain_bnd_faces_dir: Py<PyArray1<u8>>,

    pub ap_is_immersed_face: Py<PyArray1<bool>>,
    pub ap_dist_owner_to_bnd: Py<PyArray1<f64>>,
    pub ap_dist_neighbour_to_bnd: Py<PyArray1<f64>>,
    pub ap_owner_far_cell_id: Py<PyArray1<usize>>,
    pub ap_neighbour_far_cell_id: Py<PyArray1<usize>>,
    pub ap_owner_weights: Py<PyArray2<f64>>,
    pub ap_neighbour_weights: Py<PyArray2<f64>>,
    pub ap_owner_bnd_anchor_id: Py<PyArray1<usize>>,
    pub ap_owner_bnd_patch_id: Py<PyArray1<usize>>,
    pub ap_neighbour_bnd_anchor_id: Py<PyArray1<usize>>,
    pub ap_neighbour_bnd_patch_id: Py<PyArray1<usize>>,
}

impl CfdAxisProjectedMesh {
    pub fn from_core(py: Python<'_>, core_mesh: CoreCfdAxisProjectedMesh) -> Self {
        let patch_name_to_id = PyDict::new(py);
        for (index, patch_name) in core_mesh.patch_names.iter().enumerate() {
            patch_name_to_id
                .set_item(patch_name, index)
                .expect("failed to set patch_name_to_id");
        }

        Self {
            n_cells: core_mesh.n_cells,
            coordinate_type: core_mesh.coordinate_type as u8,
            patch_name_to_id: patch_name_to_id.unbind(),

            cell_centers: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_centers),
            cell_sizes: vec_k_to_py2::<f64, 3>(py, core_mesh.cell_sizes),

            internal_faces_owner: vec_to_py1(py, core_mesh.internal_faces_owner),
            internal_faces_neighbour: vec_to_py1(py, core_mesh.internal_faces_neighbour),
            internal_faces_axis: vec_to_py1(
                py,
                core_mesh
                    .internal_faces_axis
                    .into_iter()
                    .map(|axis| axis as u8)
                    .collect(),
            ),
            domain_bnd_faces_owner: vec_to_py1(py, core_mesh.domain_bnd_faces_owner),
            domain_bnd_faces_dir: vec_to_py1(
                py,
                core_mesh
                    .domain_bnd_faces_dir
                    .into_iter()
                    .map(|dir| dir as u8)
                    .collect(),
            ),

            ap_is_immersed_face: vec_to_py1(py, core_mesh.ap_is_immersed_face),
            ap_dist_owner_to_bnd: vec_to_py1(py, core_mesh.ap_dist_owner_to_bnd),
            ap_dist_neighbour_to_bnd: vec_to_py1(py, core_mesh.ap_dist_neighbour_to_bnd),
            ap_owner_far_cell_id: vec_to_py1(py, core_mesh.ap_owner_far_cell_id),
            ap_neighbour_far_cell_id: vec_to_py1(py, core_mesh.ap_neighbour_far_cell_id),
            ap_owner_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_owner_weights),
            ap_neighbour_weights: vec_k_to_py2::<f64, 3>(py, core_mesh.ap_neighbour_weights),
            ap_owner_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_owner_bnd_anchor_id),
            ap_owner_bnd_patch_id: vec_to_py1(py, core_mesh.ap_owner_bnd_patch_id),
            ap_neighbour_bnd_anchor_id: vec_to_py1(py, core_mesh.ap_neighbour_bnd_anchor_id),
            ap_neighbour_bnd_patch_id: vec_to_py1(py, core_mesh.ap_neighbour_bnd_patch_id),
        }
    }
}
