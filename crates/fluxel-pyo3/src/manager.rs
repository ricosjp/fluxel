//! One-shot IBM mesh generation (`FluxelManager`).

use crate::cfd_mesh::{BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh};
use crate::stl::load_stl;
use fluxel_core::Forest as CoreForest;
use fluxel_geometry::{BoundingBox as CoreBoundingBox, Geometry};
use fluxel_ibm::{mesh::IBMMesh, solver, CellType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;

/// Use case 1: one-stop manager for generating IBM mesh data.
#[pyclass]
pub struct FluxelManager {
    core_bbox: CoreBoundingBox,
    base_res: [u32; 3],
    n_leaf_refinement: u8,
}

impl FluxelManager {
    /// Shared preprocessing pipeline used by both GCIBM and APIBM mesh builders.
    fn prepare_forest_and_mesh(
        &self,
        stl_path: &str,
        target_level: u8,
    ) -> PyResult<(CoreForest, Geometry, IBMMesh)> {
        if !Path::new(stl_path).exists() {
            return Err(PyValueError::new_err(format!(
                "STL file not found: {}",
                stl_path
            )));
        }

        let (vertices, indices) = load_stl(stl_path)?;
        let ibm_mesh = IBMMesh::from_vertices_and_indices(&vertices, &indices);

        let geom = Geometry::new(self.core_bbox, self.base_res);

        let mut forest = CoreForest::new(self.base_res);
        forest.populate_root_cells();

        for _ in 0..target_level {
            let current_cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh);

            let mut refine_flags = vec![false; forest.num_cells()];
            let mut should_refine = false;
            (0..forest.num_cells()).for_each(|global_id| {
                if current_cell_types[global_id] == CellType::Intersect {
                    let key = forest.keys()[global_id];
                    if key.level() < target_level {
                        refine_flags[global_id] = true;
                        should_refine = true;
                    }
                }
            });

            if !should_refine {
                break;
            }

            forest.refine_by_flags(&refine_flags);
        }

        forest.enforce_2_to_1_balance();
        forest.uniform_refinement(self.n_leaf_refinement);

        Ok((forest, geom, ibm_mesh))
    }
}

#[pymethods]
impl FluxelManager {
    #[new]
    #[pyo3(signature = (bbox, base_res, n_leaf_refinement = 3))]
    pub fn new(bbox: &BoundingBox, base_res: [u32; 3], n_leaf_refinement: u8) -> Self {
        let core_bbox = CoreBoundingBox::new(bbox.min, bbox.max);
        Self {
            core_bbox,
            base_res,
            n_leaf_refinement,
        }
    }

    /// Loads an STL file and automatically runs BVH construction, AMR, 2:1 balancing,
    /// and inside/outside classification, then returns a CFD-ready mesh.
    pub fn build_ghost_cell_mesh(
        &self,
        py: Python<'_>,
        stl_path: &str,
        target_level: u8,
        fluid_seed_point: [f64; 3],
    ) -> PyResult<CfdGhostCellMesh> {
        let (forest, geom, ibm_mesh) = self.prepare_forest_and_mesh(stl_path, target_level)?;

        let mut cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh);

        solver::flood_fill_inside_outside(&forest, &geom, &mut cell_types, fluid_seed_point)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let gc_data = solver::compute_ghost_cell_geometry(
            &forest,
            &geom,
            &ibm_mesh,
            &cell_types,
            fluid_seed_point,
        );

        let core_mesh = fluxel_export::builder::build_ghost_cell_mesh(&forest, &geom, gc_data);

        Ok(CfdGhostCellMesh::from_core(py, core_mesh))
    }

    /// Builds mesh data for APIBM.
    pub fn build_axis_projected_mesh(
        &self,
        py: Python<'_>,
        stl_path: &str,
        target_level: u8,
    ) -> PyResult<CfdAxisProjectedMesh> {
        let (forest, geom, ibm_mesh) = self.prepare_forest_and_mesh(stl_path, target_level)?;

        let cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh);

        let core_mesh = fluxel_export::builder::build_axis_projected_mesh(
            &forest,
            &geom,
            &ibm_mesh,
            &cell_types,
        );

        Ok(CfdAxisProjectedMesh::from_core(py, core_mesh))
    }
}
