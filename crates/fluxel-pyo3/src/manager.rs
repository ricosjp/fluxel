//! One-shot IBM mesh generation (`FluxelManager`).

use crate::cfd_mesh::{BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh};
use fluxel_core::Forest as CoreForest;
use fluxel_geometry::{BoundingBox as CoreBoundingBox, Geometry};
use fluxel_ibm::{mesh::IBMMesh, solver, CellType};
use fluxel_sfc::MAX_LEVEL;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::Path;

/// Use case 1: one-stop manager for generating IBM mesh data.
#[pyclass]
pub struct FluxelManager {
    core_bbox: CoreBoundingBox,
    base_res: [u32; 3],
    n_leaf_refinement: u8,
}

type RefinementRegion = ([f64; 3], [f64; 3], u8);

fn cell_intersects_bbox(center: [f64; 3], size: [f64; 3], min: [f64; 3], max: [f64; 3]) -> bool {
    let half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0];
    let cell_min = [
        center[0] - half[0],
        center[1] - half[1],
        center[2] - half[2],
    ];
    let cell_max = [
        center[0] + half[0],
        center[1] + half[1],
        center[2] + half[2],
    ];

    cell_min[0] < max[0]
        && cell_max[0] > min[0]
        && cell_min[1] < max[1]
        && cell_max[1] > min[1]
        && cell_min[2] < max[2]
        && cell_max[2] > min[2]
}

impl FluxelManager {
    fn load_ibm_mesh(mesh_path: Option<&str>) -> PyResult<IBMMesh> {
        match mesh_path {
            Some(mesh_path) => {
                let path = Path::new(mesh_path);
                if !path.exists() {
                    return Err(PyValueError::new_err(format!(
                        "Mesh file not found: {}",
                        mesh_path
                    )));
                }

                let loader = match path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .as_deref()
                {
                    Some("stl") => IBMMesh::from_stl_file,
                    Some("obj") => IBMMesh::from_obj_file,
                    Some(other) => {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported mesh extension '.{other}'. Use .stl or .obj."
                        )));
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "Mesh file has no extension. Use .stl or .obj.",
                        ));
                    }
                };

                loader(path).map_err(|e| PyValueError::new_err(e.to_string()))
            }
            None => {
                // Place a dummy triangle far outside the domain when mesh input is omitted.
                let dummy_v = vec![
                    [1e10, 1e10, 1e10],
                    [1e10 + 1.0, 1e10, 1e10],
                    [1e10, 1e10 + 1.0, 1e10],
                ];
                let dummy_i = vec![[0, 1, 2]];
                Ok(IBMMesh::from_vertices_indices_and_patches(
                    &dummy_v,
                    &dummy_i,
                    vec!["empty".to_string()],
                    vec![0],
                ))
            }
        }
    }

    /// Shared preprocessing pipeline used by both GCIBM and APIBM mesh builders.
    fn prepare_forest_and_mesh(
        &self,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: &[RefinementRegion],
    ) -> PyResult<(CoreForest, Geometry, IBMMesh)> {
        let ibm_mesh = Self::load_ibm_mesh(mesh_path)?;

        let geom = Geometry::new(self.core_bbox, self.base_res);

        let mut forest = CoreForest::new(self.base_res);
        forest.populate_root_cells();

        for _ in 0..target_level {
            let current_cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh);

            let refine_flags: Vec<bool> = forest
                .keys()
                .par_iter()
                .zip(current_cell_types.par_iter())
                .map(|(key, cell_type)| {
                    *cell_type == CellType::Intersect && key.level() < target_level
                })
                .collect();

            if !refine_flags.par_iter().any(|&flag| flag) {
                break;
            }

            forest.refine_by_flags(&refine_flags);
        }

        Self::refine_regions_to_level(&mut forest, &geom, refinement_regions);

        forest.enforce_2_to_1_balance();
        forest.uniform_refinement(self.n_leaf_refinement);

        Ok((forest, geom, ibm_mesh))
    }

    fn refine_regions_to_level(
        forest: &mut CoreForest,
        geom: &Geometry,
        refinement_regions: &[RefinementRegion],
    ) {
        loop {
            let refine_flags: Vec<bool> = forest
                .keys()
                .par_iter()
                .map(|key| {
                    let logical = key.to_logical();
                    let (center, size) = geom.cell_bounds(&logical);

                    refinement_regions.iter().any(|&(min, max, target_level)| {
                        key.level() < target_level
                            && key.level() < MAX_LEVEL
                            && cell_intersects_bbox(center, size, min, max)
                    })
                })
                .collect();

            if !refine_flags.par_iter().any(|&flag| flag) {
                break;
            }

            forest.refine_by_flags(&refine_flags);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_refinement_refines_intersecting_cells_to_target_level() {
        let bbox = CoreBoundingBox::new([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);
        let mut forest = CoreForest::new([2, 1, 1]);
        forest.populate_root_cells();

        FluxelManager::refine_regions_to_level(
            &mut forest,
            &geom,
            &[([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1)],
        );

        assert_eq!(forest.num_cells(), 9);
        assert_eq!(
            forest.keys().iter().filter(|key| key.level() == 1).count(),
            8
        );
        assert_eq!(
            forest.keys().iter().filter(|key| key.level() == 0).count(),
            1
        );
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
    #[pyo3(signature = (mesh_path, target_level, fluid_seed_point, refinement_regions = None))]
    pub fn build_ghost_cell_mesh(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        fluid_seed_point: [f64; 3],
        refinement_regions: Option<Vec<RefinementRegion>>,
    ) -> PyResult<CfdGhostCellMesh> {
        let refinement_regions = refinement_regions.unwrap_or_default();
        let (forest, geom, ibm_mesh) =
            self.prepare_forest_and_mesh(mesh_path, target_level, &refinement_regions)?;

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

        let core_mesh =
            fluxel_export::builder::build_ghost_cell_mesh(&forest, &geom, &ibm_mesh, gc_data);

        Ok(CfdGhostCellMesh::from_core(py, core_mesh))
    }

    /// Builds mesh data for APIBM.
    #[pyo3(signature = (mesh_path, target_level, refinement_regions = None))]
    pub fn build_axis_projected_mesh(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RefinementRegion>>,
    ) -> PyResult<CfdAxisProjectedMesh> {
        let refinement_regions = refinement_regions.unwrap_or_default();
        let (forest, geom, ibm_mesh) =
            self.prepare_forest_and_mesh(mesh_path, target_level, &refinement_regions)?;

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
