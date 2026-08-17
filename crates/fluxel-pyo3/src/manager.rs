//! One-shot IBM mesh generation (`FluxelManager`).

use crate::apibm_session::ApibmSession;
use crate::cfd_mesh::{BoundingBox, CfdAxisProjectedMesh, CfdGhostCellMesh};
use crate::pipeline::{prepare_forest_and_mesh, MeshBuildConfig, RefinementRegion};
use fluxel_geometry::BoundingBox as CoreBoundingBox;
use fluxel_ibm::solver;
use parry3d_f64::math::Pose;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Use case 1: one-stop manager for generating IBM mesh data.
#[pyclass]
pub struct FluxelManager {
    pub(crate) config: MeshBuildConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::refine_regions_to_level;
    use fluxel_core::Forest as CoreForest;
    use fluxel_geometry::Geometry;

    #[test]
    fn region_refinement_refines_intersecting_cells_to_target_level() {
        let bbox = CoreBoundingBox::new([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let geom = Geometry::new(bbox, [2, 1, 1]);
        let mut forest = CoreForest::new([2, 1, 1]);
        forest.populate_root_cells();

        refine_regions_to_level(
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
            config: MeshBuildConfig {
                core_bbox,
                base_res,
                n_leaf_refinement,
            },
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
        let identity = Pose::identity();
        let (forest, geom, ibm_mesh) = prepare_forest_and_mesh(
            &self.config,
            mesh_path,
            target_level,
            &refinement_regions,
            &identity,
        )?;

        let mut cell_types =
            solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh, &identity);

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

    /// Builds mesh data for APIBM (one-shot; no session state is retained).
    #[pyo3(signature = (mesh_path, target_level, refinement_regions = None))]
    pub fn build_axis_projected_mesh(
        &self,
        py: Python<'_>,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RefinementRegion>>,
    ) -> PyResult<CfdAxisProjectedMesh> {
        let refinement_regions = refinement_regions.unwrap_or_default();
        let identity = Pose::identity();
        let (forest, geom, ibm_mesh) = prepare_forest_and_mesh(
            &self.config,
            mesh_path,
            target_level,
            &refinement_regions,
            &identity,
        )?;

        let cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh, &identity);

        let core_mesh = fluxel_export::builder::build_axis_projected_mesh(
            &forest,
            &geom,
            &ibm_mesh,
            &cell_types,
            &identity,
        );

        Ok(CfdAxisProjectedMesh::from_core(py, core_mesh))
    }

    /// Creates a stateful APIBM session that retains the forest for moving-boundary updates.
    #[pyo3(signature = (mesh_path, target_level, refinement_regions = None))]
    pub fn create_axis_projected_session(
        &self,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RefinementRegion>>,
    ) -> PyResult<ApibmSession> {
        ApibmSession::new(&self.config, mesh_path, target_level, refinement_regions)
    }
}
