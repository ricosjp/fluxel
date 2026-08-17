//! Stateful APIBM session for rigid moving-boundary updates.

use crate::cfd_mesh::CfdAxisProjectedMesh;
use crate::pipeline::{
    build_forest_for_ibm, prepare_forest_and_mesh, MeshBuildConfig, RefinementRegion,
};
use fluxel_core::Forest as CoreForest;
use fluxel_export::CfdAxisProjectedMesh as CoreCfdAxisProjectedMesh;
use fluxel_export::{has_under_refined_intersect_cells, rebuild_axis_projected_ib};
use fluxel_geometry::Geometry;
use fluxel_ibm::mesh::IBMMesh;
use fluxel_ibm::pose_from_translation_quaternion;
use fluxel_ibm::solver;
use parry3d_f64::math::Pose;
use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;

/// Stateful APIBM session retaining forest + IBM geometry for `update_ib` / `remesh`.
#[pyclass]
pub struct ApibmSession {
    config: MeshBuildConfig,
    forest: CoreForest,
    geom: Geometry,
    ibm_mesh: IBMMesh,
    pose: Pose,
    mesh: CoreCfdAxisProjectedMesh,
    target_level: u8,
    refinement_regions: Vec<RefinementRegion>,
}

impl ApibmSession {
    /// Builds a session at identity pose: AMR forest, IBM mesh, and initial CFD snapshot.
    pub(crate) fn new(
        config: &MeshBuildConfig,
        mesh_path: Option<&str>,
        target_level: u8,
        refinement_regions: Option<Vec<RefinementRegion>>,
    ) -> PyResult<Self> {
        let refinement_regions = refinement_regions.unwrap_or_default();
        let identity = Pose::identity();
        let (forest, geom, ibm_mesh) = prepare_forest_and_mesh(
            config,
            mesh_path,
            target_level,
            &refinement_regions,
            &identity,
        )?;

        let cell_types = solver::mark_intersecting_cells(&forest, &geom, &ibm_mesh, &identity);
        let mesh = fluxel_export::builder::build_axis_projected_mesh(
            &forest,
            &geom,
            &ibm_mesh,
            &cell_types,
            &identity,
        );

        Ok(Self {
            config: *config,
            forest,
            geom,
            ibm_mesh,
            pose: identity,
            mesh,
            target_level,
            refinement_regions,
        })
    }

    /// Replaces pose components that were provided; omitted axes keep the current values.
    fn set_pose_from_args(
        &mut self,
        translation: Option<[f64; 3]>,
        rotation_quaternion: Option<[f64; 4]>,
    ) {
        if translation.is_none() && rotation_quaternion.is_none() {
            return;
        }
        let t = translation.unwrap_or([
            self.pose.translation.x,
            self.pose.translation.y,
            self.pose.translation.z,
        ]);
        let q = rotation_quaternion.unwrap_or([
            self.pose.rotation.w,
            self.pose.rotation.x,
            self.pose.rotation.y,
            self.pose.rotation.z,
        ]);
        self.pose = pose_from_translation_quaternion(t, q);
    }

    /// Emits a Python `UserWarning` when the IB intersects cells coarser than `target_level`.
    fn emit_under_refined_warning(
        py: Python<'_>,
        warn: bool,
        forest: &CoreForest,
        cell_types: &[fluxel_ibm::CellType],
        target_level: u8,
    ) -> PyResult<()> {
        if !warn {
            return Ok(());
        }
        if has_under_refined_intersect_cells(forest, cell_types, target_level) {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "Immersed boundary intersects cells below target_level; \
                     consider remesh() to restore refinement near the boundary.",
                    py.get_type::<PyUserWarning>(),
                ),
            )?;
        }
        Ok(())
    }
}

#[pymethods]
impl ApibmSession {
    /// Current CFD mesh snapshot (new Python object each call).
    #[getter]
    fn mesh(&self, py: Python<'_>) -> CfdAxisProjectedMesh {
        CfdAxisProjectedMesh::from_core(py, self.mesh.clone())
    }

    /// Current rigid translation applied to the IBM mesh.
    #[getter]
    fn translation(&self) -> [f64; 3] {
        [
            self.pose.translation.x,
            self.pose.translation.y,
            self.pose.translation.z,
        ]
    }

    /// Current rigid rotation quaternion `[w, x, y, z]`.
    #[getter]
    fn rotation_quaternion(&self) -> [f64; 4] {
        [
            self.pose.rotation.w,
            self.pose.rotation.x,
            self.pose.rotation.y,
            self.pose.rotation.z,
        ]
    }

    /// Updates immersed-boundary data only (background mesh topology fixed).
    ///
    /// `translation` and `rotation_quaternion` set the absolute rigid pose of the IB mesh
    /// (`[w, x, y, z]`). Defaults keep the current pose components when omitted.
    #[pyo3(signature = (
        translation = None,
        rotation_quaternion = None,
        warn_outside_refinement = true
    ))]
    fn update_ib(
        &mut self,
        py: Python<'_>,
        translation: Option<[f64; 3]>,
        rotation_quaternion: Option<[f64; 4]>,
        warn_outside_refinement: bool,
    ) -> PyResult<CfdAxisProjectedMesh> {
        self.set_pose_from_args(translation, rotation_quaternion);

        let cell_types = rebuild_axis_projected_ib(
            &self.forest,
            &self.geom,
            &self.ibm_mesh,
            &mut self.mesh,
            &self.pose,
        );

        Self::emit_under_refined_warning(
            py,
            warn_outside_refinement,
            &self.forest,
            &cell_types,
            self.target_level,
        )?;

        Ok(CfdAxisProjectedMesh::from_core(py, self.mesh.clone()))
    }

    /// Rebuilds the AMR background mesh and IB payload for the current (or updated) pose.
    #[pyo3(signature = (
        target_level = None,
        refinement_regions = None,
        translation = None,
        rotation_quaternion = None,
        warn_outside_refinement = true
    ))]
    fn remesh(
        &mut self,
        py: Python<'_>,
        target_level: Option<u8>,
        refinement_regions: Option<Vec<RefinementRegion>>,
        translation: Option<[f64; 3]>,
        rotation_quaternion: Option<[f64; 4]>,
        warn_outside_refinement: bool,
    ) -> PyResult<CfdAxisProjectedMesh> {
        self.set_pose_from_args(translation, rotation_quaternion);
        if let Some(level) = target_level {
            self.target_level = level;
        }
        if let Some(regions) = refinement_regions {
            self.refinement_regions = regions;
        }

        let (forest, geom) = build_forest_for_ibm(
            &self.config,
            &self.ibm_mesh,
            self.target_level,
            &self.refinement_regions,
            &self.pose,
        )?;
        self.forest = forest;
        self.geom = geom;

        let cell_types =
            solver::mark_intersecting_cells(&self.forest, &self.geom, &self.ibm_mesh, &self.pose);
        self.mesh = fluxel_export::builder::build_axis_projected_mesh(
            &self.forest,
            &self.geom,
            &self.ibm_mesh,
            &cell_types,
            &self.pose,
        );

        Self::emit_under_refined_warning(
            py,
            warn_outside_refinement,
            &self.forest,
            &cell_types,
            self.target_level,
        )?;

        Ok(CfdAxisProjectedMesh::from_core(py, self.mesh.clone()))
    }
}
