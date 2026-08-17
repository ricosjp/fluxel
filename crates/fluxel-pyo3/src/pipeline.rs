//! Shared AMR preprocessing helpers used by [`FluxelManager`] and [`ApibmSession`].

use fluxel_core::Forest as CoreForest;
use fluxel_geometry::{BoundingBox as CoreBoundingBox, Geometry};
use fluxel_ibm::{mesh::IBMMesh, solver, CellType};
use fluxel_sfc::MAX_LEVEL;
use parry3d_f64::math::Pose;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::Path;

pub(crate) type RefinementRegion = ([f64; 3], [f64; 3], u8);

/// Domain / refinement settings shared by one-shot builds and sessions.
#[derive(Clone, Copy)]
pub(crate) struct MeshBuildConfig {
    pub core_bbox: CoreBoundingBox,
    pub base_res: [u32; 3],
    pub n_leaf_refinement: u8,
}

/// Returns `true` if the cell AABB overlaps the axis-aligned box `[min, max]`.
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

/// Loads an STL / OBJ IBM mesh, or a dummy triangle far outside the domain when
/// `mesh_path` is `None`.
pub(crate) fn load_ibm_mesh(mesh_path: Option<&str>) -> PyResult<IBMMesh> {
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

/// Refines cells that intersect any region until each region reaches its target
/// level.
pub(crate) fn refine_regions_to_level(
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

/// Builds an AMR forest for an already-loaded IBM mesh under `pose`.
pub(crate) fn build_forest_for_ibm(
    config: &MeshBuildConfig,
    ibm_mesh: &IBMMesh,
    target_level: u8,
    refinement_regions: &[RefinementRegion],
    pose: &Pose,
) -> PyResult<(CoreForest, Geometry)> {
    let geom = Geometry::new(config.core_bbox, config.base_res);

    let mut forest = CoreForest::new(config.base_res);
    forest.populate_root_cells();

    for _ in 0..target_level {
        let current_cell_types = solver::mark_intersecting_cells(&forest, &geom, ibm_mesh, pose);

        let refine_flags: Vec<bool> = forest
            .keys()
            .par_iter()
            .zip(current_cell_types.par_iter())
            .map(|(key, cell_type)| *cell_type == CellType::Intersect && key.level() < target_level)
            .collect();

        if !refine_flags.par_iter().any(|&flag| flag) {
            break;
        }

        forest.refine_by_flags(&refine_flags);
    }

    refine_regions_to_level(&mut forest, &geom, refinement_regions);

    forest.enforce_2_to_1_balance();
    forest.uniform_refinement(config.n_leaf_refinement);

    Ok((forest, geom))
}

/// Shared preprocessing pipeline used by both GCIBM and APIBM mesh builders.
pub(crate) fn prepare_forest_and_mesh(
    config: &MeshBuildConfig,
    mesh_path: Option<&str>,
    target_level: u8,
    refinement_regions: &[RefinementRegion],
    pose: &Pose,
) -> PyResult<(CoreForest, Geometry, IBMMesh)> {
    let ibm_mesh = load_ibm_mesh(mesh_path)?;
    build_forest_for_ibm(config, &ibm_mesh, target_level, refinement_regions, pose)
        .map(|(forest, geom)| (forest, geom, ibm_mesh))
}
