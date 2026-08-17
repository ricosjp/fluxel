//! Fluxel export module.
//!
//! Extracts and builds flat Structure-of-Arrays (SoA) layouts from a `Forest` (space-filling curve
//! ordering) for direct use in CFD solvers.

pub mod builder;
pub mod mesh;

pub use builder::{
    build_axis_projected_mesh, build_ghost_cell_mesh, fill_ap_ibm_face_data,
    has_under_refined_intersect_cells, rebuild_axis_projected_ib,
};
pub use mesh::{ApIbmFaceData, CfdAxisProjectedMesh, CfdGhostCellMesh};
