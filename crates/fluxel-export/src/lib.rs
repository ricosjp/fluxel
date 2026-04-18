//! Fluxel export module.
//!
//! Extracts and builds flat Structure-of-Arrays (SoA) layouts from a `Forest` (space-filling curve
//! ordering) for direct use in CFD solvers.

pub mod builder;
pub mod mesh;

pub use builder::{build_axis_projected_mesh, build_ghost_cell_mesh};
pub use mesh::{CfdAxisProjectedMesh, CfdGhostCellMesh};
