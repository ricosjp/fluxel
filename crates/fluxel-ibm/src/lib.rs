//! Fluxel IBM (Immersed Boundary Method) module.
//!
//! This crate handles intersection tests between boundary geometry (e.g., STL)
//! and orthogonal AMR meshes, inside/outside classification (solid voxelization),
//! and ghost-cell geometry construction.

pub mod mesh;
pub mod solver;
pub mod types;

pub use mesh::IBMMesh;
pub use solver::{mark_intersecting_cells, resolve_apibm_face};
pub use types::*;
