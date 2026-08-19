//! Fluxel IBM (Immersed Boundary Method) module.
//!
//! This crate handles intersection tests between boundary geometry (e.g., STL)
//! and orthogonal AMR meshes, inside/outside classification (solid voxelization),
//! and ghost-cell geometry construction.

pub mod mesh;
pub mod pose;
pub mod solver;
pub mod types;

pub use mesh::IBMMesh;
pub use pose::{pose_from_translation_quaternion, quaternion_from_axis_angle};
pub use solver::{mark_intersecting_cells, resolve_apibm_face};
pub use types::*;
