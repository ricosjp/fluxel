//! Immutable background topology shared by every immersed-boundary method.
mod background;
mod build;
mod id;
pub use background::{BackgroundMesh, CellGeometry, GridContext, MeshError, MeshTopology};
pub use id::{CellIndex, FaceIndex, MeshId};
