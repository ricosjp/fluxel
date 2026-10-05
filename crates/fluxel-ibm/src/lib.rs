//! Boundary geometry and method-specific IBM computations. No file or Python I/O.
pub mod apibm;
pub mod classify;
mod error;
pub mod gcibm;
pub mod surface;
pub use apibm::{compute_apibm, ApIbmData};
pub use classify::{classify_intersections, intersect_cells, intersect_domain, IntersectionMask};
pub use error::IbmError;
pub use gcibm::{compute_gcibm, GcDiagnostics, GhostCellData};
pub use surface::{Boundary, BoundaryRevision, BoundaryState, BoundarySurface, PatchTable};
