//! Cylindrical annular sector with positive inner radius.
//!
//! File roles match [`crate::cartesian`]: [`domain`] is the physical mapping and
//! [`locate`] turns a world point into logical octree coordinates. [`metrics`]
//! holds the annular volume and face measures, which have no Cartesian file
//! because a Cartesian cell is a box. [`intersect`] is the exact wedge test used
//! after the shared axis-aligned candidate filter.

pub mod domain;
mod intersect;
pub mod locate;
pub mod metrics;

pub use domain::{CylindricalGeometry, ParameterInterval};
pub use metrics::{sector_faces, CellMetrics, FaceMetrics, ParameterFace, QuadraturePoint};
