//! Geometry utilities for mapping logical octree cells to physical space.
//!
//! This crate binds the logical coordinate domain (`0 .. 2^MAX_LEVEL` per tree axis)
//! to world coordinates. [`cartesian`] and [`cylindrical`] each split that mapping
//! into a physical domain and a point-location query. [`SpatialDomain`] selects one
//! of them for the mesh and immersed-boundary methods.

pub mod cartesian;
pub mod cylindrical;

pub use cartesian::domain::{BoundingBox, Geometry};
pub use cartesian::locate::get_global_id_from_phys;

pub mod coordinate;
pub mod error;
pub mod pose;
pub mod spatial;
pub use coordinate::CoordinateType;
pub use cylindrical::{
    CellMetrics, CylindricalGeometry, FaceMetrics, ParameterFace, ParameterInterval,
    QuadraturePoint,
};
pub use error::GeometryError;
pub use pose::{quaternion_from_axis_angle, RigidPose};
pub use spatial::{CellMeasure, ParameterBox, SpatialDomain};

/// Historical path for the Cartesian domain module.
pub use cartesian::domain;
/// Historical path for Cartesian point location.
pub use cartesian::locate;
