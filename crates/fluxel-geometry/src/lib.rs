//! Geometry utilities for mapping logical octree cells to physical space.
//!
//! This crate binds the logical coordinate domain (`0 .. 2^MAX_LEVEL` per tree axis)
//! to world coordinates. Given a global [`BoundingBox`] and base tree resolution, it
//! provides deterministic world-space center/size evaluation for each logical cell.

pub mod domain;

pub use domain::{BoundingBox, Geometry};
