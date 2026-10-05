//! Cartesian mapping from the logical octree to world coordinates.
//!
//! [`domain`] stores the physical box and cell bounds. [`locate`] resolves a
//! world point to a forest cell. Cylindrical code uses the same two roles under
//! [`crate::cylindrical`].

pub mod domain;
pub mod locate;
