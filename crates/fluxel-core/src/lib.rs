//! Core AMR infrastructure: [`Forest`] and algorithms over a sorted [`fluxel_sfc::Key`] array.
//!
//! The forest stores no explicit parent/child pointers; space-filling-curve order is the single
//! source of truth for fast neighbour queries, point location, and bulk refine/coarsen.

pub mod balance;
pub mod enums;
pub mod forest;
pub mod neighbour;
pub mod query;
pub mod refine;

pub use enums::{Axis, CoordinateType, Direction};
pub use forest::Forest;
