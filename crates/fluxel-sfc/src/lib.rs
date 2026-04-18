/// Maximum refinement depth for octree cells (`0` = coarsest root, `MAX_LEVEL` = finest).
///
/// Logical coordinates use 32-bit unsigned integers; `level` counts how many times the domain
/// has been subdivided from the root cell.
pub const MAX_LEVEL: u8 = 32;

pub mod key;
pub mod morton;

pub use key::{Key, LogicalCell};
