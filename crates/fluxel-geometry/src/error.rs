use std::fmt;
#[derive(Debug, Clone, PartialEq)]
/// Invalid physical-domain or rigid-transform input.
pub enum GeometryError {
    /// Nonfinite, reversed, empty or overflowing domain extents.
    InvalidBounds,
    /// Invalid root counts or nonpositive/nonfinite physical root widths.
    InvalidResolution,
    /// Nonfinite pose/axis-angle inputs, or a zero quaternion.
    InvalidPose,
    /// Cylindrical bounds are nonfinite, empty, nonpositive in radius, or wider than one turn.
    InvalidCylindricalDomain,
}
impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidBounds => "bounds must be finite with min < max on every axis",
            Self::InvalidResolution => {
                "base_res must be nonzero with at most 2^26 trees and finite positive cell sizes"
            }
            Self::InvalidPose => "pose must be finite and quaternion must have nonzero length",
            Self::InvalidCylindricalDomain => {
                "cylindrical domain requires finite bounds, 0 < r_min < r_max, a positive extent of at most one turn, and root angle at most π/2"
            }
        })
    }
}
impl std::error::Error for GeometryError {}
