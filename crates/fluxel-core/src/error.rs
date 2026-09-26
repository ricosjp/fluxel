//! Errors for validated forest operations.
use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq)]
/// Invalid forest input or an operation exceeding its representation/resource limits.
pub enum ForestError {
    /// Zero components, overflow, or more than 2^26 root trees.
    InvalidResolution,
    /// Flag count differs from the current leaf count.
    InvalidFlags,
    /// Requested refinement exceeds MAX_LEVEL.
    LevelLimit,
    /// Requested leaf count exceeds the configured limit or overflows.
    CellLimit { limit: usize },
    /// Keys fail to cover all roots exactly once in sorted order.
    IncompleteCoverage,
}
impl fmt::Display for ForestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResolution => {
                write!(f, "base_res must be nonzero and contain at most 2^26 trees")
            }
            Self::InvalidFlags => write!(f, "Flags length must match num_cells"),
            Self::LevelLimit => write!(f, "refinement exceeds MAX_LEVEL"),
            Self::CellLimit { limit } => write!(f, "mesh exceeds cell limit {limit}"),
            Self::IncompleteCoverage => write!(f, "forest must cover every root tree exactly once"),
        }
    }
}
impl std::error::Error for ForestError {}
/// Return the root count, rejecting zero components, overflow, or more than 2^26 roots.
pub fn validate_resolution(resolution: [u32; 3]) -> Result<usize, ForestError> {
    let count = resolution
        .into_iter()
        .try_fold(1u64, |n, r| {
            (r > 0).then(|| n.checked_mul(u64::from(r))).flatten()
        })
        .filter(|&n| n <= 1 << 26)
        .ok_or(ForestError::InvalidResolution)?;
    Ok(count as usize)
}
