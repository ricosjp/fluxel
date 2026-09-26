//! A complete forest whose balance cannot be invalidated through mutable access.
use crate::{Forest, ForestError};

#[derive(Debug)]
/// Proof of complete root coverage and 2:1 balance, exposing no mutable forest access.
pub struct BalancedForest(Forest);

impl Forest {
    /// Establishes the invariant once, before the final uniform refinement.
    /// Consumes the forest; returns coverage or balancing errors. The caller retains no original on failure.
    pub fn into_balanced(mut self) -> Result<BalancedForest, ForestError> {
        self.validate_coverage()?;
        self.enforce_2_to_1_balance()?;
        Ok(BalancedForest(self))
    }
}

impl BalancedForest {
    /// Uniform splitting preserves both complete coverage and 2:1 balance.
    /// Consumes the proof; errors on final level or cell limit, without returning the consumed forest.
    pub fn uniform_refinement(mut self, levels: u8) -> Result<Self, ForestError> {
        self.0.uniform_refinement(levels)?;
        Ok(self)
    }

    /// Consume the proof and recover the owned forest; later mutation may invalidate balance.
    pub fn into_forest(self) -> Forest {
        self.0
    }
}
