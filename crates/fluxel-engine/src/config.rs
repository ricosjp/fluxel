use crate::BuildError;
use fluxel_core::validate_resolution;
use fluxel_geometry::{BoundingBox, Geometry};
use fluxel_sfc::MAX_LEVEL;

#[derive(Debug, Clone, Copy)]
/// Resource policy applied to every build and inherited by session remeshing.
pub struct BuildLimits {
    /// Maximum leaf-cell count, including balancing and final uniform refinement.
    /// The default is `usize::MAX`. This is a rejection threshold, not a requested
    /// mesh size or a cap on bytes, faces, temporary allocations, or resident memory.
    pub max_cells: usize,
}
impl Default for BuildLimits {
    fn default() -> Self {
        Self {
            max_cells: usize::MAX,
        }
    }
}
#[derive(Debug, Clone)]
/// Validated domain and root-grid settings shared by builds and sessions.
pub struct MeshBuildConfig {
    pub(crate) bbox: BoundingBox,
    pub(crate) base_res: [u32; 3],
    pub(crate) leaf_refinement: u8,
    pub(crate) limits: BuildLimits,
}
impl MeshBuildConfig {
    /// Validate physical bounds, positive root resolution, and final leaf refinement.
    /// `base_res` counts root trees along X/Y/Z; `leaf_refinement` adds uniform octree
    /// levels after surface/region refinement and balancing. Lengths use caller units.
    ///
    /// # Errors
    /// Rejects more than 2^26 roots, roots exceeding `limits.max_cells`, nonpositive
    /// or nonfinite physical root widths, and refinement above `MAX_LEVEL`.
    /// The combined surface/region and leaf level is checked when building a grid.
    pub fn new(
        bbox: BoundingBox,
        base_res: [u32; 3],
        leaf_refinement: u8,
        limits: BuildLimits,
    ) -> Result<Self, BuildError> {
        let roots = validate_resolution(base_res)?;
        Geometry::new(bbox, base_res)?;
        if roots > limits.max_cells {
            return Err(fluxel_core::ForestError::CellLimit {
                limit: limits.max_cells,
            }
            .into());
        }
        validate_level(leaf_refinement)?;
        Ok(Self {
            bbox,
            base_res,
            leaf_refinement,
            limits,
        })
    }
}
#[derive(Debug, Clone)]
/// A physical box with a minimum requested level before final uniform refinement.
pub struct RefinementRegion {
    pub(crate) bounds: BoundingBox,
    pub(crate) level: u8,
}
impl RefinementRegion {
    /// Request refinement for cells with positive-volume overlap with `bounds`.
    /// Face-only contact is excluded; portions outside the domain have no effect.
    /// Finer cells are never coarsened. Errors if `level` exceeds `MAX_LEVEL`.
    pub fn new(bounds: BoundingBox, level: u8) -> Result<Self, BuildError> {
        validate_level(level)?;
        Ok(Self { bounds, level })
    }
}
#[derive(Debug, Clone)]
/// Surface and regional targets; neither limits the final balanced/uniform cell levels.
pub struct RefinementPlan {
    pub(crate) target_level: u8,
    pub(crate) regions: Vec<RefinementRegion>,
}
impl RefinementPlan {
    /// Create a surface target and region requests; an empty region list adds none.
    /// Errors if `target_level` exceeds `MAX_LEVEL`. Compatibility with a build's
    /// uniform refinement is checked later; this constructor allocates no mesh.
    pub fn new(target_level: u8, regions: Vec<RefinementRegion>) -> Result<Self, BuildError> {
        validate_level(target_level)?;
        Ok(Self {
            target_level,
            regions,
        })
    }
    /// Surface target before final uniform refinement.
    pub fn target_level(&self) -> u8 {
        self.target_level
    }
    /// Reject requested levels whose sum with final uniform refinement exceeds MAX_LEVEL.
    pub(crate) fn validate(&self, config: &MeshBuildConfig) -> Result<(), BuildError> {
        let max = self
            .regions
            .iter()
            .map(|r| r.level)
            .fold(self.target_level, u8::max);
        if max > MAX_LEVEL - config.leaf_refinement {
            return Err(fluxel_core::ForestError::LevelLimit.into());
        }
        Ok(())
    }
}
/// Reject a level above the logical key representation limit.
pub(crate) fn validate_level(level: u8) -> Result<(), BuildError> {
    if level > MAX_LEVEL {
        return Err(fluxel_core::ForestError::LevelLimit.into());
    }
    Ok(())
}
