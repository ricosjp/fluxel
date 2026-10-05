use crate::BuildError;
use fluxel_core::validate_resolution;
use fluxel_geometry::{BoundingBox, CylindricalGeometry, Geometry, SpatialDomain};
use fluxel_sfc::{LogicalCell, MAX_LEVEL};

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
pub(crate) enum BuiltDomain {
    Cartesian(Geometry),
    Cylindrical(CylindricalGeometry),
}
#[derive(Debug, Clone)]
/// Validated domain and root-grid settings shared by builds and sessions.
pub struct MeshBuildConfig {
    pub(crate) domain: BuiltDomain,
    pub(crate) base_res: [u32; 3],
    pub(crate) leaf_refinement: u8,
    pub(crate) limits: BuildLimits,
    /// Logical axes that wrap across the root grid: Cartesian X/Y/Z or cylindrical
    /// ``(r, θ, z)``.
    pub(crate) periodic: [bool; 3],
}
impl MeshBuildConfig {
    /// Validate physical bounds, positive root resolution, and final leaf refinement.
    /// `base_res` counts root trees along X/Y/Z; `leaf_refinement` adds uniform octree
    /// levels after surface/region refinement and balancing. Lengths use caller units.
    /// `periodic` is `[x, y, z]` and wraps the outer faces of each true axis.
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
        periodic: [bool; 3],
    ) -> Result<Self, BuildError> {
        let roots = validate_resolution(base_res)?;
        if roots > limits.max_cells {
            return Err(fluxel_core::ForestError::CellLimit {
                limit: limits.max_cells,
            }
            .into());
        }
        validate_level(leaf_refinement)?;
        Ok(Self {
            domain: BuiltDomain::Cartesian(Geometry::new(bbox, base_res)?),
            base_res,
            leaf_refinement,
            limits,
            periodic,
        })
    }

    /// Annular sector or full turn about world Z.
    #[allow(clippy::too_many_arguments)]
    ///
    /// `theta_extent` is radians. A full turn requires `base_res[1] >= 4` so each
    /// root angle is at most `π/2`. `periodic` is `[r, θ, z]` and wraps each true
    /// axis across the root grid. Lengths use caller units.
    ///
    /// # Errors
    /// Rejects the same root and level limits as [`Self::new`], plus a nonpositive
    /// radius, an angular extent outside `(0, 2π]`, or a root angle above `π/2`.
    pub fn cylindrical(
        origin: [f64; 3],
        r_min: f64,
        r_max: f64,
        theta_start: f64,
        theta_extent: f64,
        z_min: f64,
        z_max: f64,
        base_res: [u32; 3],
        leaf_refinement: u8,
        limits: BuildLimits,
        periodic: [bool; 3],
    ) -> Result<Self, BuildError> {
        let roots = validate_resolution(base_res)?;
        if roots > limits.max_cells {
            return Err(fluxel_core::ForestError::CellLimit {
                limit: limits.max_cells,
            }
            .into());
        }
        validate_level(leaf_refinement)?;
        let geometry = CylindricalGeometry::from_extent(
            origin,
            r_min,
            r_max,
            theta_start,
            theta_extent,
            z_min,
            z_max,
        )?;
        geometry.cell_interval(&LogicalCell::root(0), base_res)?;
        Ok(Self {
            domain: BuiltDomain::Cylindrical(geometry),
            base_res,
            leaf_refinement,
            limits,
            periodic,
        })
    }

    /// True when builds use parameter intervals `(r, θ, z)` rather than world boxes.
    pub fn is_cylindrical(&self) -> bool {
        matches!(self.domain, BuiltDomain::Cylindrical(_))
    }

    pub(crate) fn spatial_domain(&self) -> SpatialDomain {
        match &self.domain {
            BuiltDomain::Cartesian(geometry) => SpatialDomain::Cartesian(geometry.clone()),
            BuiltDomain::Cylindrical(geometry) => SpatialDomain::Cylindrical(*geometry),
        }
    }

    /// Logical axes that wrap, in domain order.
    pub(crate) fn periodic_axes(&self) -> [bool; 3] {
        self.periodic
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
/// Parameter interval `(r, θ, z)` or, for a Cartesian build, world `(x, y, z)`.
///
/// A full-turn region may cross the angular seam by giving `max[1] < min[1]`.
/// That wrap is rejected on a sector and on a Cartesian domain.
pub struct ParameterRegion {
    pub(crate) min: [f64; 3],
    pub(crate) max: [f64; 3],
    pub(crate) level: u8,
}
impl ParameterRegion {
    /// Request refinement where the cell parameter box overlaps this interval.
    ///
    /// Radial and axial limits must be strictly ordered. The angular component may
    /// wrap. Errors if `level` exceeds `MAX_LEVEL` or a non-angular limit is empty.
    pub fn new(min: [f64; 3], max: [f64; 3], level: u8) -> Result<Self, BuildError> {
        validate_level(level)?;
        if !min.iter().chain(max.iter()).all(|value| value.is_finite())
            || min[0] >= max[0]
            || min[2] >= max[2]
            || min[1] == max[1]
        {
            return Err(BuildError::InvalidInput(
                "parameter region requires finite bounds, ordered r and z, and a nonzero angle"
                    .into(),
            ));
        }
        Ok(Self { min, max, level })
    }
    /// Inclusive lower corner.
    pub fn min(&self) -> [f64; 3] {
        self.min
    }
    /// Upper corner. The angular component may be below `min[1]` when the region wraps.
    pub fn max(&self) -> [f64; 3] {
        self.max
    }
    /// Requested level before final uniform refinement.
    pub fn level(&self) -> u8 {
        self.level
    }
}
#[derive(Debug, Clone)]
/// Surface and regional targets; neither limits the final balanced/uniform cell levels.
pub struct RefinementPlan {
    pub(crate) target_level: u8,
    pub(crate) regions: Vec<RefinementRegion>,
    pub(crate) parameter_regions: Vec<ParameterRegion>,
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
            parameter_regions: Vec::new(),
        })
    }
    /// Surface target plus parameter-interval regions. World-box regions stay empty.
    pub fn from_parameter(
        target_level: u8,
        parameter_regions: Vec<ParameterRegion>,
    ) -> Result<Self, BuildError> {
        validate_level(target_level)?;
        Ok(Self {
            target_level,
            regions: Vec::new(),
            parameter_regions,
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
            .map(|region| region.level)
            .chain(self.parameter_regions.iter().map(|region| region.level))
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
