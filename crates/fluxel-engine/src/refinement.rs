//! Refinement decisions are application policy; splitting and balancing remain in core.
use crate::{BuildError, MeshBuildConfig, RefinementPlan, RefinementRegion};
use fluxel_core::Forest;
use fluxel_geometry::Geometry;
use fluxel_ibm::{intersect_cells, BoundaryState};
use fluxel_mesh::GridContext;
use rayon::prelude::*;

/// Split cells with positive-volume overlap until all regional targets are met.
/// Multiple regions select the finest requested level. Mutates progressively:
/// a later limit error may follow earlier successful splits. Call on a candidate
/// forest when rollback is required. Does not balance or coarsen.
pub(crate) fn refine_regions(
    forest: &mut Forest,
    geometry: &Geometry,
    regions: &[RefinementRegion],
) -> Result<(), BuildError> {
    loop {
        let flags: Vec<_> = forest
            .keys()
            .par_iter()
            .map(|key| {
                let (center, size) = geometry.cell_bounds(&key.to_logical());
                regions.iter().any(|region| {
                    key.level() < region.level && region.bounds.overlaps_cell(center, size)
                })
            })
            .collect();
        if !flags.iter().any(|&f| f) {
            break;
        }
        forest.refine_by_flags(&flags)?;
    }
    Ok(())
}
/// Build roots, refine surfaces and regions, balance, uniformly split, then extract topology.
/// All work is local. Invalid combined levels, cell limits or boundary queries fail
/// without changing inputs. The balanced wrapper avoids a second neighbour scan.
pub(crate) fn build_grid(
    config: &MeshBuildConfig,
    boundary: &BoundaryState,
    plan: &RefinementPlan,
) -> Result<GridContext, BuildError> {
    plan.validate(config)?;
    let geometry = Geometry::new(config.bbox, config.base_res)?;
    let mut forest = Forest::with_cell_limit(config.base_res, config.limits.max_cells)?;
    forest.populate_root_cells();
    for _ in 0..plan.target_level {
        let mask = intersect_cells(&forest, &geometry, boundary)?;
        let flags: Vec<_> = forest
            .keys()
            .iter()
            .zip(mask)
            .map(|(key, hit)| hit && key.level() < plan.target_level)
            .collect();
        if !flags.iter().any(|&f| f) {
            break;
        }
        forest.refine_by_flags(&flags)?;
    }
    refine_regions(&mut forest, &geometry, &plan.regions)?;
    let forest = forest
        .into_balanced()?
        .uniform_refinement(config.leaf_refinement)?;
    Ok(GridContext::from_balanced(forest, geometry)?)
}
