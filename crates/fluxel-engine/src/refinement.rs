//! Refinement decisions are application policy; splitting and balancing remain in core.
use crate::{BuildError, MeshBuildConfig, ParameterRegion, RefinementPlan, RefinementRegion};
use fluxel_core::Forest;
use fluxel_geometry::{Geometry, ParameterBox, SpatialDomain};
use fluxel_ibm::{intersect_domain, BoundaryState};
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

fn refine_parameter_regions(
    forest: &mut Forest,
    domain: &SpatialDomain,
    regions: &[ParameterRegion],
) -> Result<(), BuildError> {
    if regions.is_empty() {
        return Ok(());
    }
    loop {
        let flags: Result<Vec<_>, BuildError> = forest
            .keys()
            .par_iter()
            .enumerate()
            .map(|(id, key)| {
                let cell = domain
                    .parameter_box(forest, id)
                    .map_err(|error| BuildError::InvalidInput(error.to_string()))?;
                Ok(regions.iter().any(|region| {
                    key.level() < region.level() && parameter_overlap(domain, cell, region)
                }))
            })
            .collect();
        let flags = flags?;
        if !flags.iter().any(|&flag| flag) {
            break;
        }
        forest.refine_by_flags(&flags)?;
    }
    Ok(())
}

fn parameter_overlap(domain: &SpatialDomain, cell: ParameterBox, region: &ParameterRegion) -> bool {
    let min = region.min();
    let max = region.max();
    if !(cell.min[0] < max[0]
        && min[0] < cell.max[0]
        && cell.min[2] < max[2]
        && min[2] < cell.max[2])
    {
        return false;
    }
    match domain {
        SpatialDomain::Cartesian(_) => cell.min[1] < max[1] && min[1] < cell.max[1],
        SpatialDomain::Cylindrical(geometry) => {
            let start = geometry.theta_start();
            let end = start + geometry.theta_extent();
            if min[1] < max[1] {
                cell.min[1] < max[1] && min[1] < cell.max[1]
            } else if geometry.is_full_turn() {
                angular_overlap(cell.min[1], cell.max[1], min[1], end)
                    || angular_overlap(cell.min[1], cell.max[1], start, max[1])
            } else {
                false
            }
        }
    }
}

fn angular_overlap(cell_min: f64, cell_max: f64, region_min: f64, region_max: f64) -> bool {
    cell_min < region_max && region_min < cell_max
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
    if config.is_cylindrical() && !plan.regions.is_empty() {
        return Err(BuildError::InvalidInput(
            "cylindrical builds accept parameter regions, not world boxes".into(),
        ));
    }
    let domain = config.spatial_domain();
    let mut forest = Forest::with_periodic_cell_limit(
        config.base_res,
        config.limits.max_cells,
        config.periodic_axes(),
    )?;
    forest.populate_root_cells();
    // Surface hits are split up to the target, then regions, balance, and uniform leaves.
    for _ in 0..plan.target_level {
        let mask = intersect_domain(&forest, &domain, boundary)?;
        let flags: Vec<_> = forest
            .keys()
            .iter()
            .zip(mask)
            .map(|(key, hit)| hit && key.level() < plan.target_level)
            .collect();
        if !flags.iter().any(|&flag| flag) {
            break;
        }
        forest.refine_by_flags(&flags)?;
    }
    if let SpatialDomain::Cartesian(geometry) = &domain {
        refine_regions(&mut forest, geometry, &plan.regions)?;
    }
    refine_parameter_regions(&mut forest, &domain, &plan.parameter_regions)?;
    let forest = forest
        .into_balanced()?
        .uniform_refinement(config.leaf_refinement)?;
    Ok(match domain {
        SpatialDomain::Cartesian(geometry) => GridContext::from_balanced(forest, geometry)?,
        SpatialDomain::Cylindrical(geometry) => {
            GridContext::from_cylindrical(forest.into_forest(), geometry)?
        }
    })
}
