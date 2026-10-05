use crate::{
    ApibmMesh, BuildError, BuildOutput, BuildReport, MeshBuildConfig, MeshSnapshot, RefinementPlan,
};
use fluxel_geometry::RigidPose;
use fluxel_ibm::{
    classify_intersections, compute_apibm, compute_gcibm, ApIbmData, Boundary, BoundaryState,
    GhostCellData, IntersectionMask,
};
use fluxel_mesh::GridContext;
use std::sync::Arc;

/// Count final-grid cells, faces, and intersecting levels without geometry queries.
/// `mask` must come from this grid and boundary. Comparisons use `target_level`
/// directly, without adding the final uniform refinement count.
pub(crate) fn report(grid: &GridContext, mask: &IntersectionMask, target_level: u8) -> BuildReport {
    let levels: Vec<_> = grid
        .forest()
        .keys()
        .iter()
        .zip(mask.values())
        .filter(|(_, hit)| **hit)
        .map(|(key, _)| key.level())
        .collect();
    BuildReport {
        n_cells: grid.background().n_cells(),
        n_internal_faces: grid.background().topology().n_internal_faces(),
        under_refined_cells: levels.iter().filter(|&&l| l < target_level).count(),
        minimum_intersect_level: levels.into_iter().min(),
        ..Default::default()
    }
}
/// Recompute the intersection mask and APIBM payload while sharing the background.
/// Propagates query/classification errors without changing `grid` or `boundary`.
pub(crate) fn apibm_on_grid(
    grid: &GridContext,
    boundary: &BoundaryState,
    target_level: u8,
) -> Result<(ApibmMesh, BuildReport), BuildError> {
    let mask = classify_intersections(grid, boundary)?;
    let payload = compute_apibm(grid, boundary, &mask)?;
    let mut report = report(grid, &mask, target_level);
    report.n_ibm_records = payload.dist_owner_to_bnd().len();
    Ok((
        MeshSnapshot {
            background: grid.background().clone(),
            patches: boundary.patches().clone(),
            boundary_revision: boundary.revision(),
            payload: Arc::new(payload),
        },
        report,
    ))
}
/// Build a new grid and APIBM payload as one candidate, retaining the forest for a session.
/// Failure discards local work; inputs remain unchanged.
pub(crate) fn prepare_apibm(
    config: &MeshBuildConfig,
    boundary: &BoundaryState,
    plan: &RefinementPlan,
) -> Result<(GridContext, ApibmMesh, BuildReport), BuildError> {
    let grid = crate::refinement::build_grid(config, boundary, plan)?;
    let (mesh, report) = apibm_on_grid(&grid, boundary, plan.target_level)?;
    Ok((grid, mesh, report))
}
/// Build an immutable background and compressed axis-projected payload.
///
/// The boundary uses its original coordinates (identity pose). Surface refinement
/// and region refinement precede 2:1 balancing and final uniform leaf refinement.
/// `Boundary::None` skips surface refinement and yields an all-false face mask.
/// The returned snapshot owns shared immutable arrays; its indices belong only
/// to that background. The report records final-grid intersection/refinement counts.
///
/// # Errors
/// Returns `BuildError` for invalid combined refinement levels, cell-count limits,
/// invalid background geometry, or failed boundary queries. No caller state changes.
///
/// # Example
/// ```
/// use fluxel_engine::{build_apibm, BuildLimits, MeshBuildConfig, RefinementPlan};
/// use fluxel_geometry::BoundingBox;
/// use fluxel_ibm::Boundary;
/// let config = MeshBuildConfig::new(
///     BoundingBox::new([0.0; 3], [1.0; 3])?, [1; 3], 0,
///     BuildLimits { max_cells: 8 },
///     [false; 3],
/// )?;
/// let output = build_apibm(&config, Boundary::None, RefinementPlan::new(0, vec![])?)?;
/// assert_eq!(output.mesh.background().n_cells(), 1);
/// # Ok::<(), fluxel_engine::BuildError>(())
/// ```
pub fn build_apibm(
    config: &MeshBuildConfig,
    boundary: Boundary,
    plan: RefinementPlan,
) -> Result<BuildOutput<ApIbmData>, BuildError> {
    let boundary = BoundaryState::new(boundary, RigidPose::identity());
    let (_, mesh, report) = prepare_apibm(config, &boundary, &plan)?;
    Ok(BuildOutput { mesh, report })
}
/// Build a background, final fluid classification, and ghost stencils.
///
/// Refinement follows [`build_apibm`]. `seed` is a finite physical point in the
/// half-open domain; its containing cell must not intersect the surface. Even with
/// `Boundary::None`, the seed is checked before returning an all-fluid, no-ghost mesh.
/// The result includes fallback and unresolved-stencil counts in its report.
/// Unresolved rows reference the ghost cell itself with weight one; callers must
/// inspect the report before treating all rows as valid fluid interpolation.
///
/// # Errors
/// Returns `BuildError` for invalid levels, cell limits, invalid seed location or
/// seed cell, and failed geometry queries. Interpolation fallback alone is not an
/// error. No caller state changes.
pub fn build_gcibm(
    config: &MeshBuildConfig,
    boundary: Boundary,
    plan: RefinementPlan,
    seed: [f64; 3],
) -> Result<BuildOutput<GhostCellData>, BuildError> {
    let boundary = BoundaryState::new(boundary, RigidPose::identity());
    let grid = crate::refinement::build_grid(config, &boundary, &plan)?;
    let mask = classify_intersections(&grid, &boundary)?;
    let (payload, diagnostics) = compute_gcibm(&grid, &boundary, &mask, seed)?;
    let mut report = report(&grid, &mask, plan.target_level);
    report.n_ibm_records = payload.gc_cell_ids().len();
    report.interpolation_fallbacks = diagnostics.interpolation_fallbacks;
    report.unresolved_stencils = diagnostics.unresolved_stencils;
    let mesh = MeshSnapshot {
        background: grid.background().clone(),
        boundary_revision: boundary.revision(),
        patches: boundary.patches().clone(),
        payload: Arc::new(payload),
    };
    Ok(BuildOutput { mesh, report })
}
