//! Ghost-cell geometry with explicit final classification and diagnostics.
pub mod classification;
mod payload;
mod stencil;
use crate::{BoundaryState, IbmError, IntersectionMask};
use fluxel_mesh::GridContext;
pub use payload::GhostCellData;
use rayon::prelude::*;
#[derive(Debug, Default, Clone, Copy)]
/// Interpolation outcomes; fallback includes unresolved rows, not just successful IDW.
pub struct GcDiagnostics {
    /// Number of rows using inverse-distance fallback instead of linear least squares.
    pub interpolation_fallbacks: usize,
    /// Number of rows with zero fluid candidates; they self-reference and need caller handling.
    pub unresolved_stencils: usize,
}
/// Classify fluid cells, then build rows for non-fluid cells touching fluid by a face.
/// The seed uses physical coordinates and must occupy a non-intersecting cell in
/// the half-open domain. Output rows follow increasing forest cell index; unused
/// slots in each eight-column stencil have zero weight.
///
/// Weighted linear least squares falls back to inverse-distance interpolation.
/// With no candidates the row references itself with weight one; this is not a
/// valid fluid stencil. Callers must inspect the returned diagnostics. A boundary
/// of None still validates the seed, then returns all-fluid flags and no ghost rows.
///
/// # Errors
/// Rejects stale masks, seeds outside the domain (including nonfinite points),
/// seeds in intersecting cells, and failed geometry queries. Interpolation fallback
/// and unresolved rows are reported, not raised. Inputs remain unchanged.
pub fn compute_gcibm(
    grid: &GridContext,
    boundary: &BoundaryState,
    mask: &IntersectionMask,
    seed: [f64; 3],
) -> Result<(GhostCellData, GcDiagnostics), IbmError> {
    let classification = classification::classify_fluid(grid, boundary, mask, seed)?;
    let mut data = GhostCellData {
        gc_is_fluid: classification.values().to_vec(),
        ..Default::default()
    };
    let mut diagnostics = GcDiagnostics::default();
    let Some(surface) = boundary.surface() else {
        return Ok((data, diagnostics));
    };
    let rows: Result<Vec<_>, _> = (0..grid.background().n_cells())
        .into_par_iter()
        .map(|id| {
            stencil::try_build_ghost_stencil_row(
                id,
                grid.forest(),
                grid.domain(),
                surface,
                &boundary.pose(),
                classification.values(),
            )
        })
        .collect();
    for row in rows?.into_iter().flatten() {
        data.gc_cell_ids.push(row.cell_id);
        data.gc_bnd_anchor_ids.push(row.bnd_anchor_id);
        data.gc_bnd_patch_ids.push(row.bnd_patch_id);
        data.gc_bnd_intercepts.push(row.bnd_intercept);
        data.gc_image_points.push(row.image_point);
        data.gc_interp_stencil_indices.push(row.stencil_indices);
        data.gc_interp_stencil_weights.push(row.stencil_weights);
        diagnostics.interpolation_fallbacks += usize::from(row.fallback);
        diagnostics.unresolved_stencils += usize::from(row.unresolved);
    }
    Ok((data, diagnostics))
}
