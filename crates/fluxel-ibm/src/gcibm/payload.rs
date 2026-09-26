/// Data structure for ghost cell interpolation stencils.
#[derive(Debug, Clone, Default)]
/// `gc_is_fluid` spans all background cells. Other arrays share the order of
/// `gc_cell_ids`; stencil rows have eight slots. Ignore zero-weight slots.
/// Linear-fit weights can be negative. Unresolved rows self-reference with weight
/// one; pair this data with the diagnostics returned by compute_gcibm.
pub struct GhostCellData {
    /// Per-background-cell classification: `true` means fluid.
    pub(crate) gc_is_fluid: Vec<bool>,
    /// Forest cell indices for computational ghost cells.
    pub(crate) gc_cell_ids: Vec<usize>,
    /// Boundary anchor ids (face ids on the surface mesh).
    pub(crate) gc_bnd_anchor_ids: Vec<usize>,
    /// Boundary patch ids (patch ids on the surface mesh).
    pub(crate) gc_bnd_patch_ids: Vec<usize>,
    /// Boundary intercept points per ghost cell.
    pub(crate) gc_bnd_intercepts: Vec<[f64; 3]>,
    /// Mirrored image-point coordinates per ghost cell.
    pub(crate) gc_image_points: Vec<[f64; 3]>,
    /// Interpolation stencil indices (`N_ghost` x 8).
    pub(crate) gc_interp_stencil_indices: Vec<[usize; 8]>,
    /// Interpolation stencil weights (`N_ghost` x 8).
    pub(crate) gc_interp_stencil_weights: Vec<[f64; 8]>,
}

impl GhostCellData {
    /// True for fluid background cells, in forest order.
    pub fn gc_is_fluid(&self) -> &[bool] {
        &self.gc_is_fluid
    }
    /// Indices of computational ghost cells into this background.
    pub fn gc_cell_ids(&self) -> &[usize] {
        &self.gc_cell_ids
    }
    /// Nearest boundary triangle ID for each ghost row.
    pub fn gc_bnd_anchor_ids(&self) -> &[usize] {
        &self.gc_bnd_anchor_ids
    }
    /// Patch-table index at each nearest boundary point.
    pub fn gc_bnd_patch_ids(&self) -> &[usize] {
        &self.gc_bnd_patch_ids
    }
    /// Nearest boundary points in physical coordinates.
    pub fn gc_bnd_intercepts(&self) -> &[[f64; 3]] {
        &self.gc_bnd_intercepts
    }
    /// Physical reflected points: twice the intercept minus the ghost center.
    pub fn gc_image_points(&self) -> &[[f64; 3]] {
        &self.gc_image_points
    }
    /// Eight background indices per row; only nonzero-weight slots are meaningful.
    pub fn gc_interp_stencil_indices(&self) -> &[[usize; 8]] {
        &self.gc_interp_stencil_indices
    }
    /// Eight interpolation weights per row; see the type-level fallback contract.
    pub fn gc_interp_stencil_weights(&self) -> &[[f64; 8]] {
        &self.gc_interp_stencil_weights
    }
}
