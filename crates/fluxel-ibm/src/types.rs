//! Core data types for IBM processing.

/// Cell state used by the IBM classification pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CellType {
    /// Fluid region (active in flow computation).
    Fluid = 0,
    /// Solid region (excluded from flow computation).
    Solid = 1,
    /// Cell geometrically intersecting the boundary (intersect cell).
    Intersect = 2,
}

/// Data structure for ghost cell interpolation stencils.
#[derive(Debug, Clone)]
pub struct GhostCellData {
    /// Cell classification (`0`=Fluid, `1`=Solid).
    pub gc_is_fluid: Vec<bool>,
    /// Forest cell indices for computational ghost cells.
    pub gc_cell_ids: Vec<usize>,
    /// Boundary anchor ids (face ids on the surface mesh).
    pub gc_bnd_anchor_ids: Vec<usize>,
    /// Boundary patch ids (patch ids on the surface mesh).
    pub gc_bnd_patch_ids: Vec<usize>,
    /// Boundary intercept points per ghost cell.
    pub gc_bnd_intercepts: Vec<[f64; 3]>,
    /// Mirrored image-point coordinates per ghost cell.
    pub gc_image_points: Vec<[f64; 3]>,
    /// Interpolation stencil indices (`N_ghost` x 8).
    pub gc_interp_stencil_indices: Vec<[usize; 8]>,
    /// Interpolation stencil weights (`N_ghost` x 8).
    pub gc_interp_stencil_weights: Vec<[f64; 8]>,
}
