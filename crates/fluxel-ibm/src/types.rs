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
    /// Boundary intercept points per ghost cell.
    pub gc_bnd_intercepts: Vec<[f64; 3]>,
    /// Mirrored image-point coordinates per ghost cell.
    pub gc_image_points: Vec<[f64; 3]>,
    /// Interpolation stencil indices (`N_ghost` x 8).
    pub gc_interp_stencil_indices: Vec<[usize; 8]>,
    /// Interpolation stencil weights (`N_ghost` x 8).
    pub gc_interp_stencil_weights: Vec<[f64; 8]>,
}

/// Axis-projected per-face data used by APIBM reconstruction.
#[derive(Debug, Clone, Default)]
pub struct AxisProjectedData {
    /// Interior-face flags for boundary intersection along X.
    pub ap_x_has_bnd: Vec<bool>,
    /// Owner-center to boundary distance for X faces.
    pub ap_x_dist_owner_to_bnd: Vec<f64>,
    /// Neighbour-center to boundary distance for X faces.
    pub ap_x_dist_neighbour_to_bnd: Vec<f64>,
    /// Owner-side far-cell index for X faces.
    pub ap_x_owner_far_cell_id: Vec<usize>,
    /// Neighbour-side far-cell index for X faces.
    pub ap_x_neighbour_far_cell_id: Vec<usize>,
    /// APIBM reconstruction weights on the owner side for X faces.
    pub ap_x_owner_weights: Vec<[f64; 3]>,
    /// APIBM reconstruction weights on the neighbour side for X faces.
    pub ap_x_neighbour_weights: Vec<[f64; 3]>,
    /// Boundary anchor id for owner side of X faces.
    pub ap_x_owner_bnd_anchor_id: Vec<usize>,
    /// Boundary anchor id for neighbour side of X faces.
    pub ap_x_neighbour_bnd_anchor_id: Vec<usize>,

    /// Interior-face flags for boundary intersection along Y.
    pub ap_y_has_bnd: Vec<bool>,
    /// Owner-center to boundary distance for Y faces.
    pub ap_y_dist_owner_to_bnd: Vec<f64>,
    /// Neighbour-center to boundary distance for Y faces.
    pub ap_y_dist_neighbour_to_bnd: Vec<f64>,
    /// Owner-side far-cell index for Y faces.
    pub ap_y_owner_far_cell_id: Vec<usize>,
    /// Neighbour-side far-cell index for Y faces.
    pub ap_y_neighbour_far_cell_id: Vec<usize>,
    /// APIBM reconstruction weights on the owner side for Y faces.
    pub ap_y_owner_weights: Vec<[f64; 3]>,
    /// APIBM reconstruction weights on the neighbour side for Y faces.
    pub ap_y_neighbour_weights: Vec<[f64; 3]>,
    /// Boundary anchor id for owner side of Y faces.
    pub ap_y_owner_bnd_anchor_id: Vec<usize>,
    /// Boundary anchor id for neighbour side of Y faces.
    pub ap_y_neighbour_bnd_anchor_id: Vec<usize>,

    /// Interior-face flags for boundary intersection along Z.
    pub ap_z_has_bnd: Vec<bool>,
    /// Owner-center to boundary distance for Z faces.
    pub ap_z_dist_owner_to_bnd: Vec<f64>,
    /// Neighbour-center to boundary distance for Z faces.
    pub ap_z_dist_neighbour_to_bnd: Vec<f64>,
    /// Owner-side far-cell index for Z faces.
    pub ap_z_owner_far_cell_id: Vec<usize>,
    /// Neighbour-side far-cell index for Z faces.
    pub ap_z_neighbour_far_cell_id: Vec<usize>,
    /// APIBM reconstruction weights on the owner side for Z faces.
    pub ap_z_owner_weights: Vec<[f64; 3]>,
    /// APIBM reconstruction weights on the neighbour side for Z faces.
    pub ap_z_neighbour_weights: Vec<[f64; 3]>,
    /// Boundary anchor id for owner side of Z faces.
    pub ap_z_owner_bnd_anchor_id: Vec<usize>,
    /// Boundary anchor id for neighbour side of Z faces.
    pub ap_z_neighbour_bnd_anchor_id: Vec<usize>,
}
