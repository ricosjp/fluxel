//! CFD-oriented mesh data structures for export.
//! Interior faces are stored in a **combined** list with an explicit per-face axis tag
//! (`internal_faces_axis`). Domain boundary faces use `bnd_faces_dir` (six canonical directions).
//! Face normals and areas can be derived from the axis when needed.

use fluxel_core::{Axis, CoordinateType, Direction};

/// SoA mesh layout for GCIBM (Ghost-Cell Immersed Boundary Method).
#[derive(Debug, Default, Clone)]
pub struct CfdGhostCellMesh {
    // --- Base topology (shared) ---
    pub n_cells: usize,
    pub coordinate_type: CoordinateType,
    pub cell_centers: Vec<[f64; 3]>,
    pub cell_sizes: Vec<[f64; 3]>,

    // --- Combined topology ---
    pub internal_faces_owner: Vec<usize>,
    pub internal_faces_neighbour: Vec<usize>,
    pub internal_faces_axis: Vec<Axis>,

    pub bnd_faces_owner: Vec<usize>,
    pub bnd_faces_dir: Vec<Direction>,

    // --- GCIBM-specific fields ---
    /// Encoded cell category: `true` = Fluid, `false` = Solid.
    pub gc_is_fluid: Vec<bool>,

    /// Length `N_ghost` (one entry per computational ghost cell): mesh cell index for each ghost.
    pub gc_cell_ids: Vec<usize>,
    /// Face anchor id for each ghost cell (length `N_ghost`).
    pub gc_bnd_anchor_ids: Vec<usize>,
    /// Closest boundary point (intercept) per ghost cell (length `N_ghost`).
    pub gc_bnd_intercepts: Vec<[f64; 3]>,
    /// Image-point coordinates per ghost cell (length `N_ghost`).
    pub gc_image_points: Vec<[f64; 3]>,

    /// (`N_ghost`, 8) Indices of fluid cells used for interpolation.
    pub gc_interp_stencil_indices: Vec<[usize; 8]>,
    /// (`N_ghost`, 8) Weights for the corresponding interpolation stencil.
    pub gc_interp_stencil_weights: Vec<[f64; 8]>,
}

/// SoA mesh layout for APIBM (Axis-Projected Immersed Boundary Method).
#[derive(Debug, Default, Clone)]
pub struct CfdAxisProjectedMesh {
    // --- Base topology (shared) ---
    pub n_cells: usize,
    pub coordinate_type: CoordinateType,
    pub cell_centers: Vec<[f64; 3]>,
    pub cell_sizes: Vec<[f64; 3]>,

    // --- Combined topology ---
    pub internal_faces_owner: Vec<usize>,
    pub internal_faces_neighbour: Vec<usize>,
    pub internal_faces_axis: Vec<Axis>,

    pub bnd_faces_owner: Vec<usize>,
    pub bnd_faces_dir: Vec<Direction>,

    // --- APIBM-specific fields ---
    pub ap_has_bnd: Vec<bool>,
    pub ap_dist_owner_to_bnd: Vec<f64>,
    pub ap_dist_neighbour_to_bnd: Vec<f64>,
    pub ap_owner_far_cell_id: Vec<usize>,
    pub ap_neighbour_far_cell_id: Vec<usize>,
    pub ap_owner_weights: Vec<[f64; 3]>,
    pub ap_neighbour_weights: Vec<[f64; 3]>,
    pub ap_owner_bnd_anchor_id: Vec<usize>,
    pub ap_neighbour_bnd_anchor_id: Vec<usize>,
}
