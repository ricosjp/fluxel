//! CFD-oriented mesh data structures for export.
//! Uses an **axis-split** layout: interior faces are registered only along axis
//! directions from each cell. Face normals and areas can be derived from the axis when needed,
//! reducing memory compared to a full face list with explicit normals.

/// SoA mesh layout for GCIBM (Ghost-Cell Immersed Boundary Method).
#[derive(Debug, Default, Clone)]
pub struct CfdGhostCellMesh {
    // --- Base topology (shared) ---
    pub cell_centers: Vec<[f64; 3]>,
    pub cell_sizes: Vec<[f64; 3]>,

    // Topology along X
    pub x_faces_owner: Vec<usize>,
    pub x_faces_neighbour: Vec<usize>,
    pub x_bnd_minus_owner: Vec<usize>,
    pub x_bnd_plus_owner: Vec<usize>,

    // Topology along Y
    pub y_faces_owner: Vec<usize>,
    pub y_faces_neighbour: Vec<usize>,
    pub y_bnd_minus_owner: Vec<usize>,
    pub y_bnd_plus_owner: Vec<usize>,

    // Topology along Z
    pub z_faces_owner: Vec<usize>,
    pub z_faces_neighbour: Vec<usize>,
    pub z_bnd_minus_owner: Vec<usize>,
    pub z_bnd_plus_owner: Vec<usize>,

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
#[derive(Default, Debug, Clone)]
pub struct CfdAxisProjectedMesh {
    // --- Base topology (shared) ---
    pub cell_centers: Vec<[f64; 3]>,
    pub cell_sizes: Vec<[f64; 3]>,

    // Topology along X
    pub x_faces_owner: Vec<usize>,
    pub x_faces_neighbour: Vec<usize>,
    pub x_bnd_minus_owner: Vec<usize>,
    pub x_bnd_plus_owner: Vec<usize>,

    // Topology along Y
    pub y_faces_owner: Vec<usize>,
    pub y_faces_neighbour: Vec<usize>,
    pub y_bnd_minus_owner: Vec<usize>,
    pub y_bnd_plus_owner: Vec<usize>,

    // Topology along Z
    pub z_faces_owner: Vec<usize>,
    pub z_faces_neighbour: Vec<usize>,
    pub z_bnd_minus_owner: Vec<usize>,
    pub z_bnd_plus_owner: Vec<usize>,

    // --- APIBM-specific fields ---
    pub ap_x_has_bnd: Vec<bool>,
    pub ap_x_dist_owner_to_bnd: Vec<f64>,
    pub ap_x_dist_neighbour_to_bnd: Vec<f64>,
    pub ap_x_owner_far_cell_id: Vec<usize>,
    pub ap_x_neighbour_far_cell_id: Vec<usize>,
    pub ap_x_owner_weights: Vec<[f64; 3]>,
    pub ap_x_neighbour_weights: Vec<[f64; 3]>,
    pub ap_x_owner_bnd_anchor_id: Vec<usize>,
    pub ap_x_neighbour_bnd_anchor_id: Vec<usize>,

    pub ap_y_has_bnd: Vec<bool>,
    pub ap_y_dist_owner_to_bnd: Vec<f64>,
    pub ap_y_dist_neighbour_to_bnd: Vec<f64>,
    pub ap_y_owner_far_cell_id: Vec<usize>,
    pub ap_y_neighbour_far_cell_id: Vec<usize>,
    pub ap_y_owner_weights: Vec<[f64; 3]>,
    pub ap_y_neighbour_weights: Vec<[f64; 3]>,
    pub ap_y_owner_bnd_anchor_id: Vec<usize>,
    pub ap_y_neighbour_bnd_anchor_id: Vec<usize>,

    pub ap_z_has_bnd: Vec<bool>,
    pub ap_z_dist_owner_to_bnd: Vec<f64>,
    pub ap_z_dist_neighbour_to_bnd: Vec<f64>,
    pub ap_z_owner_far_cell_id: Vec<usize>,
    pub ap_z_neighbour_far_cell_id: Vec<usize>,
    pub ap_z_owner_weights: Vec<[f64; 3]>,
    pub ap_z_neighbour_weights: Vec<[f64; 3]>,
    pub ap_z_owner_bnd_anchor_id: Vec<usize>,
    pub ap_z_neighbour_bnd_anchor_id: Vec<usize>,
}
