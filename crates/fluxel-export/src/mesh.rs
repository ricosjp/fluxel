//! CFD-oriented mesh data structures for export.
//! Interior faces are stored in a **combined** list with an explicit per-face axis tag
//! (`internal_faces_axis`). Domain boundary faces use `domain_bnd_faces_dir` (six canonical directions).
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
    pub patch_names: Vec<String>,

    // --- Combined topology ---
    pub internal_faces_owner: Vec<usize>,
    pub internal_faces_neighbour: Vec<usize>,
    pub internal_faces_axis: Vec<Axis>,

    pub domain_bnd_faces_owner: Vec<usize>,
    pub domain_bnd_faces_dir: Vec<Direction>,

    // --- GCIBM-specific fields ---
    /// Encoded cell category: `true` = Fluid, `false` = Solid.
    pub gc_is_fluid: Vec<bool>,

    /// Length `N_ghost` (one entry per computational ghost cell): mesh cell index for each ghost.
    pub gc_cell_ids: Vec<usize>,
    /// Face anchor id for each ghost cell (length `N_ghost`).
    pub gc_bnd_anchor_ids: Vec<usize>,
    /// Face patch id for each ghost cell (length `N_ghost`).
    pub gc_bnd_patch_ids: Vec<usize>,
    /// Closest boundary point (intercept) per ghost cell (length `N_ghost`).
    pub gc_bnd_intercepts: Vec<[f64; 3]>,
    /// Image-point coordinates per ghost cell (length `N_ghost`).
    pub gc_image_points: Vec<[f64; 3]>,

    /// (`N_ghost`, 8) Indices of fluid cells used for interpolation.
    pub gc_interp_stencil_indices: Vec<[usize; 8]>,
    /// (`N_ghost`, 8) Weights for the corresponding interpolation stencil.
    pub gc_interp_stencil_weights: Vec<[f64; 8]>,
}

/// Compressed axis-projected immersed-boundary payload for internal faces.
///
/// `is_immersed_face` has length `N_internal_faces`. All other fields are compressed to
/// length `N_immersed = count(is_immersed_face)` and store data only for immersed faces,
/// in the same order as `true` entries in `is_immersed_face`.
///
/// Near-boundary flags mark cell-center Dirichlet-constraint candidates when
/// `d / delta_x <= delta_x / L`, using each side's cell width along the face axis
/// and the largest computational-domain extent `L`. The solver must apply any
/// constraints; the flags leave the physical distances unchanged.
#[derive(Debug, Default, Clone)]
pub struct ApIbmFaceData {
    pub is_immersed_face: Vec<bool>,
    /// Physical owner-center distance to the first boundary hit, possibly zero.
    pub dist_owner_to_bnd: Vec<f64>,
    /// Physical neighbour-center distance to the first boundary hit, possibly zero.
    pub dist_neighbour_to_bnd: Vec<f64>,
    /// Whether the owner center is a Dirichlet-constraint candidate.
    pub owner_near_boundary: Vec<bool>,
    /// Whether the neighbour center is a Dirichlet-constraint candidate.
    pub neighbour_near_boundary: Vec<bool>,
    pub owner_bnd_anchor_id: Vec<usize>,
    pub owner_bnd_patch_id: Vec<usize>,
    pub neighbour_bnd_anchor_id: Vec<usize>,
    pub neighbour_bnd_patch_id: Vec<usize>,
}

/// SoA mesh layout for APIBM (Axis-Projected Immersed Boundary Method).
#[derive(Debug, Default, Clone)]
pub struct CfdAxisProjectedMesh {
    // --- Base topology (shared) ---
    pub n_cells: usize,
    pub coordinate_type: CoordinateType,
    pub cell_centers: Vec<[f64; 3]>,
    pub cell_sizes: Vec<[f64; 3]>,
    pub patch_names: Vec<String>,

    // --- Combined topology ---
    pub internal_faces_owner: Vec<usize>,
    pub internal_faces_neighbour: Vec<usize>,
    pub internal_faces_axis: Vec<Axis>,

    pub domain_bnd_faces_owner: Vec<usize>,
    pub domain_bnd_faces_dir: Vec<Direction>,

    // --- APIBM-specific fields ---
    pub ap: ApIbmFaceData,
}
