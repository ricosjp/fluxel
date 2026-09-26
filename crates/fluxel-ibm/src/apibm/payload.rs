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
pub struct ApIbmData {
    pub(crate) is_immersed_face: Vec<bool>,
    /// Physical owner-center distance to the first boundary hit, possibly zero.
    pub(crate) dist_owner_to_bnd: Vec<f64>,
    /// Physical neighbour-center distance to the first boundary hit, possibly zero.
    pub(crate) dist_neighbour_to_bnd: Vec<f64>,
    /// Whether the owner center is a Dirichlet-constraint candidate.
    pub(crate) owner_near_boundary: Vec<bool>,
    /// Whether the neighbour center is a Dirichlet-constraint candidate.
    pub(crate) neighbour_near_boundary: Vec<bool>,
    pub(crate) owner_bnd_anchor_id: Vec<usize>,
    pub(crate) owner_bnd_patch_id: Vec<usize>,
    pub(crate) neighbour_bnd_anchor_id: Vec<usize>,
    pub(crate) neighbour_bnd_patch_id: Vec<usize>,
}

impl ApIbmData {
    /// One flag per internal face; true only when both axis rays hit.
    pub fn is_immersed_face(&self) -> &[bool] {
        &self.is_immersed_face
    }
    /// Compressed physical first-hit distances from owner centers, including zero.
    pub fn dist_owner_to_bnd(&self) -> &[f64] {
        &self.dist_owner_to_bnd
    }
    /// Compressed physical first-hit distances from neighbour centers, including zero.
    pub fn dist_neighbour_to_bnd(&self) -> &[f64] {
        &self.dist_neighbour_to_bnd
    }
    /// Compressed owner constraint-candidate flags; see the type-level threshold.
    pub fn owner_near_boundary(&self) -> &[bool] {
        &self.owner_near_boundary
    }
    /// Compressed neighbour constraint-candidate flags; see the type-level threshold.
    pub fn neighbour_near_boundary(&self) -> &[bool] {
        &self.neighbour_near_boundary
    }
    /// Surface triangle IDs hit by owner rays, in compressed face order.
    pub fn owner_bnd_anchor_id(&self) -> &[usize] {
        &self.owner_bnd_anchor_id
    }
    /// Patch-table indices hit by owner rays, in compressed face order.
    pub fn owner_bnd_patch_id(&self) -> &[usize] {
        &self.owner_bnd_patch_id
    }
    /// Surface triangle IDs hit by neighbour rays, possibly different from owner hits.
    pub fn neighbour_bnd_anchor_id(&self) -> &[usize] {
        &self.neighbour_bnd_anchor_id
    }
    /// Patch-table indices hit by neighbour rays, in compressed face order.
    pub fn neighbour_bnd_patch_id(&self) -> &[usize] {
        &self.neighbour_bnd_patch_id
    }
}
