/// Compressed axis-projected immersed-boundary payload for internal faces.
///
/// `is_immersed_face` has length `N_internal_faces`. All other fields are compressed to
/// length `N_immersed = count(is_immersed_face)` and store data only for immersed faces,
/// in the same order as `true` entries in `is_immersed_face`.
///
/// Distances are path lengths from the sample to the first hit. Cartesian paths
/// follow the world axis. Cylindrical `r` and `z` follow `e_r` and `e_z`.
/// Cylindrical `θ` is the arc length `r|Δθ|`.
///
/// Near-boundary flags mark cell-center Dirichlet-constraint candidates when
/// `d / width <= width / L`, using that side's cell width on the face axis and
/// the largest domain extent `L`. The solver must apply any constraints; the
/// flags leave the distances unchanged.
#[derive(Debug, Default, Clone)]
pub struct ApIbmData {
    pub(crate) is_immersed_face: Vec<bool>,
    /// Owner path length to the first hit, possibly zero.
    pub(crate) dist_owner_to_bnd: Vec<f64>,
    /// Neighbour path length to the first hit, possibly zero.
    pub(crate) dist_neighbour_to_bnd: Vec<f64>,
    /// Whether the owner center is a Dirichlet-constraint candidate.
    pub(crate) owner_near_boundary: Vec<bool>,
    /// Whether the neighbour center is a Dirichlet-constraint candidate.
    pub(crate) neighbour_near_boundary: Vec<bool>,
    pub(crate) owner_bnd_anchor_id: Vec<usize>,
    pub(crate) owner_bnd_patch_id: Vec<usize>,
    pub(crate) neighbour_bnd_anchor_id: Vec<usize>,
    pub(crate) neighbour_bnd_patch_id: Vec<usize>,
    /// World intersection on the owner side.
    pub(crate) owner_bnd_point: Vec<[f64; 3]>,
    /// Unit path tangent on the owner side, in world components.
    pub(crate) owner_bnd_tangent: Vec<[f64; 3]>,
    /// Unit triangle normal on the owner side, in world components.
    pub(crate) owner_bnd_normal: Vec<[f64; 3]>,
    /// World intersection on the neighbour side.
    pub(crate) neighbour_bnd_point: Vec<[f64; 3]>,
    /// Unit path tangent on the neighbour side, in world components.
    pub(crate) neighbour_bnd_tangent: Vec<[f64; 3]>,
    /// Unit triangle normal on the neighbour side, in world components.
    pub(crate) neighbour_bnd_normal: Vec<[f64; 3]>,
}

impl ApIbmData {
    /// One flag per internal face; true only when both axis rays hit.
    pub fn is_immersed_face(&self) -> &[bool] {
        &self.is_immersed_face
    }
    /// Compressed owner path lengths to the first hit, including zero.
    pub fn dist_owner_to_bnd(&self) -> &[f64] {
        &self.dist_owner_to_bnd
    }
    /// Compressed neighbour path lengths to the first hit, including zero.
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
    /// Owner-side world intersections. The path length is not the chord to this point.
    pub fn owner_bnd_point(&self) -> &[[f64; 3]] {
        &self.owner_bnd_point
    }
    /// Owner-side unit tangents of the search path.
    pub fn owner_bnd_tangent(&self) -> &[[f64; 3]] {
        &self.owner_bnd_tangent
    }
    /// Owner-side unit surface normals from triangle winding.
    pub fn owner_bnd_normal(&self) -> &[[f64; 3]] {
        &self.owner_bnd_normal
    }
    /// Neighbour-side world intersections.
    pub fn neighbour_bnd_point(&self) -> &[[f64; 3]] {
        &self.neighbour_bnd_point
    }
    /// Neighbour-side unit tangents of the search path.
    pub fn neighbour_bnd_tangent(&self) -> &[[f64; 3]] {
        &self.neighbour_bnd_tangent
    }
    /// Neighbour-side unit surface normals from triangle winding.
    pub fn neighbour_bnd_normal(&self) -> &[[f64; 3]] {
        &self.neighbour_bnd_normal
    }
}
