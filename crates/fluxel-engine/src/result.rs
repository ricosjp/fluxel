use fluxel_ibm::{ApIbmData, BoundaryRevision, GhostCellData, PatchTable};
use fluxel_mesh::BackgroundMesh;
use std::sync::Arc;
#[derive(Debug, Clone)]
/// Immutable background, patch table and method payload for one boundary revision.
/// Cloning shares Arc storage. A snapshot outlives its producer; IDs and row indices
/// must be interpreted with its own background, never a remeshed one.
pub struct MeshSnapshot<P> {
    pub(crate) background: Arc<BackgroundMesh>,
    pub(crate) boundary_revision: BoundaryRevision,
    pub(crate) patches: Arc<PatchTable>,
    pub(crate) payload: Arc<P>,
}
impl<P> MeshSnapshot<P> {
    /// Shared cell geometry and face topology.
    pub fn background(&self) -> &Arc<BackgroundMesh> {
        &self.background
    }
    /// Boundary state revision used for this payload.
    pub fn boundary_revision(&self) -> BoundaryRevision {
        self.boundary_revision
    }
    /// Patch IDs index this shared name table.
    pub fn patches(&self) -> &Arc<PatchTable> {
        &self.patches
    }
    /// Shared method-specific data aligned with this background.
    pub fn payload(&self) -> &Arc<P> {
        &self.payload
    }
}
/// Snapshot carrying compressed axis-projected face data.
pub type ApibmMesh = MeshSnapshot<ApIbmData>;
/// Snapshot carrying fluid classification and ghost-cell interpolation data.
pub type GcibmMesh = MeshSnapshot<GhostCellData>;
#[derive(Debug, Default, Clone)]
/// Counts for a completed build/update; reporting does not itself emit warnings.
pub struct BuildReport {
    /// Total background leaf cells, including non-fluid cells.
    pub n_cells: usize,
    /// Number of owner/neighbour internal-face records.
    pub n_internal_faces: usize,
    /// Immersed faces for APIBM, or ghost-cell rows for GCIBM.
    pub n_ibm_records: usize,
    /// Intersecting final cells below the surface target (without adding leaf levels).
    pub under_refined_cells: usize,
    /// Minimum level among intersecting final cells; None if there are none.
    pub minimum_intersect_level: Option<u8>,
    /// GCIBM rows using inverse-distance fallback, including unresolved rows; zero for APIBM.
    pub interpolation_fallbacks: usize,
    /// GCIBM rows with no fluid candidates; these self-reference and require caller handling.
    pub unresolved_stencils: usize,
}
#[derive(Debug)]
/// Completed snapshot and diagnostic report; no active session is retained.
pub struct BuildOutput<P> {
    /// Immutable generated mesh.
    pub mesh: MeshSnapshot<P>,
    /// Build diagnostics; currently not exposed by the one-shot Python API.
    pub report: BuildReport,
}
