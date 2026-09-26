//! Prepare and commit updates atomically. Preparing never mutates the current state.
use crate::{
    ApibmMesh, BuildError, BuildReport, MeshBuildConfig, RefinementPlan, RefinementRegion,
};
use fluxel_geometry::RigidPose;
use fluxel_ibm::{Boundary, BoundaryState};
use fluxel_mesh::GridContext;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default, Clone, Copy)]
/// Absolute pose components; `None` retains the current component.
/// Rotation is about the original origin and precedes translation. Quaternions
/// use `[w, x, y, z]` and are normalized; Rust rejects zero or nonfinite inputs.
pub struct PoseUpdate {
    /// Absolute displacement in the boundary coordinate units; None retains it.
    pub translation: Option<[f64; 3]>,
    /// Absolute rotation in w/x/y/z order; None retains it.
    pub quaternion: Option<[f64; 4]>,
}
impl PoseUpdate {
    /// Merge omitted components with the current pose, then validate and normalize.
    fn apply(self, current: RigidPose) -> Result<RigidPose, BuildError> {
        Ok(RigidPose::new(
            self.translation.unwrap_or(current.translation()),
            self.quaternion.unwrap_or(current.quaternion()),
        )?)
    }
}
#[derive(Debug, Default)]
/// Optional changes to apply while rebuilding a session background from roots.
pub struct RemeshRequest {
    /// New surface target, or None to keep the current target.
    pub target_level: Option<u8>,
    /// None keeps the current regions; Some(empty) clears them.
    pub regions: Option<Vec<RefinementRegion>>,
    /// Absolute pose changes, with omitted components preserved.
    pub pose: PoseUpdate,
}
#[derive(Debug)]
/// Uncommitted candidate tied to one session and its current revision.
/// Dropping it has no effect on the session. Its snapshot can be inspected or
/// cloned before commit, but successful inspection does not commit the candidate.
pub struct PreparedUpdate {
    session_id: u64,
    base_revision: u64,
    grid: Option<GridContext>,
    boundary: BoundaryState,
    plan: RefinementPlan,
    mesh: ApibmMesh,
    report: BuildReport,
}
impl PreparedUpdate {
    /// Candidate snapshot; it is not the current session state until committed.
    pub fn mesh(&self) -> &ApibmMesh {
        &self.mesh
    }
    /// Diagnostics computed for this candidate.
    pub fn report(&self) -> &BuildReport {
        &self.report
    }
}
#[derive(Debug)]
/// Stateful APIBM builder retaining the forest and boundary for rigid updates.
/// Snapshots share immutable arrays and remain valid after updates or session drop.
/// Mutations require exclusive access; no interior concurrent mutation is supported.
pub struct ApibmSession {
    id: u64,
    revision: u64,
    config: MeshBuildConfig,
    grid: GridContext,
    boundary: BoundaryState,
    plan: RefinementPlan,
    mesh: ApibmMesh,
    report: BuildReport,
}
impl ApibmSession {
    /// Build the initial mesh with identity boundary pose and revision zero.
    /// Stores the configuration, including cell limits, for subsequent remeshing.
    ///
    /// # Errors
    /// Propagates the same refinement and geometry errors as [`crate::build_apibm`].
    pub fn new(
        config: MeshBuildConfig,
        boundary: Boundary,
        plan: RefinementPlan,
    ) -> Result<Self, BuildError> {
        let boundary = BoundaryState::new(boundary, RigidPose::identity());
        let (grid, mesh, report) = crate::build::prepare_apibm(&config, &boundary, &plan)?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("session ID space exhausted");
        Ok(Self {
            id,
            revision: 0,
            config,
            grid,
            boundary,
            plan,
            mesh,
            report,
        })
    }
    /// Current immutable snapshot; clone it to retain it independently of this session.
    pub fn mesh(&self) -> &ApibmMesh {
        &self.mesh
    }
    /// Current absolute rigid pose of the boundary.
    pub fn pose(&self) -> RigidPose {
        self.boundary.pose()
    }
    /// Diagnostics from the last successful initialization or commit.
    pub fn report(&self) -> &BuildReport {
        &self.report
    }
    /// Compute a new boundary payload on the unchanged background without mutation.
    /// Cell/face indices and background ID stay fixed. No automatic remeshing or
    /// warning emission occurs; inspect the candidate's under-refinement report.
    ///
    /// # Errors
    /// Rejects invalid poses and propagates boundary query errors. Current mesh,
    /// pose, settings, and revision stay unchanged, including on success until commit.
    pub fn prepare_update(&self, pose: PoseUpdate) -> Result<PreparedUpdate, BuildError> {
        let boundary = self.boundary.with_pose(pose.apply(self.pose())?);
        let (mesh, report) =
            crate::build::apibm_on_grid(&self.grid, &boundary, self.plan.target_level)?;
        Ok(PreparedUpdate {
            session_id: self.id,
            base_revision: self.revision,
            grid: None,
            boundary,
            plan: self.plan.clone(),
            mesh,
            report,
        })
    }
    /// Build a replacement grid and payload without changing the current session.
    /// Omitted targets/regions keep their values; an explicit empty region list clears
    /// them. The original cell limit and uniform refinement remain in effect. The
    /// candidate has a new mesh ID; never reuse old cell/face indices for it. Solution
    /// fields are not transferred.
    ///
    /// # Errors
    /// Rejects invalid pose/levels, cell-limit violations, and geometry query failures.
    /// Current state remains unchanged until a successful [`Self::commit`].
    pub fn prepare_remesh(&self, request: RemeshRequest) -> Result<PreparedUpdate, BuildError> {
        let plan = RefinementPlan::new(
            request.target_level.unwrap_or(self.plan.target_level),
            request.regions.unwrap_or_else(|| self.plan.regions.clone()),
        )?;
        let boundary = self.boundary.with_pose(request.pose.apply(self.pose())?);
        let (grid, mesh, report) = crate::build::prepare_apibm(&self.config, &boundary, &plan)?;
        Ok(PreparedUpdate {
            session_id: self.id,
            base_revision: self.revision,
            grid: Some(grid),
            boundary,
            plan,
            mesh,
            report,
        })
    }
    /// Atomically install a candidate and advance the session revision.
    /// Previously returned snapshots remain valid. Consumes the candidate; after one
    /// commit, other candidates prepared from the old revision become stale.
    ///
    /// # Errors
    /// Returns `BuildError::StaleUpdate` for another session's candidate, an obsolete
    /// revision, or revision-counter exhaustion. Rejection leaves this session intact.
    pub fn commit(&mut self, update: PreparedUpdate) -> Result<(), BuildError> {
        if update.session_id != self.id || update.base_revision != self.revision {
            return Err(BuildError::StaleUpdate);
        }
        let next = self
            .revision
            .checked_add(1)
            .ok_or(BuildError::StaleUpdate)?;
        if let Some(grid) = update.grid {
            self.grid = grid;
        }
        self.boundary = update.boundary;
        self.plan = update.plan;
        self.mesh = update.mesh;
        self.report = update.report;
        self.revision = next;
        Ok(())
    }
    /// Prepare and immediately commit a fixed-grid update, returning the current mesh.
    /// Errors match [`Self::prepare_update`] and [`Self::commit`]; failures leave state
    /// unchanged. For application-side validation before commit, use those two methods.
    pub fn update_ib(&mut self, pose: PoseUpdate) -> Result<&ApibmMesh, BuildError> {
        let update = self.prepare_update(pose)?;
        self.commit(update)?;
        Ok(&self.mesh)
    }
    /// Prepare and immediately commit a rebuilt grid, returning the current mesh.
    /// Errors match [`Self::prepare_remesh`] and [`Self::commit`]. Indices may change;
    /// no solver fields are transferred. Failures leave the previous state intact.
    pub fn remesh(&mut self, request: RemeshRequest) -> Result<&ApibmMesh, BuildError> {
        let update = self.prepare_remesh(request)?;
        self.commit(update)?;
        Ok(&self.mesh)
    }
}
