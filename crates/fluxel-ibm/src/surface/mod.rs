//! Validated surfaces and backend-independent boundary states.
mod parry;
use crate::IbmError;
use fluxel_geometry::RigidPose;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Process-local boundary-state identity; changes even when a new pose equals the old pose.
pub struct BoundaryRevision(u64);
impl BoundaryRevision {
    fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(
            NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("boundary revision space exhausted"),
        )
    }
}
#[derive(Debug)]
/// Unique surface patch names; array position is the patch ID.
pub struct PatchTable {
    names: Vec<String>,
}
impl PatchTable {
    /// Ordered patch names; empty for Boundary::None.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}
#[derive(Debug)]
/// Validated, immutable triangle surface with backend acceleration and patch assignments.
pub struct BoundarySurface {
    backend: parry::SurfaceBackend,
    patches: Arc<PatchTable>,
    anchor_to_patch: Vec<usize>,
}
impl BoundarySurface {
    /// Build a surface from physical vertices, zero-based triangles and patch assignments.
    /// Triangle order defines anchor IDs. `anchor_to_patch` contains one index into
    /// `names` per triangle. No unit conversion or closed/manifold-surface repair occurs.
    /// Parity-based inside/outside classification is intended for well-defined surfaces;
    /// structural validation alone does not guarantee that geometric property.
    ///
    /// # Errors
    /// Rejects empty triangles, nonfinite vertices, invalid vertex/patch indices,
    /// duplicate patch names, degenerate triangles, and backend construction failures.
    pub fn new(
        vertices: &[[f64; 3]],
        indices: &[[u32; 3]],
        names: Vec<String>,
        anchor_to_patch: Vec<usize>,
    ) -> Result<Self, IbmError> {
        if indices.is_empty()
            || vertices.iter().flatten().any(|x| !x.is_finite())
            || indices
                .iter()
                .flatten()
                .any(|&i| i as usize >= vertices.len())
            || anchor_to_patch.len() != indices.len()
            || anchor_to_patch.iter().any(|&p| p >= names.len())
        {
            return Err(IbmError::InvalidSurface(
                "surface requires finite vertices, nonempty triangles and valid index/patch arrays"
                    .into(),
            ));
        }
        if names.iter().collect::<std::collections::HashSet<_>>().len() != names.len() {
            return Err(IbmError::InvalidSurface(
                "patch names must be unique".into(),
            ));
        }
        let backend = parry::SurfaceBackend::new(vertices, indices)?;
        Ok(Self {
            backend,
            patches: Arc::new(PatchTable { names }),
            anchor_to_patch,
        })
    }
    /// Surface patch table shared with snapshots.
    pub fn patches(&self) -> &Arc<PatchTable> {
        &self.patches
    }
    /// Number of triangles and valid anchor IDs.
    pub fn triangle_count(&self) -> usize {
        self.anchor_to_patch.len()
    }
    /// Test a physical cell box against the posed surface; propagate backend query failure.
    pub(crate) fn intersects(
        &self,
        center: [f64; 3],
        size: [f64; 3],
        pose: &RigidPose,
    ) -> Result<bool, IbmError> {
        self.backend.intersects(center, size, pose)
    }
    /// Return the first posed-surface hit within `length`, or None if absent.
    /// Origin and length use world units; direction must be unit length for the returned
    /// ray parameter to be a physical distance. Maps the triangle hit to its patch ID.
    pub(crate) fn ray(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        length: f64,
        pose: &RigidPose,
    ) -> Result<Option<RayHit>, IbmError> {
        self.backend
            .ray(origin, direction, length, pose)
            .map(|hit| {
                hit.map(|(distance, anchor)| RayHit {
                    distance,
                    anchor,
                    patch: self.anchor_to_patch[anchor],
                })
            })
    }
    /// Return the nearest posed-surface point, triangle and patch, or a query error.
    pub(crate) fn closest(
        &self,
        point: [f64; 3],
        pose: &RigidPose,
    ) -> Result<ClosestPoint, IbmError> {
        let (point, anchor) = self.backend.closest(point, pose)?;
        Ok(ClosestPoint {
            point,
            anchor,
            patch: self.anchor_to_patch[anchor],
        })
    }
    /// Classify a physical point using majority parity over multiple posed-surface rays.
    /// This is a finite-ray heuristic, not a proof for open, self-intersecting or
    /// extreme-scale surfaces. Uses the backend's existing tolerances and ray reach.
    pub(crate) fn contains(&self, point: [f64; 3], pose: &RigidPose) -> bool {
        self.backend.contains(point, pose)
    }
}
pub(crate) struct RayHit {
    pub distance: f64,
    pub anchor: usize,
    pub patch: usize,
}
pub(crate) struct ClosestPoint {
    pub point: [f64; 3],
    pub anchor: usize,
    pub patch: usize,
}
#[derive(Debug, Clone, Default)]
/// Optional immersed geometry; absence is explicit and never a synthetic surface.
pub enum Boundary {
    #[default]
    /// No immersed boundary; background domain faces still exist.
    None,
    /// Shared immutable triangle surface.
    Surface(Arc<BoundarySurface>),
}
#[derive(Debug, Clone)]
/// A shared boundary at one absolute pose, with a fresh classification revision.
pub struct BoundaryState {
    boundary: Boundary,
    pose: RigidPose,
    revision: BoundaryRevision,
    patches: Arc<PatchTable>,
}
impl BoundaryState {
    /// Create a boundary state with a fresh revision and its surface patch table.
    pub fn new(boundary: Boundary, pose: RigidPose) -> Self {
        let patches = match &boundary {
            Boundary::None => Arc::new(PatchTable { names: vec![] }),
            Boundary::Surface(s) => s.patches.clone(),
        };
        Self {
            boundary,
            pose,
            revision: BoundaryRevision::fresh(),
            patches,
        }
    }
    /// Share the surface at an absolute pose; always allocate a new revision.
    pub fn with_pose(&self, pose: RigidPose) -> Self {
        Self::new(self.boundary.clone(), pose)
    }
    /// Absolute rigid transform of this state.
    pub fn pose(&self) -> RigidPose {
        self.pose
    }
    /// Identity used to reject stale intersection masks.
    pub fn revision(&self) -> BoundaryRevision {
        self.revision
    }
    /// Shared patch table; empty if no boundary.
    pub fn patches(&self) -> &Arc<PatchTable> {
        &self.patches
    }
    /// The underlying surface, or None for explicit boundary absence.
    pub fn surface(&self) -> Option<&BoundarySurface> {
        match &self.boundary {
            Boundary::None => None,
            Boundary::Surface(s) => Some(s),
        }
    }
}
