//! IDs are scoped to a background mesh, never persistent across remeshing.
use std::sync::atomic::{AtomicU64, Ordering};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Process-local background identity, not a stable identifier for serialized meshes.
pub struct MeshId(u64);
impl MeshId {
    pub(crate) fn fresh() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(
            NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("mesh ID space exhausted"),
        )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// Cell-array offset scoped to its originating background mesh.
pub struct CellIndex(pub(crate) usize);
impl CellIndex {
    /// Raw array offset; the caller must retain the corresponding mesh identity.
    pub fn index(self) -> usize {
        self.0
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// Face-array offset scoped to its originating background mesh.
pub struct FaceIndex(pub(crate) usize);
impl FaceIndex {
    /// Raw array offset; the caller must retain the corresponding mesh identity.
    pub fn index(self) -> usize {
        self.0
    }
}
