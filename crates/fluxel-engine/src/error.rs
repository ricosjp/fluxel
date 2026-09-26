use fluxel_core::ForestError;
use fluxel_geometry::GeometryError;
use fluxel_ibm::IbmError;
use fluxel_mesh::MeshError;
use std::fmt;
#[derive(Debug)]
/// Failures during validation, loading, mesh generation or session commit.
pub enum BuildError {
    /// Invalid application input, such as an unsupported file extension.
    InvalidInput(String),
    /// Surface file could not be read.
    Io(std::io::Error),
    /// Surface file could not be parsed.
    Parse(String),
    /// Core resolution, refinement, coverage or cell-limit failure.
    Forest(ForestError),
    /// Invalid physical bounds, mapping or pose.
    Geometry(GeometryError),
    /// Background mesh validation failure.
    Mesh(MeshError),
    /// Boundary or IBM computation failure.
    Ibm(IbmError),
    /// Foreign/obsolete candidate, or exhausted session revision counter.
    StaleUpdate,
}
impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) | Self::Parse(s) => f.write_str(s),
            Self::Io(e) => e.fmt(f),
            Self::Forest(e) => e.fmt(f),
            Self::Geometry(e) => e.fmt(f),
            Self::Mesh(e) => e.fmt(f),
            Self::Ibm(e) => e.fmt(f),
            Self::StaleUpdate => {
                f.write_str("update belongs to a different session or an obsolete revision")
            }
        }
    }
}
impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Forest(e) => Some(e),
            Self::Geometry(e) => Some(e),
            Self::Mesh(e) => Some(e),
            Self::Ibm(e) => Some(e),
            _ => None,
        }
    }
}
impl From<ForestError> for BuildError {
    fn from(e: ForestError) -> Self {
        Self::Forest(e)
    }
}
impl From<GeometryError> for BuildError {
    fn from(e: GeometryError) -> Self {
        Self::Geometry(e)
    }
}
impl From<MeshError> for BuildError {
    fn from(e: MeshError) -> Self {
        Self::Mesh(e)
    }
}
impl From<IbmError> for BuildError {
    fn from(e: IbmError) -> Self {
        Self::Ibm(e)
    }
}
