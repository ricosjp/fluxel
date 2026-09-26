use crate::{CellIndex, MeshId};
use fluxel_core::{Axis, BalancedForest, Direction, Forest, ForestError};
use fluxel_geometry::{CoordinateType, Geometry};
use std::{fmt, sync::Arc};

#[derive(Debug)]
/// Invalid forest/geometry inputs for immutable background construction.
pub enum MeshError {
    Forest(ForestError),
    Unbalanced,
    GeometryMismatch,
}
impl From<ForestError> for MeshError {
    fn from(e: ForestError) -> Self {
        Self::Forest(e)
    }
}
impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forest(e) => e.fmt(f),
            Self::Unbalanced => f.write_str("background mesh requires a 2:1 balanced forest"),
            Self::GeometryMismatch => f.write_str("forest and geometry resolutions do not match"),
        }
    }
}
impl std::error::Error for MeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Forest(e) => Some(e),
            _ => None,
        }
    }
}
#[derive(Debug, Default)]
/// Face connectivity in deterministic cell/axis/neighbour traversal order.
/// Each internal record joins a negative-axis owner to a positive-axis neighbour.
/// Coarse/fine interfaces have one record per neighbour. Domain records represent
/// faces with no neighbour; surface patches belong to the IBM payload, not here.
pub struct MeshTopology {
    pub(crate) internal_owner: Vec<CellIndex>,
    pub(crate) internal_neighbour: Vec<CellIndex>,
    pub(crate) internal_axis: Vec<Axis>,
    pub(crate) boundary_owner: Vec<CellIndex>,
    pub(crate) boundary_direction: Vec<Direction>,
}
impl MeshTopology {
    /// Negative-axis cell for each internal-face record.
    pub fn internal_owner(&self) -> &[CellIndex] {
        &self.internal_owner
    }
    /// Positive-axis cell for each internal-face record.
    pub fn internal_neighbour(&self) -> &[CellIndex] {
        &self.internal_neighbour
    }
    /// X/Y/Z axis aligned with each internal-face record.
    pub fn internal_axis(&self) -> &[Axis] {
        &self.internal_axis
    }
    /// Cell adjacent to each outer-domain face.
    pub fn boundary_owner(&self) -> &[CellIndex] {
        &self.boundary_owner
    }
    /// Outward signed axis direction for each domain face.
    pub fn boundary_direction(&self) -> &[Direction] {
        &self.boundary_direction
    }
    /// Number of internal-face records, including coarse/fine subfaces.
    pub fn n_internal_faces(&self) -> usize {
        self.internal_owner.len()
    }
}
#[derive(Debug, Default)]
/// Physical cell centers and side lengths in forest key order and caller length units.
pub struct CellGeometry {
    pub(crate) centers: Vec<[f64; 3]>,
    pub(crate) sizes: Vec<[f64; 3]>,
}
impl CellGeometry {
    /// Physical X/Y/Z cell centers, one row per background cell.
    pub fn centers(&self) -> &[[f64; 3]] {
        &self.centers
    }
    /// Physical X/Y/Z side lengths, one row per background cell.
    pub fn sizes(&self) -> &[[f64; 3]] {
        &self.sizes
    }
}
#[derive(Debug)]
/// Immutable topology and geometry with a process-local identity, shareable across IBM methods.
pub struct BackgroundMesh {
    pub(crate) id: MeshId,
    pub(crate) topology: MeshTopology,
    pub(crate) geometry: CellGeometry,
}
impl BackgroundMesh {
    /// Identity of this background; a rebuild receives a new ID.
    pub fn id(&self) -> MeshId {
        self.id
    }
    /// Internal and domain-boundary face connectivity.
    pub fn topology(&self) -> &MeshTopology {
        &self.topology
    }
    /// Physical cell centers and widths.
    pub fn geometry(&self) -> &CellGeometry {
        &self.geometry
    }
    /// Number of background leaf cells.
    pub fn n_cells(&self) -> usize {
        self.geometry.centers.len()
    }
    /// Cartesian; other coordinate mappings are not implemented.
    pub fn coordinate_type(&self) -> CoordinateType {
        CoordinateType::Cartesian
    }
}
/// Couples a validated forest to its derived arrays. Only immutable references escape.
/// Snapshots retain only `background`, not the forest or its search state.
#[derive(Debug)]
pub struct GridContext {
    forest: Forest,
    geometry: Geometry,
    background: Arc<BackgroundMesh>,
}
impl GridContext {
    /// Consume a forest and derive a background after validating resolution, coverage and balance.
    /// Returns GeometryMismatch for unequal root resolutions, a Forest error for bad
    /// coverage, or Unbalanced for level gaps above one. Does not repair the forest.
    pub fn new(forest: Forest, geometry: Geometry) -> Result<Self, MeshError> {
        if forest.base_resolution() != geometry.base_resolution() {
            return Err(MeshError::GeometryMismatch);
        }
        forest.validate_coverage()?;
        if !forest.is_balanced() {
            return Err(MeshError::Unbalanced);
        }
        Self::build(forest, geometry)
    }
    /// Consumes proof of complete coverage and balance, avoiding another neighbour scan.
    /// Still checks root-resolution agreement; consumes both inputs on success or failure.
    pub fn from_balanced(forest: BalancedForest, geometry: Geometry) -> Result<Self, MeshError> {
        Self::build(forest.into_forest(), geometry)
    }
    /// Extract deterministic immutable arrays after coverage/balance have been established.
    fn build(forest: Forest, geometry: Geometry) -> Result<Self, MeshError> {
        if forest.base_resolution() != geometry.base_resolution() {
            return Err(MeshError::GeometryMismatch);
        }
        let background = Arc::new(crate::build::build_background(&forest, &geometry));
        Ok(Self {
            forest,
            geometry,
            background,
        })
    }
    /// Validated forest with immutable access only.
    pub fn forest(&self) -> &Forest {
        &self.forest
    }
    /// Physical mapping used to derive this background.
    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    /// Shared background arrays; cloning the Arc does not retain the forest.
    pub fn background(&self) -> &Arc<BackgroundMesh> {
        &self.background
    }
}
