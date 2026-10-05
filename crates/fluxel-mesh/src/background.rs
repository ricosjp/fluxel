use crate::{CellIndex, MeshId};
use fluxel_core::{Axis, BalancedForest, Direction, Forest, ForestError};
use fluxel_geometry::{CoordinateType, CylindricalGeometry, Geometry, ParameterBox, SpatialDomain};
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
            Self::GeometryMismatch => f.write_str("forest and geometry do not match"),
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
    /// Parameter box of each internal face. The face axis has equal limits.
    pub(crate) internal_bounds: Vec<ParameterBox>,
    /// Area vector from the owner toward the neighbour.
    pub(crate) internal_area: Vec<[f64; 3]>,
    /// Periodic wraps crossed by the positive neighbour. Zero when the axis does not wrap.
    pub(crate) internal_winding: Vec<i8>,
    pub(crate) boundary_owner: Vec<CellIndex>,
    pub(crate) boundary_direction: Vec<Direction>,
    /// Parameter box of each domain face.
    pub(crate) boundary_bounds: Vec<ParameterBox>,
    /// Outward area vector of each domain face.
    pub(crate) boundary_area: Vec<[f64; 3]>,
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
    /// Parameter box of each internal face, including coarse/fine partial faces.
    pub fn internal_bounds(&self) -> &[ParameterBox] {
        &self.internal_bounds
    }
    /// Area vector of each internal face, pointing from owner toward neighbour.
    pub fn internal_area(&self) -> &[[f64; 3]] {
        &self.internal_area
    }
    /// Periodic wraps of each internal face. A full-turn seam has winding 1.
    pub fn internal_winding(&self) -> &[i8] {
        &self.internal_winding
    }
    /// Cell adjacent to each outer-domain face.
    pub fn boundary_owner(&self) -> &[CellIndex] {
        &self.boundary_owner
    }
    /// Outward signed axis direction for each domain face.
    pub fn boundary_direction(&self) -> &[Direction] {
        &self.boundary_direction
    }
    /// Parameter box of each domain face.
    pub fn boundary_bounds(&self) -> &[ParameterBox] {
        &self.boundary_bounds
    }
    /// Outward area vector of each domain face.
    pub fn boundary_area(&self) -> &[[f64; 3]] {
        &self.boundary_area
    }
    /// Number of internal-face records, including coarse/fine subfaces.
    pub fn n_internal_faces(&self) -> usize {
        self.internal_owner.len()
    }
}
#[derive(Debug, Default)]
/// Sample points and coordinate-aligned widths in forest key order.
///
/// Cartesian widths are the X/Y/Z edge lengths. Cylindrical widths are
/// `[Δr, r Δθ, Δz]`, and each sample is the parameter midpoint mapped to world.
pub struct CellGeometry {
    pub(crate) centers: Vec<[f64; 3]>,
    pub(crate) sizes: Vec<[f64; 3]>,
    pub(crate) volumes: Vec<f64>,
    pub(crate) centroids: Vec<[f64; 3]>,
    pub(crate) corners: Vec<[[f64; 3]; 8]>,
}
impl CellGeometry {
    /// World sample point of each background cell.
    ///
    /// Cartesian samples are the box centers. Cylindrical samples are parameter
    /// midpoints mapped to world, which differ from the volume centroids.
    pub fn centers(&self) -> &[[f64; 3]] {
        &self.centers
    }
    /// Coordinate-aligned widths of each background cell.
    ///
    /// Cartesian widths are the edge lengths. Cylindrical widths are
    /// `[Δr, r Δθ, Δz]`, not the world AABB and not a volume factorization.
    pub fn sizes(&self) -> &[[f64; 3]] {
        &self.sizes
    }
    /// Exact cell volumes.
    pub fn volumes(&self) -> &[f64] {
        &self.volumes
    }
    /// Volume centroids in world coordinates.
    pub fn centroids(&self) -> &[[f64; 3]] {
        &self.centroids
    }
    /// Eight world corners of each cell, in VTK hexahedron order.
    ///
    /// Cylindrical corners are the parameter-box corners. Edges between them are
    /// straight display edges, while a constant-radius face is curved.
    pub fn corners(&self) -> &[[[f64; 3]; 8]] {
        &self.corners
    }
}
#[derive(Debug)]
/// Immutable topology and geometry with a process-local identity, shareable across IBM methods.
pub struct BackgroundMesh {
    pub(crate) id: MeshId,
    pub(crate) topology: MeshTopology,
    pub(crate) geometry: CellGeometry,
    pub(crate) coordinate_type: CoordinateType,
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
    /// Coordinate system used to derive centers, widths, and face connectivity.
    pub fn coordinate_type(&self) -> CoordinateType {
        self.coordinate_type
    }
}
/// Couples a validated forest to its derived arrays. Only immutable references escape.
/// Snapshots retain only `background`, not the forest or its search state.
#[derive(Debug)]
pub struct GridContext {
    forest: Forest,
    domain: SpatialDomain,
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
        Self::from_domain(forest, SpatialDomain::Cartesian(geometry))
    }
    /// Build a cylindrical background. Root angle must be at most `π/2`.
    ///
    /// Forest periodicity is authoritative: any of ``(r, θ, z)`` may wrap.
    pub fn from_cylindrical(
        forest: Forest,
        geometry: CylindricalGeometry,
    ) -> Result<Self, MeshError> {
        Self::from_domain(forest, SpatialDomain::Cylindrical(geometry))
    }
    /// Consumes proof of complete coverage and balance, avoiding another neighbour scan.
    /// Still checks root-resolution agreement; consumes both inputs on success or failure.
    pub fn from_balanced(forest: BalancedForest, geometry: Geometry) -> Result<Self, MeshError> {
        let forest = forest.into_forest();
        if forest.base_resolution() != geometry.base_resolution() {
            return Err(MeshError::GeometryMismatch);
        }
        Self::build(forest, SpatialDomain::Cartesian(geometry))
    }
    fn from_domain(forest: Forest, domain: SpatialDomain) -> Result<Self, MeshError> {
        forest.validate_coverage()?;
        if !forest.is_balanced() {
            return Err(MeshError::Unbalanced);
        }
        Self::build(forest, domain)
    }
    /// Extract deterministic immutable arrays after coverage/balance have been established.
    fn build(forest: Forest, domain: SpatialDomain) -> Result<Self, MeshError> {
        let background = Arc::new(crate::build::build_background(&forest, &domain)?);
        Ok(Self {
            forest,
            domain,
            background,
        })
    }
    /// Validated forest with immutable access only.
    pub fn forest(&self) -> &Forest {
        &self.forest
    }
    /// Physical mapping used to derive this background.
    pub fn domain(&self) -> &SpatialDomain {
        &self.domain
    }
    /// Shared background arrays; cloning the Arc does not retain the forest.
    pub fn background(&self) -> &Arc<BackgroundMesh> {
        &self.background
    }
}
