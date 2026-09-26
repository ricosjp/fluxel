use std::fmt;
#[derive(Debug)]
/// Surface validation, query, seed or classification-provenance failure.
pub enum IbmError {
    /// Invalid vertices, triangles or patch assignments.
    InvalidSurface(String),
    /// Backend query or feature decoding failed.
    Query(String),
    /// Nonfinite seed or no containing cell in the half-open domain.
    SeedOutside,
    /// The seed cell intersects the boundary; the point need not be on the surface.
    SeedOnBoundary,
    /// Mask grid ID or boundary revision differs from the computation input.
    StaleClassification,
}
impl fmt::Display for IbmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSurface(s) | Self::Query(s) => f.write_str(s),
            Self::SeedOutside => f.write_str("The fluid seed point `fluid_seed_point` lies outside the computational domain (bounding box)."),
            Self::SeedOnBoundary => f.write_str("The fluid seed point `fluid_seed_point` lies on an intersect (wall) cell; specify a seed clearly inside the fluid region."),
            Self::StaleClassification => f.write_str("classification belongs to a different grid or boundary state"),
        }
    }
}
impl std::error::Error for IbmError {}
