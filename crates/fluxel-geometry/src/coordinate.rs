/// Coordinate type for the domain.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateType {
    #[default]
    Cartesian = 0, // Default
    Cylindrical = 1,
    Spherical = 2,
}
