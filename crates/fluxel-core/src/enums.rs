/// Coordinate type for the domain.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateType {
    #[default]
    Cartesian = 0, // Default
    Cylindrical = 1,
    Spherical = 2,
}

/// Axis for the domain.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    /// All axes.
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Index for 3D component access: X→0, Y→1, Z→2.
    pub const fn as_index(self) -> usize {
        self as u8 as usize
    }

    /// Splits the axis into the two directions that define it.
    pub const fn split_into_directions(self) -> (Direction, Direction) {
        match self {
            Self::X => (Direction::XMinus, Direction::XPlus),
            Self::Y => (Direction::YMinus, Direction::YPlus),
            Self::Z => (Direction::ZMinus, Direction::ZPlus),
        }
    }
}

/// One of the six axis-aligned face directions between octree cells.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5,
}

impl Direction {
    /// All directions.
    pub const ALL: [Self; 6] = [
        Self::XMinus,
        Self::XPlus,
        Self::YMinus,
        Self::YPlus,
        Self::ZMinus,
        Self::ZPlus,
    ];

    /// Returns the coordinate axis this direction is aligned with.
    pub const fn axis(self) -> Axis {
        match self {
            Self::XMinus | Self::XPlus => Axis::X,
            Self::YMinus | Self::YPlus => Axis::Y,
            Self::ZMinus | Self::ZPlus => Axis::Z,
        }
    }

    /// Returns the opposite face direction (same axis, inverted sign).
    pub const fn opposite(self) -> Self {
        match self {
            Self::XMinus => Self::XPlus,
            Self::XPlus => Self::XMinus,
            Self::YMinus => Self::YPlus,
            Self::YPlus => Self::YMinus,
            Self::ZMinus => Self::ZPlus,
            Self::ZPlus => Self::ZMinus,
        }
    }

    /// Unit sign along the axis: `-1` for *Minus, `+1` for *Plus.
    pub const fn sign(self) -> i8 {
        if (self as u8) & 1 == 0 {
            -1
        } else {
            1
        }
    }
}
