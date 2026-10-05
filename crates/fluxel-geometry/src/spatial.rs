//! Selects [`crate::cartesian`] or [`crate::cylindrical`] for the mesh and immersed-boundary methods.

use crate::{
    get_global_id_from_phys, CoordinateType, CylindricalGeometry, Geometry, ParameterFace,
    ParameterInterval,
};
use fluxel_core::Forest;

/// Inclusive parameter box of a cell or face.
///
/// Cartesian components are world `(x, y, z)`. Cylindrical components are `(r, θ, z)`.
/// A face has equal minimum and maximum on its constant axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterBox {
    /// Inclusive lower corner.
    pub min: [f64; 3],
    /// Inclusive upper corner. Cell boxes use the geometric outer corner.
    pub max: [f64; 3],
}

/// Sample, widths, volume, centroid, corners, and parameter box of one cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellMeasure {
    /// World image of the parameter midpoint.
    pub sample: [f64; 3],
    /// Coordinate-aligned widths. Cylindrical values are `[Δr, r Δθ, Δz]`.
    pub widths: [f64; 3],
    /// Exact cell volume.
    pub volume: f64,
    /// Volume centroid in world coordinates.
    pub centroid: [f64; 3],
    /// Eight parameter corners mapped to world, in VTK hexahedron order.
    pub corners: [[f64; 3]; 8],
    /// Parameter box of the cell.
    pub parameter: ParameterBox,
}

/// Physical domain paired with one forest.
#[derive(Debug, Clone)]
pub enum SpatialDomain {
    /// Axis-aligned octree mapped by equal root boxes.
    Cartesian(Geometry),
    /// Annular sector about an axis parallel to world Z.
    Cylindrical(CylindricalGeometry),
}

impl SpatialDomain {
    /// Coordinate system stored with the background mesh.
    pub fn coordinate_type(&self) -> CoordinateType {
        match self {
            Self::Cartesian(_) => CoordinateType::Cartesian,
            Self::Cylindrical(_) => CoordinateType::Cylindrical,
        }
    }

    /// Cartesian mapping, when this domain uses one.
    pub fn cartesian(&self) -> Option<&Geometry> {
        match self {
            Self::Cartesian(geometry) => Some(geometry),
            Self::Cylindrical(_) => None,
        }
    }

    /// Cylindrical mapping, when this domain uses one.
    pub fn cylindrical(&self) -> Option<&CylindricalGeometry> {
        match self {
            Self::Cylindrical(geometry) => Some(geometry),
            Self::Cartesian(_) => None,
        }
    }

    /// Largest world length used by the Gibou near-boundary test.
    pub fn domain_length(&self) -> f64 {
        match self {
            Self::Cartesian(geometry) => {
                let bounds = geometry.bounding_box();
                (0..3)
                    .map(|axis| bounds.max()[axis] - bounds.min()[axis])
                    .fold(0.0, f64::max)
            }
            Self::Cylindrical(geometry) => geometry.domain_length(),
        }
    }

    /// World sample point and coordinate-aligned widths of one cell.
    ///
    /// Cartesian widths are the edge lengths. Cylindrical widths are
    /// `[Δr, r Δθ, Δz]`, and the sample is the parameter midpoint mapped to world.
    pub fn cell_sample(
        &self,
        forest: &Forest,
        cell_id: usize,
    ) -> Result<([f64; 3], [f64; 3]), crate::GeometryError> {
        let logical = forest.keys()[cell_id].to_logical();
        match self {
            Self::Cartesian(geometry) => Ok(geometry.cell_bounds(&logical)),
            Self::Cylindrical(geometry) => {
                let interval = geometry.cell_interval(&logical, forest.base_resolution())?;
                let metrics = geometry.cell_metrics(interval)?;
                Ok((metrics.sample_point, metrics.representative_widths))
            }
        }
    }

    /// World AABB center and side lengths used as an intersection candidate filter.
    pub fn world_box(
        &self,
        forest: &Forest,
        cell_id: usize,
    ) -> Result<([f64; 3], [f64; 3]), crate::GeometryError> {
        let logical = forest.keys()[cell_id].to_logical();
        match self {
            Self::Cartesian(geometry) => Ok(geometry.cell_bounds(&logical)),
            Self::Cylindrical(geometry) => {
                let interval = geometry.cell_interval(&logical, forest.base_resolution())?;
                Ok(geometry.world_aabb(interval))
            }
        }
    }

    /// Orthonormal frame for stencil offsets at a world sample.
    ///
    /// Cartesian offsets stay aligned with X/Y/Z. Cylindrical offsets follow
    /// `(e_r, e_θ, e_z)` at the sample's angle.
    pub fn sample_frame(&self, sample: [f64; 3]) -> [[f64; 3]; 3] {
        match self {
            Self::Cartesian(_) => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            Self::Cylindrical(geometry) => {
                let theta = geometry
                    .param_from_world(sample)
                    .map(|parameter| parameter[1])
                    .unwrap_or_else(|| geometry.theta_start());
                geometry.local_frame(theta)
            }
        }
    }

    /// Finest cell containing a world point, or None outside the half-open domain.
    pub fn locate(&self, forest: &Forest, point: [f64; 3]) -> Option<usize> {
        match self {
            Self::Cartesian(geometry) => get_global_id_from_phys(forest, geometry, point),
            Self::Cylindrical(geometry) => {
                let (tree_id, x, y, z) = geometry.logical_point(point, forest.base_resolution())?;
                let key = forest.find_cell_containing(tree_id, x, y, z)?;
                forest.keys().binary_search(&key).ok()
            }
        }
    }

    /// Parameter box of one cell. The upper corner is the geometric face, included.
    pub fn parameter_box(
        &self,
        forest: &Forest,
        cell_id: usize,
    ) -> Result<ParameterBox, crate::GeometryError> {
        let logical = forest.keys()[cell_id].to_logical();
        match self {
            Self::Cartesian(geometry) => {
                let (center, size) = geometry.cell_bounds(&logical);
                Ok(box_from_center(center, size))
            }
            Self::Cylindrical(geometry) => {
                let interval = geometry.cell_interval(&logical, forest.base_resolution())?;
                Ok(ParameterBox {
                    min: interval.min,
                    max: interval.max,
                })
            }
        }
    }

    /// Volume, centroid, corners, and the sample already stored on the mesh.
    pub fn cell_measure(
        &self,
        forest: &Forest,
        cell_id: usize,
    ) -> Result<CellMeasure, crate::GeometryError> {
        let parameter = self.parameter_box(forest, cell_id)?;
        match self {
            Self::Cartesian(_) => {
                let size = sub(parameter.max, parameter.min);
                let sample = midpoint(parameter.min, parameter.max);
                Ok(CellMeasure {
                    sample,
                    widths: size,
                    volume: size[0] * size[1] * size[2],
                    centroid: sample,
                    corners: mapped_corners(parameter, |point| point),
                    parameter,
                })
            }
            Self::Cylindrical(geometry) => {
                let interval = ParameterInterval {
                    min: parameter.min,
                    max: parameter.max,
                };
                let metrics = geometry.cell_metrics(interval)?;
                Ok(CellMeasure {
                    sample: metrics.sample_point,
                    widths: metrics.representative_widths,
                    volume: metrics.volume,
                    centroid: metrics.centroid,
                    corners: mapped_corners(parameter, |point| {
                        geometry.world_from_param(point[0], point[1], point[2])
                    }),
                    parameter,
                })
            }
        }
    }

    /// Area vector of a parameter face.
    ///
    /// `sign` is `+1` toward increasing parameter and `-1` toward decreasing parameter.
    /// The face box is constant on one axis.
    pub fn face_area_vector(
        &self,
        face: ParameterBox,
        sign: f64,
    ) -> Result<[f64; 3], crate::GeometryError> {
        let axis = face_axis(face);
        let span = sub(face.max, face.min);
        match self {
            Self::Cartesian(_) => {
                let area = match axis {
                    0 => span[1] * span[2],
                    1 => span[0] * span[2],
                    _ => span[0] * span[1],
                };
                let mut vector = [0.0; 3];
                vector[axis] = sign * area;
                Ok(vector)
            }
            Self::Cylindrical(geometry) => {
                let curved = match axis {
                    0 => ParameterFace::Radius {
                        radius: face.min[0],
                        theta: [face.min[1], face.max[1]],
                        z: [face.min[2], face.max[2]],
                    },
                    1 => ParameterFace::Theta {
                        theta: face.min[1],
                        radius: [face.min[0], face.max[0]],
                        z: [face.min[2], face.max[2]],
                    },
                    _ => ParameterFace::Z {
                        z: face.min[2],
                        radius: [face.min[0], face.max[0]],
                        theta: [face.min[1], face.max[1]],
                    },
                };
                let metrics = geometry.face_metrics(curved)?;
                Ok(scale(metrics.area_vector, sign))
            }
        }
    }
}

fn box_from_center(center: [f64; 3], size: [f64; 3]) -> ParameterBox {
    ParameterBox {
        min: [
            center[0] - 0.5 * size[0],
            center[1] - 0.5 * size[1],
            center[2] - 0.5 * size[2],
        ],
        max: [
            center[0] + 0.5 * size[0],
            center[1] + 0.5 * size[1],
            center[2] + 0.5 * size[2],
        ],
    }
}

fn mapped_corners(parameter: ParameterBox, map: impl Fn([f64; 3]) -> [f64; 3]) -> [[f64; 3]; 8] {
    let [x0, y0, z0] = parameter.min;
    let [x1, y1, z1] = parameter.max;
    [
        map([x0, y0, z0]),
        map([x1, y0, z0]),
        map([x1, y1, z0]),
        map([x0, y1, z0]),
        map([x0, y0, z1]),
        map([x1, y0, z1]),
        map([x1, y1, z1]),
        map([x0, y1, z1]),
    ]
}

fn face_axis(face: ParameterBox) -> usize {
    (0..3)
        .min_by(|&left, &right| {
            let left_span = (face.max[left] - face.min[left]).abs();
            let right_span = (face.max[right] - face.min[right]).abs();
            left_span.partial_cmp(&right_span).unwrap()
        })
        .unwrap_or(0)
}

fn midpoint(min: [f64; 3], max: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ]
}

fn sub(max: [f64; 3], min: [f64; 3]) -> [f64; 3] {
    [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}
