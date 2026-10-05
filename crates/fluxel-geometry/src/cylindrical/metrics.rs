//! Cell volume and face area of an annular sector.
//!
//! Cartesian cell widths live on [`crate::cartesian::domain::Geometry::cell_bounds`].
//! These measures are the cylindrical counterpart, including curved-face quadrature.

use super::domain::{CylindricalGeometry, ParameterInterval};
use crate::GeometryError;

/// One parameter-aligned face of an annular sector.
///
/// [`FaceMetrics::area_vector`] points toward increasing `r`, `θ`, or `z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterFace {
    /// Patch at constant radius.
    Radius {
        /// Face radius.
        radius: f64,
        /// Inclusive/exclusive angular interval `[start, end)`.
        theta: [f64; 2],
        /// Inclusive/exclusive axial interval `[start, end)`.
        z: [f64; 2],
    },
    /// Patch at constant angle.
    Theta {
        /// Face angle.
        theta: f64,
        /// Inclusive/exclusive radial interval `[start, end)`.
        radius: [f64; 2],
        /// Inclusive/exclusive axial interval `[start, end)`.
        z: [f64; 2],
    },
    /// Patch at constant axial coordinate.
    Z {
        /// Face axial coordinate.
        z: f64,
        /// Inclusive/exclusive radial interval `[start, end)`.
        radius: [f64; 2],
        /// Inclusive/exclusive angular interval `[start, end)`.
        theta: [f64; 2],
    },
}

/// Volume, sample point, and volume centroid of one annular sector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellMetrics {
    /// Exact sector volume.
    pub volume: f64,
    /// Image of the parameter-interval midpoint. This is not the volume centroid.
    pub sample_point: [f64; 3],
    /// Volume centroid in world coordinates.
    pub centroid: [f64; 3],
    /// Physical lengths `[Δr, r_mid Δθ, Δz]`. These are not world AABB widths.
    pub representative_widths: [f64; 3],
}

/// Scalar area, integrated normal, and area centroid of one face.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceMetrics {
    /// Positive scalar area.
    pub area: f64,
    /// Integrated unit normal, `∫ n dA`, in the increasing-parameter direction.
    pub area_vector: [f64; 3],
    /// Area centroid in world coordinates. A curved face may place this off the surface.
    pub centroid: [f64; 3],
    /// Positive area weights. Their normals reconstruct [`Self::area_vector`].
    pub quadrature: Vec<QuadraturePoint>,
}

/// One area sample on a face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraturePoint {
    /// World position.
    pub point: [f64; 3],
    /// Positive area weight.
    pub weight: f64,
    /// Unit normal in the increasing-parameter direction.
    pub normal: [f64; 3],
}

impl CylindricalGeometry {
    /// Exact volume, midpoint image, and volume centroid of an annular sector.
    ///
    /// # Errors
    /// Rejects a nonpositive radius, an empty interval, or an angle of at least one turn.
    pub fn cell_metrics(&self, interval: ParameterInterval) -> Result<CellMetrics, GeometryError> {
        let [r_minus, theta_minus, z_minus] = interval.min;
        let [r_plus, theta_plus, z_plus] = interval.max;
        let d_radius = r_plus - r_minus;
        let d_theta = theta_plus - theta_minus;
        let d_z = z_plus - z_minus;
        if r_minus <= 0.0
            || d_radius <= 0.0
            || d_theta <= 0.0
            || d_theta >= std::f64::consts::TAU
            || d_z <= 0.0
            || ![r_minus, r_plus, theta_minus, theta_plus, z_minus, z_plus]
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        let radius_mid = 0.5 * (r_minus + r_plus);
        let theta_mid = 0.5 * (theta_minus + theta_plus);
        let z_mid = 0.5 * (z_minus + z_plus);
        let radial_centroid = mean_cubic_radius(r_minus, r_plus) * sinc(0.5 * d_theta);
        Ok(CellMetrics {
            volume: 0.5 * (r_plus * r_plus - r_minus * r_minus) * d_theta * d_z,
            sample_point: self.world_from_param(radius_mid, theta_mid, z_mid),
            centroid: self.world_from_param(radial_centroid, theta_mid, z_mid),
            representative_widths: [d_radius, radius_mid * d_theta, d_z],
        })
    }

    /// Scalar area, integrated normal, area centroid, and quadrature for one face.
    ///
    /// Quadrature weights sum to the scalar area. Weighted unit normals sum to the
    /// area vector. On planar faces one centroid sample integrates linear world
    /// fields exactly. A constant-radius face uses eight Gauss points in angle,
    /// which integrates a linear world field through a right angle to roundoff.
    ///
    /// # Errors
    /// Rejects a nonpositive radius, an empty patch, or an angle of at least one turn.
    pub fn face_metrics(&self, face: ParameterFace) -> Result<FaceMetrics, GeometryError> {
        match face {
            ParameterFace::Radius { radius, theta, z } => self.radius_face(radius, theta, z),
            ParameterFace::Theta { theta, radius, z } => self.theta_face(theta, radius, z),
            ParameterFace::Z { z, radius, theta } => self.z_face(z, radius, theta),
        }
    }
    fn radius_face(
        &self,
        radius: f64,
        theta: [f64; 2],
        z: [f64; 2],
    ) -> Result<FaceMetrics, GeometryError> {
        let d_theta = theta[1] - theta[0];
        let d_z = z[1] - z[0];
        if radius <= 0.0
            || d_theta <= 0.0
            || d_theta >= std::f64::consts::TAU
            || d_z <= 0.0
            || ![radius, theta[0], theta[1], z[0], z[1]]
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        let theta_mid = 0.5 * (theta[0] + theta[1]);
        let z_mid = 0.5 * (z[0] + z[1]);
        let area = radius * d_theta * d_z;
        let area_vector = [
            radius * d_z * (theta[1].sin() - theta[0].sin()),
            radius * d_z * (theta[0].cos() - theta[1].cos()),
            0.0,
        ];
        let centroid = self.world_from_param(radius * sinc(0.5 * d_theta), theta_mid, z_mid);
        let mut quadrature = Vec::with_capacity(8);
        for node in gauss_legendre8() {
            let sample_theta = theta_mid + 0.5 * d_theta * node.position;
            let [e_r, _, _] = self.local_frame(sample_theta);
            quadrature.push(QuadraturePoint {
                point: self.world_from_param(radius, sample_theta, z_mid),
                weight: radius * (0.5 * d_theta) * node.weight * d_z,
                normal: e_r,
            });
        }
        Ok(FaceMetrics {
            area,
            area_vector,
            centroid,
            quadrature,
        })
    }

    fn theta_face(
        &self,
        theta: f64,
        radius: [f64; 2],
        z: [f64; 2],
    ) -> Result<FaceMetrics, GeometryError> {
        let d_radius = radius[1] - radius[0];
        let d_z = z[1] - z[0];
        if radius[0] <= 0.0
            || d_radius <= 0.0
            || d_z <= 0.0
            || ![theta, radius[0], radius[1], z[0], z[1]]
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        let area = d_radius * d_z;
        let [_, e_theta, _] = self.local_frame(theta);
        let area_vector = scale(e_theta, area);
        let centroid =
            self.world_from_param(0.5 * (radius[0] + radius[1]), theta, 0.5 * (z[0] + z[1]));
        Ok(planar_face(area, area_vector, centroid, e_theta))
    }

    fn z_face(
        &self,
        z: f64,
        radius: [f64; 2],
        theta: [f64; 2],
    ) -> Result<FaceMetrics, GeometryError> {
        let d_radius = radius[1] - radius[0];
        let d_theta = theta[1] - theta[0];
        if radius[0] <= 0.0
            || d_radius <= 0.0
            || d_theta <= 0.0
            || d_theta >= std::f64::consts::TAU
            || ![z, radius[0], radius[1], theta[0], theta[1]]
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(GeometryError::InvalidCylindricalDomain);
        }
        let area = 0.5 * (radius[1] * radius[1] - radius[0] * radius[0]) * d_theta;
        let area_vector = [0.0, 0.0, area];
        let radial_centroid = mean_cubic_radius(radius[0], radius[1]) * sinc(0.5 * d_theta);
        let centroid = self.world_from_param(radial_centroid, 0.5 * (theta[0] + theta[1]), z);
        Ok(planar_face(area, area_vector, centroid, [0.0, 0.0, 1.0]))
    }
}

/// Six parameter faces of a cell, ordered `r-`, `r+`, `θ-`, `θ+`, `z-`, `z+`.
///
/// Each face still carries the increasing-parameter normal. The outward normal
/// on a minimum face is the opposite of that vector.
pub fn sector_faces(interval: ParameterInterval) -> [ParameterFace; 6] {
    let [r_minus, theta_minus, z_minus] = interval.min;
    let [r_plus, theta_plus, z_plus] = interval.max;
    [
        ParameterFace::Radius {
            radius: r_minus,
            theta: [theta_minus, theta_plus],
            z: [z_minus, z_plus],
        },
        ParameterFace::Radius {
            radius: r_plus,
            theta: [theta_minus, theta_plus],
            z: [z_minus, z_plus],
        },
        ParameterFace::Theta {
            theta: theta_minus,
            radius: [r_minus, r_plus],
            z: [z_minus, z_plus],
        },
        ParameterFace::Theta {
            theta: theta_plus,
            radius: [r_minus, r_plus],
            z: [z_minus, z_plus],
        },
        ParameterFace::Z {
            z: z_minus,
            radius: [r_minus, r_plus],
            theta: [theta_minus, theta_plus],
        },
        ParameterFace::Z {
            z: z_plus,
            radius: [r_minus, r_plus],
            theta: [theta_minus, theta_plus],
        },
    ]
}

fn planar_face(
    area: f64,
    area_vector: [f64; 3],
    centroid: [f64; 3],
    normal: [f64; 3],
) -> FaceMetrics {
    FaceMetrics {
        area,
        area_vector,
        centroid,
        quadrature: vec![QuadraturePoint {
            point: centroid,
            weight: area,
            normal,
        }],
    }
}

/// `sin(t) / t`, with a Taylor value at the origin.
fn sinc(angle: f64) -> f64 {
    let square = angle * angle;
    if square < 1e-8 {
        1.0 - square / 6.0
    } else {
        angle.sin() / angle
    }
}

/// `(2/3) (r_+^3 - r_-^3) / (r_+^2 - r_-^2)` without cancelling the radial gap.
fn mean_cubic_radius(r_minus: f64, r_plus: f64) -> f64 {
    (2.0 / 3.0) * (r_plus * r_plus + r_plus * r_minus + r_minus * r_minus) / (r_minus + r_plus)
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

#[derive(Clone, Copy)]
struct GaussNode {
    position: f64,
    weight: f64,
}

/// Eight-point Gauss–Legendre rule on `[-1, 1]`, exact through degree 15.
fn gauss_legendre8() -> [GaussNode; 8] {
    const POSITIVE: [(f64, f64); 4] = [
        (0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (0.960_289_856_497_536_3, 0.101_228_536_290_376_26),
    ];
    let mut nodes = [GaussNode {
        position: 0.0,
        weight: 0.0,
    }; 8];
    for (index, (position, weight)) in POSITIVE.into_iter().enumerate() {
        nodes[index] = GaussNode {
            position: -position,
            weight,
        };
        nodes[7 - index] = GaussNode { position, weight };
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::super::domain::CylindricalGeometry;
    use super::*;

    fn sector() -> CylindricalGeometry {
        CylindricalGeometry::new(
            [1.0, 2.0, 3.0],
            1.0,
            2.0,
            -0.25,
            std::f64::consts::FRAC_PI_2,
            4.0,
            7.0,
        )
        .unwrap()
    }

    fn whole_interval(domain: &CylindricalGeometry) -> ParameterInterval {
        ParameterInterval {
            min: [domain.r_min(), domain.theta_start(), domain.z_min()],
            max: [
                domain.r_max(),
                domain.theta_start() + domain.theta_extent(),
                domain.z_max(),
            ],
        }
    }
    #[test]
    fn volume_matches_the_annular_sector_formula() {
        let domain = sector();
        let metrics = domain.cell_metrics(whole_interval(&domain)).unwrap();
        let expected = 0.5 * (4.0 - 1.0) * domain.theta_extent() * 3.0;
        assert!((metrics.volume - expected).abs() < 1e-12);
        let sample =
            domain.world_from_param(1.5, domain.theta_start() + 0.5 * domain.theta_extent(), 5.5);
        assert!(sample
            .iter()
            .zip(metrics.sample_point)
            .all(|(expected, actual)| (expected - actual).abs() < 1e-12));
        let sample_radius = (metrics.sample_point[0] - domain.origin()[0])
            .hypot(metrics.sample_point[1] - domain.origin()[1]);
        let centroid_radius = (metrics.centroid[0] - domain.origin()[0])
            .hypot(metrics.centroid[1] - domain.origin()[1]);
        assert!((sample_radius - 1.5).abs() < 1e-12);
        assert!((centroid_radius - sample_radius).abs() > 1e-3);
    }

    #[test]
    fn octants_preserve_volume_and_area_vectors_close() {
        let domain = sector();
        let parent = whole_interval(&domain);
        let parent_metrics = domain.cell_metrics(parent).unwrap();
        let mid = [
            0.5 * (parent.min[0] + parent.max[0]),
            0.5 * (parent.min[1] + parent.max[1]),
            0.5 * (parent.min[2] + parent.max[2]),
        ];
        let mut volume = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let min = [
                        if i == 0 { parent.min[0] } else { mid[0] },
                        if j == 0 { parent.min[1] } else { mid[1] },
                        if k == 0 { parent.min[2] } else { mid[2] },
                    ];
                    let max = [
                        if i == 0 { mid[0] } else { parent.max[0] },
                        if j == 0 { mid[1] } else { parent.max[1] },
                        if k == 0 { mid[2] } else { parent.max[2] },
                    ];
                    volume += domain
                        .cell_metrics(ParameterInterval { min, max })
                        .unwrap()
                        .volume;
                }
            }
        }
        assert!((volume - parent_metrics.volume).abs() < 1e-12);

        let mut closure = [0.0; 3];
        for (index, face) in sector_faces(parent).into_iter().enumerate() {
            let metrics = domain.face_metrics(face).unwrap();
            let sign = if index % 2 == 0 { -1.0 } else { 1.0 };
            for (slot, component) in closure.iter_mut().zip(metrics.area_vector) {
                *slot += sign * component;
            }
            let weight: f64 = metrics.quadrature.iter().map(|point| point.weight).sum();
            assert!((weight - metrics.area).abs() < 1e-12);
            let mut integrated = [0.0; 3];
            for point in &metrics.quadrature {
                for (slot, component) in integrated.iter_mut().zip(point.normal) {
                    *slot += point.weight * component;
                }
            }
            for (actual, expected) in integrated.iter().zip(metrics.area_vector) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        assert!(closure.iter().all(|component| component.abs() < 1e-12));
    }

    #[test]
    fn radius_face_quadrature_integrates_a_linear_world_field() {
        let domain = sector();
        let interval = whole_interval(&domain);
        let face = sector_faces(interval)[1];
        let metrics = domain.face_metrics(face).unwrap();
        let mut numerical = [0.0; 3];
        for point in &metrics.quadrature {
            let relative_x = point.point[0] - domain.origin()[0];
            for (slot, component) in numerical.iter_mut().zip(point.normal) {
                *slot += point.weight * relative_x * component;
            }
        }
        let radius = domain.r_max();
        let theta_minus = domain.theta_start();
        let theta_plus = theta_minus + domain.theta_extent();
        let d_theta = domain.theta_extent();
        let d_z = domain.z_max() - domain.z_min();
        let integral_cos_sq =
            0.5 * d_theta + 0.25 * ((2.0 * theta_plus).sin() - (2.0 * theta_minus).sin());
        let integral_cos_sin = -0.25 * ((2.0 * theta_plus).cos() - (2.0 * theta_minus).cos());
        let scale = radius * radius * d_z;
        let analytic = [scale * integral_cos_sq, scale * integral_cos_sin, 0.0];
        for (actual, expected) in numerical.iter().zip(analytic) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
