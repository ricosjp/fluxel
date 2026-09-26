//! The only module that depends on Parry's shape, pose and feature representation.
use crate::IbmError;
use fluxel_geometry::RigidPose;
use parry3d_f64::{
    math::{Pose, Rotation, Vector},
    query::{self, PointQuery, Ray, RayCast},
    shape::{Cuboid, FeatureId, TriMesh},
};
#[derive(Debug)]
pub(super) struct SurfaceBackend {
    mesh: TriMesh,
}
/// Convert the validated backend-independent w/x/y/z transform to Parry conventions.
fn pose(value: &RigidPose) -> Pose {
    let [w, x, y, z] = value.quaternion();
    let [tx, ty, tz] = value.translation();
    Pose::from_parts(Vector::new(tx, ty, tz), Rotation::from_xyzw(x, y, z, w))
}
/// Convert an X/Y/Z array into the private backend vector type.
fn vector([x, y, z]: [f64; 3]) -> Vector {
    Vector::new(x, y, z)
}
impl SurfaceBackend {
    /// Build acceleration after index validation by BoundarySurface; reject degenerate/nonfinite triangles.
    pub fn new(vertices: &[[f64; 3]], indices: &[[u32; 3]]) -> Result<Self, IbmError> {
        for tri in indices {
            let a = vector(vertices[tri[0] as usize]);
            let b = vector(vertices[tri[1] as usize]);
            let c = vector(vertices[tri[2] as usize]);
            let normal = (b - a).cross(c - a);
            if !normal.is_finite() || normal.length_squared() == 0.0 {
                return Err(IbmError::InvalidSurface("degenerate triangle".into()));
            }
        }
        let mesh = TriMesh::new(
            vertices.iter().copied().map(vector).collect(),
            indices.to_vec(),
        )
        .map_err(|e| IbmError::InvalidSurface(e.to_string()))?;
        Ok(Self { mesh })
    }
    /// Decode front/back face IDs to triangle order; non-face features are query errors.
    fn anchor(&self, feature: FeatureId) -> Result<usize, IbmError> {
        match feature {
            FeatureId::Face(id) => Ok(id as usize % self.mesh.indices().len()),
            _ => Err(IbmError::Query(
                "surface query returned a non-face feature".into(),
            )),
        }
    }
    /// Query a posed surface against a physical cell AABB; map unsupported backend queries to IbmError.
    pub fn intersects(
        &self,
        center: [f64; 3],
        size: [f64; 3],
        transform: &RigidPose,
    ) -> Result<bool, IbmError> {
        let cell = Cuboid::new(vector(size) / 2.0);
        let cell_pose = Pose::translation(center[0], center[1], center[2]);
        query::intersection_test(&cell_pose, &cell, &pose(transform), &self.mesh)
            .map_err(|e| IbmError::Query(e.to_string()))
    }
    /// Find the first hit within the ray parameter limit and decode its triangle ID; None means no hit.
    pub fn ray(
        &self,
        origin: [f64; 3],
        direction: [f64; 3],
        length: f64,
        transform: &RigidPose,
    ) -> Result<Option<(f64, usize)>, IbmError> {
        self.mesh
            .cast_ray_and_get_normal(
                &pose(transform),
                &Ray::new(vector(origin), vector(direction)),
                length,
                false,
            )
            .map(|hit| {
                self.anchor(hit.feature)
                    .map(|anchor| (hit.time_of_impact, anchor))
            })
            .transpose()
    }
    /// Project a world-space point onto the posed triangle surface and decode its anchor.
    pub fn closest(
        &self,
        point: [f64; 3],
        transform: &RigidPose,
    ) -> Result<([f64; 3], usize), IbmError> {
        let (projection, feature) = self
            .mesh
            .project_point_and_get_feature(&pose(transform), vector(point));
        Ok((
            [projection.point.x, projection.point.y, projection.point.z],
            self.anchor(feature)?,
        ))
    }
    /// Use majority parity over 13 directions (at least 7 odd counts).
    ///     Each ray has reach 1e6 in world units and advances 1e-7 past each hit.
    ///     These fixed heuristics are scale-sensitive and assume meaningful surface
    ///     inside/outside; open/self-intersecting surfaces are not repaired.
    pub fn contains(&self, point: [f64; 3], transform: &RigidPose) -> bool {
        let p = vector(point);
        let pose_value = pose(transform);
        let pose = &pose_value;
        let dirs = [
            Vector::new(1.0, 0.11, 0.05).normalize(),
            Vector::new(-1.0, -0.05, 0.11).normalize(),
            Vector::new(0.05, 1.0, -0.11).normalize(),
            Vector::new(-0.11, -1.0, 0.05).normalize(),
            Vector::new(0.11, 0.05, 1.0).normalize(),
            Vector::new(-0.05, -0.11, -1.0).normalize(),
            Vector::new(0.577, 0.577, 0.577).normalize(),
            Vector::new(-0.577, -0.577, -0.577).normalize(),
            Vector::new(0.577, -0.577, 0.577).normalize(),
            Vector::new(-0.577, 0.577, -0.577).normalize(),
            Vector::new(0.577, 0.577, -0.577).normalize(),
            Vector::new(-0.577, -0.577, 0.577).normalize(),
            Vector::new(-0.577, 0.577, 0.577).normalize(),
        ];

        let inside_count = dirs
            .iter()
            .filter(|dir| {
                let dir = **dir;
                let mut intersections = 0;
                let mut current_p = p;
                let mut max_dist = 1e6;

                while let Some(toi) =
                    self.mesh
                        .cast_ray(pose, &Ray::new(current_p, dir), max_dist, false)
                {
                    intersections += 1;
                    current_p += dir * (toi + 1e-7);
                    max_dist -= toi + 1e-7;
                    if max_dist <= 0.0 {
                        break;
                    }
                }

                intersections % 2 != 0
            })
            .count();

        inside_count >= 7
    }
}
