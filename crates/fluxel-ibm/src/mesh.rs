//! Wrapper types for STL-like meshes and BVH acceleration.

use parry3d_f64::math::Vector;
use parry3d_f64::shape::TriMesh;

/// BVH-backed mesh wrapper for efficient intersection queries on complex 3D geometry.
pub struct IBMMesh {
    /// `parry3d` triangle mesh with an internally built BVH.
    pub bvh: TriMesh,
}

impl IBMMesh {
    /// Builds an `IBMMesh` from vertex and triangle-index arrays.
    ///
    /// STL loading utilities can be implemented as thin wrappers around this constructor.
    pub fn from_vertices_and_indices(vertices: &[[f64; 3]], indices: &[[u32; 3]]) -> Self {
        let points: Vec<Vector> = vertices
            .iter()
            .map(|v| Vector::new(v[0], v[1], v[2]))
            .collect();

        Self {
            bvh: TriMesh::new(points, indices.to_vec()).expect("Failed to build TriMesh"),
        }
    }
}
