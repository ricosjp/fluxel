//! Wrapper types for STL-like meshes and BVH acceleration.

use parry3d_f64::math::Vector;
use parry3d_f64::shape::TriMesh;
use std::collections::HashMap;
use std::path::Path;

/// BVH-backed mesh wrapper for efficient intersection queries on complex 3D geometry.
pub struct IBMMesh {
    /// `parry3d` triangle mesh with an internally built BVH.
    pub bvh: TriMesh,

    // Patch name list
    pub patch_names: Vec<String>,

    // Face patch ids (length = number of faces)
    pub anchor_to_patch_id: Vec<usize>,
}

impl IBMMesh {
    /// Builds an `IBMMesh` from vertex and triangle-index arrays.
    ///
    /// STL loading utilities can be implemented as thin wrappers around this constructor.
    pub fn from_vertices_indices_and_patches(
        vertices: &[[f64; 3]],
        indices: &[[u32; 3]],
        patch_names: Vec<String>,
        anchor_to_patch_id: Vec<usize>,
    ) -> Self {
        let points: Vec<Vector> = vertices
            .iter()
            .map(|v| Vector::new(v[0], v[1], v[2]))
            .collect();

        Self {
            bvh: TriMesh::new(points, indices.to_vec()).expect("Failed to build TriMesh"),
            patch_names,
            anchor_to_patch_id,
        }
    }

    pub fn from_stl_file(path: &Path) -> Result<Self, String> {
        let mut file =
            std::fs::File::open(path).map_err(|e| format!("Failed to open STL file: {}", e))?;

        let stl =
            stl_io::read_stl(&mut file).map_err(|e| format!("Failed to read STL file: {}", e))?;

        let vertices: Vec<[f64; 3]> = stl
            .vertices
            .into_iter()
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();

        let indices: Vec<[u32; 3]> = stl
            .faces
            .into_iter()
            .map(|f| {
                [
                    f.vertices[0] as u32,
                    f.vertices[1] as u32,
                    f.vertices[2] as u32,
                ]
            })
            .collect();

        let patch_names: Vec<String> = vec!["_default".to_string(); indices.len()];
        let anchor_to_patch_id: Vec<usize> = vec![0; indices.len()];

        Ok(Self::from_vertices_indices_and_patches(
            &vertices,
            &indices,
            patch_names,
            anchor_to_patch_id,
        ))
    }

    pub fn from_obj_file(path: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read OBJ file: {}", e))?;
        let obj_set = wavefront_obj::obj::parse(text)
            .map_err(|e| format!("Failed to parse OBJ file: {}", e))?;

        let mut vertices = Vec::<[f64; 3]>::new();
        let mut indices = Vec::<[u32; 3]>::new();
        let mut patch_name_map = HashMap::<String, usize>::new();
        let mut anchor_to_patch_id = Vec::<usize>::new();

        for obj in obj_set.objects {
            let mut local_to_global = Vec::<u32>::with_capacity(obj.vertices.len());
            for v in &obj.vertices {
                let gid = vertices.len() as u32;
                vertices.push([v.x, v.y, v.z]);
                local_to_global.push(gid);
            }

            for geometry in obj.geometry {
                for shape in geometry.shapes {
                    // Get patch name and id
                    let group_name = shape
                        .groups
                        .first()
                        .map(|s| s.as_str())
                        .unwrap_or("_default");
                    let patch_name = if obj.name.is_empty() {
                        group_name.to_string()
                    } else {
                        format!("{}_{}", obj.name, group_name)
                    };
                    let patch_id = if let Some(&id) = patch_name_map.get(&patch_name) {
                        id
                    } else {
                        patch_name_map.insert(patch_name, patch_name_map.len());
                        patch_name_map.len() - 1
                    };

                    match shape.primitive {
                        wavefront_obj::obj::Primitive::Triangle(i0, i1, i2) => {
                            let g0 = local_to_global[i0.0];
                            let g1 = local_to_global[i1.0];
                            let g2 = local_to_global[i2.0];

                            indices.push([g0, g1, g2]);
                            anchor_to_patch_id.push(patch_id);
                        }
                        _ => {
                            continue;
                        }
                    }
                }
            }
        }

        let mut patch_entries: Vec<(String, usize)> = patch_name_map.into_iter().collect();
        patch_entries.sort_by_key(|(_, id)| *id);
        let patch_names: Vec<String> = patch_entries.into_iter().map(|(name, _)| name).collect();

        Ok(Self::from_vertices_indices_and_patches(
            &vertices,
            &indices,
            patch_names,
            anchor_to_patch_id,
        ))
    }
}
