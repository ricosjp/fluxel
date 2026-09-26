//! File formats are adapters into the IBM surface model.
use crate::BuildError;
use fluxel_ibm::{Boundary, BoundarySurface};
use std::{collections::HashMap, path::Path, sync::Arc};
/// Load a surface by case-insensitive STL/OBJ extension, or return Boundary::None.
/// Available with the `mesh-io` feature. Coordinates are used without unit conversion.
///
/// # Errors
/// Rejects unsupported extensions, unreadable files, parse failures and invalid
/// surfaces. Delegates patch conventions to [`load_stl`] and [`load_obj`].
pub fn load_boundary(path: Option<&Path>) -> Result<Boundary, BuildError> {
    let Some(path) = path else {
        return Ok(Boundary::None);
    };
    let surface = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("stl") => load_stl(path)?,
        Some("obj") => load_obj(path)?,
        _ => {
            return Err(BuildError::InvalidInput(
                "Unsupported mesh extension. Use .stl or .obj.".into(),
            ))
        }
    };
    Ok(Boundary::Surface(Arc::new(surface)))
}
/// Read an STL surface with a single `_default` patch (ID zero).
/// Preserves parser triangle order as anchor IDs and does not rescale coordinates.
/// Errors on I/O, parsing, u32 index overflow, or invalid/degenerate surface data.
pub fn load_stl(path: &Path) -> Result<BoundarySurface, BuildError> {
    let mut file = std::fs::File::open(path).map_err(BuildError::Io)?;

    let stl = stl_io::read_stl(&mut file).map_err(|e| BuildError::Parse(e.to_string()))?;

    let vertices: Vec<[f64; 3]> = stl
        .vertices
        .into_iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();

    if vertices.len() > u32::MAX as usize {
        return Err(BuildError::InvalidInput(
            "surface exceeds u32 vertex index capacity".into(),
        ));
    }
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

    let patch_names: Vec<String> = vec!["_default".to_string()];
    let anchor_to_patch_id: Vec<usize> = vec![0; indices.len()];

    BoundarySurface::new(&vertices, &indices, patch_names, anchor_to_patch_id)
        .map_err(BuildError::from)
}

/// Read OBJ triangles with patch names derived from object and first group names.
/// Uses `_default` when a group is absent; a nonempty object name prefixes the
/// group with an underscore. Non-triangle primitives are skipped. Triangle order
/// defines anchor IDs. Errors on I/O/parsing, invalid indices or surface geometry.
pub fn load_obj(path: &Path) -> Result<BoundarySurface, BuildError> {
    let text = std::fs::read_to_string(path).map_err(BuildError::Io)?;
    let obj_set = wavefront_obj::obj::parse(text).map_err(|e| BuildError::Parse(e.to_string()))?;

    let mut vertices = Vec::<[f64; 3]>::new();
    let mut indices = Vec::<[u32; 3]>::new();
    let mut patch_name_map = HashMap::<String, usize>::new();
    let mut anchor_to_patch_id = Vec::<usize>::new();

    for obj in obj_set.objects {
        let mut local_to_global = Vec::<u32>::with_capacity(obj.vertices.len());
        for v in &obj.vertices {
            let gid = u32::try_from(vertices.len()).map_err(|_| {
                BuildError::InvalidInput("surface exceeds u32 vertex index capacity".into())
            })?;
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
                        let index = |i: usize| {
                            local_to_global.get(i).copied().ok_or_else(|| {
                                BuildError::Parse("OBJ vertex index out of range".into())
                            })
                        };
                        let g0 = index(i0.0)?;
                        let g1 = index(i1.0)?;
                        let g2 = index(i2.0)?;

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

    BoundarySurface::new(&vertices, &indices, patch_names, anchor_to_patch_id)
        .map_err(BuildError::from)
}
