//! File formats are adapters into the IBM surface model.
mod obj;
mod stl;

use crate::BuildError;
use fluxel_ibm::Boundary;
use std::{path::Path, sync::Arc};

pub use obj::load_obj;
pub use stl::load_stl;

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
