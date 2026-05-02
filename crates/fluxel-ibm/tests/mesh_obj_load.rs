//! Integration: load a minimal Wavefront OBJ from disk (file I/O and parsing).

use fluxel_ibm::IBMMesh;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn from_obj_file_loads_single_triangle() {
    let mut file = NamedTempFile::new().expect("temp file");
    writeln!(
        file,
        "o test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"
    )
    .unwrap();

    let mesh = IBMMesh::from_obj_file(file.path()).expect("parse obj");
    assert_eq!(mesh.bvh.indices().len(), 1);
    assert!(!mesh.patch_names.is_empty());
}
