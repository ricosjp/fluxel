#![cfg(feature = "mesh-io")]
use fluxel_engine::io::{load_boundary, load_obj, load_stl};
use std::io::Write;
use tempfile::NamedTempFile;
#[test]
fn obj_patch_names_and_triangle_order_are_preserved() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(
        file,
        "o test\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\ng a\nf 1 2 3\ng b\nf 1 3 4"
    )
    .unwrap();
    let surface = load_obj(file.path()).unwrap();
    assert_eq!(surface.triangle_count(), 2);
    assert_eq!(surface.patches().names(), &["test_a", "test_b"]);
}
#[test]
fn stl_default_patch_and_invalid_files() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file,"solid test\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid test").unwrap();
    let surface = load_stl(file.path()).unwrap();
    assert_eq!(surface.patches().names(), &["_default"]);
    assert!(load_boundary(Some(file.path())).is_err());
    let empty = NamedTempFile::new().unwrap();
    assert!(load_obj(empty.path()).is_err());
}
