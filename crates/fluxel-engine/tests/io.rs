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
fn facet_at(z: &str) -> String {
    format!(
        "facet normal 0 0 1\nouter loop\nvertex 0 0 {z}\nvertex 1 0 {z}\nvertex 0 1 {z}\nendloop\nendfacet\n"
    )
}
#[test]
fn ascii_stl_uses_the_solid_name_as_the_patch() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "solid test\n{}endsolid test", facet_at("0")).unwrap();
    let surface = load_stl(file.path()).unwrap();
    assert_eq!(surface.triangle_count(), 1);
    assert_eq!(surface.patches().names(), &["test"]);
    assert!(load_boundary(Some(file.path())).is_err());
}
#[test]
fn ascii_stl_keeps_every_solid_and_shares_repeated_names() {
    let mut file = NamedTempFile::new().unwrap();
    write!(
        file,
        "solid left wall\n{}endsolid left wall\n\nsolid back\n{}endsolid back\nsolid left wall\n{}endsolid left wall\n",
        facet_at("0"),
        facet_at("1"),
        facet_at("2")
    )
    .unwrap();
    let surface = load_stl(file.path()).unwrap();
    assert_eq!(surface.triangle_count(), 3);
    assert_eq!(surface.patches().names(), &["left wall", "back"]);
}
#[test]
fn ascii_stl_without_a_solid_name_uses_the_default_patch() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "solid\n{}endsolid", facet_at("0")).unwrap();
    let surface = load_stl(file.path()).unwrap();
    assert_eq!(surface.patches().names(), &["_default"]);
}
#[test]
fn binary_stl_uses_the_default_patch() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&[0; 80]).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    for value in [
        0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ] {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.write_all(&0u16.to_le_bytes()).unwrap();
    let surface = load_stl(file.path()).unwrap();
    assert_eq!(surface.triangle_count(), 1);
    assert_eq!(surface.patches().names(), &["_default"]);
}
#[test]
fn malformed_ascii_stl_and_empty_obj_are_rejected() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "solid test\nnot-a-facet\nendsolid test").unwrap();
    assert!(load_stl(file.path()).is_err());
    let empty = NamedTempFile::new().unwrap();
    assert!(load_obj(empty.path()).is_err());
}
