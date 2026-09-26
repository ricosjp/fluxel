"""Enforce crate boundaries independently of individual numerical algorithms."""

import tomllib
from pathlib import Path


def test_crate_dependencies_follow_the_layer_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]
    allowed = {
        "fluxel-sfc": set(),
        "fluxel-core": {"fluxel-sfc"},
        "fluxel-geometry": {"fluxel-core", "fluxel-sfc"},
        "fluxel-mesh": {"fluxel-core", "fluxel-geometry"},
        "fluxel-ibm": {"fluxel-core", "fluxel-geometry", "fluxel-mesh"},
        "fluxel-engine": {
            "fluxel-core",
            "fluxel-sfc",
            "fluxel-geometry",
            "fluxel-mesh",
            "fluxel-ibm",
        },
        "fluxel-pyo3": {
            "fluxel-engine",
            "fluxel-geometry",
            "fluxel-mesh",
            "fluxel-ibm",
        },
    }
    names = set()
    for member in workspace["members"]:
        manifest = tomllib.loads((root / member / "Cargo.toml").read_text())
        name = manifest["package"]["name"]
        names.add(name)
        dependencies = manifest.get("dependencies", {})
        internal = {dep for dep in dependencies if dep.startswith("fluxel-")}
        assert internal <= allowed[name]
        if name != "fluxel-pyo3":
            assert not {"pyo3", "numpy"} & dependencies.keys()
        if name != "fluxel-engine":
            assert not {"stl_io", "wavefront_obj"} & dependencies.keys()
    assert names == allowed.keys()
    assert not (root / "crates/fluxel-export/Cargo.toml").exists()
