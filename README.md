# Fluxel

An extremely fast hierarchical grid mesh generator for CFD solvers. The core is implemented in Rust (adaptive mesh refinement on a space-filling curve, IBM geometry, export), with Python bindings built via [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/).

## Workspace layout

| Crate | Role |
|-------|------|
| `fluxel-core` | AMR forest and algorithms over SFC-ordered keys |
| `fluxel-sfc` | Space-filling curve utilities |
| `fluxel-geometry` | Geometry helpers |
| `fluxel-ibm` | Immersed boundary meshing |
| `fluxel-export` | Export utilities |
| `fluxel-pyo3` | Python bindings (`fluxel._fluxel`) |

## License

[Apache License 2.0](./LICENSE).
