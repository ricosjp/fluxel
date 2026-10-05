# Fluxel

A hierarchical grid mesh generator for CFD solvers, with a Rust AMR and immersed-boundary engine and Python bindings built using PyO3 and Maturin.

## Workspace layout

| Crate | Responsibility |
|-------|----------------|
| `fluxel-sfc` | Morton keys and logical cells |
| `fluxel-core` | Forest, neighbours, refinement, coarsening, validated 2:1 balance |
| `fluxel-geometry` | Physical bounds, Cartesian mapping, point location, rigid poses |
| `fluxel-mesh` | Immutable background topology and cell geometry, shared across methods |
| `fluxel-ibm` | Validated boundary surfaces, intersection classification, APIBM and GCIBM payloads |
| `fluxel-engine` | Build policy, file loading, manual AMR, reports, transactional sessions |
| `fluxel-pyo3` | Existing Python API, error/warning translation, NumPy publication |

The engine has no Python dependencies. STL/OBJ loading is enabled by its `mesh-io` feature; in-memory builds work without that feature. `fluxel-export` has been removed: topology is built once in `fluxel-mesh`, method-specific data belongs to `fluxel-ibm`, and the engine composes the results.

Python names, arguments, array layouts and default writable snapshots are preserved. Rust APIs have been redesigned without compatibility wrappers.

## Cell limits in YAML

The APIBM and GCIBM export scripts accept an optional top-level YAML setting:

```yaml
max_cells: 1000000
```

It must be a positive integer; omitting it or using `null` preserves the default
without a user-specified cell limit. The scripts pass it to
`FluxelManager(..., max_cells=...)`. The limit applies to initial cells, surface
and region refinement, 2:1 balancing, and final uniform refinement, including
session remeshing. An operation that would exceed the limit raises `ValueError`;
it does not automatically reduce refinement to fit. This limits cell count,
not total memory usage.

## Moving boundaries and array ownership

```python
from fluxel import BoundingBox, FluxelManager

manager = FluxelManager(BoundingBox([0, 0, 0], [1, 1, 1]), [4, 4, 4], 0)
session = manager.create_axis_projected_session(None, target_level=0)

writable = session.mesh  # independent writable arrays
before = session.snapshot()  # read-only shared snapshot
after = session.update_ib([0.1, 0, 0], copy=False)
rebuilt = session.remesh(target_level=1, copy=False)
```

`update_ib` shares the Rust background grid and rebuilds only intersection and IBM data. `remesh` creates a new grid; cell and face indices must not be reused across remeshing. Existing snapshots remain valid after either operation or destruction of the session. Pose arguments are absolute, quaternion order is `[w, x, y, z]`, and omitted pose components retain their values.

The default `mesh`, `update_ib` and `remesh` results retain independent writable arrays. Opting into `snapshot()` or `copy=False` uses immutable NumPy storage cached by background ID and boundary revision. Initial publication copies to immutable byte buffers; repeated reads and fixed-grid updates reuse the background buffers. This is safe shared publication, not an end-to-end zero-copy claim. Fresh ndarray headers and patch dictionaries isolate user changes to array metadata from the cache.

Updates are prepared without changing current state. Python conversion and warnings complete before commit, so invalid inputs, failed remeshing and warnings promoted to exceptions leave the session unchanged. Concurrent or reentrant mutation of one Python session is rejected.

## Rust use

```rust
use fluxel_engine::{build_apibm, BuildLimits, MeshBuildConfig, RefinementPlan};
use fluxel_geometry::BoundingBox;
use fluxel_ibm::Boundary;

let config = MeshBuildConfig::new(
    BoundingBox::new([0.0; 3], [1.0; 3])?,
    [4; 3],
    0,
    BuildLimits { max_cells: 1_000_000 },
)?;
let result = build_apibm(&config, Boundary::None, RefinementPlan::new(0, vec![])?)?;
let centers = result.mesh.background().geometry().centers();
```

`Boundary::None` is explicit and creates no synthetic surface. `BoundarySurface::new` accepts in-memory vertices, triangles and patch assignments; `fluxel_engine::io::load_boundary` loads STL/OBJ. Results contain immutable background and payload data plus a report with refinement and GCIBM fallback diagnostics. `ApibmSession::prepare_update` / `prepare_remesh` and `commit` expose the same transactional lifecycle to Rust callers.

Geometry currently supports Cartesian grids. GCIBM retains the existing weighted linear least-squares interpolation and inverse-distance fallback; fallback and unresolved-stencil counts are reported. Fluxel supplies geometry and constraint candidates, leaving boundary conditions and solution transfer after remeshing to the solver.

## Development

```sh
cargo test --workspace
cargo test -p fluxel-engine --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
uv run maturin develop --release
uv run pytest tests
```

Build the native extension before running Python tests. Tests cover numerical cases, shared topology, stale candidate rejection, failure rollback, array ownership and crate dependency rules. Large geometry benchmarks remain opt-in with `make benchmark`.

## License

[Apache License 2.0](./LICENSE).
