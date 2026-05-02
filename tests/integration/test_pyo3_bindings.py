"""Integration tests for PyO3 bindings (requires built native extension).

Run after `maturin develop` or installing the package with the extension.
"""

from __future__ import annotations

import fluxel


def test_fluxel_import_and_bounding_box() -> None:
    b = fluxel.BoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    assert b.min[0] == -1.0
    assert b.max[2] == 1.0


def test_fluxel_module_exports_core_classes() -> None:
    for name in (
        "BoundingBox",
        "CfdGhostCellMesh",
        "CfdAxisProjectedMesh",
        "FluxelManager",
        "Forest",
    ):
        assert hasattr(fluxel, name), f"missing {name}"
