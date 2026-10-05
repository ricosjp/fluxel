"""Integration tests for PyO3 bindings (requires built native extension).

Run after `maturin develop` or installing the package with the extension.
"""

from __future__ import annotations

import fluxel


def test_fluxel_import_and_bounding_box() -> None:
    b = fluxel.BoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    assert b.min == (-1.0, -1.0, -1.0)
    assert b.max == (1.0, 1.0, 1.0)


def test_cylindrical_matches_bounding_box_construction() -> None:
    domain = fluxel.Cylindrical(
        (0.0, 0.0, 0.0), 1.0, 2.0, 0.0, 6.283185307179586, 0.0, 1.0
    )
    assert domain.origin == (0.0, 0.0, 0.0)
    assert domain.r_min == 1.0
    assert domain.theta_extent == 6.283185307179586
    fluxel.FluxelManager(domain, (2, 4, 1), 0)


def test_manager_accepts_periodic_axes() -> None:
    box = fluxel.BoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    fluxel.FluxelManager(box, (2, 2, 1), 0, periodic=(True, False, True))
    cylinder = fluxel.Cylindrical(
        (0.0, 0.0, 0.0), 1.0, 2.0, 0.0, 1.0, 0.0, 1.0
    )
    fluxel.FluxelManager(
        cylinder, (1, 4, 1), 0, periodic=(False, True, False)
    )


def test_fluxel_module_exports_core_classes() -> None:
    for name in (
        "BoundingBox",
        "Cylindrical",
        "CfdGhostCellMesh",
        "CfdAxisProjectedMesh",
        "ApIbmFaceData",
        "ApibmSession",
        "FluxelManager",
        "Bool3",
        "Float3",
        "Forest",
        "Int3",
        "quaternion_from_axis_angle",
    ):
        assert hasattr(fluxel, name), f"missing {name}"
