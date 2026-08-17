"""Integration tests for APIBM moving-boundary session APIs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fluxel import BoundingBox, FluxelManager


@pytest.fixture
def manager() -> FluxelManager:
    bbox = BoundingBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    return FluxelManager(bbox, base_res=[4, 4, 4], n_leaf_refinement=0)


def test_create_session_and_update_ib_keeps_topology(
    manager: FluxelManager,
) -> None:
    session = manager.create_axis_projected_session(None, target_level=0)
    mesh0 = session.mesh
    n_faces = len(mesh0.internal_faces_owner)
    n_cells = mesh0.n_cells

    mesh1 = session.update_ib(
        translation=[0.01, 0.0, 0.0],
        warn_outside_refinement=False,
    )

    assert mesh1.n_cells == n_cells
    assert len(mesh1.internal_faces_owner) == n_faces
    assert len(mesh1.ap.is_immersed_face) == n_faces
    assert list(session.translation) == pytest.approx([0.01, 0.0, 0.0])


def test_update_ib_warn_outside_refinement_can_be_silenced(
    manager: FluxelManager,
) -> None:
    session = manager.create_axis_projected_session(None, target_level=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.update_ib(
            translation=[0.0, 0.0, 0.0],
            warn_outside_refinement=False,
        )
    assert caught == []


def test_ap_payload_is_nested(manager: FluxelManager) -> None:
    mesh = manager.build_axis_projected_mesh(None, target_level=0)
    assert hasattr(mesh, "ap")
    assert isinstance(mesh.ap.is_immersed_face, np.ndarray)
    assert len(mesh.ap.is_immersed_face) == len(mesh.internal_faces_owner)
