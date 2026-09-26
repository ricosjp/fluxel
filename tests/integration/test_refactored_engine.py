"""Public contracts for the Rust engine and immutable snapshot publication."""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pytest

from fluxel import (
    BoundingBox,
    CfdAxisProjectedMesh,
    FluxelManager,
    Forest,
    quaternion_from_axis_angle,
)


@pytest.fixture
def plane(tmp_path: Path) -> str:
    path = tmp_path / "wall.obj"
    path.write_text(
        "o wall\nv 0.6 -1 -1\nv 0.6 2 -1\nv 0.6 2 2\nv 0.6 -1 2\n"
        "g front\nf 1 2 3\ng back\nf 1 3 4\n"
    )
    return str(path)


@pytest.fixture
def manager() -> FluxelManager:
    return FluxelManager(BoundingBox([0, 0, 0], [1, 1, 1]), [2, 2, 2], 0)


def assert_same_background(
    a: CfdAxisProjectedMesh, b: CfdAxisProjectedMesh
) -> None:
    for name in (
        "cell_centers",
        "cell_sizes",
        "internal_faces_owner",
        "internal_faces_neighbour",
        "internal_faces_axis",
        "domain_bnd_faces_owner",
        "domain_bnd_faces_dir",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_ap_gc_topology_and_array_contract(
    manager: FluxelManager, plane: str
) -> None:
    ap = manager.build_axis_projected_mesh(plane, 1)
    gc_mesh = manager.build_ghost_cell_mesh(plane, 1, [0.1, 0.1, 0.1])
    assert_same_background(ap, gc_mesh)
    assert ap.cell_centers.shape == (ap.n_cells, 3)
    assert ap.cell_centers.dtype == np.dtype("float64")
    assert ap.internal_faces_owner.dtype == np.dtype(np.uintp)
    assert ap.internal_faces_axis.dtype == np.dtype("uint8")
    assert ap.ap.is_immersed_face.dtype == np.dtype("bool")
    assert len(ap.ap.dist_owner_to_bnd) == np.count_nonzero(
        ap.ap.is_immersed_face
    )
    assert ap.patch_name_to_id == {"wall_front": 0, "wall_back": 1}
    assert gc_mesh.gc_interp_stencil_indices.shape[1] == 8
    assert gc_mesh.gc_interp_stencil_weights.shape[1] == 8


def test_none_has_no_boundary_under_any_pose(manager: FluxelManager) -> None:
    session = manager.create_axis_projected_session(None, 1)
    mesh = session.update_ib([-1e10, -1e10, -1e10])
    assert not mesh.ap.is_immersed_face.any()
    assert mesh.patch_name_to_id == {"empty": 0}
    gc_mesh = manager.build_ghost_cell_mesh(None, 1, [0.1, 0.1, 0.1])
    assert gc_mesh.gc_is_fluid.all()
    assert gc_mesh.gc_interp_stencil_indices.shape == (0, 8)
    assert gc_mesh.gc_bnd_intercepts.shape == (0, 3)


def test_shared_snapshots_are_immutable_and_keep_their_storage(
    manager: FluxelManager, plane: str
) -> None:
    session = manager.create_axis_projected_session(plane, 1)
    before = session.snapshot()
    saved_centers = before.cell_centers.copy()
    saved_payload = before.ap.dist_owner_to_bnd.copy()
    after = session.update_ib([10, 0, 0], copy=False)
    assert np.shares_memory(before.cell_centers, after.cell_centers)
    assert len(after.ap.dist_owner_to_bnd) == 0
    assert len(saved_payload) > 0
    for array in (
        before.cell_centers,
        before.internal_faces_owner,
        before.ap.dist_owner_to_bnd,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
        with pytest.raises(ValueError):
            array.flat[0] = 0
        base = array.base
        while isinstance(base, np.ndarray):
            with pytest.raises(ValueError):
                base.flags.writeable = True
            base = base.base
        assert isinstance(base, bytes)
    rebuilt = session.remesh(copy=False)
    assert not np.shares_memory(before.cell_centers, rebuilt.cell_centers)
    del session
    gc.collect()
    np.testing.assert_array_equal(before.cell_centers, saved_centers)
    np.testing.assert_array_equal(before.ap.dist_owner_to_bnd, saved_payload)


def test_snapshot_headers_and_patch_dicts_do_not_mutate_the_cache(
    manager: FluxelManager, plane: str
) -> None:
    session = manager.create_axis_projected_session(plane, 1)
    first = session.snapshot()
    shape = first.cell_centers.shape
    first.cell_centers.shape = (first.cell_centers.size,)
    first.ap.is_immersed_face.shape = (1, -1)
    first.patch_name_to_id.clear()
    second = session.snapshot()
    assert second.cell_centers.shape == shape
    assert second.ap.is_immersed_face.ndim == 1
    assert second.patch_name_to_id == {"wall_front": 0, "wall_back": 1}


def test_legacy_snapshots_stay_writable_and_independent(
    manager: FluxelManager, plane: str
) -> None:
    session = manager.create_axis_projected_session(plane, 1)
    before = session.mesh
    original = before.cell_centers.copy()
    before.cell_centers[:] = -999
    before.ap.dist_owner_to_bnd[:] = -999
    after = session.update_ib(warn_outside_refinement=False)
    assert after.cell_centers.flags.writeable
    assert not np.shares_memory(before.cell_centers, after.cell_centers)
    np.testing.assert_array_equal(after.cell_centers, original)
    assert (after.ap.dist_owner_to_bnd >= 0).all()


@pytest.mark.parametrize("copy", [True, False])
def test_warning_as_error_rolls_back_pose_mesh_and_settings(
    manager: FluxelManager, plane: str, copy: bool
) -> None:
    session = manager.create_axis_projected_session(plane, 1)
    before = session.mesh
    # Initial refinement is at x >= 0.5; move the wall to an unrefined root.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            session.update_ib([-0.4, 0, 0], copy=copy)
    assert list(session.translation) == [0, 0, 0]
    assert_same_background(before, session.mesh)
    np.testing.assert_array_equal(
        before.ap.dist_owner_to_bnd, session.mesh.ap.dist_owner_to_bnd
    )
    with pytest.raises(ValueError):
        session.remesh(target_level=33, translation=[0.2, 0, 0], copy=copy)
    assert list(session.translation) == [0, 0, 0]
    assert_same_background(before, session.mesh)
    session.remesh(copy=copy)
    assert_same_background(before, session.mesh)


def test_warning_can_be_silenced_and_pose_is_absolute(
    manager: FluxelManager, plane: str
) -> None:
    session = manager.create_axis_projected_session(plane, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.update_ib([-0.4, 0, 0], warn_outside_refinement=False)
    assert not caught
    session.update_ib([-0.4, 0, 0], warn_outside_refinement=False)
    assert list(session.translation) == pytest.approx([-0.4, 0, 0])
    session.update_ib(
        rotation_quaternion=[0, 0, 0, 0], warn_outside_refinement=False
    )
    assert list(session.translation) == pytest.approx([-0.4, 0, 0])
    assert list(session.rotation_quaternion) == [1, 0, 0, 0]


def test_regions_none_keeps_and_empty_clears(manager: FluxelManager) -> None:
    session = manager.create_axis_projected_session(
        None, 0, [([0, 0, 0], [0.2, 0.2, 0.2], 1)]
    )
    count = session.mesh.n_cells
    assert count > 8
    assert session.remesh(refinement_regions=None).n_cells == count
    assert session.remesh(refinement_regions=[]).n_cells == 8


@pytest.mark.parametrize(
    "base", [[0, 1, 1], [2**26 + 1, 1, 1], [2**32 - 1] * 3]
)
def test_invalid_resolution_raises_value_error(base: list[int]) -> None:
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])
    with pytest.raises(ValueError):
        FluxelManager(bbox, base)
    with pytest.raises(ValueError):
        Forest(bbox, base)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_inputs_are_rejected_without_mutating_session(
    manager: FluxelManager, value: float
) -> None:
    with pytest.raises(ValueError):
        BoundingBox([0, 0, 0], [value, 1, 1])
    session = manager.create_axis_projected_session(None, 0)
    with pytest.raises(ValueError):
        session.update_ib([value, 0, 0])
    with pytest.raises(ValueError):
        session.update_ib(rotation_quaternion=[value, 0, 0, 0])
    assert list(session.translation) == [0, 0, 0]
    with pytest.raises(ValueError):
        manager.build_ghost_cell_mesh(None, 0, [value, 0, 0])
    with pytest.raises(ValueError):
        quaternion_from_axis_angle([value, 0, 0], 1)


def test_manual_region_uses_the_automatic_policy() -> None:
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])
    forest = Forest(bbox, [2, 1, 1])
    forest.refine_by_bbox([0, 0, 0], [0.5, 1, 1], 1)
    mesh = FluxelManager(bbox, [2, 1, 1], 0).build_axis_projected_mesh(
        None, 0, [([0, 0, 0], [0.5, 1, 1], 1)]
    )
    assert forest.num_cells() == mesh.n_cells == 9
    with pytest.raises(ValueError):
        forest.refine_by_flags([True])
    with pytest.raises(ValueError):
        forest.uniform_refinement(33)
    assert forest.num_cells() == 9
