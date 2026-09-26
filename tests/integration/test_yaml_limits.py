"""Cell limits survive the YAML -> Python -> Rust configuration path."""

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from fluxel import BoundingBox, FluxelManager


@pytest.fixture(params=["apibm", "gcibm"])
def exporter(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    return importlib.import_module(f"examples.export_{request.param}")


def write_config(tmp_path: Path, limit: str = "") -> Path:
    surface = tmp_path / "wall.obj"
    surface.write_text(
        "v 0.6 -1 -1\nv 0.6 2 -1\nv 0.6 2 2\nv 0.6 -1 2\nf 1 2 3\nf 1 3 4\n"
    )
    path = tmp_path / "mesh.yaml"
    path.write_text(
        f"input_mesh: {surface}\n"
        f"output_vtu: {tmp_path / 'mesh.vtu'}\n"
        "bounding_box: {min: [0, 0, 0], max: [1, 1, 1]}\n"
        "base_resolution: [1, 1, 1]\n"
        "target_level: 0\n"
        "n_leaf_refinement: 1\n"
        "fluid_seed_point: [0.1, 0.1, 0.1]\n"
        f"{limit}"
    )
    return path


@pytest.mark.parametrize("limit", ["", "max_cells: null\n", "max_cells: 8\n"])
def test_yaml_limit_defaults_and_exact_bound(
    exporter: ModuleType, tmp_path: Path, limit: str
) -> None:
    method = exporter.__name__.rsplit("_", 1)[1]
    cfg = getattr(exporter, f"load_{method}_config")(
        write_config(tmp_path, limit)
    )
    assert cfg.max_cells == (8 if "8" in limit else None)
    manager = FluxelManager(
        BoundingBox(cfg.bounding_box.min, cfg.bounding_box.max),
        cfg.base_resolution,
        cfg.n_leaf_refinement,
        max_cells=cfg.max_cells,
    )
    if method == "apibm":
        mesh = manager.build_axis_projected_mesh(str(cfg.input_mesh), 0)
    else:
        mesh = manager.build_ghost_cell_mesh(
            str(cfg.input_mesh), 0, cfg.fluid_seed_point
        )
    assert mesh.n_cells == 8


@pytest.mark.parametrize("limit", ["0", "-1", "1.5", "true", '"8"'])
def test_yaml_rejects_invalid_cell_limits(
    exporter: ModuleType, tmp_path: Path, limit: str
) -> None:
    method = exporter.__name__.rsplit("_", 1)[1]
    with pytest.raises(ValidationError, match="max_cells"):
        getattr(exporter, f"load_{method}_config")(
            write_config(tmp_path, f"max_cells: {limit}\n")
        )


def test_export_script_passes_limit_to_engine(
    exporter: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_config(tmp_path, "max_cells: 7\n")
    monkeypatch.setattr(
        sys, "argv", [exporter.__file__, "--config", str(config)]
    )
    with pytest.raises(ValueError, match="mesh exceeds cell limit 7"):
        runpy.run_path(exporter.__file__, run_name="__main__")
    assert not (tmp_path / "mesh.vtu").exists()


def test_cell_limit_checks_roots_and_survives_session_remeshing() -> None:
    bbox = BoundingBox([0, 0, 0], [1, 1, 1])
    with pytest.raises(ValueError, match="cell limit 7"):
        FluxelManager(bbox, [2, 2, 2], 0, max_cells=7)
    session = FluxelManager(
        bbox, [1, 1, 1], 0, max_cells=8
    ).create_axis_projected_session(None, 0)
    regions = [([0, 0, 0], [1, 1, 1], 2)]
    with pytest.raises(ValueError, match="cell limit 8"):
        session.remesh(refinement_regions=regions)
    assert session.snapshot().n_cells == 1
    assert session.remesh().n_cells == 1
    regions = [([0, 0, 0], [1, 1, 1], 1)]
    assert session.remesh(refinement_regions=regions).n_cells == 8
