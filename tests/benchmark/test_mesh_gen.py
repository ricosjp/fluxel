"""Benchmarks for common geometry routines."""

import logging
import pathlib
from collections.abc import Callable, Generator

import pytest
from pydantic import BaseModel

import fluxel

logger = logging.getLogger(__name__)

DRIVAER_CONFIG = {
    "input_stl": "data/stl/DrivAer.stl",
    "target_level": 3,
    "n_leaf_refinement": 3,
    "base_resolution": [8, 4, 2],
    "bounding_box": {
        "min": [-5.0, -4.0, 0.0],
        "max": [11.0, 4.0, 4.0],
    },
    "fluid_seed_point": [-4.0, 0.0, 2.0],
}


class BoundingBoxConfig(BaseModel, frozen=True):
    """Axis-aligned domain bounding box (min / max corners)."""

    min: list[float, float, float]
    max: list[float, float, float]


class DrivAerConfig(BaseModel, frozen=True):
    """DrivAer configuration."""

    input_stl: pathlib.Path
    target_level: int
    n_leaf_refinement: int
    base_resolution: list[int, int, int]
    bounding_box: BoundingBoxConfig
    fluid_seed_point: list[float, float, float]


def benchmark_with_group(func: Callable) -> Callable:
    """Attach a pytest-benchmark group matching the function name."""
    return pytest.mark.benchmark(group=func.__name__)(func)


@pytest.mark.benchmark
@benchmark_with_group
def test_gcibm_mesh_generation(benchmark: Generator):
    """Benchmark GCIBM mesh generation."""
    cfg = DrivAerConfig.model_validate(DRIVAER_CONFIG)
    manager = fluxel.FluxelManager(
        bbox=fluxel.BoundingBox(
            min=cfg.bounding_box.min, max=cfg.bounding_box.max
        ),
        base_res=cfg.base_resolution,
        n_leaf_refinement=cfg.n_leaf_refinement,
    )

    def build_ghost_cell_mesh():
        _ = manager.build_ghost_cell_mesh(
            str(cfg.input_stl), cfg.target_level, cfg.fluid_seed_point
        )

    benchmark(build_ghost_cell_mesh)


@pytest.mark.benchmark
@benchmark_with_group
def test_apibm_mesh_generation(benchmark: Generator):
    """Benchmark APIBM mesh generation."""
    cfg = DrivAerConfig.model_validate(DRIVAER_CONFIG)
    manager = fluxel.FluxelManager(
        bbox=fluxel.BoundingBox(
            min=cfg.bounding_box.min, max=cfg.bounding_box.max
        ),
        base_res=cfg.base_resolution,
        n_leaf_refinement=cfg.n_leaf_refinement,
    )

    def build_axis_projected_mesh():
        _ = manager.build_axis_projected_mesh(
            str(cfg.input_stl), cfg.target_level
        )

    benchmark(build_axis_projected_mesh)
