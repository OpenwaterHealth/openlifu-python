from __future__ import annotations

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.geo import Point


@pytest.fixture()
def example_point() -> Point:
    return Point(
        id = "example_point",
        name="Example point",
        color=(0.,0.7, 0.2),
        radius=1.5,
        position=np.array([-10.,0,25]),
        dims = ("R", "A", "S"),
        units = "m",
    )

@pytest.mark.parametrize("compact_representation", [True, False])
@pytest.mark.parametrize("default_point", [True, False])
def test_serialize_deserialize_point(example_point : Point, compact_representation: bool, default_point: bool):
    """Verify that turning a point into json and then re-constructing it gets back to the original point"""
    point = Point() if default_point else example_point
    reconstructed_point = point.from_json(point.to_json(compact_representation))
    assert dataclasses_are_equal(point, reconstructed_point)
