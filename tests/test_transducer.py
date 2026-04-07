from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.xdc import Element, Transducer, TransducerArray
from openlifu.xdc.transducerarray import (
    get_angle_from_gap,
    get_gap_from_angle,
    get_roc_from_angle,
)


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer.from_file(Path(__file__).parent/'resources/example_db/transducers/example_transducer/example_transducer.json')

def load_transducer_array(transducer_array_id : str) -> TransducerArray:
    """Load an example TransducerArray given the transducer ID."""
    return TransducerArray.from_file(Path(__file__).parent/f'resources/example_db/transducers/{transducer_array_id}/{transducer_array_id}.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_transducer(example_transducer : Transducer, compact_representation: bool):
    reconstructed_transducer = example_transducer.from_json(example_transducer.to_json(compact_representation))
    dataclasses_are_equal(example_transducer, reconstructed_transducer)

def test_get_polydata_color_options(example_transducer : Transducer):
    """Ensure that the color is set correctly on the polydata"""
    polydata_with_default_color = example_transducer.get_polydata()
    point_scalars = polydata_with_default_color.GetPointData().GetScalars()
    assert point_scalars is None

    polydata_with_given_color = example_transducer.get_polydata(facecolor=[0,1,1,0.5])
    point_scalars = polydata_with_given_color.GetPointData().GetScalars()
    assert point_scalars is not None

def test_default_transducer():
    """Ensure it is possible to construct a default transducer"""
    Transducer()

def test_convert_transform():
    transducer = Transducer(units='cm')
    transform = transducer.convert_transform(
        matrix = np.array([
            [1,0,0,2],
            [0,1,0,3],
            [0,0,1,4],
            [0,0,0,1],
        ], dtype=float),
        units = "m",
    )
    expected_transform = np.array([
        [1,0,0,200],
        [0,1,0,300],
        [0,0,1,400],
        [0,0,0,1],
    ], dtype=float)
    assert np.allclose(transform,expected_transform)

def test_get_effective_origin():
    transducer = Transducer.gen_matrix_array(nx=3, ny=2, units='cm')
    effective_origin_with_all_active = transducer.get_effective_origin(apodizations = np.ones(transducer.numelements()))
    assert np.allclose(effective_origin_with_all_active, np.zeros(3))

    rng = np.random.default_rng()
    element_index_to_turn_on = rng.integers(transducer.numelements())
    apodizations_with_just_one_element = np.zeros(transducer.numelements())
    apodizations_with_just_one_element[element_index_to_turn_on] = 0.5 # It is allowed to be a number between 0 and 1
    assert np.allclose(
        transducer.get_effective_origin(apodizations = apodizations_with_just_one_element, units = "um"),
        transducer.get_positions(units="um")[element_index_to_turn_on],
    )

def test_get_standoff_transform_in_units():
    standoff_transform_in_mm = np.array([
            [-0.1,0.9,0,20],
            [0.9,0.1,0,30],
            [0,0,1,40],
            [0,0,0,1],
    ])
    standoff_transform_in_cm = np.array([
            [-0.1,0.9,0,2],
            [0.9,0.1,0,3],
            [0,0,1,4],
            [0,0,0,1],
    ])
    transducer = Transducer(units='mm')
    transducer.standoff_transform = standoff_transform_in_mm
    assert np.allclose(
        transducer.get_standoff_transform_in_units("cm"),
        standoff_transform_in_cm,
    )

def test_read_data_types(example_transducer:Transducer):
    assert isinstance(example_transducer.standoff_transform, np.ndarray)
    if len(example_transducer.elements) > 0:
        assert isinstance(example_transducer.elements[0], Element)

@pytest.mark.parametrize(
    "transducer_array_id",
    [
        "example_transducer_array",
        "example_transducer_array2",
    ]
)
def test_transducer_array_to_transducer_data_types(transducer_array_id):
    transducer_array : TransducerArray = load_transducer_array(transducer_array_id)
    transducer = transducer_array.to_transducer()
    assert isinstance(transducer.standoff_transform, np.ndarray)
    assert not hasattr(transducer, "impulse_response")
    assert not hasattr(transducer, "impulse_dt")
    if len(transducer.elements) > 0:
        assert isinstance(transducer.elements[0], Element)


def test_transducer_calc_output_interpolates_dictionary_sensitivity():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [1.0, 3.0]},
    )
    transducer.elements[0].sensitivity = 1.0
    cycles = 3
    dt = 1e-7

    output_mid = transducer.calc_output(cycles=cycles, frequency=200e3, dt=dt)
    output_low = transducer.calc_output(cycles=cycles, frequency=100e3, dt=dt)

    n_samples_mid = int(np.round(cycles / (200e3 * dt)))
    n_samples_low = int(np.round(cycles / (100e3 * dt)))
    t_mid = np.arange(n_samples_mid) * dt
    t_low = np.arange(n_samples_low) * dt
    expected_mid = 2.0 * np.sin(2 * np.pi * 200e3 * t_mid)
    expected_low = 1.0 * np.sin(2 * np.pi * 100e3 * t_low)

    np.testing.assert_allclose(output_mid[0], expected_mid)
    np.testing.assert_allclose(output_low[0], expected_low)


def test_legacy_sensitivity_mapping_is_normalized_to_schema():
    transducer = Transducer(
        sensitivity={"100000.0": 1.0, "300000.0": 3.0},
    )
    assert transducer.sensitivity == {
        "freq_Hz": [100000.0, 300000.0],
        "values_Pa_per_V": [1.0, 3.0],
    }


def test_element_calc_output_generates_signal_from_scalar_input():
    element = Element(sensitivity=2.0)
    cycles = 4
    frequency = 100e3
    dt = 1e-7
    n_samples = int(np.round(cycles / (frequency * dt)))

    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt, amplitude=3.0)
    t = np.arange(n_samples) * dt
    expected = 2.0 * 3.0 * np.sin(2 * np.pi * frequency * t)

    np.testing.assert_allclose(output, expected)


def test_element_calc_output_enforces_cycles_duration_for_generated_signal():
    element = Element(sensitivity=1.0)
    cycles = 1
    frequency = 200e3
    dt = 1e-6
    n_samples = int(np.round(cycles / (frequency * dt)))
    output = element.calc_output(cycles=cycles, frequency=frequency, dt=dt)
    t = np.arange(n_samples) * dt
    expected = np.sin(2 * np.pi * frequency * t)

    assert len(output) == n_samples
    np.testing.assert_allclose(output, expected)


def test_merge_pushes_transducer_sensitivity_into_elements():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [2.0, 4.0]},
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [3.0, 6.0]},
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    merged = Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)

    assert merged.sensitivity == 1.0
    assert merged.elements[0].sensitivity == {"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [10.0, 20.0]}
    assert merged.elements[1].sensitivity == {"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [21.0, 42.0]}


def test_merge_rejects_mismatched_sensitivity_keys():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [2.0, 4.0]},
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 400e3], "values_Pa_per_V": [3.0, 6.0]},
    )
    transducer_a.elements[0].sensitivity = {"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [5.0, 7.0]}
    transducer_b.elements[0].sensitivity = {"freq_Hz": [100e3, 400e3], "values_Pa_per_V": [11.0, 13.0]}

    with pytest.raises(ValueError, match="different frequency keys"):
        Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)


@pytest.mark.parametrize(
    ("width", "dth", "roc"),
    [
        (8.0, 0.08, 25.0),
        (10.0, 0.12, 30.0),
        (12.0, 0.18, 45.0),
    ],
)
def test_concave_geometry_helpers_are_mutual_inverses(width: float, dth: float, roc: float):
    gap = get_gap_from_angle(width, dth, roc)
    recovered_roc = get_roc_from_angle(width, gap, dth)
    recovered_dth = get_angle_from_gap(width, gap, roc)
    recovered_gap = get_gap_from_angle(width, recovered_dth, roc)

    assert np.isclose(recovered_roc, roc)
    assert np.isclose(recovered_dth, dth)
    assert np.isclose(recovered_gap, gap)


def test_get_concave_cylinder_computes_gap_from_dth_and_roc_layout_spacing():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 8.0
    dth = 0.12
    roc = 25.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=2,
        cols=1,
        width=width,
        dth=dth,
        roc=roc,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    expected_gap = get_gap_from_angle(width, dth, roc)
    y_spacing = np.abs(positions[1, 1] - positions[0, 1])

    assert np.isclose(y_spacing, width + expected_gap)


def test_get_concave_cylinder_handles_zero_dth_without_roc():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    width = 10.0
    gap = 2.0
    array = TransducerArray.get_concave_cylinder(
        base,
        rows=1,
        cols=2,
        width=width,
        gap=gap,
        dth=0.0,
        units="mm",
    )
    merged = array.to_transducer()
    positions = merged.get_positions(units="mm")

    x_spacing = np.abs(positions[1, 0] - positions[0, 0])
    z_values = positions[:, 2]

    assert np.isclose(x_spacing, width + gap)
    np.testing.assert_allclose(z_values, np.zeros_like(z_values))


def test_get_concave_cylinder_rejects_gap_dth_roc_together():
    base = Transducer.gen_matrix_array(nx=1, ny=1, units="mm")
    with pytest.raises(ValueError, match="cannot specify all of gap, dth, and roc"):
        TransducerArray.get_concave_cylinder(
            base,
            rows=1,
            cols=2,
            width=10.0,
            gap=1.0,
            dth=0.2,
            roc=20.0,
            units="mm",
        )


def test_transducer_calc_output_combines_frequency_dependent_sensitivities():
    transducer = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [2.0, 4.0]},
    )
    transducer.elements[0].sensitivity = {"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [5.0, 9.0]}

    frequency = 200e3
    dt = 1e-7
    cycles = 3
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = transducer.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 21.0 * expected_drive)


def test_transducer_array_to_transducer_preserves_frequency_dependent_sensitivities():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [2.0, 4.0]},
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={"freq_Hz": [100e3, 300e3], "values_Pa_per_V": [1.0, 3.0]},
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    array = TransducerArray.get_concave_cylinder(
        [transducer_a, transducer_b],
        rows=1,
        cols=2,
        width=10.0,
        gap=0.0,
        units="mm",
    )
    merged = array.to_transducer()

    frequency = 200e3
    dt = 1e-7
    cycles = 2
    n_samples = int(np.round(cycles / (frequency * dt)))
    t = np.arange(n_samples) * dt
    expected_drive = np.sin(2 * np.pi * frequency * t)

    output = merged.calc_output(cycles=cycles, frequency=frequency, dt=dt)

    np.testing.assert_allclose(output[0], 15.0 * expected_drive)
    np.testing.assert_allclose(output[1], 14.0 * expected_drive)
