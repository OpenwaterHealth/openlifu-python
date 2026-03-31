from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu.xdc import Element, Transducer, TransducerArray


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
        sensitivity={100e3: 1.0, 300e3: 3.0},
    )
    transducer.elements[0].sensitivity = 1.0
    input_signal = np.array([1.0, -1.0, 0.5], dtype=float)

    output_mid = transducer.calc_output(input_signal, cycles=3, frequency=200e3, dt=1e-7)
    output_low = transducer.calc_output(input_signal, cycles=3, frequency=100e3, dt=1e-7)

    np.testing.assert_allclose(output_mid[0, :len(input_signal)], 2.0 * input_signal)
    np.testing.assert_allclose(output_low[0, :len(input_signal)], 1.0 * input_signal)


def test_element_calc_output_generates_signal_from_scalar_input():
    element = Element(sensitivity=2.0)
    cycles = 4
    frequency = 100e3
    dt = 1e-7
    n_samples = int(np.round(cycles / (frequency * dt)))

    output = element.calc_output(3.0, cycles=cycles, frequency=frequency, dt=dt)
    t = np.arange(n_samples) * dt
    expected = 2.0 * 3.0 * np.sin(2 * np.pi * frequency * t)

    np.testing.assert_allclose(output, expected)


def test_element_calc_output_enforces_cycles_duration_for_array_input():
    element = Element(sensitivity=1.0)
    cycles = 1
    frequency = 200e3
    dt = 1e-6
    n_samples = int(np.round(cycles / (frequency * dt)))
    input_signal = np.arange(20, dtype=float)

    output = element.calc_output(input_signal, cycles=cycles, frequency=frequency, dt=dt)

    assert len(output) == n_samples
    np.testing.assert_allclose(output, input_signal[:n_samples])


def test_merge_pushes_transducer_sensitivity_into_elements():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={100e3: 2.0, 300e3: 4.0},
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={100e3: 3.0, 300e3: 6.0},
    )
    transducer_a.elements[0].sensitivity = 5.0
    transducer_b.elements[0].sensitivity = 7.0

    merged = Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)

    assert merged.sensitivity == 1.0
    assert merged.elements[0].sensitivity == {100e3: 10.0, 300e3: 20.0}
    assert merged.elements[1].sensitivity == {100e3: 21.0, 300e3: 42.0}


def test_merge_rejects_mismatched_sensitivity_keys():
    transducer_a = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={100e3: 2.0, 300e3: 4.0},
    )
    transducer_b = Transducer.gen_matrix_array(
        nx=1,
        ny=1,
        units="mm",
        sensitivity={100e3: 3.0, 400e3: 6.0},
    )
    transducer_a.elements[0].sensitivity = {100e3: 5.0, 300e3: 7.0}
    transducer_b.elements[0].sensitivity = {100e3: 11.0, 400e3: 13.0}

    with pytest.raises(ValueError, match="different frequency keys"):
        Transducer.merge([transducer_a, transducer_b], merge_mismatched_sensitivity=True)
