from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xa
from helpers import dataclasses_are_equal
from pytest_mock import MockerFixture

from openlifu import Point, Pulse, Sequence, Solution, Transducer
from openlifu.bf.focal_patterns import SinglePoint
from openlifu.plan import SolutionAnalysis
from openlifu.xdc.element import Element


@pytest.fixture()
def example_transducer() -> Transducer:
    return Transducer(
        id="trans_456",
        name="Test Transducer",
        elements=[
            Element(index=1, position=[-14, -14, 0], units="mm"),
            Element(index=2, position=[-2, -2, 0], units="mm"),
            Element(index=3, position=[2, 2, 0], units="mm"),
            Element(index=4, position=[14, 14, 0], units="mm")
        ],
        frequency=1e6,
        units="mm"
    )


@pytest.fixture()
def example_focal_pattern_single() -> SinglePoint:
    return SinglePoint(target_pressure=1.0e6, units="Pa")


@pytest.fixture()
def example_solution() -> Solution:
    rng = np.random.default_rng(147)
    return Solution(
        id="sol_001",
        name="Test Solution",
        protocol_id="prot_123",
        transducer=Transducer(
            id="trans_456",
            name="Test Transducer",
            elements=[
                Element(index=1, position=[-14, -14, 0], units="m"),
                Element(index=2, position=[-2, -2, 0], units="m"),
                Element(index=3, position=[2, 2, 0], units="m"),
                Element(index=4, position=[14, 14, 0], units="m")
            ],
            frequency=1e6,
            units="m"
        ),
        date_created=datetime(2024, 1, 1, 12, 0),
        description="This is a test solution for a unit test.",
        delays=np.array([[0.0, 1.0, 2.0, 3.0]]),
        apodizations=np.array([[0.5, 0.75, 1.0, 0.85]]),
        pulse=Pulse(frequency=42),
        sequence=Sequence(pulse_count=27, pulse_interval=2, pulse_train_interval=2*27+5),
        foci=[Point(id="test_focus_point")],
        target=Point(id="test_target_point"),
        simulation_result=xa.Dataset(
            {
                'p_min': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'p_max': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "Pa"}
                ),
                'intensity': xa.DataArray(
                    data=rng.random((1, 3, 2, 3)),
                    dims=["focal_point_index", "x", "y", "z"],
                    attrs={'units': "W/cm^2"}
                )
            },
            coords={
                'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
                'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'focal_point_index': [0]
            }
        )
    )


def test_default_solution():
    """Ensure it is possible to construct a default Solution"""
    Solution()


@pytest.mark.parametrize("compact_representation", [True, False])
@pytest.mark.parametrize("include_simulation_data", [True, False])
@pytest.mark.parametrize("default_solution", [True, False])
def test_json_serialize_deserialize_solution(
    example_solution: Solution,
    compact_representation: bool,
    include_simulation_data: bool,
    default_solution: bool
):
    """Verify that turning a Solution into json and then re-constructing it gets back to the original solution"""

    # Default solution serializes a bit differently because it's full of None values, so we test both cases
    solution = Solution() if default_solution else example_solution
    solution_json = solution.to_json(include_simulation_data=include_simulation_data, compact=compact_representation)
    if include_simulation_data:
        solution_reconstructed = Solution.from_json(solution_json)
    else:
        solution_reconstructed = Solution.from_json(solution_json, simulation_result=solution.simulation_result)
    assert dataclasses_are_equal(solution_reconstructed, solution)


def test_save_load_solution(example_solution: Solution, tmp_path: Path):
    """Test that a solution can be saved to and loaded from disk faithfully."""
    json_filepath = tmp_path/"some_directory"/"example_soln.json"
    example_solution.to_files(json_filepath)
    assert dataclasses_are_equal(Solution.from_files(json_filepath), example_solution)


def test_save_load_solution_custom_dataset_filepath(example_solution: Solution, tmp_path: Path):
    """Test that a solution can be saved to and loaded from disk faithfully, this time with a custom path for simulation data."""
    json_filepath = tmp_path/"some_directory"/"example_soln.json"
    nc_filepath = tmp_path/"some_other_directory"/"sim_output.nc"
    example_solution.to_files(json_filepath, nc_filepath)
    assert dataclasses_are_equal(Solution.from_files(json_filepath, nc_filepath), example_solution)


def test_num_foci(example_solution:Solution):
    """Ensure that the number of foci in the test solution matches the number of foci provided in the simuluation and beamform data."""
    num_foci = example_solution.num_foci()
    assert len(example_solution.foci) == num_foci
    assert len(example_solution.simulation_result['focal_point_index']) == num_foci
    assert example_solution.delays.shape[0] == num_foci
    assert example_solution.apodizations.shape[0] == num_foci


def test_solution_get_focus_counts_with_explicit_focus_order(example_solution: Solution):
    """Test that explicit focus_order is counted per focus."""
    example_solution.foci = [
        Point(id="focus_1"),
        Point(id="focus_2"),
        Point(id="focus_3"),
    ]
    example_solution.sequence = Sequence(
        pulse_count=6,
        pulse_interval=1,
        pulse_train_interval=6,
        focus_order=[1, 3, 2, 1, 3, 1],
    )

    np.testing.assert_array_equal(example_solution.get_focus_counts(), np.array([3, 1, 2]))


def test_solution_compute_balanced_focus_counts_preserves_total_and_weights_inverse_metric(example_solution: Solution):
    """Test ISPTA balancing count allocation preserves pulse count and favors lower metrics."""
    example_solution.foci = [
        Point(id="focus_1"),
        Point(id="focus_2"),
        Point(id="focus_3"),
    ]

    focus_counts = example_solution.compute_balanced_focus_counts(
        balance_metric_values=np.array([10, 20, 10]),
        pulse_count=8,
    )

    assert np.sum(focus_counts) == 8
    assert np.all(focus_counts >= 1)
    assert focus_counts[1] < focus_counts[0]
    assert focus_counts[1] < focus_counts[2]


def test_solution_build_focus_order_minimizes_repeats(example_solution: Solution):
    """Test focus order construction preserves counts and avoids repeats when possible."""
    example_solution.foci = [
        Point(id="focus_1"),
        Point(id="focus_2"),
        Point(id="focus_3"),
    ]
    focus_counts = np.array([3, 2, 3])

    focus_order = example_solution.build_focus_order(focus_counts)

    assert len(focus_order) == np.sum(focus_counts)
    np.testing.assert_array_equal(
        np.bincount(focus_order, minlength=example_solution.num_foci() + 1)[1:],
        focus_counts,
    )
    assert all(focus_order[i] != focus_order[i + 1] for i in range(len(focus_order) - 1))

@pytest.mark.parametrize("compact_representation", [True, False])
def test_json_serialize_deserialize_solution_analysis(compact_representation: bool):
    """Verify that turning a SolutionAnalysis into json and then re-constructing it gets back to the original"""
    analysis = SolutionAnalysis(mainlobe_isppa_Wcm2=[1,2],beamwidth_ax_6dB_mm=[3,4], MI=5)
    analysis_json = analysis.to_json(compact=compact_representation)
    analysis_reconstructed = SolutionAnalysis.from_json(analysis_json)
    assert dataclasses_are_equal(analysis_reconstructed, analysis)

def test_solution_analyze_data_types(example_solution:Solution):
    """Test that solution analysis field are all floats or lists of floats as expected"""
    analysis = example_solution.analyze()
    for f in fields(analysis):
        value = getattr(analysis, f.name)
        if f.name == "param_constraints":
            assert isinstance(value, dict)
        elif not isinstance(value, float):
                assert isinstance(value, list)
                if len(value) > 0:
                    assert isinstance(value[0], float)


def test_solution_created_date():
    """Test that created date is recent when a solution is created."""
    tolerance = timedelta(seconds=2)

    solution = Solution()
    now = datetime.now()
    assert(now - tolerance <= solution.date_created <= now + tolerance)


def test_solution_analyze_ratios(example_solution: Solution):
    """Test the calculation of mainlobe to sidelobe ratios in Solution.analyze()"""
    solution = example_solution
    # Use only one focus point for simplicity
    solution.foci = [Point(id="test_focus_point", position=np.array([0, 0, 0.05]), units="m")] # Centered focus
    solution.delays = np.array([[0.0, 0.0, 0.0, 0.0]])
    solution.apodizations = np.array([[1.0, 1.0, 1.0, 1.0]])

    # Create a simple simulation result
    # Dimensions: focal_point_index, lat, ele, ax
    # Coords: lat=[-0.01, 0, 0.01], ele=[0], ax=[0.04, 0.05, 0.06] (units in m)
    lat_coords = np.array([-0.01, 0, 0.01])
    ele_coords = np.array([0])
    ax_coords = np.array([0.04, 0.05, 0.06])

    p_min_data = np.zeros((1, 3, 1, 3))  # Initialize with zeros
    intensity_data = np.zeros((1, 3, 1, 3)) # Initialize with zeros

    # Case 1: Normal values
    # Mainlobe at (lat=0, ele=0, ax=0.05)
    # Sidelobe at (lat=0.01, ele=0, ax=0.06) -> make this distinct
    p_min_data[0, 1, 0, 1] = 1.0e6  # Mainlobe pressure = 1 MPa
    p_min_data[0, 2, 0, 2] = 0.5e6  # Sidelobe pressure = 0.5 MPa
    intensity_data[0, 1, 0, 1] = 10.0  # Mainlobe intensity = 10 W/cm^2
    intensity_data[0, 2, 0, 2] = 2.0   # Sidelobe intensity = 2 W/cm^2

    solution.simulation_result = xa.Dataset(
        {
            'p_min': xa.DataArray(data=p_min_data, dims=["focal_point_index", "x", "y", "z"], attrs={'units': "Pa"}),
            'p_max': xa.DataArray(data=p_min_data, dims=["focal_point_index", "x", "y", "z"], attrs={'units': "Pa"}), # Keep p_max same for simplicity
            'intensity': xa.DataArray(data=intensity_data, dims=["focal_point_index", "x", "y", "z"], attrs={'units': "W/cm^2"})
        },
        coords={
            'x': xa.DataArray(dims=["x"], data=lat_coords, attrs={'units': "m"}),
            'y': xa.DataArray(dims=["y"], data=ele_coords, attrs={'units': "m"}),
            'z': xa.DataArray(dims=["z"], data=ax_coords, attrs={'units': "m"}),
            'focal_point_index': [0]
        }
    )

    from openlifu.plan.solution_analysis import SolutionAnalysisOptions
    # Ensure options make the chosen voxels the max in their respective regions
    # Focus is at (0,0,0.05). Mainlobe radius will capture p_min_data[0,1,0,1]
    # Sidelobe radius will capture p_min_data[0,2,0,2] if it's outside mainlobe.
    options = SolutionAnalysisOptions(
        mainlobe_radius=0.005, # 5mm, should capture central voxel
        sidelobe_radius=0.005, # 5mm, but mask is > sidelobe_radius, so it's outside this
                                 # and aspect ratio matters.
        mainlobe_aspect_ratio=(1,1,1), # Make it spherical for simplicity
        sidelobe_zmin=0.001, # well below focus
        distance_units="m" # Matching simulation_result
    )
    # The transducer effective origin also plays a role in get_mask.
    # For example_transducer, with all elements at Z=0 and centered apodization, origin is approx (0,0,0)
    # Let's assume default transducer origin is [0,0,0] for get_mask calculations.

    analysis = solution.analyze(options=options)

    # Check Case 1
    # Expected mainlobe pnp = 1.0 MPa (from p_min_data[0,1,0,1] = 1e6 Pa)
    # Expected sidelobe pnp = 0.5 MPa (from p_min_data[0,2,0,2] = 0.5e6 Pa)
    # Expected pressure ratio = 0.5 / 1.0 = 0.5

    # Expected mainlobe isppa = 10 W/cm^2
    # Expected sidelobe isppa = 2 W/cm^2
    # Expected intensity ratio = 2.0 / 10.0 = 0.2

    # Due to potential small variations from interpolation/masking, use approx
    assert np.isclose(analysis.mainlobe_pnp_MPa[0], 1.0), f"Mainlobe PNP was {analysis.mainlobe_pnp_MPa[0]}"
    assert analysis.sidelobe_pnp_MPa[0] > 0, "Sidelobe PNP was zero, check mask/data for Case 1." # Ensure it's not zero
    # Using the actual found sidelobe value for ratio assertion robustness
    expected_pressure_ratio_case1 = analysis.sidelobe_pnp_MPa[0] / analysis.mainlobe_pnp_MPa[0]
    assert np.isclose(analysis.sidelobe_to_mainlobe_pressure_ratio[0], expected_pressure_ratio_case1), \
        f"Pressure ratio calculation error. Got {analysis.sidelobe_to_mainlobe_pressure_ratio[0]}, expected {expected_pressure_ratio_case1}"

    assert np.isclose(analysis.mainlobe_isppa_Wcm2[0], 10.0), f"Mainlobe ISPPA was {analysis.mainlobe_isppa_Wcm2[0]}"
    assert analysis.sidelobe_isppa_Wcm2[0] > 0, "Sidelobe ISPPA was zero, check mask/data for Case 1." # Ensure it's not zero
    expected_intensity_ratio_case1 = analysis.sidelobe_isppa_Wcm2[0] / analysis.mainlobe_isppa_Wcm2[0]
    assert np.isclose(analysis.sidelobe_to_mainlobe_intensity_ratio[0], expected_intensity_ratio_case1), \
        f"Intensity ratio calculation error. Got {analysis.sidelobe_to_mainlobe_intensity_ratio[0]}, expected {expected_intensity_ratio_case1}"


    # Case 2: Sidelobe pressure is zero, Mainlobe is non-zero
    p_min_data_case2 = np.zeros((1, 3, 1, 3))
    p_min_data_case2[0, 1, 0, 1] = 1.0e6  # Mainlobe pressure = 1 MPa
    p_min_data_case2[0, 2, 0, 2] = 0.0    # Sidelobe pressure = 0 MPa
    solution.simulation_result['p_min'].data = p_min_data_case2
    solution.simulation_result['intensity'].data = intensity_data # Keep intensity same

    analysis_case2 = solution.analyze(options=options)
    assert analysis_case2.mainlobe_pnp_MPa[0] == 1.0
    assert analysis_case2.sidelobe_pnp_MPa[0] == 0.0
    assert analysis_case2.sidelobe_to_mainlobe_pressure_ratio[0] == 0.0 # 0 / 1.0 == 0.0

    # Case 3: Sidelobe intensity is zero, Mainlobe is non-zero
    intensity_data_case3 = np.zeros((1,3,1,3))
    intensity_data_case3[0, 1, 0, 1] = 10.0  # Mainlobe intensity = 10 W/cm^2
    intensity_data_case3[0, 2, 0, 2] = 0.0   # Sidelobe intensity = 0 W/cm^2
    solution.simulation_result['p_min'].data = p_min_data # Reset p_min
    solution.simulation_result['intensity'].data = intensity_data_case3

    analysis_case3 = solution.analyze(options=options)
    assert analysis_case3.mainlobe_isppa_Wcm2[0] == 10.0
    assert analysis_case3.sidelobe_isppa_Wcm2[0] == 0.0
    assert analysis_case3.sidelobe_to_mainlobe_intensity_ratio[0] == 0.0 # 0 / 10.0 == 0.0

    # Case 4: Mainlobe pressure is zero, Sidelobe is non-zero
    p_min_data_case4 = np.zeros((1, 3, 1, 3))
    p_min_data_case4[0, 1, 0, 1] = 0.0    # Mainlobe pressure = 0 MPa
    p_min_data_case4[0, 2, 0, 2] = 0.5e6  # Sidelobe pressure = 0.5 MPa
    solution.simulation_result['p_min'].data = p_min_data_case4
    solution.simulation_result['intensity'].data = intensity_data # Keep intensity same

    analysis_case4 = solution.analyze(options=options)
    assert analysis_case4.mainlobe_pnp_MPa[0] == 0.0
    # Ensure sidelobe is picked up as non-zero
    assert analysis_case4.sidelobe_pnp_MPa[0] > 0, "Sidelobe PNP was zero for Case 4, expected non-zero."
    assert analysis_case4.sidelobe_to_mainlobe_pressure_ratio[0] == np.inf # 0.5 / 0 == inf

    # Case 5: Mainlobe intensity is zero, Sidelobe is non-zero
    intensity_data_case5 = np.zeros((1,3,1,3))
    intensity_data_case5[0, 1, 0, 1] = 0.0   # Mainlobe intensity = 0 W/cm^2
    intensity_data_case5[0, 2, 0, 2] = 2.0  # Sidelobe intensity = 2 W/cm^2
    solution.simulation_result['p_min'].data = p_min_data # Reset p_min
    solution.simulation_result['intensity'].data = intensity_data_case5
    analysis_case5 = solution.analyze(options=options)
    assert analysis_case5.mainlobe_isppa_Wcm2[0] == 0.0
    assert analysis_case5.sidelobe_isppa_Wcm2[0] > 0, "Sidelobe ISPPA was zero for Case 5, expected non-zero."
    assert analysis_case5.sidelobe_to_mainlobe_intensity_ratio[0] == np.inf # 2.0 / 0 == inf

    # Case 6: Mainlobe and Sidelobe pressure are zero
    p_min_data_case6 = np.zeros((1, 3, 1, 3))
    solution.simulation_result['p_min'].data = p_min_data_case6
    solution.simulation_result['intensity'].data = intensity_data # Keep intensity same

    analysis_case6 = solution.analyze(options=options)
    assert analysis_case6.mainlobe_pnp_MPa[0] == 0.0
    assert analysis_case6.sidelobe_pnp_MPa[0] == 0.0
    assert np.isnan(analysis_case6.sidelobe_to_mainlobe_pressure_ratio[0]) # 0 / 0 == nan

    # Case 7: Mainlobe and Sidelobe intensity are zero
    intensity_data_case7 = np.zeros((1,3,1,3))
    solution.simulation_result['p_min'].data = p_min_data # Reset p_min
    solution.simulation_result['intensity'].data = intensity_data_case7
    analysis_case7 = solution.analyze(options=options)
    assert analysis_case7.mainlobe_isppa_Wcm2[0] == 0.0
    assert analysis_case7.sidelobe_isppa_Wcm2[0] == 0.0
    assert np.isnan(analysis_case7.sidelobe_to_mainlobe_intensity_ratio[0]) # 0 / 0 == nan


@pytest.mark.parametrize("force_cpu", [True, False])
@pytest.mark.parametrize("gpu_is_available", [True, False])
def test_solution_simulate_gpu_handling(
    mocker: MockerFixture,
    example_solution: Solution,
    force_cpu: bool,
    gpu_is_available: bool,
):
    """Test that the correct GPU flag is passed to run_simulation in solution.simulate"""
    example_simulation_output = xa.Dataset(
        {
            'p_min': xa.DataArray(data=np.empty((3, 2, 3)), dims=["x", "y", "z"], attrs={'units': "Pa"}),
            'p_max': xa.DataArray(data=np.empty((3, 2, 3)), dims=["x", "y", "z"], attrs={'units': "Pa"}),
            'intensity': xa.DataArray(data=np.empty((3, 2, 3)), dims=["x", "y", "z"], attrs={'units': "W/cm^2"}),
        },
        coords={
            'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
            'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
            'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
        },
    )
    mocker.patch(
        "openlifu.plan.solution.gpu_available",
        return_value=gpu_is_available,
    )
    run_simulation_mock = mocker.patch(
        "openlifu.plan.solution.run_simulation",
        return_value=example_simulation_output
    )

    # Create params dataset for simulation
    params = xa.Dataset(
        {
            'density': xa.DataArray(data=np.ones((3, 2, 3)), dims=["x", "y", "z"], attrs={'units': "kg/m^3"}),
            'sound_speed': xa.DataArray(data=np.ones((3, 2, 3)) * 1500, dims=["x", "y", "z"], attrs={'units': "m/s"}),
        },
        coords={
            'x': xa.DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
            'y': xa.DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"}),
            'z': xa.DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
        },
    )

    example_solution.simulate(
        params=params,
        _force_cpu=force_cpu,
    )

    args, kwargs = run_simulation_mock.call_args
    # If force_cpu is True, GPU should always be False
    # Otherwise, GPU should match gpu_available()
    if force_cpu:
        assert not(kwargs['gpu'])
    else:
        assert kwargs['gpu'] == gpu_is_available
