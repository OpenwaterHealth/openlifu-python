from __future__ import annotations

import logging

import pytest

from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.plan.solution_analysis import (
    SolutionAnalysis,
    SolutionAnalysisOptions,
    model_tx_temperature_rise,
)

# ---- Tests for SolutionAnalysis ----

@pytest.fixture()
def example_solution_analysis() -> SolutionAnalysis:
    return SolutionAnalysis(
        mainlobe_pnp_MPa=[1.1, 1.2],
        mainlobe_isppa_Wcm2=[10.0, 12.0],
        mainlobe_ispta_mWcm2=[500.0, 520.0],
        beamwidth_lat_3dB_mm=[1.5, 1.6],
        beamwidth_ele_3dB_mm=[2.0, 2.1],
        beamwidth_ax_3dB_mm=[3.0, 3.1],
        beamwidth_lat_6dB_mm=[1.8, 1.9],
        beamwidth_ele_6dB_mm=[2.5, 2.6],
        beamwidth_ax_6dB_mm=[3.5, 3.6],
        sidelobe_pnp_MPa=[0.5, 0.6],
        sidelobe_isppa_Wcm2=[5.0, 5.5],
        sidelobe_to_mainlobe_pressure_ratio=[0.5/1.1, 0.6/1.2], # approx 0.45, 0.5
        sidelobe_to_mainlobe_intensity_ratio=[5.0/10.0, 5.5/12.0], # 0.5, approx 0.458
        global_pnp_MPa=[1.3],
        global_isppa_Wcm2=[13.0],
        p0_MPa=[1.0, 1.1],
        TIC=0.7,
        power_W=25.0,
        MI=1.2,
        global_ispta_mWcm2=540.0,
        param_constraints={
            "global_pnp_MPa": ParameterConstraint(
                operator="<=",
                warning_value=1.4,
                error_value=1.6
            )
        }
    )

def test_to_dict_from_dict_solution_analysis(example_solution_analysis: SolutionAnalysis):
    sa_dict = example_solution_analysis.to_dict()
    new_solution = SolutionAnalysis.from_dict(sa_dict)
    assert new_solution == example_solution_analysis

@pytest.mark.parametrize("compact", [True, False])
def test_serialize_deserialize_solution_analysis(example_solution_analysis: SolutionAnalysis, compact: bool):
    json_str = example_solution_analysis.to_json(compact)
    deserialized = SolutionAnalysis.from_json(json_str)
    assert deserialized == example_solution_analysis


# ---- Tests for SolutionAnalysisOptions ----


@pytest.fixture()
def example_solution_analysis_options() -> SolutionAnalysisOptions:
    return SolutionAnalysisOptions(
        standoff_sound_speed=1480.0,
        standoff_density=990.0,
        ref_sound_speed=1540.0,
        ref_density=1020.0,
        mainlobe_aspect_ratio=(1.0, 1.0, 4.0),
        mainlobe_radius=2.0e-3,
        beamwidth_radius=4.0e-3,
        sidelobe_radius=2.5e-3,
        sidelobe_zmin=0.5e-3,
        distance_units="mm",
        param_constraints={
            "mainlobe_radius": ParameterConstraint(
                operator=">=",
                warning_value=1.5e-3,
                error_value=1.0e-3
            )
        }
    )

def test_to_dict_from_dict_solution_analysis_options(example_solution_analysis_options: SolutionAnalysisOptions):
    options_dict = example_solution_analysis_options.to_dict()
    new_options = SolutionAnalysisOptions.from_dict(options_dict)
    assert new_options == example_solution_analysis_options


# ---- Tests for model_tx_temperature_rise ----

# Valid mid-range parameters used as a baseline throughout the tests.
# P = voltage^2 * duty_cycle = 14^2 * 1.0 = 196 V^2 (valid range: 50-500)
_BASE_VOLTAGE = 14.0       # V
_LOW_VOLTAGE = 8.0         # V  → P = 64 V^2 (near low end of valid range)
_HIGH_VOLTAGE = 21.0       # V  → P = 441 V^2 (near high end of valid range)
_BASE_T0 = 30.0            # °C (mid-range of valid 20-40 °C)
_BASE_FREQ = 400.0         # kHz (centre of valid 380-420 kHz)


def test_low_voltage_less_heating_than_high_voltage():
    """Higher voltage (more power) should produce a greater temperature rise."""
    t = 120.0  # mid-range time, well within valid 1-600 s
    rise_low = model_tx_temperature_rise(_LOW_VOLTAGE, t, T0_degC=_BASE_T0, frequency_kHz=_BASE_FREQ)
    rise_high = model_tx_temperature_rise(_HIGH_VOLTAGE, t, T0_degC=_BASE_T0, frequency_kHz=_BASE_FREQ)
    assert rise_low < rise_high, (
        f"Expected lower voltage ({_LOW_VOLTAGE} V, rise={rise_low:.4f} °C) to produce "
        f"less heating than higher voltage ({_HIGH_VOLTAGE} V, rise={rise_high:.4f} °C)"
    )


def test_little_heating_right_after_start():
    """Temperature rise at t=1 s (just started) should be much less than at t=300 s."""
    rise_early = model_tx_temperature_rise(_BASE_VOLTAGE, t_sec=1.0, T0_degC=_BASE_T0)
    rise_later = model_tx_temperature_rise(_BASE_VOLTAGE, t_sec=300.0, T0_degC=_BASE_T0)
    assert rise_early < rise_later, (
        f"Expected rise at t=1 s ({rise_early:.4f} °C) to be less than rise at t=300 s ({rise_later:.4f} °C)"
    )
    # Additionally confirm the early rise is small in absolute terms
    assert rise_early < 5.0, (
        f"Expected temperature rise at t=1 s to be < 5 °C, got {rise_early:.4f} °C"
    )


def test_temperature_rises_monotonically_with_time():
    """Temperature rise must be strictly increasing across a span of time points."""
    times = [1.0, 10.0, 60.0, 180.0, 360.0, 600.0]
    rises = [model_tx_temperature_rise(_BASE_VOLTAGE, t, T0_degC=_BASE_T0) for t in times]
    for i in range(len(rises) - 1):
        assert rises[i] < rises[i + 1], (
            f"Temperature rise not monotonically increasing: "
            f"rise[{times[i]} s]={rises[i]:.4f} >= rise[{times[i+1]} s]={rises[i+1]:.4f}"
        )


def test_temperature_rises_monotonically_with_voltage():
    """Temperature rise must increase with voltage (at fixed time and other params)."""
    # Voltages chosen so that P = V^2 stays within the valid 50-500 V^2 range
    voltages = [8.0, 11.0, 14.0, 17.0, 21.0]
    t = 120.0
    rises = [model_tx_temperature_rise(v, t, T0_degC=_BASE_T0) for v in voltages]
    for i in range(len(rises) - 1):
        assert rises[i] < rises[i + 1], (
            f"Temperature rise not monotonically increasing with voltage: "
            f"rise[{voltages[i]} V]={rises[i]:.4f} >= rise[{voltages[i+1]} V]={rises[i+1]:.4f}"
        )


def test_lower_duty_cycle_produces_less_heating():
    """Reducing duty cycle reduces effective power and therefore temperature rise."""
    t = 120.0
    voltage = _BASE_VOLTAGE
    rise_full = model_tx_temperature_rise(voltage, t, duty_cycle=1.0, T0_degC=_BASE_T0)
    rise_half = model_tx_temperature_rise(voltage, t, duty_cycle=0.5, T0_degC=_BASE_T0)
    assert rise_half < rise_full, (
        f"Expected 50 % duty cycle ({rise_half:.4f} °C) to produce less heating "
        f"than 100 % duty cycle ({rise_full:.4f} °C)"
    )


def test_lower_apodization_produces_less_heating():
    """Partial apodization reduces effective power and therefore temperature rise."""
    t = 120.0
    rise_full = model_tx_temperature_rise(_BASE_VOLTAGE, t, apodization_fraction=1.0, T0_degC=_BASE_T0)
    rise_half = model_tx_temperature_rise(_BASE_VOLTAGE, t, apodization_fraction=0.5, T0_degC=_BASE_T0)
    assert rise_half < rise_full, (
        f"Expected apodization=0.5 ({rise_half:.4f} °C) to produce less heating "
        f"than apodization=1.0 ({rise_full:.4f} °C)"
    )


@pytest.mark.parametrize("bad_T0", [19.9, 40.1])
def test_warning_emitted_for_T0_out_of_range(bad_T0, caplog):
    """A warning must be logged when T0 is outside the valid 20-40 °C range."""
    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution_analysis"):
        model_tx_temperature_rise(_BASE_VOLTAGE, t_sec=60.0, T0_degC=bad_T0)
    assert any("T0" in record.message or "temperature" in record.message.lower()
               for record in caplog.records), (
        f"Expected a warning about T0 out of range for T0={bad_T0} °C"
    )


@pytest.mark.parametrize(("bad_voltage","bad_duty_cycle"), [
    (6.0, 1.0),   # P = 36 < 50
    (25.0, 1.0),  # P = 625 > 500
])
def test_warning_emitted_for_power_out_of_range(bad_voltage, bad_duty_cycle, caplog):
    """A warning must be logged when the squared-voltage power is outside 50-500 V^2."""
    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution_analysis"):
        model_tx_temperature_rise(bad_voltage, t_sec=60.0, duty_cycle=bad_duty_cycle, T0_degC=_BASE_T0)
    assert any("voltage" in record.message.lower() or "squared" in record.message.lower() or "v^2" in record.message.lower()
               for record in caplog.records), (
        f"Expected a warning about power out of range for voltage={bad_voltage} V"
    )


@pytest.mark.parametrize("bad_time", [0.5, 601.0])
def test_warning_emitted_for_time_out_of_range(bad_time, caplog):
    """A warning must be logged when t is outside the valid 1-600 s range."""
    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution_analysis"):
        model_tx_temperature_rise(_BASE_VOLTAGE, t_sec=bad_time, T0_degC=_BASE_T0)
    assert any("time" in record.message.lower() or "seconds" in record.message.lower()
               for record in caplog.records), (
        f"Expected a warning about time out of range for t={bad_time} s"
    )


@pytest.mark.parametrize("bad_freq", [379.9, 420.1])
def test_warning_emitted_for_frequency_out_of_range(bad_freq, caplog):
    """A warning must be logged when frequency is outside the valid 380-420 kHz range."""
    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution_analysis"):
        model_tx_temperature_rise(_BASE_VOLTAGE, t_sec=60.0, frequency_kHz=bad_freq, T0_degC=_BASE_T0)
    assert any("frequency" in record.message.lower() or "khz" in record.message.lower()
               for record in caplog.records), (
        f"Expected a warning about frequency out of range for freq={bad_freq} kHz"
    )


def test_no_warnings_for_valid_inputs(caplog):
    """No warnings should be emitted when all inputs are within their valid ranges."""
    with caplog.at_level(logging.WARNING, logger="openlifu.plan.solution_analysis"):
        model_tx_temperature_rise(
            voltage=_BASE_VOLTAGE,
            t_sec=60.0,
            duty_cycle=1.0,
            apodization_fraction=1.0,
            frequency_kHz=_BASE_FREQ,
            T0_degC=_BASE_T0,
        )
    assert len(caplog.records) == 0, (
        f"Unexpected warnings for valid inputs: {[r.message for r in caplog.records]}"
    )
