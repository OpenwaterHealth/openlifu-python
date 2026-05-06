from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from openlifu.bf.delay_methods import DelayMethod, SimulationCorrected


class TestSimulationCorrectedConstruction:
    """Test basic construction of SimulationCorrected instances."""

    def test_default_construction(self):
        method = SimulationCorrected()
        assert isinstance(method, SimulationCorrected)
        assert isinstance(method, DelayMethod)
        assert method.c0 == 1500.0
        assert method.cfl == 0.3
        assert method.n_cycles == 3
        assert method.gpu is True

    def test_custom_params(self):
        method = SimulationCorrected(c0=1480.0, cfl=0.2, n_cycles=5, gpu=False)
        assert method.c0 == 1480.0
        assert method.cfl == 0.2
        assert method.n_cycles == 5
        assert method.gpu is False


class TestSimulationCorrectedValidation:
    """Test parameter validation."""

    def test_invalid_c0_negative(self):
        with pytest.raises(ValueError, match="greater than 0"):
            SimulationCorrected(c0=-100.0)

    def test_invalid_c0_zero(self):
        with pytest.raises(ValueError, match="greater than 0"):
            SimulationCorrected(c0=0.0)

    def test_invalid_c0_type(self):
        with pytest.raises(TypeError):
            SimulationCorrected(c0="fast")

    def test_invalid_cfl_too_high(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            SimulationCorrected(cfl=1.0)

    def test_invalid_cfl_zero(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            SimulationCorrected(cfl=0.0)

    def test_invalid_cfl_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            SimulationCorrected(cfl=-0.1)

    def test_invalid_n_cycles_zero(self):
        with pytest.raises(ValueError, match="at least 1"):
            SimulationCorrected(n_cycles=0)

    def test_invalid_n_cycles_type(self):
        with pytest.raises(TypeError):
            SimulationCorrected(n_cycles=2.5)

    def test_n_cycles_float_int_coercion(self):
        # A float that is exactly an integer should be accepted
        method = SimulationCorrected(n_cycles=3.0)
        assert method.n_cycles == 3
        assert isinstance(method.n_cycles, int)


class TestSimulationCorrectedSerialization:
    """Test to_dict / from_dict round-trip serialization."""

    def test_to_dict(self):
        method = SimulationCorrected(c0=1480.0, cfl=0.25, n_cycles=4, gpu=False)
        d = method.to_dict()
        assert d['class'] == 'SimulationCorrected'
        assert d['c0'] == 1480.0
        assert d['cfl'] == 0.25
        assert d['n_cycles'] == 4
        assert d['gpu'] is False

    def test_from_dict(self):
        d = {
            'class': 'SimulationCorrected',
            'c0': 1480.0,
            'cfl': 0.25,
            'n_cycles': 4,
            'gpu': False,
        }
        method = DelayMethod.from_dict(d)
        assert isinstance(method, SimulationCorrected)
        assert method.c0 == 1480.0
        assert method.cfl == 0.25
        assert method.n_cycles == 4
        assert method.gpu is False

    def test_round_trip(self):
        original = SimulationCorrected(c0=1450.0, cfl=0.35, n_cycles=5, gpu=True)
        d = original.to_dict()
        restored = DelayMethod.from_dict(d)
        assert isinstance(restored, SimulationCorrected)
        assert restored.c0 == original.c0
        assert restored.cfl == original.cfl
        assert restored.n_cycles == original.n_cycles
        assert restored.gpu == original.gpu

    def test_round_trip_defaults(self):
        original = SimulationCorrected()
        d = original.to_dict()
        restored = DelayMethod.from_dict(d)
        assert isinstance(restored, SimulationCorrected)
        assert restored.c0 == original.c0
        assert restored.cfl == original.cfl
        assert restored.n_cycles == original.n_cycles
        assert restored.gpu == original.gpu


class TestSimulationCorrectedTable:
    """Test the to_table method."""

    def test_to_table(self):
        method = SimulationCorrected()
        table = method.to_table()
        assert len(table) == 5
        assert table.iloc[0]['Name'] == 'Type'
        assert table.iloc[0]['Value'] == 'SimulationCorrected'
        assert table.iloc[1]['Name'] == 'Default Sound Speed'
        assert table.iloc[1]['Value'] == 1500.0


class TestSimulationCorrectedBehavior:
    """Test delay calculation logic by mocking the k-wave simulation layer."""

    def test_delays_from_known_arrival_times(self):
        """When _run_reciprocal_simulation returns known arrival times,
        calc_delays should return max(arrival) - arrival for each element."""
        arrival_times = np.array([0.001, 0.002, 0.003])
        expected_delays = np.array([0.002, 0.001, 0.0])

        method = SimulationCorrected()
        with patch.object(
            SimulationCorrected,
            "_run_reciprocal_simulation",
            return_value=arrival_times,
        ), patch("importlib.util.find_spec", return_value=True):
            delays = method.calc_delays(
                arr=None, target=None, params=None, transform=None
            )
        np.testing.assert_allclose(delays, expected_delays)

    def test_fallback_on_simulation_failure(self):
        """When _run_reciprocal_simulation raises, calc_delays should
        fall back to Direct geometric delays without crashing."""
        method = SimulationCorrected()
        with patch.object(
            SimulationCorrected,
            "_run_reciprocal_simulation",
            side_effect=RuntimeError("mocked failure"),
        ), patch("importlib.util.find_spec", return_value=True), patch.object(
            SimulationCorrected,
            "_fallback_delays",
            return_value=np.array([0.0, 0.0, 0.0]),
        ) as mock_fallback:
            delays = method.calc_delays(
                arr=None, target=None, params=None, transform=None
            )
        mock_fallback.assert_called_once()
        np.testing.assert_array_equal(delays, [0.0, 0.0, 0.0])

    def test_fallback_when_kwave_missing(self):
        """When k-wave is not installed, calc_delays should fall back
        to Direct geometric delays."""
        method = SimulationCorrected()
        with patch("importlib.util.find_spec", return_value=None), patch.object(
            SimulationCorrected,
            "_fallback_delays",
            return_value=np.array([0.0, 0.0]),
        ) as mock_fallback:
            delays = method.calc_delays(
                arr=None, target=None, params=None, transform=None
            )
        mock_fallback.assert_called_once()
        np.testing.assert_array_equal(delays, [0.0, 0.0])
