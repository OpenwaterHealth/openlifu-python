from __future__ import annotations

from pathlib import Path

import pytest

from openlifu.io import LIFUInterface
from openlifu.plan.solution import Solution


# Test LIFUInterface in test_mode
@pytest.fixture()
def lifu_interface():
    interface = LIFUInterface(TX_test_mode=True, HV_test_mode=True, run_async=False)
    assert isinstance(interface, LIFUInterface)
    yield interface
    interface.close()

# load example solution
@pytest.fixture()
def example_solution():
    return Solution.from_files(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/solutions/example_solution/example_solution.json")

# Test LIFUInterface with example solution
def test_lifu_interface_with_solution(lifu_interface, example_solution):
    assert all(lifu_interface.is_device_connected())
    # Load the example solution
    lifu_interface.set_solution(example_solution.to_dict())

# Create invalid duty cycle solution
@pytest.fixture()
def invalid_duty_cycle_solution():
    solution = Solution.from_files(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/solutions/example_solution/example_solution.json")
    solution.pulse.duration = 0.06
    solution.sequence.pulse_interval = 0.1
    return solution

# Test LIFUInterface with invalid solution
def test_lifu_interface_with_invalid_solution(lifu_interface, invalid_duty_cycle_solution):
    with pytest.raises(ValueError, match=R"Sequence duty cycle"):
        lifu_interface.set_solution(invalid_duty_cycle_solution.to_dict())

# Create invalid voltage solution
@pytest.fixture()
def invalid_voltage_solution():
    solution = Solution.from_files(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/solutions/example_solution/example_solution.json")
    solution.voltage = 1000  # Set voltage above maximum
    return solution

# Test LIFUInterface with invalid voltage solution
def test_lifu_interface_with_invalid_voltage_solution(lifu_interface, invalid_voltage_solution):
    with pytest.raises(ValueError, match=R"exceeds maximum allowed voltage"):
        lifu_interface.set_solution(invalid_voltage_solution.to_dict())

# Create too long sequence solution
@pytest.fixture()
def too_long_sequence_solution():
    solution = Solution.from_files(Path(__file__).parent/"resources/example_db/subjects/example_subject/sessions/example_session/solutions/example_solution/example_solution.json")
    solution.voltage = 40
    solution.pulse.duration = 0.05
    solution.sequence.pulse_interval = 0.1
    solution.sequence.pulse_count = 10
    solution.sequence.pulse_train_interval = 1.0
    solution.sequence.pulse_train_count = 600
    return solution

# Test LIFUInterface with too long sequence solution
def test_lifu_interface_with_too_long_sequence_solution(lifu_interface, too_long_sequence_solution):
    with pytest.raises(ValueError, match=R"exceeds maximum allowed voltage"):
        lifu_interface.set_solution(too_long_sequence_solution.to_dict())
