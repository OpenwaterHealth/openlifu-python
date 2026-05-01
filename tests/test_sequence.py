from __future__ import annotations

import pytest
from helpers import dataclasses_are_equal

from openlifu import Sequence


def test_dict_undict_sequence():
    """Test that conversion between Sequence and dict works"""
    sequence = Sequence(pulse_interval=2, pulse_count=5, pulse_train_interval=11, pulse_train_count=3, focus_order=[1, 2, 1, 3, 2])
    reconstructed_sequence = Sequence.from_dict(sequence.to_dict())
    assert dataclasses_are_equal(sequence, reconstructed_sequence)

@pytest.mark.parametrize(
    ("focus_order", "error_type", "match"),
    [
        ([], ValueError, "must not be empty"),
        ([1, 2], ValueError, "length must match pulse count"),
        ([1, 2, 1.5], TypeError, "entries must be integers"),
        ([1, 2, 0], ValueError, "entries must be positive"),
        ([1, 2, -1], ValueError, "entries must be positive"),
    ],
)
def test_sequence_focus_order_validation(focus_order, error_type, match):
    """Test validation of focus_order values."""
    with pytest.raises(error_type, match=match):
        Sequence(pulse_count=3, pulse_train_interval=3, focus_order=focus_order)
