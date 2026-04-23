from __future__ import annotations

from helpers import dataclasses_are_equal

from openlifu.bf import Sequence


def test_dict_undict_sequence():
    """Test that conversion between Sequence and dict works"""
    sequence = Sequence(pulse_interval=2, pulse_count=5, pulse_train_interval=11, pulse_train_count=3)
    reconstructed_sequence = Sequence.from_dict(sequence.to_dict())
    assert dataclasses_are_equal(sequence, reconstructed_sequence)
