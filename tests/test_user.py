from __future__ import annotations

from pathlib import Path

import pytest

from openlifu.db import User


@pytest.fixture()
def example_user() -> User:
    return User.from_file(Path(__file__).parent/'resources/example_db/users/example_user/example_user.json')

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_user(example_user : User, compact_representation: bool):
    assert example_user.from_json(example_user.to_json(compact_representation)) == example_user

def test_default_user():
    """Ensure it is possible to construct a default user"""
    User()
