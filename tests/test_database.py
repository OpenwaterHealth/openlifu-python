from __future__ import annotations

import logging
import shutil
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from helpers import dataclasses_are_equal
from vtk import vtkImageData, vtkPolyData

from openlifu.db import Session, Subject, User
from openlifu.db.database import Database, OnConflictOpts
from openlifu.db.session import TransducerTrackingResult
from openlifu.geo import ArrayTransform, Point
from openlifu.nav.photoscan import Photoscan
from openlifu.plan import Protocol, Run, Solution
from openlifu.util.volume_conversion import is_dicom_file_or_directory
from openlifu.xdc import Transducer


@pytest.fixture()
def example_database(tmp_path:Path) -> Database:
    """Example database in a temporary directory; appropriate to use when testing write operations."""
    shutil.copytree(Path(__file__).parent/'resources/example_db', tmp_path/"example_db")
    return Database(tmp_path/"example_db")

@pytest.fixture()
def example_session(example_database : Database) -> Session:
    return Session.from_file(
        filename = Path(example_database.path)/"subjects/example_subject/sessions/example_session/example_session.json",
    )

@pytest.fixture()
def example_subject(example_database : Database) -> Subject:
    return Subject.from_file(
        filename = Path(example_database.path)/"subjects/example_subject/example_subject.json",
    )

@pytest.fixture()
def example_transducer(example_database : Database) -> Transducer:
    return Transducer.from_file(
        filename = Path(example_database.path)/"transducers/example_transducer/example_transducer.json",
        )

def test_new_database(tmp_path:Path):
    """Test that a new empty database can be created that more or less works"""
    db1 = Database.initialize_empty_database(tmp_path/"db1")
    db2 = Database.initialize_empty_database(str(tmp_path/"db2")) # make sure using string also works
    assert len(db1.get_protocol_ids()) == 0
    assert len(db1.get_user_ids()) == 0
    assert len(db1.get_subject_ids()) == 0
    assert len(db1.get_transducer_ids()) == 0

@pytest.fixture()
def example_transducer_tracking_result() -> TransducerTrackingResult:
    return TransducerTrackingResult(photoscan_id="example_photoscan",
                                    transducer_to_volume_transform = ArrayTransform(np.eye(4),"mm"),
                                    photoscan_to_volume_transform = ArrayTransform(np.eye(4),"mm"))

def test_write_protocol(example_database: Database):
    protocol = Protocol(name="bleh", id="a_protocol_called_bleh")

    # Protocol id is not in list initially
    assert protocol.id not in example_database.get_protocol_ids()

    # Can add a new protocol, and it loads back in correctly.
    example_database.write_protocol(protocol)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert dataclasses_are_equal(reloaded_protocol,protocol)

    # Protocol id is now in the list
    assert protocol.id in example_database.get_protocol_ids()

    # Error raised when the protocol already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_protocol(protocol, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    protocol.name = "new_name"
    example_database.write_protocol(protocol, on_conflict=OnConflictOpts.SKIP)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert reloaded_protocol.name == "bleh"

    # Overwrite option
    protocol.name = "new_name"
    example_database.write_protocol(protocol, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_protocol = example_database.load_protocol(protocol.id)
    assert reloaded_protocol.name == "new_name"

def test_delete_protocol(example_database: Database):
    # Write a protocol
    protocol = Protocol(name="bleh", id="a_protocol_to_be_deleted")
    example_database.write_protocol(protocol)
    assert protocol.id in example_database.get_protocol_ids()

    # Protocol is deleted
    example_database.delete_protocol(protocol.id)
    assert protocol.id not in example_database.get_protocol_ids()
    with pytest.raises(FileNotFoundError):
        example_database.load_protocol(protocol.id)

    # Error option
    with pytest.raises(ValueError, match="does not exist in the database"):
        example_database.delete_protocol("non_existent_protocol", on_conflict=OnConflictOpts.ERROR)

    # Skip option
    example_database.delete_protocol("non_existent_protocol", on_conflict=OnConflictOpts.SKIP)

    # Invalid option
    with pytest.raises(ValueError, match="Invalid"):
        example_database.delete_protocol("non_existent_protocol", on_conflict=OnConflictOpts.OVERWRITE)

def test_write_user(example_database: Database):
    user = User(name="thelegend27", password_hash="abc", id="a_user_called_thelegend27")

    # User id is not in list initially
    assert user.id not in example_database.get_user_ids()

    # Can add a new user, and it loads back in correctly.
    example_database.write_user(user)
    reloaded_user = example_database.load_user(user.id)
    assert dataclasses_are_equal(reloaded_user,user)

    # User id is now in the list
    assert user.id in example_database.get_user_ids()

    # Error raised when the user already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_user(user, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    user.name = "new_name"
    example_database.write_user(user, on_conflict=OnConflictOpts.SKIP)
    reloaded_user = example_database.load_user(user.id)
    assert reloaded_user.name == "thelegend27"

    # Overwrite option
    user.name = "new_name"
    example_database.write_user(user, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_user = example_database.load_user(user.id)
    assert reloaded_user.name == "new_name"

def test_delete_user(example_database: Database):
    # Write a user
    user = User(name="thelegend27", id="a_user_to_be_deleted")
    example_database.write_user(user)
    assert user.id in example_database.get_user_ids()

    # User is deleted
    example_database.delete_user(user.id)
    assert user.id not in example_database.get_user_ids()
    with pytest.raises(FileNotFoundError):
        example_database.load_user(user.id)

    # Error option
    with pytest.raises(ValueError, match="does not exist in the database"):
        example_database.delete_user("non_existent_user", on_conflict=OnConflictOpts.ERROR)

    # Skip option
    example_database.delete_user("non_existent_user", on_conflict=OnConflictOpts.SKIP)

    # Invalid option
    with pytest.raises(ValueError, match="Invalid"):
        example_database.delete_user("non_existent_user", on_conflict=OnConflictOpts.OVERWRITE)

def test_load_all_users(example_database: Database):
    previous_number_of_users_in_database = len(example_database.load_all_users())

    # Create a user and write it to the database
    user = User(name="thelegend28", id="additional_user_to_be_loaded_then_deleted")
    example_database.write_user(user)

    # Load all users and check if they match
    loaded_users = example_database.load_all_users()
    assert len(loaded_users) == 1 + previous_number_of_users_in_database
    assert any(dataclasses_are_equal(u, user) for u in loaded_users)

    example_database.delete_user(user.id, on_conflict=OnConflictOpts.ERROR)

    loaded_users = example_database.load_all_users()
    assert len(loaded_users) == previous_number_of_users_in_database
    assert not any(dataclasses_are_equal(u, user) for u in loaded_users)

def test_load_session_from_file(example_session : Session, example_database : Database):

    # Test that Session loaded via Session.from_file is correct
    session = example_session
    assert session.name == "Example Session"
    assert session.volume_id == "example_volume"
    assert session.transducer_id == "example_transducer"
    assert session.protocol_id == "example_protocol"
    assert session.array_transform.matrix.shape == (4,4)
    assert session.array_transform.units == "mm"

    # Test that the Session loaded via the Database is identical
    session_from_database = example_database.load_session(
        example_database.load_subject(session.subject_id),
        session.id,
    )
    assert dataclasses_are_equal(session_from_database,session)

def test_write_subject(example_database : Database):
    subject = Subject(id="bleh",name="Seb Jectson")

    # Can add a new subject, and it loads back in correctly.
    example_database.write_subject(subject)
    reloaded_subject = example_database.load_subject("bleh")
    assert subject == reloaded_subject

    # Empty sessions file is created (overlaps with test_write_subject_associated_object_structure_created but it's okay)
    sessions_filename = example_database.get_sessions_filename(subject.id)
    assert sessions_filename.exists()
    assert sessions_filename.is_file()
    assert sessions_filename.name == "sessions.json"
    session_ids = example_database.get_session_ids(subject.id)
    assert session_ids == []

    # Add a session so we can later test that overwriting a subject doesn't wipe out the session
    session = Session(id="jectson_session_1", subject_id=subject.id)
    example_database.write_session(subject, session)

    # Error raised when the subject already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_subject(subject, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    subject.name = "Deb Jectson"
    example_database.write_subject(subject, on_conflict=OnConflictOpts.SKIP)
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Seb Jectson"

    # Overwrite option
    example_database.write_subject(subject, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_subject = example_database.load_subject("bleh")
    assert reloaded_subject.name == "Deb Jectson"

    # Ensure that after overwrite of a subject the sessions are still there
    assert session.id in example_database.get_session_ids(subject.id)
    assert dataclasses_are_equal(
        example_database.load_session(subject, session.id),
        session
    )

def test_write_subject_associated_object_structure_created(example_database : Database):
    """Test that when you create a new subject, the file structure needed for other objects that can be written
    under that subject is also created."""
    subject = Subject(id="bleh",name="Seb Jectson")
    example_database.write_subject(subject)

    assert example_database.get_sessions_filename(subject.id).is_file()
    assert example_database.get_volumes_filename(subject.id).is_file()

def test_write_session(example_database: Database, example_subject: Subject):
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)

    # Can add a new session, and it loads back in correctly.
    example_database.write_session(example_subject, session)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert dataclasses_are_equal(reloaded_session,session)

    # Add a solution and a run to later test that overwriting a session doesn't wipe them out
    solution = Solution(id="please_keep_me")
    run = Run(id="please_keep_me_too")
    example_database.write_solution(session,solution)
    example_database.write_run(run,session)

    # Error raised when the session already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.SKIP)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "bleh"

    # Overwrite option
    session.name = "new_name"
    example_database.write_session(example_subject, session, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_session = example_database.load_session(example_subject, session.id)
    assert reloaded_session.name == "new_name"

    # Ensure that after overwrite of a session the runs and solutions are still there
    assert solution.id in example_database.get_solution_ids(session.subject_id, session.id)
    assert dataclasses_are_equal(
        example_database.load_solution(session, solution.id),
        solution,
    )
    assert run.id in example_database.get_run_ids(session.subject_id, session.id)

    # When writing to a new subject
    new_subject = Subject(id="bleh_new",name="Deb Jectson")
    example_database.write_subject(new_subject, on_conflict=OnConflictOpts.OVERWRITE)
    session = Session(name="bleh", id='a_session',subject_id=new_subject.id)
    example_database.write_session(new_subject, session)
    reloaded_session = example_database.load_session(new_subject, session.id)
    assert reloaded_session.name == "bleh"

def test_write_session_associated_object_structure_created(example_database: Database, example_subject: Subject):
    """Test that when you create a new session, the file structure needed for other objects that can be written
    under that session is also created."""
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)
    example_database.write_session(example_subject, session)

    assert example_database.get_solutions_filename(example_subject.id, session.id).is_file()
    assert example_database.get_runs_filename(example_subject.id, session.id).is_file()

def test_write_session_with_invalid_photoscan_id(example_database: Database, example_subject: Subject, example_transducer_tracking_result):
    """ Test that when you write a session with a transducer tracking result associated with an
      invalid photoscan, an error is raised."""
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)
    session.transducer_tracking_results = [example_transducer_tracking_result]
    example_transducer_tracking_result.photoscan_id = "bogus_photoscan"
    with pytest.raises(ValueError, match="been associated with this session"):
        example_database.write_session(example_subject, session)

def test_write_session_with_transducer_tracking_results(example_database: Database, example_subject: Subject, example_transducer_tracking_result):
    """ Test that when there is a transducer tracking result class associated with a session, the session
    is correctly written to file."""
    session = Session(name="bleh", id='example_session',subject_id=example_subject.id)
    session.transducer_tracking_results = [example_transducer_tracking_result]
    example_database.write_session(example_subject, session, on_conflict = OnConflictOpts.OVERWRITE)

def test_delete_session(example_database: Database, example_subject: Subject):
    # Write a session
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)

    session_id = session.id
    subject_id = example_subject.id

    # Can add a new session, and it loads back in correctly.
    example_database.write_session(example_subject, session)

    assert session.id in example_database.get_session_ids(example_subject.id)

    # Session is deleted
    example_database.delete_session(subject_id, session_id)
    assert session.id not in example_database.get_session_ids(subject_id)
    with pytest.raises(FileNotFoundError):
        example_database.load_session(example_subject, session_id)

    # Error option
    with pytest.raises(ValueError, match="does not exist in the database"):
        example_database.delete_session(subject_id, "non_existent_session", on_conflict=OnConflictOpts.ERROR)

    # Skip option
    example_database.delete_session(subject_id, "non_existent_session", on_conflict=OnConflictOpts.SKIP)

    # Invalid option
    with pytest.raises(ValueError, match="Invalid"):
        example_database.delete_session(subject_id, "non_existent_session", on_conflict=OnConflictOpts.OVERWRITE)

def test_write_run(example_database: Database, tmp_path:Path):
    subject_id = "example_subject"
    session_id = "example_session"
    protocol_id = "example_protocol"
    run_id = "example_run_2"
    success_flag = True
    note = "Test note"
    solution_id = "example_solution"
    subject = example_database.load_subject(subject_id)
    session = example_database.load_session(subject, session_id)
    protocol = example_database.load_protocol(protocol_id)
    run = Run(id=run_id, success_flag=success_flag, note=note, session_id=session_id, solution_id=solution_id)

    # Can add a new run
    example_database.write_run(run, session, protocol)
    run_file_path = tmp_path/"example_db/subjects/example_subject/sessions/example_session/runs/example_run/example_run.json"
    assert(run_file_path.is_file())

    # Error raised when the session already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_run(run, session, protocol, on_conflict=OnConflictOpts.ERROR)

    # Error raised when the user try to overwrite a run
    with pytest.raises(ValueError, match="may not be overwritten"):
        example_database.write_run(run, session, protocol, on_conflict=OnConflictOpts.OVERWRITE)

    # Make sure the Runs folder and Runs file are created
    new_session = Session(id='new_session',subject_id='example_subject')
    example_database.write_session(subject, new_session)
    new_run = Run(id=run_id, success_flag=success_flag, note=note, session_id='new_session', solution_id=solution_id)
    example_database.write_run(new_run, new_session, protocol, on_conflict=OnConflictOpts.OVERWRITE)
    runs_filename = example_database.get_runs_filename(subject.id, new_session.id)
    assert runs_filename.exists()
    assert runs_filename.is_file()
    assert runs_filename.name == "runs.json"

def test_load_session_snapshot(example_database: Database):
    subject_id = "example_subject"
    session_id = "example_session"
    run_id = "example_run"
    session = example_database.load_session_snapshot(subject_id, session_id, run_id)
    assert session.id == "example_session"

def test_load_protocol_snapshot(example_database: Database):
    subject_id = "example_subject"
    session_id = "example_session"
    run_id = "example_run"
    protocol = example_database.load_protocol_snapshot(subject_id, session_id, run_id)
    assert protocol.id == "example_protocol"

def test_write_session_mismatched_id(example_database: Database, example_subject: Subject):
    session = Session(id='a_session',subject_id='bogus_id') # The subject ID here is different from the ID in example_subject
    with pytest.raises(ValueError, match="IDs do not match"):
        example_database.write_session(example_subject, session)

@pytest.mark.parametrize(
    ("target_ids", "numbers_of_transforms", "expectation"),
    [
        # see https://docs.pytest.org/en/6.2.x/example/parametrize.html#parametrizing-conditional-raising
        ([], [], does_not_raise()),
        (["an_existing_target_id"], [1], does_not_raise()),
        (["an_existing_target_id"], [2], does_not_raise()),
        (["bogus_target_id"], [1], pytest.raises(ValueError, match="references a target")),
        (["an_existing_target_id", "bogus_target_id"], [1,1], pytest.raises(ValueError, match="references a target")),
        (["an_existing_target_id"], [0], pytest.raises(ValueError, match="provides no transforms")),
    ]
)
def test_write_session_with_invalid_fit_results(
    example_database: Database,
    example_subject: Subject,
    target_ids: List[str],
    numbers_of_transforms: List[int],
    expectation,
):
    """Verify that write_session complains appropriately about invalid virtual fit results"""
    rng = np.random.default_rng()
    session = Session(
        id="unique_id_2764592837465",
        subject_id=example_subject.id,
        targets=[Point(id="an_existing_target_id")],
        virtual_fit_results={
            target_id : [
                (True, ArrayTransform(matrix=rng.random(size=(4,4)),units="mm"))
                for _ in range(num_transforms)
            ]
            for target_id, num_transforms in zip(target_ids, numbers_of_transforms)
        },
    )
    with expectation:
        example_database.write_session(example_subject, session)

def test_session_arrays_read_correctly(example_session:Session):
    """Verify that session data that is supposed to be array type is actually array type after reading from json"""
    assert isinstance(example_session.array_transform.matrix, np.ndarray)
    for _, list_of_transforms in example_session.virtual_fit_results.items():
        for _ , array_transform in list_of_transforms:
            assert isinstance(array_transform.matrix, np.ndarray)

    for tt_result in example_session.transducer_tracking_results:
        assert isinstance(tt_result.transducer_to_volume_transform.matrix, np.ndarray)
        assert isinstance(tt_result.photoscan_to_volume_transform.matrix, np.ndarray)

@pytest.mark.parametrize("compact_representation", [True, False])
def test_serialize_deserialize_session(example_session : Session, compact_representation:bool):
    reconstructed_session = example_session.from_json(example_session.to_json(compact_representation))
    assert dataclasses_are_equal(example_session, reconstructed_session)

def test_session_to_file(example_session : Session, tmp_path:Path):
    save_path = tmp_path/"this_is_a_session.json"
    example_session.to_file(save_path)
    reloaded_session = Session.from_file(save_path)
    assert dataclasses_are_equal(example_session, reloaded_session)

def test_get_solutions_filename(example_database:Database):
    solutions_filepath = example_database.get_solutions_filename("example_subject", "example_session")
    assert solutions_filepath.exists()
    assert solutions_filepath.is_file()
    assert solutions_filepath.name == "solutions.json"

def test_get_solution_filepath(example_database:Database):
    solutions_dir = example_database.get_solution_filepath("example_subject", "example_session", "example_solution")
    assert solutions_dir.exists()
    assert solutions_dir.is_file()
    assert solutions_dir.name == "example_solution.json"

def test_get_solution_ids(example_database:Database, caplog):
    # verify that solution ids are loaded correctly
    solution_ids = example_database.get_solution_ids("example_subject", "example_session")
    assert len(solution_ids) == 1
    assert solution_ids[0] == "example_solution"

    # verify that warning is printed and empty list returned when there is no solutions file
    solutions_filepath = example_database.get_solutions_filename("example_subject", "example_session")
    solutions_filepath.unlink() # Delete file
    with caplog.at_level(logging.WARNING):
        solution_ids = example_database.get_solution_ids("example_subject", "example_session")
        assert "Solutions file not found" in caplog.text
    assert len(solution_ids) == 0

def test_get_volume_info(example_database:Database, tmp_path:Path):
    subject = "example_subject"
    volume_id = "example_volume"
    volume_info = example_database.get_volume_info(subject, volume_id)
    assert(volume_info["id"] == "example_volume")
    assert(volume_info["name"] == "EXAMPLE_VOLUME")
    assert(Path(volume_info["data_abspath"]) == \
                        tmp_path/"example_db/subjects/example_subject/volumes/example_volume/example_volume.nii")

def test_get_volume_ids(example_database:Database):
    assert(example_database.get_volume_ids("example_subject") == ["example_volume"])

def test_write_volume_ids(example_database:Database):
    example_database.write_volume_ids("example_subject", ["example_volume", "example_volume_2"])
    assert(example_database.get_volume_ids("example_subject") == ["example_volume", "example_volume_2"])

def test_get_volume_dir(example_database:Database, tmp_path:Path):
    subject_id = "example_subject"
    volume_id = "example_volume"
    assert(example_database.get_volume_dir(subject_id, volume_id) == \
                        tmp_path/f'example_db/subjects/{subject_id}/volumes/{volume_id}')

def test_write_volume(example_database:Database, tmp_path:Path):
    subject_id = "example_subject"
    volume_id = "example_volume_2"
    volume_name = "EXAMPLE_VOLUME_2"
    volume_data_path = Path(tmp_path/'test_db_files/example_volume_2.nii')
    volume_data_path.parent.mkdir(parents=True, exist_ok=True)
    volume_data_path.touch()
    example_database.write_volume(subject_id, volume_id, volume_name, volume_data_path)
    assert(example_database.get_volume_ids("example_subject") == ["example_volume", "example_volume_2"])

    volume_filepath = example_database.get_volume_metadata_filepath("example_subject", "example_volume_2")
    assert(volume_filepath.name == "example_volume_2.json")
    assert((volume_filepath.parent/"example_volume_2.nii").exists())

    # When writing to a new subject
    subject = Subject(id="bleh",name="Deb Jectson")
    example_database.write_subject(subject, on_conflict=OnConflictOpts.OVERWRITE)
    example_database.write_volume(subject.id, volume_id, volume_name, volume_data_path)

    assert(example_database.get_volume_ids("bleh") == ["example_volume_2"])

    volume_filepath = example_database.get_volume_metadata_filepath("bleh", "example_volume_2")
    assert(volume_filepath.name == "example_volume_2.json")
    assert((volume_filepath.parent/"example_volume_2.nii").exists())

def test_load_solution(example_database:Database, example_session:Session):
    with pytest.raises(FileNotFoundError,match="Solution file not found"):
        example_database.load_solution(example_session, "bogus_solution_id")

    example_solution = example_database.load_solution(example_session, "example_solution")
    assert example_solution.name == "Example Solution"
    assert "p_min" in example_solution.simulation_result.data_vars # ensure the xarray dataset got loaded too

    # ensure the simulation and beamform data was loaded for all foci
    assert len(example_solution.simulation_result['focal_point_index']) == len(example_solution.foci)
    assert example_solution.delays.shape[0] == len(example_solution.foci)
    assert example_solution.apodizations.shape[0] == len(example_solution.foci)

def test_write_solution(example_database:Database, example_session:Session):
    solution = Solution(name="bleh", id='new_solution')

    # This solution is not initially in the list of solution IDs
    assert solution.id not in example_database.get_solution_ids(example_session.subject_id, example_session.id)

    # Can add a new solution, and it loads back in correctly.
    example_database.write_solution(example_session, solution)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert dataclasses_are_equal(reloaded_solution,solution)

    # The new solution has now been added to the list of solution IDs
    assert solution.id in example_database.get_solution_ids(example_session.subject_id, example_session.id)

    # Error raised when the solution already exists
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.ERROR)

    # Skip option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.SKIP)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "bleh"

    # Overwrite option
    solution.name = "new_name"
    example_database.write_solution(example_session, solution, on_conflict=OnConflictOpts.OVERWRITE)
    reloaded_solution = example_database.load_solution(example_session, solution.id)
    assert reloaded_solution.name == "new_name"

def test_write_solution_new_session(example_database:Database, example_subject:Subject):
    """Writing a solution should be possible in a newly created session"""
    session = Session(name="bleh", id='a_session',subject_id=example_subject.id)
    solution = Solution(name="bleh", id='new_solution')
    example_database.write_session(example_subject, session)
    example_database.write_solution(session, solution)

def test_get_photoscan_absolute_filepaths_info(example_database:Database):
    subject_id = "example_subject"
    session_id = "example_session"
    photoscan_id = "example_photoscan"
    photoscan_info = example_database.get_photoscan_absolute_filepaths_info(subject_id, session_id, photoscan_id)
    assert(photoscan_info["id"] == "example_photoscan")
    assert(photoscan_info["name"] == "ExamplePhotoscan")
    assert(Path(photoscan_info["model_abspath"]).exists())
    assert(Path(photoscan_info["texture_abspath"]).exists())

def test_get_photoscan_ids(example_database:Database):
    assert(example_database.get_photoscan_ids("example_subject", "example_session") == ["example_photoscan"])

def test_write_photoscan(example_database:Database, example_session: Session, tmp_path:Path):
    model_data_path = Path(tmp_path/"test_db_files/example_photoscan_2.obj")
    model_data_path.parent.mkdir(parents=True, exist_ok=True)
    model_data_path.touch()
    texture_data_path = Path(tmp_path/"test_db_files/example_photoscan_texture_2.exr")
    texture_data_path.parent.mkdir(parents=True, exist_ok=True)
    texture_data_path.touch()
    mtl_data_path = Path(tmp_path/"test_db_files/example_photoscan.mtl")
    mtl_data_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_data_path.touch()

    photoscan = Photoscan(id = "example_photoscan_2", name =  "EXAMPLE_PHOTOSCAN_2")
    example_database.write_photoscan(example_session.subject_id, example_session.id, photoscan,
                                     model_data_filepath= model_data_path,
                                     texture_data_filepath=texture_data_path,
                                     mtl_data_filepath=mtl_data_path)
    assert(len(example_database.get_photoscan_ids("example_subject", "example_session")) == 2)
    assert("example_photoscan" in example_database.get_photoscan_ids("example_subject", "example_session"))
    assert("example_photoscan_2" in example_database.get_photoscan_ids("example_subject", "example_session"))

    photoscan_filepath = example_database.get_photoscan_metadata_filepath("example_subject","example_session","example_photoscan_2")
    assert(photoscan_filepath.name == "example_photoscan_2.json")
    assert((photoscan_filepath.parent/"example_photoscan_2.obj").exists())
    assert((photoscan_filepath.parent/"example_photoscan_texture_2.exr").exists())
    assert((photoscan_filepath.parent/"example_photoscan.mtl").exists())

    # When writing to a new subject and session
    subject = Subject(id="bleh_photoscan_test",name="Deb Jectson")
    example_database.write_subject(subject)
    session = Session(id = "bleh_session", subject_id=subject.id, name = "Bleh_Session")
    example_database.write_session(subject, session)
    with pytest.raises(ValueError, match = "file associated with photoscan"):
        example_database.write_photoscan(session.subject_id, session.id, photoscan)

    example_database.write_photoscan(session.subject_id, session.id, photoscan,
                                     model_data_path,
                                     texture_data_path,
                                     mtl_data_path)

    assert(example_database.get_photoscan_ids(subject.id,session.id) == ["example_photoscan_2"])
    photoscan_filepath = example_database.get_photoscan_metadata_filepath(subject.id, session.id, "example_photoscan_2")
    assert(photoscan_filepath.name == "example_photoscan_2.json")
    assert((photoscan_filepath.parent/"example_photoscan_2.obj").exists())
    assert((photoscan_filepath.parent/"example_photoscan_texture_2.exr").exists())

    # Test not existent filepath
    bogus_texture_file = Path(tmp_path/"test_db_files/bogus_photoscan.exr")
    photoscan.texture_abspath = bogus_texture_file
    with pytest.raises(FileNotFoundError, match="does not exist"):
        example_database.write_photoscan(example_session.subject_id, example_session.id, photoscan, model_data_path, bogus_texture_file, on_conflict=OnConflictOpts.OVERWRITE)

def test_load_photoscan(example_database:Database, example_session:Session):
    with pytest.raises(FileNotFoundError,match="Photoscan file not found"):
        example_database.load_photoscan(example_session.subject_id, example_session.id, "bogus_photoscan_id")

    example_photoscan = example_database.load_photoscan(example_session.subject_id, example_session.id, "example_photoscan")
    assert example_photoscan.name == "ExamplePhotoscan"

    example_photoscan, (model_data, texture_data) = example_database.load_photoscan(example_session.subject_id, example_session.id, "example_photoscan", load_data=True)
    assert model_data is not None
    assert texture_data is not None
    assert isinstance(model_data, vtkPolyData)
    assert isinstance(texture_data,vtkImageData)

def test_session_created_date():
    """Test that created date is recent when a session is created."""
    tolerance = timedelta(seconds=2)  # Allow for minor timing discrepancies

    session = Session()
    now = datetime.now()
    assert(now - tolerance <= session.date_created <= now + tolerance)

def test_session_date_modified_updates_on_write(example_database:Database, example_subject:Subject):
    """Test that the modified time updates when a session file is written."""
    tolerance = timedelta(seconds=2)  # Allow for minor timing discrepancies

    # Mocking time so testing only passes simulated time, not real time
    with patch('openlifu.db.session.datetime') as derptime:
        session = Session(name="qwerty", id='aoeuidhtns', subject_id=example_subject.id)
        initial_modified_time = session.date_modified

        # Update the mock to return a new time
        updated_time = datetime.now() + timedelta(seconds=1e6)
        derptime.now.return_value = updated_time
        example_database.write_session(example_subject, session)

        # Assert the modified time was updated
        assert session.date_modified - tolerance <= updated_time <= session.date_modified + tolerance
        assert session.date_modified > initial_modified_time - tolerance

def test_get_transducer_ids(example_database:Database):
    assert("example_transducer" in example_database.get_transducer_ids())

def test_write_transducer_nodata(example_database:Database, example_transducer: Transducer):
    example_transducer.id = "example_transducer_2"

    example_database.write_transducer(example_transducer)
    assert(len(example_database.get_transducer_ids()) == 4)
    assert("example_transducer" in example_database.get_transducer_ids())
    assert("example_transducer_2" in example_database.get_transducer_ids())

    transducer_filepath = example_database.get_transducer_filename("example_transducer_2")
    assert(transducer_filepath.name == "example_transducer_2.json")

def test_write_transducer(example_database:Database, example_transducer: Transducer, tmp_path:Path):
    example_transducer.id = "example_transducer_2"
    registration_surface_path = Path(tmp_path/"test_db_files/example_registration_surface.obj")
    registration_surface_path.parent.mkdir(parents=True, exist_ok=True)
    registration_surface_path.touch()
    transducer_body_path = Path(tmp_path/"test_db_files/example_transducer_body.obj")
    transducer_body_path.parent.mkdir(parents=True, exist_ok=True)
    transducer_body_path.touch()

    example_database.write_transducer(example_transducer, registration_surface_path, transducer_body_path)
    transducer_filepath = example_database.get_transducer_filename("example_transducer_2")
    assert(transducer_filepath.name == "example_transducer_2.json")
    transducer_filepaths = example_database.get_transducer_absolute_filepaths("example_transducer_2")
    assert(transducer_filepaths["id"] == "example_transducer_2")
    assert(transducer_filepaths["name"] == "Example Transducer")
    assert(Path(transducer_filepaths["registration_surface_abspath"]).exists())
    assert(Path(transducer_filepaths["transducer_body_abspath"]).exists())

    # Test not existent filepath
    bogus_body_file = Path(tmp_path/"test_db_files/bogus_transducer_body.obj")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        example_database.write_transducer(example_transducer, registration_surface_path, bogus_body_file, on_conflict=OnConflictOpts.OVERWRITE)

    # Test when previously associated data files are missing
    example_transducer.registration_surface_filename = "bogus_transducer_model.obj"
    with pytest.raises(ValueError, match="file associated with transducer"):
        example_database.write_transducer(example_transducer, on_conflict=OnConflictOpts.OVERWRITE)

@pytest.mark.parametrize("registration_surface_path", [None, "test_db_files/example_registration_surface.obj"])
@pytest.mark.parametrize("transducer_body_path", [None, "test_db_files/example_transducer_body.obj"])
def test_get_transducer_absolute_filepaths(example_database, tmp_path: Path, registration_surface_path: str | None, transducer_body_path: str | None):
    transducer = Transducer(id="transducer_for_test_get_transducer_absolute_filepaths")

    registration_surface = Path(tmp_path / registration_surface_path) if registration_surface_path else None
    transducer_body = Path(tmp_path / transducer_body_path) if transducer_body_path else None

    if registration_surface:
        registration_surface.parent.mkdir(parents=True, exist_ok=True)
        registration_surface.touch()
    if transducer_body:
        transducer_body.parent.mkdir(parents=True, exist_ok=True)
        transducer_body.touch()

    example_database.write_transducer(
        transducer=transducer,
        registration_surface_model_filepath=registration_surface,
        transducer_body_model_filepath=transducer_body,
    )

    absolute_file_paths = example_database.get_transducer_absolute_filepaths("transducer_for_test_get_transducer_absolute_filepaths")

    if registration_surface:
        reconstructed_path = Path(absolute_file_paths["registration_surface_abspath"])
        assert reconstructed_path.exists()
        assert reconstructed_path.name == registration_surface.name
    else:
        assert absolute_file_paths["registration_surface_abspath"] is None

    if transducer_body:
        reconstructed_path = Path(absolute_file_paths["transducer_body_abspath"])
        assert reconstructed_path.exists()
        assert reconstructed_path.name == transducer_body.name
    else:
        assert absolute_file_paths["transducer_body_abspath"] is None

def test_write_volume_dicom(example_database: Database):
    """Test writing a volume from DICOM file - conversion to NIfTI and storage"""
    subject_id = "example_subject"
    volume_id = "test_dicom_volume"
    volume_name = "TEST_DICOM_VOLUME"

    test_dicom_file = Path(__file__).parent / "resources" / "CT_small.dcm"
    assert test_dicom_file.exists(), "CT_small.dcm test file should exist"

    example_database.write_volume(subject_id, volume_id, volume_name, test_dicom_file)

    volume_ids = example_database.get_volume_ids(subject_id)
    assert volume_id in volume_ids

    volume_metadata_filepath = example_database.get_volume_metadata_filepath(subject_id, volume_id)
    assert volume_metadata_filepath.exists()
    assert volume_metadata_filepath.name == f"{volume_id}.json"

    # verify stored as nifti not dicom
    volume_info = example_database.get_volume_info(subject_id, volume_id)
    stored_file = Path(volume_info["data_abspath"])
    assert stored_file.exists()
    assert stored_file.suffix in [".nii", ".gz"]  # .nii.gz suffix
    assert not is_dicom_file_or_directory(stored_file)

def test_write_volume_dicom_directory(example_database: Database):
    """Test writing a volume from DICOM directory (multi-slice series) - conversion to NIfTI and storage"""
    subject_id = "example_subject"
    volume_id = "test_dicom_series_volume"
    volume_name = "TEST_DICOM_SERIES_VOLUME"

    test_dicom_dir = Path(__file__).parent / "resources" / "dicom_series"
    assert test_dicom_dir.exists(), "dicom_series directory should exist"
    assert test_dicom_dir.is_dir()
    assert len(list(test_dicom_dir.iterdir())) > 0, "dicom_series should contain files"

    example_database.write_volume(subject_id, volume_id, volume_name, test_dicom_dir)

    volume_ids = example_database.get_volume_ids(subject_id)
    assert volume_id in volume_ids

    volume_metadata_filepath = example_database.get_volume_metadata_filepath(subject_id, volume_id)
    assert volume_metadata_filepath.exists()
    assert volume_metadata_filepath.name == f"{volume_id}.json"

    # verify stored as nifti not dicom
    volume_info = example_database.get_volume_info(subject_id, volume_id)
    stored_file = Path(volume_info["data_abspath"])
    assert stored_file.exists()
    assert stored_file.suffix in [".nii", ".gz"]  # .nii.gz suffix
    assert not is_dicom_file_or_directory(stored_file)

def test_write_volume_empty_directory(example_database: Database, tmp_path: Path):
    subject_id = "example_subject"
    volume_id = "test_empty_dir_volume"
    volume_name = "TEST_EMPTY_DIR_VOLUME"

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="directory without DICOM files"):
        example_database.write_volume(subject_id, volume_id, volume_name, empty_dir)
