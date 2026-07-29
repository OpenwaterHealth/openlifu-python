"""Tests for the split-session data model: Plan, PlanningSession, SonicationSession.

These types replace the omnibus ``openlifu.db.Session`` in the split-session refactor
(see SESSION_SPLIT_DESIGN.md in the SlicerOpenLIFU repository). They are pure
dataclasses with manual to_dict/from_dict/to_json/from_json/to_file/from_file
serialization, mirroring the pattern used by :class:`~openlifu.db.Session` and
:class:`~openlifu.plan.Protocol`.

Database read/write helpers are covered further below.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu import db
from openlifu.db import Plan, PlanningSession, SonicationSession
from openlifu.db.database import Database, OnConflictOpts
from openlifu.db.session import (
    PhotoscanRegistration,
    SolutionInfo,
    TransducerTrackingResult,
)
from openlifu.geo.point import Point
from openlifu.geo.transforms import ArrayTransform
from openlifu.nav.photoscan import Photoscan
from openlifu.plan import Solution


@pytest.fixture()
def example_database(tmp_path: Path) -> Database:
    """Example database in a temporary directory; appropriate for testing write operations."""
    shutil.copytree(Path(__file__).parent / "resources/example_db", tmp_path / "example_db")
    return Database(tmp_path / "example_db")

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def _example_solution_info(sid: str = "sol_a", source: str = "virtual_fit") -> SolutionInfo:
    return SolutionInfo(
        solution_id=sid,
        protocol_id="proto",
        target_id="tgt",
        transducer_id="xdc",
        transducer_transform_source=source,
        transducer_transform_source_id="tgt:0" if source == "virtual_fit" else "tt_result_x",
        approved=True,
        computed_at=datetime(2026, 7, 24, 12, 34, 56),
        array_transform=ArrayTransform(np.eye(4), "mm"),
    )


def _example_target(tid: str = "tgt") -> Point:
    return Point(id=tid, name="Example target", position=(0.0, 0.0, 0.0), dims=("R", "A", "S"), units="mm")


def test_plan_defaults_name_from_id():
    """A Plan with only an id gets its name copied from the id (parallels Session)."""
    plan = Plan(id="my_plan")
    assert plan.id == "my_plan"
    assert plan.name == "my_plan"


def test_plan_defaults_id_from_name():
    """A Plan with only a name gets a sanitized snake-case id."""
    plan = Plan(name="My Plan #1")
    assert plan.id is not None
    assert plan.name == "My Plan #1"


def test_plan_defaults_completely_bare():
    """A Plan constructed with no id or name gets a placeholder id."""
    plan = Plan()
    assert plan.id == "plan"
    assert plan.name == "plan"


def test_plan_roundtrips_through_dict():
    plan = Plan(
        id="plan_xyz",
        name="Plan XYZ",
        subject_id="subj",
        volume_id="vol",
        protocol_id="proto",
        transducer_id="xdc",
        target=_example_target(),
        array_transform=ArrayTransform(np.eye(4), "mm"),
        pre_solutions=[_example_solution_info("sol_a"), _example_solution_info("sol_b")],
        parent_planning_session_id="ps_xyz",
        notes="finalized after review",
    )
    reloaded = Plan.from_dict(plan.to_dict())
    assert dataclasses_are_equal(reloaded, plan)


def test_plan_roundtrips_through_json():
    plan = Plan(id="plan_json", subject_id="subj", target=_example_target())
    reloaded = Plan.from_json(plan.to_json(compact=False))
    assert dataclasses_are_equal(reloaded, plan)


def test_plan_roundtrips_through_file(tmp_path: Path):
    plan = Plan(id="plan_file", subject_id="subj", target=_example_target())
    filepath = tmp_path / "plans" / plan.id / f"{plan.id}.json"
    plan.to_file(filepath)
    assert filepath.exists()
    assert dataclasses_are_equal(Plan.from_file(filepath), plan)


def test_plan_pre_solutions_default_empty_list():
    plan = Plan(id="p")
    assert plan.pre_solutions == []


def test_plan_target_can_be_none():
    """A Plan without a target is a legitimate state (e.g. under construction)."""
    plan = Plan(id="p", target=None)
    assert plan.target is None
    reloaded = Plan.from_dict(plan.to_dict())
    assert reloaded.target is None


# ---------------------------------------------------------------------------
# PlanningSession
# ---------------------------------------------------------------------------

def test_planning_session_defaults():
    """A bare PlanningSession gets a placeholder id and matching name."""
    ps = PlanningSession()
    assert ps.id == "planning_session"
    assert ps.name == "planning_session"
    assert ps.targets == []
    assert ps.virtual_fit_results == {}
    assert ps.pre_solutions == []
    assert ps.finalized_plan_ids == []


def test_planning_session_roundtrips_through_dict():
    ps = PlanningSession(
        id="ps_abc",
        name="Planning ABC",
        subject_id="subj",
        volume_id="vol",
        protocol_id="proto",
        transducer_id="xdc",
        targets=[_example_target("t1"), _example_target("t2")],
        virtual_fit_results={
            "t1": [
                (True, ArrayTransform(np.eye(4), "mm")),
                (False, ArrayTransform(np.eye(4) * 2, "mm")),
            ],
        },
        pre_solutions=[_example_solution_info("s1")],
        finalized_plan_ids=["plan_1", "plan_2"],
    )
    reloaded = PlanningSession.from_dict(ps.to_dict())
    assert dataclasses_are_equal(reloaded, ps)


def test_planning_session_roundtrips_through_file(tmp_path: Path):
    ps = PlanningSession(id="ps_file", subject_id="subj")
    filepath = tmp_path / "planning_sessions" / ps.id / f"{ps.id}.json"
    ps.to_file(filepath)
    assert filepath.exists()
    assert dataclasses_are_equal(PlanningSession.from_file(filepath), ps)


def test_planning_session_update_modified_time_moves_forward():
    ps = PlanningSession(id="ps")
    before = ps.date_modified
    ps.update_modified_time(datetime(2027, 1, 1))
    assert ps.date_modified == datetime(2027, 1, 1)
    assert ps.date_modified != before


def test_planning_session_accepts_single_target_as_point():
    """Convenience: passing a single Point in targets is normalized to a list."""
    target = _example_target()
    ps = PlanningSession(id="ps", targets=target)
    assert ps.targets == [target]


# ---------------------------------------------------------------------------
# SonicationSession
# ---------------------------------------------------------------------------

def test_sonication_session_defaults():
    ss = SonicationSession()
    assert ss.id == "sonication_session"
    assert ss.name == "sonication_session"
    assert ss.plan_id is None
    assert ss.photoscan_ids == []
    assert ss.photoscan_registrations == []
    assert ss.transducer_tracking_results == []
    assert ss.solution is None
    assert ss.run_ids == []


def test_sonication_session_no_volume_id_field():
    """SonicationSession does NOT have a volume_id field. Volume comes from the Plan.

    Guards against reintroducing the field, which was explicitly decided against in
    the session-split design (rationale: a different volume means a different Plan).
    """
    ss = SonicationSession()
    assert not hasattr(ss, "volume_id")


def test_sonication_session_no_target_field():
    """SonicationSession does NOT have a target field. Target comes from the Plan."""
    ss = SonicationSession()
    assert not hasattr(ss, "target")
    assert not hasattr(ss, "targets")


def test_sonication_session_solution_is_optional_solutioninfo_not_list():
    """SonicationSession.solution is Optional[SolutionInfo], NOT a list.

    Multi-solution generation and selection is deliberately out of scope for the
    initial split-session refactor.
    """
    ss = SonicationSession(id="ss")
    ss.solution = _example_solution_info("sol_final")
    assert isinstance(ss.solution, SolutionInfo)
    # And it does NOT have a `solutions` list field masquerading in parallel:
    assert not hasattr(ss, "solutions")


def test_sonication_session_roundtrips_through_dict():
    reg = PhotoscanRegistration(
        photoscan_id="pscan1",
        transform=ArrayTransform(np.eye(4), "mm"),
        approval=True,
        id="pscan1__pr__00",
    )
    tt = TransducerTrackingResult(
        photoscan_id="pscan1",
        transducer_to_volume_transform=ArrayTransform(np.eye(4), "mm"),
        photoscan_registration_id="pscan1__pr__00",
        approval=True,
        id="tt_result_1",
        target_id="tgt",
    )
    ss = SonicationSession(
        id="ss_abc",
        name="Sonication ABC",
        subject_id="subj",
        plan_id="plan_xyz",
        photoscan_ids=["pscan1"],
        photoscan_registrations=[reg],
        transducer_tracking_results=[tt],
        solution=_example_solution_info("sol_final", source="localization"),
        run_ids=["run1", "run2"],
    )
    reloaded = SonicationSession.from_dict(ss.to_dict())
    assert dataclasses_are_equal(reloaded, ss)


def test_sonication_session_roundtrips_through_file(tmp_path: Path):
    ss = SonicationSession(id="ss_file", subject_id="subj", plan_id="plan_xyz")
    filepath = tmp_path / "sonication_sessions" / ss.id / f"{ss.id}.json"
    ss.to_file(filepath)
    assert filepath.exists()
    assert dataclasses_are_equal(SonicationSession.from_file(filepath), ss)


def test_sonication_session_solution_none_roundtrips():
    ss = SonicationSession(id="ss", solution=None)
    reloaded = SonicationSession.from_dict(ss.to_dict())
    assert reloaded.solution is None


def test_sonication_session_update_modified_time_moves_forward():
    ss = SonicationSession(id="ss")
    before = ss.date_modified
    ss.update_modified_time(datetime(2027, 1, 1))
    assert ss.date_modified == datetime(2027, 1, 1)
    assert ss.date_modified != before


# ---------------------------------------------------------------------------
# Cross-type sanity
# ---------------------------------------------------------------------------

def test_plan_and_planningsession_share_solutioninfo_structure():
    """A SolutionInfo can be moved between a PlanningSession.pre_solutions list and a
    Plan.pre_solutions list without modification. This is the load-bearing invariant
    for finalize_plan()."""
    si = _example_solution_info("sol_shared")
    ps = PlanningSession(id="ps", pre_solutions=[si])
    plan = Plan(id="plan", subject_id="subj", pre_solutions=[si])

    ps_reloaded = PlanningSession.from_dict(ps.to_dict())
    plan_reloaded = Plan.from_dict(plan.to_dict())
    assert dataclasses_are_equal(ps_reloaded.pre_solutions[0], plan_reloaded.pre_solutions[0])


@pytest.mark.parametrize("cls", [Plan, PlanningSession, SonicationSession])
def test_new_types_are_exported_from_openlifu_db(cls):
    """Plan, PlanningSession, SonicationSession are all exported from ``openlifu.db``."""
    assert getattr(db, cls.__name__) is cls


# ---------------------------------------------------------------------------
# Database read/write for the new session types.
# ---------------------------------------------------------------------------

def _make_photoscan(pid: str, subject_dir: Path) -> tuple[Photoscan, Path, Path, Path]:
    """Create a Photoscan with three sidecar files ready to be written."""
    tmp_dir = subject_dir / f"__stage_{pid}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model = tmp_dir / f"{pid}.obj"
    texture = tmp_dir / f"{pid}.png"
    mtl = tmp_dir / f"{pid}.mtl"
    model.write_text("v 0 0 0\n")
    texture.write_bytes(b"\x89PNG\r\n\x1a\n")
    mtl.write_text("newmtl mat\n")
    photoscan = Photoscan(id=pid, name=pid, photoscan_approved=False)
    return photoscan, model, texture, mtl


# --- Plan --------------------------------------------------------------------

def test_write_load_delete_plan(example_database: Database):
    subject_id = "example_subject"
    plan = Plan(
        id="plan_a",
        name="Plan A",
        subject_id=subject_id,
        volume_id="example_volume",
        protocol_id="example_protocol",
        transducer_id="example_transducer",
        target=_example_target(),
    )
    assert plan.id not in example_database.get_plan_ids(subject_id)

    example_database.write_plan(subject_id, plan)
    reloaded = example_database.load_plan(subject_id, plan.id)
    assert dataclasses_are_equal(reloaded, plan)
    assert plan.id in example_database.get_plan_ids(subject_id)

    # ERROR on conflict
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_plan(subject_id, plan, on_conflict=OnConflictOpts.ERROR)

    # SKIP keeps original
    plan.name = "New Name"
    example_database.write_plan(subject_id, plan, on_conflict=OnConflictOpts.SKIP)
    assert example_database.load_plan(subject_id, plan.id).name == "Plan A"

    # OVERWRITE replaces it
    example_database.write_plan(subject_id, plan, on_conflict=OnConflictOpts.OVERWRITE)
    assert example_database.load_plan(subject_id, plan.id).name == "New Name"

    # Delete removes files and drops from index
    example_database.delete_plan(subject_id, plan.id)
    assert plan.id not in example_database.get_plan_ids(subject_id)
    assert not example_database.get_plan_dir(subject_id, plan.id).exists()

    # ERROR on delete of missing
    with pytest.raises(ValueError, match="does not exist"):
        example_database.delete_plan(subject_id, plan.id)

    # SKIP on delete of missing
    example_database.delete_plan(subject_id, plan.id, on_conflict=OnConflictOpts.SKIP)


def test_write_plan_rejects_mismatched_subject_id(example_database: Database):
    plan = Plan(id="plan_x", subject_id="wrong_subject", target=_example_target())
    with pytest.raises(ValueError, match="does not match"):
        example_database.write_plan("example_subject", plan)


def test_load_plan_missing(example_database: Database):
    with pytest.raises(FileNotFoundError, match="Plan file not found"):
        example_database.load_plan("example_subject", "bogus_plan_id")


# --- PlanningSession ---------------------------------------------------------

def test_write_load_delete_planning_session(example_database: Database):
    subject_id = "example_subject"
    ps = PlanningSession(
        id="ps_a",
        name="Planning A",
        subject_id=subject_id,
        volume_id="example_volume",
        protocol_id="example_protocol",
        transducer_id="example_transducer",
        targets=[_example_target("t1")],
        virtual_fit_results={
            "t1": [(True, ArrayTransform(np.eye(4), "mm"))],
        },
    )
    assert ps.id not in example_database.get_planning_session_ids(subject_id)

    example_database.write_planning_session(subject_id, ps)
    reloaded = example_database.load_planning_session(subject_id, ps.id)
    assert dataclasses_are_equal(reloaded.targets, ps.targets)
    assert dataclasses_are_equal(
        reloaded.virtual_fit_results["t1"][0][1], ps.virtual_fit_results["t1"][0][1],
    )
    assert reloaded.name == ps.name
    assert ps.id in example_database.get_planning_session_ids(subject_id)

    # ERROR on conflict
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_planning_session(subject_id, ps, on_conflict=OnConflictOpts.ERROR)

    # SKIP
    ps.name = "Renamed"
    example_database.write_planning_session(subject_id, ps, on_conflict=OnConflictOpts.SKIP)
    assert example_database.load_planning_session(subject_id, ps.id).name == "Planning A"

    # OVERWRITE
    example_database.write_planning_session(subject_id, ps, on_conflict=OnConflictOpts.OVERWRITE)
    assert example_database.load_planning_session(subject_id, ps.id).name == "Renamed"

    # Delete
    example_database.delete_planning_session(subject_id, ps.id)
    assert ps.id not in example_database.get_planning_session_ids(subject_id)
    assert not example_database.get_planning_session_dir(subject_id, ps.id).exists()

    with pytest.raises(ValueError, match="does not exist"):
        example_database.delete_planning_session(subject_id, ps.id)
    example_database.delete_planning_session(subject_id, ps.id, on_conflict=OnConflictOpts.SKIP)


def test_write_planning_session_rejects_vf_referencing_unknown_target(example_database: Database):
    subject_id = "example_subject"
    ps = PlanningSession(
        id="ps_bad_vf",
        subject_id=subject_id,
        targets=[_example_target("t1")],
        virtual_fit_results={"t_missing": [(True, ArrayTransform(np.eye(4), "mm"))]},
    )
    with pytest.raises(ValueError, match="virtual_fit_results references target"):
        example_database.write_planning_session(subject_id, ps)


def test_write_planning_session_rejects_vf_with_no_transforms(example_database: Database):
    subject_id = "example_subject"
    ps = PlanningSession(
        id="ps_empty_vf",
        subject_id=subject_id,
        targets=[_example_target("t1")],
        virtual_fit_results={"t1": []},
    )
    with pytest.raises(ValueError, match="provides no"):
        example_database.write_planning_session(subject_id, ps)


def test_write_planning_session_rejects_mismatched_subject_id(example_database: Database):
    ps = PlanningSession(id="ps_x", subject_id="wrong")
    with pytest.raises(ValueError, match="does not match"):
        example_database.write_planning_session("example_subject", ps)


def test_load_planning_session_missing(example_database: Database):
    with pytest.raises(FileNotFoundError, match="PlanningSession file not found"):
        example_database.load_planning_session("example_subject", "bogus_id")


# --- SonicationSession -------------------------------------------------------

def test_write_load_delete_sonication_session(example_database: Database):
    subject_id = "example_subject"
    reg = PhotoscanRegistration(
        photoscan_id="pscan1",
        transform=ArrayTransform(np.eye(4), "mm"),
        approval=True,
        id="pscan1__pr__00",
    )
    tt = TransducerTrackingResult(
        photoscan_id="pscan1",
        transducer_to_volume_transform=ArrayTransform(np.eye(4), "mm"),
        photoscan_registration_id="pscan1__pr__00",
        approval=True,
        id="tt_1",
        target_id="tgt",
    )
    ss = SonicationSession(
        id="ss_a",
        name="Sonication A",
        subject_id=subject_id,
        plan_id="plan_a",
        photoscan_ids=["pscan1"],
        photoscan_registrations=[reg],
        transducer_tracking_results=[tt],
        solution=_example_solution_info("sol_final", source="localization"),
    )
    assert ss.id not in example_database.get_sonication_session_ids(subject_id)

    example_database.write_sonication_session(subject_id, ss)
    reloaded = example_database.load_sonication_session(subject_id, ss.id)
    assert reloaded.plan_id == "plan_a"
    assert reloaded.photoscan_ids == ["pscan1"]
    assert dataclasses_are_equal(reloaded.transducer_tracking_results[0], tt)
    assert dataclasses_are_equal(reloaded.solution, ss.solution)
    assert ss.id in example_database.get_sonication_session_ids(subject_id)

    # ERROR
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_sonication_session(subject_id, ss, on_conflict=OnConflictOpts.ERROR)

    # SKIP
    ss.name = "Renamed"
    example_database.write_sonication_session(subject_id, ss, on_conflict=OnConflictOpts.SKIP)
    assert example_database.load_sonication_session(subject_id, ss.id).name == "Sonication A"

    # OVERWRITE
    example_database.write_sonication_session(subject_id, ss, on_conflict=OnConflictOpts.OVERWRITE)
    assert example_database.load_sonication_session(subject_id, ss.id).name == "Renamed"

    # Delete
    example_database.delete_sonication_session(subject_id, ss.id)
    assert ss.id not in example_database.get_sonication_session_ids(subject_id)
    assert not example_database.get_sonication_session_dir(subject_id, ss.id).exists()

    with pytest.raises(ValueError, match="does not exist"):
        example_database.delete_sonication_session(subject_id, ss.id)
    example_database.delete_sonication_session(subject_id, ss.id, on_conflict=OnConflictOpts.SKIP)


def test_write_sonication_session_rejects_mismatched_subject_id(example_database: Database):
    ss = SonicationSession(id="ss_x", subject_id="wrong", plan_id="plan_x")
    with pytest.raises(ValueError, match="does not match"):
        example_database.write_sonication_session("example_subject", ss)


def test_load_sonication_session_missing(example_database: Database):
    with pytest.raises(FileNotFoundError, match="SonicationSession file not found"):
        example_database.load_sonication_session("example_subject", "bogus_id")


# --- Subject-scoped Photoscan storage ----------------------------------------

def test_write_load_delete_photoscan_at_subject_scope(example_database: Database, tmp_path: Path):
    subject_id = "example_subject"
    photoscan, model, texture, mtl = _make_photoscan("subj_pscan", tmp_path)

    assert example_database.get_subject_photoscan_ids(subject_id) == []

    example_database.write_photoscan_at_subject_scope(
        subject_id, photoscan,
        model_data_filepath=str(model),
        texture_data_filepath=str(texture),
        mtl_data_filepath=str(mtl),
    )
    reloaded = example_database.load_photoscan_at_subject_scope(subject_id, photoscan.id)
    assert reloaded.id == photoscan.id
    assert reloaded.name == photoscan.name
    assert photoscan.id in example_database.get_subject_photoscan_ids(subject_id)

    # Files are on disk at subject scope, not under a session
    photoscan_dir = example_database.get_subject_photoscan_dir(subject_id, photoscan.id)
    assert (photoscan_dir / model.name).exists()
    assert (photoscan_dir / texture.name).exists()
    assert (photoscan_dir / mtl.name).exists()

    # get_photoscan_absolute_filepaths_info_at_subject_scope returns absolute paths
    info = example_database.get_photoscan_absolute_filepaths_info_at_subject_scope(
        subject_id, photoscan.id,
    )
    assert info["id"] == photoscan.id
    assert info["model_abspath"].exists()
    assert info["texture_abspath"].exists()
    assert info["mtl_abspath"].exists()

    # ERROR on conflict
    with pytest.raises(ValueError, match="already exists"):
        example_database.write_photoscan_at_subject_scope(
            subject_id, photoscan,
            model_data_filepath=str(model),
            on_conflict=OnConflictOpts.ERROR,
        )

    # SKIP keeps original
    photoscan.name = "renamed"
    example_database.write_photoscan_at_subject_scope(
        subject_id, photoscan, on_conflict=OnConflictOpts.SKIP,
    )
    assert example_database.load_photoscan_at_subject_scope(subject_id, photoscan.id).name == "subj_pscan"

    # OVERWRITE without new files reuses existing on-disk files
    example_database.write_photoscan_at_subject_scope(
        subject_id, photoscan, on_conflict=OnConflictOpts.OVERWRITE,
    )
    assert example_database.load_photoscan_at_subject_scope(subject_id, photoscan.id).name == "renamed"

    # Delete
    example_database.delete_photoscan_at_subject_scope(subject_id, photoscan.id)
    assert photoscan.id not in example_database.get_subject_photoscan_ids(subject_id)
    assert not photoscan_dir.exists()

    with pytest.raises(ValueError, match="does not exist"):
        example_database.delete_photoscan_at_subject_scope(subject_id, photoscan.id)
    example_database.delete_photoscan_at_subject_scope(
        subject_id, photoscan.id, on_conflict=OnConflictOpts.SKIP,
    )


def test_load_photoscan_at_subject_scope_missing(example_database: Database):
    with pytest.raises(FileNotFoundError, match="Photoscan file not found at subject scope"):
        example_database.load_photoscan_at_subject_scope("example_subject", "bogus_pscan")


def test_subject_scoped_photoscan_does_not_collide_with_session_scoped(
    example_database: Database, tmp_path: Path,
):
    """A subject-scoped photoscan and a legacy session-scoped photoscan with the same id coexist.

    Guards against the two storage paths accidentally sharing directory space.
    """
    subject_id = "example_subject"
    photoscan, model, texture, _ = _make_photoscan("shared_pscan", tmp_path)

    # Write under subject scope.
    example_database.write_photoscan_at_subject_scope(
        subject_id, photoscan,
        model_data_filepath=str(model),
        texture_data_filepath=str(texture),
    )
    subj_dir = example_database.get_subject_photoscan_dir(subject_id, photoscan.id)
    assert subj_dir.exists()

    # Legacy session-scoped path is completely disjoint.
    legacy_dir = example_database.get_photoscan_metadata_filepath(
        subject_id, "example_session", photoscan.id,
    ).parent
    assert legacy_dir != subj_dir


# --- Integration: cross-type references (finalize-plan sketch) ---------------

def test_planning_session_and_plan_reference_same_subject_scoped_solution(
    example_database: Database,
):
    """A single Solution at subject scope can be referenced by both a PlanningSession's
    pre_solutions list and a finalized Plan's pre_solutions list. Round-tripping through
    the database preserves the shared SolutionInfo on both sides.

    This is a sketch of the finalize_plan() invariant that will be implemented in a
    later commit: finalizing does NOT copy the underlying Solution -- it copies only
    the SolutionInfo reference.
    """
    subject_id = "example_subject"
    solution = Solution(name="shared", id="shared_sol")
    example_database.write_solution_at_subject_scope(subject_id, solution)

    si = _example_solution_info("shared_sol")

    ps = PlanningSession(
        id="ps_shared",
        subject_id=subject_id,
        targets=[_example_target()],
        pre_solutions=[si],
    )
    example_database.write_planning_session(subject_id, ps)

    plan = Plan(
        id="plan_shared",
        subject_id=subject_id,
        target=_example_target(),
        pre_solutions=[si],
        parent_planning_session_id=ps.id,
    )
    example_database.write_plan(subject_id, plan)

    ps_reloaded = example_database.load_planning_session(subject_id, ps.id)
    plan_reloaded = example_database.load_plan(subject_id, plan.id)

    # Both point at the same solution_id.
    assert ps_reloaded.pre_solutions[0].solution_id == "shared_sol"
    assert plan_reloaded.pre_solutions[0].solution_id == "shared_sol"

    # And the solution exists exactly once on disk.
    solution_ids = example_database.get_subject_solution_ids(subject_id)
    assert solution_ids.count("shared_sol") == 1
