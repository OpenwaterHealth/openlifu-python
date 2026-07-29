"""Tests for the split-session data model: Plan, PlanningSession, SonicationSession.

These types replace the omnibus ``openlifu.db.Session`` in the split-session refactor
(see SESSION_SPLIT_DESIGN.md in the SlicerOpenLIFU repository). They are pure
dataclasses with manual to_dict/from_dict/to_json/from_json/to_file/from_file
serialization, mirroring the pattern used by :class:`~openlifu.db.Session` and
:class:`~openlifu.plan.Protocol`.

Database read/write helpers are covered separately.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from helpers import dataclasses_are_equal

from openlifu import db
from openlifu.db import Plan, PlanningSession, SonicationSession
from openlifu.db.session import (
    PhotoscanRegistration,
    SolutionInfo,
    TransducerTrackingResult,
)
from openlifu.geo.point import Point
from openlifu.geo.transforms import ArrayTransform

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
