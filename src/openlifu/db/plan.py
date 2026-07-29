from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List

import numpy as np

from openlifu.db.session import SolutionInfo
from openlifu.geo.point import Point
from openlifu.geo.transforms import ArrayTransform
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.json import PYFUSEncoder
from openlifu.util.strings import sanitize


@dataclass
class Plan:
    """Immutable finalized treatment plan produced by a :class:`PlanningSession`.

    A ``Plan`` pins down: which target, which transducer at which pose (the virtual-fit
    pose that was approved on the parent PlanningSession), and against which protocol.
    Optionally carries :class:`SolutionInfo` references to pre-solutions computed at
    planning time so the sonication operator can review expected pressures before
    treatment.

    A ``Plan`` is frozen once written. Editing means "create a new Plan from an updated
    PlanningSession". A single PlanningSession may finalize multiple Plans over time.

    A :class:`SonicationSession` references a ``Plan`` by ``id`` (frozen input); the
    Plan's ``target``, ``volume_id``, ``protocol_id``, ``transducer_id``, and
    ``array_transform`` are the reference "what we're trying to hit" for the sonication.
    """

    id: Annotated[str | None, OpenLIFUFieldData("Plan ID", "ID of this plan")] = None
    """ID of this plan"""

    name: Annotated[str | None, OpenLIFUFieldData("Plan name", "Plan name")] = None
    """Plan name"""

    subject_id: Annotated[str | None, OpenLIFUFieldData("Subject ID", "ID of the parent subject of this plan")] = None
    """ID of the parent subject of this plan"""

    volume_id: Annotated[str | None, OpenLIFUFieldData("Volume ID", "ID of the subject volume this plan was finalized against")] = None
    """ID of the subject volume this plan was finalized against"""

    protocol_id: Annotated[str | None, OpenLIFUFieldData("Protocol ID", "ID of the protocol this plan was finalized against")] = None
    """ID of the protocol this plan was finalized against"""

    transducer_id: Annotated[str | None, OpenLIFUFieldData("Transducer ID", "ID of the transducer this plan was finalized against")] = None
    """ID of the transducer this plan was finalized against"""

    target: Annotated[Point | None, OpenLIFUFieldData("Target", "The Point target this plan committed to sonicating.")] = None
    """The :class:`Point` target this plan committed to sonicating."""

    array_transform: Annotated[ArrayTransform, OpenLIFUFieldData("Array transform", "The transducer affine transform matrix with units representing the approved virtual-fit pose this plan committed to.")] = field(default_factory=lambda: ArrayTransform(np.eye(4), "mm"))
    """The transducer affine transform matrix with units representing the approved
    virtual-fit pose this plan committed to."""

    pre_solutions: Annotated[List[SolutionInfo], OpenLIFUFieldData("Pre-solutions", "SolutionInfo references to pre-solutions computed at planning time. The actual Solution files live at subject scope (``subjects/{sid}/solutions/``) and are shared with the parent PlanningSession's pre_solutions list. Empty if the user finalized the plan without computing pre-solutions.")] = field(default_factory=list)
    """:class:`SolutionInfo` references to pre-solutions computed at planning time.

    The actual :class:`~openlifu.plan.Solution` files live at subject scope
    (``subjects/{sid}/solutions/``) and are shared with the parent
    :class:`PlanningSession`'s ``pre_solutions`` list. Empty if the user finalized the
    plan without computing pre-solutions.
    """

    parent_planning_session_id: Annotated[str | None, OpenLIFUFieldData("Parent planning session ID", "ID of the PlanningSession that finalized this Plan. Provenance only; a Plan does not depend on its parent still existing.")] = None
    """ID of the PlanningSession that finalized this Plan. Provenance only; a Plan does
    not depend on its parent still existing."""

    date_created: Annotated[datetime, OpenLIFUFieldData("Date created", "Date the plan was finalized")] = field(default_factory=datetime.now)
    """Date the plan was finalized"""

    notes: Annotated[str, OpenLIFUFieldData("Plan notes", "Free-form notes recorded at finalization time.")] = ""
    """Free-form notes recorded at finalization time."""

    attrs: Annotated[dict, OpenLIFUFieldData("Custom attributes", "Dictionary of additional custom attributes to save with the plan")] = field(default_factory=dict)
    """Dictionary of additional custom attributes to save with the plan"""

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "plan"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id

    @staticmethod
    def from_file(filename) -> Plan:
        """Load a Plan from a JSON file."""
        with open(filename) as f:
            return Plan.from_dict(json.load(f))

    @staticmethod
    def from_dict(d: Dict) -> Plan:
        """Reconstruct a Plan from its dictionary representation."""
        d = dict(d)  # shallow copy; we mutate below
        if "date_created" in d and isinstance(d["date_created"], str):
            d["date_created"] = datetime.fromisoformat(d["date_created"])
        if "array_transform" in d and isinstance(d["array_transform"], dict):
            d["array_transform"] = ArrayTransform.from_dict(d["array_transform"])
        if "target" in d and isinstance(d["target"], dict):
            d["target"] = Point.from_dict(d["target"])
        if "pre_solutions" in d:
            d["pre_solutions"] = [
                s if isinstance(s, SolutionInfo) else SolutionInfo(**s)
                for s in d["pre_solutions"]
            ]
        return Plan(**d)

    def to_dict(self) -> Dict:
        """Serialize the Plan to a dictionary."""
        d = copy.deepcopy(self.__dict__)
        d["date_created"] = d["date_created"].isoformat()
        d["array_transform"] = asdict(d["array_transform"])
        if d["target"] is not None:
            d["target"] = d["target"].to_dict()
        d["pre_solutions"] = [asdict(s) for s in d["pre_solutions"]]
        return d

    @staticmethod
    def from_json(json_string: str) -> Plan:
        """Load a Plan from a JSON string."""
        return Plan.from_dict(json.loads(json_string))

    def to_json(self, compact: bool) -> str:
        """Serialize a Plan to a JSON string.

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(",", ":"), cls=PYFUSEncoder)
        return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename) -> None:
        """Write the Plan to a JSON file, creating parent directories as needed."""
        Path(filename).parent.parent.mkdir(exist_ok=True)  # plans directory
        Path(filename).parent.mkdir(exist_ok=True)         # {plan_id} directory
        with open(filename, "w") as f:
            f.write(self.to_json(compact=False))
