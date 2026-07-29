from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List

from openlifu.db.session import (
    PhotoscanRegistration,
    SolutionInfo,
    TransducerTrackingResult,
)
from openlifu.geo.transforms import ArrayTransform
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.json import PYFUSEncoder
from openlifu.util.strings import sanitize


@dataclass
class SonicationSession:
    """At-treatment-time session.

    References a :class:`Plan` by id (immutable input); the Plan's ``target``,
    ``volume_id``, ``protocol_id``, ``transducer_id``, and ``array_transform`` are the
    reference "what we're trying to hit". The SonicationSession itself owns "what we
    actually did": photoscan registrations, transducer-tracking results, the final
    :class:`~openlifu.plan.Solution` reference, and the list of :class:`~openlifu.plan.Run`
    ids performed against this session.

    Unlike :class:`PlanningSession` there is at most one solution per SonicationSession
    (``solution: Optional[SolutionInfo]``, not a list). Recomputing replaces it in
    place. Multi-solution generation and selection is a plausible future feature but
    is deliberately out of scope for the initial split-session refactor.

    Photoscans are physically stored at subject scope
    (``subjects/{sid}/photoscans/{pid}/``) but logically owned by this session via
    :attr:`photoscan_ids`. Two SonicationSessions on the same subject each own their
    own list.
    """

    id: Annotated[str | None, OpenLIFUFieldData("Sonication session ID", "ID of this sonication session")] = None
    """ID of this sonication session"""

    name: Annotated[str | None, OpenLIFUFieldData("Sonication session name", "Sonication session name")] = None
    """Sonication session name"""

    subject_id: Annotated[str | None, OpenLIFUFieldData("Subject ID", "ID of the parent subject of this sonication session")] = None
    """ID of the parent subject of this sonication session"""

    plan_id: Annotated[str | None, OpenLIFUFieldData("Plan ID", "ID of the Plan this sonication session was launched from. Immutable frozen reference; volume/target/protocol/transducer/array_transform come from the Plan.")] = None
    """ID of the :class:`Plan` this sonication session was launched from. Immutable
    frozen reference; ``volume_id``, ``target``, ``protocol_id``, ``transducer_id``,
    and ``array_transform`` all come from the Plan."""

    date_created: Annotated[datetime, OpenLIFUFieldData("Date created", "Date of creation of the sonication session")] = field(default_factory=datetime.now)
    """Date of creation of the sonication session"""

    date_modified: Annotated[datetime, OpenLIFUFieldData("Date modified", "Date of last modification of the sonication session")] = field(default_factory=datetime.now)
    """Date of last modification of the sonication session"""

    photoscan_ids: Annotated[List[str], OpenLIFUFieldData("Photoscan IDs", "IDs of photoscans owned by this sonication session. Photoscan files live at subject scope; a photoscan_id in this list is the assertion that this session owns that photoscan.")] = field(default_factory=list)
    """IDs of photoscans owned by this sonication session.

    Photoscan files live at subject scope (``subjects/{sid}/photoscans/{pid}/``);
    a photoscan_id in this list is the assertion that this session owns that
    photoscan. Two SonicationSessions on the same subject each own their own list.
    """

    photoscan_registrations: Annotated[List[PhotoscanRegistration], OpenLIFUFieldData("Photoscan registrations", "List of photoscan-to-volume registrations stored on this sonication session. Each transducer tracking result references one of these registrations by ``photoscan_registration_id``.")] = field(default_factory=list)
    """List of photoscan-to-volume registrations stored on this sonication session.
    Each transducer tracking result references one of these registrations by
    ``photoscan_registration_id``."""

    transducer_tracking_results: Annotated[List[TransducerTrackingResult], OpenLIFUFieldData("Transducer tracking results", "List of transducer tracking results computed on this sonication session")] = field(default_factory=list)
    """List of transducer tracking results computed on this sonication session"""

    solution: Annotated[SolutionInfo | None, OpenLIFUFieldData("Solution", "SolutionInfo reference to the ONE final solution computed for this sonication session, or None. Actual Solution files live at subject scope. Recomputing replaces this in place; the run_ids history captures what was actually delivered under previous solutions.")] = None
    """:class:`SolutionInfo` reference to the ONE final solution computed for this
    sonication session, or ``None``.

    Actual :class:`~openlifu.plan.Solution` files live at subject scope
    (``subjects/{sid}/solutions/``). Recomputing replaces this in place; the
    :attr:`run_ids` history captures what was actually delivered under previous
    solutions. Multi-solution generation and selection is a plausible future
    feature but is deliberately out of scope for the initial split-session refactor.
    """

    run_ids: Annotated[List[str], OpenLIFUFieldData("Run IDs", "IDs of Runs performed against this sonication session, in chronological order. Actual Run JSON files live under the session's runs/ subdirectory.")] = field(default_factory=list)
    """IDs of :class:`~openlifu.plan.Run`\\ s performed against this sonication session,
    in chronological order. Actual Run JSON files live under the session's ``runs/``
    subdirectory."""

    attrs: Annotated[dict, OpenLIFUFieldData("Custom attributes", "Dictionary of additional custom attributes to save to the sonication session")] = field(default_factory=dict)
    """Dictionary of additional custom attributes to save to the sonication session"""

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "sonication_session"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id

    @staticmethod
    def from_file(filename) -> SonicationSession:
        """Load a SonicationSession from a JSON file."""
        with open(filename) as f:
            return SonicationSession.from_dict(json.load(f))

    @staticmethod
    def from_dict(d: Dict) -> SonicationSession:
        """Reconstruct a SonicationSession from its dictionary representation."""
        d = dict(d)  # shallow copy; we mutate below
        if "date_created" in d and isinstance(d["date_created"], str):
            d["date_created"] = datetime.fromisoformat(d["date_created"])
        if "date_modified" in d and isinstance(d["date_modified"], str):
            d["date_modified"] = datetime.fromisoformat(d["date_modified"])
        if "photoscan_registrations" in d:
            d["photoscan_registrations"] = [
                pr if isinstance(pr, PhotoscanRegistration) else PhotoscanRegistration(
                    photoscan_id=pr["photoscan_id"],
                    transform=ArrayTransform.from_dict(pr["transform"]),
                    approval=pr.get("approval", False),
                    id=pr.get("id"),
                )
                for pr in d["photoscan_registrations"]
            ]
        if "transducer_tracking_results" in d:
            d["transducer_tracking_results"] = [
                t if isinstance(t, TransducerTrackingResult) else TransducerTrackingResult(
                    photoscan_id=t["photoscan_id"],
                    transducer_to_volume_transform=ArrayTransform.from_dict(t["transducer_to_volume_transform"]),
                    photoscan_registration_id=t.get("photoscan_registration_id"),
                    approval=t.get("approval", False),
                    id=t.get("id"),
                    target_id=t.get("target_id"),
                )
                for t in d["transducer_tracking_results"]
            ]
        if "solution" in d and d["solution"] is not None and not isinstance(d["solution"], SolutionInfo):
            d["solution"] = SolutionInfo(**d["solution"])
        return SonicationSession(**d)

    def to_dict(self) -> Dict:
        """Serialize the SonicationSession to a dictionary."""
        d = copy.deepcopy(self.__dict__)
        d["date_created"] = d["date_created"].isoformat()
        d["date_modified"] = d["date_modified"].isoformat()
        d["photoscan_registrations"] = [asdict(pr) for pr in d["photoscan_registrations"]]
        d["transducer_tracking_results"] = [asdict(t) for t in d["transducer_tracking_results"]]
        d["solution"] = asdict(d["solution"]) if d["solution"] is not None else None
        return d

    @staticmethod
    def from_json(json_string: str) -> SonicationSession:
        """Load a SonicationSession from a JSON string."""
        return SonicationSession.from_dict(json.loads(json_string))

    def to_json(self, compact: bool) -> str:
        """Serialize a SonicationSession to a JSON string."""
        if compact:
            return json.dumps(self.to_dict(), separators=(",", ":"), cls=PYFUSEncoder)
        return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename) -> None:
        """Write the SonicationSession to a JSON file, creating parent directories as needed."""
        Path(filename).parent.parent.mkdir(exist_ok=True)  # sonication_sessions directory
        Path(filename).parent.mkdir(exist_ok=True)         # {sonication_session_id} directory
        with open(filename, "w") as f:
            f.write(self.to_json(compact=False))

    def update_modified_time(self, time: datetime | None = None) -> None:
        if time is None:
            time = datetime.now()
        self.date_modified = time
