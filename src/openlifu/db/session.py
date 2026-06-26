from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

import numpy as np

from openlifu.geo.point import Point
from openlifu.geo.transforms import ArrayTransform
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.json import PYFUSEncoder
from openlifu.util.strings import sanitize


@dataclass
class PhotoscanRegistration:
    """A registration of a photoscan model into the volume coordinate frame.

    A photoscan may have multiple registration attempts stored on the session; at most one is
    expected to be approved at a time, though this is not enforced at the dataclass level.
    Downstream transducer-tracking results refer back to a specific registration by ``id``.
    """

    photoscan_id: Annotated[str, OpenLIFUFieldData("Photoscan ID", "ID of the photoscan that this registration applies to")]
    """ID of the photoscan that this registration applies to"""

    transform: Annotated[ArrayTransform, OpenLIFUFieldData("Photoscan-to-volume transform", "Transform that registers the photoscan model to the volume's skin segmentation")]
    """Transform that registers the photoscan model to the volume's skin segmentation"""

    approval: Annotated[bool, OpenLIFUFieldData("Registration approved?", "Whether this photoscan registration has been approved by the user.")] = False
    """Whether this photoscan registration has been approved by the user."""

    id: Annotated[str | None, OpenLIFUFieldData("Registration ID", "Stable identifier for this photoscan registration, unique within the session. Survives reordering or deletion of other registrations so that downstream references (e.g. transducer tracking results) remain valid.")] = None
    """Stable identifier for this photoscan registration, unique within the session."""


@dataclass
class TransducerTrackingResult:
    """
    Class representing the results of running the transducer tracking
    algorithm.

    Each result registers a transducer pose against a volume in the context of a particular
    photoscan registration. The photoscan-to-volume transform that the tracking was performed
    against is stored separately as a :class:`PhotoscanRegistration` and referenced here by
    ``photoscan_registration_id``.
    """

    photoscan_id: Annotated[str, OpenLIFUFieldData("Photoscan ID", "ID of the photoscan object used for transducer tracking")]
    """ID of the photoscan object used for transducer tracking"""

    transducer_to_volume_transform: Annotated[ArrayTransform, OpenLIFUFieldData("Transducer to volume transform", "Transform output by transducer tracking algorithm to register the transducer surface to the volume")]
    """Transform output by transducer tracking algorithm to register the transducer surface to the volume"""

    photoscan_registration_id: Annotated[str | None, OpenLIFUFieldData("Photoscan registration ID", "ID of the PhotoscanRegistration this tracking result was computed against. May be None for legacy entries imported from sessions saved before the registration concept was split out.")] = None
    """ID of the :class:`PhotoscanRegistration` this tracking result was computed against."""

    approval: Annotated[bool, OpenLIFUFieldData("Tracking approved?", "Approval state of the transducer tracking result. `True` means the user has provided some kind of confirmation that the transform result agrees with reality.")] = False
    """Approval state of the transducer tracking result. ``True`` means the user has provided
    some kind of confirmation that the transform result agrees with reality."""

    id: Annotated[str | None, OpenLIFUFieldData("Result ID", "Stable identifier for this tracking result, unique within the session. Survives reordering or deletion of other results so that downstream references (e.g. solutions) remain valid.")] = None
    """Stable identifier for this tracking result, unique within the session. Survives reordering or
    deletion of other results so that downstream references (e.g. solutions) remain valid."""

    target_id: Annotated[str | None, OpenLIFUFieldData("Target ID", "ID of the openlifu Point target this tracking result was computed for, if any.")] = None
    """ID of the openlifu Point target this tracking result was computed for, if any."""

@dataclass
class Session:
    """
    Class representing an openlifu session, which consists essentially of a patient scan, a protocol
    to use, potential targets for sonication, and a transducer situated in the patient space.
    """

    id: Annotated[str | None, OpenLIFUFieldData("Session ID", "ID of this session")] = None
    """ID of this session"""

    subject_id: Annotated[str | None, OpenLIFUFieldData("Subject ID", "ID of the parent subject of this session")] = None
    """ID of the parent subject of this session"""

    name: Annotated[str | None, OpenLIFUFieldData("Session name", "Session name")] = None
    """Session name"""

    date_created: Annotated[datetime, OpenLIFUFieldData("Date created", "Date of creation of the session")] = field(default_factory=datetime.now)
    """Date of creation of the session"""

    date_modified: Annotated[datetime, OpenLIFUFieldData("Date modified", "Date of modification of the session")] = field(default_factory=datetime.now)
    """Date of modification of the session"""

    protocol_id: Annotated[str | None, OpenLIFUFieldData("Protocol ID", "ID of the protocol used for this session")] = None
    """ID of the protocol used for this session"""

    volume_id: Annotated[str | None, OpenLIFUFieldData("Volume ID", "ID of the subject volume associated with this session")] = None
    """ID of the subject volume associated with this session"""

    transducer_id: Annotated[str | None, OpenLIFUFieldData("Transducer ID", "ID of the transducer associated with this session")] = None
    """ID of the transducer associated with this session"""

    solution_id: Annotated[str, OpenLIFUFieldData("Solution ID", "ID of the most recently computed sonication Solution for this session, or '' if there is none. Cleared whenever the array_transform changes because a Solution is only valid for the transducer pose it was computed against.")] = ""
    """ID of the most recently computed sonication ``Solution`` for this session, or ``""`` if there is none.

    Cleared whenever the ``array_transform`` changes (manual move, virtual fit, transducer tracking), because a
    ``Solution`` is only valid for the specific transducer pose it was computed against. Consumers loading a
    session can use this to fetch the persisted ``Solution`` (and its analysis) from the database rather than
    re-running the simulation.
    """

    array_transform: Annotated[ArrayTransform, OpenLIFUFieldData("Array transform", "The transducer affine transform matrix with units, situating the transducer in space")] = field(default_factory=lambda: ArrayTransform(np.eye(4), "mm"))
    """The transducer affine transform matrix with units, situating the transducer in space"""

    targets: Annotated[List[Point], OpenLIFUFieldData("Targets", "Targets saved to this session")] = field(default_factory=list)
    """Targets saved to this session"""

    markers: Annotated[List[Point], OpenLIFUFieldData("Markers", "Registration markers saved to this session")] = field(default_factory=list)
    """Registration markers saved to this session"""

    photoscans: Annotated[List[str], OpenLIFUFieldData("Photoscan IDs", "IDs of photoscans that belong to this session. Each ID corresponds to a Photoscan stored under the session's ``photoscans/`` directory in the database.")] = field(default_factory=list)
    """IDs of photoscans that belong to this session. Each ID corresponds to a Photoscan
    stored under the session's ``photoscans/`` directory in the database. This is the
    authoritative list used to decide which photoscans to keep on save; legacy sessions
    that omit this field are auto-populated from the on-disk index on load."""

    photocollections: Annotated[List[str], OpenLIFUFieldData("Photocollection reference numbers", "Reference numbers of photocollections that belong to this session. Each entry corresponds to a directory under the session's ``photocollections/`` directory in the database.")] = field(default_factory=list)
    """Reference numbers (scan IDs) of photocollections that belong to this session.
    Each entry corresponds to a directory under the session's ``photocollections/``
    directory in the database. Legacy sessions that omit this field are auto-populated
    from the on-disk index on load."""

    attrs: Annotated[dict, OpenLIFUFieldData("Custom attributes", "Dictionary of additional custom attributes to save to the session")] = field(default_factory=dict)
    """Dictionary of additional custom attributes to save to the session"""

    virtual_fit_results: Annotated[Dict[str, List[Tuple[bool, ArrayTransform]]], OpenLIFUFieldData("Virtual fit results", "Virtual fit results. This is a dictionary mapping target IDs to a list of (approval, transform) pairs")] = field(default_factory=dict)
    """Virtual fit results. This is a dictionary mapping target IDs to a list of (approval, transform) pairs,
    where:

        `approval` is a boolean indicating whether the specific virtual fit `transform` has been approved for sonication, and
        `transform` is a transducer transform generated by the virtual fit for that target.

    The idea is that the list of transforms would be ordered from best to worst, and should of course
    contain at least one transform. The "approval" associated with each transform indicates whether the transform
    has been approved for sonication.
    """

    transducer_tracking_results: Annotated[List[TransducerTrackingResult], OpenLIFUFieldData("Tracking results", "List of any transducer tracking results")] = field(default_factory=list)
    """List of any transducer tracking results"""

    photoscan_registrations: Annotated[List[PhotoscanRegistration], OpenLIFUFieldData("Photoscan registrations", "List of photoscan-to-volume registrations stored on this session.")] = field(default_factory=list)
    """List of photoscan-to-volume registrations stored on this session. Each transducer tracking
    result references one of these registrations by ``photoscan_registration_id``."""

    def __post_init__(self):
        if self.id is None and self.name is None:
            self.id = "session"
        if self.id is None:
            self.id = sanitize(self.name, "snake")
        if self.name is None:
            self.name = self.id
        if isinstance(self.targets, Point):
            self.targets = [self.targets]
        else:
            self.targets = list(self.targets)
        if isinstance(self.markers, Point):
            self.markers = [self.markers]
        else:
            self.markers = list(self.markers)

    @staticmethod
    def from_file(filename):
        """
        Create a Session from a file

        :param filename: Name of the file to read
        :returns: Session object
        """
        with open(filename) as f:
            return Session.from_dict(json.load(f))

    @staticmethod
    def from_dict(d:Dict):
        """
        Create a session from a dictionary

        :param d: Dictionary of session parameters
        :returns: Session object
        """
        if 'date_created' in d:
            d['date_created'] = datetime.fromisoformat(d['date_created'])
        if 'date_modified' in d:
            d['date_modified'] = datetime.fromisoformat(d['date_modified'])
        if 'volume' in d:
            raise ValueError("Sessions no longer recognize a volume attribute -- it is now volume_id.")
        if 'array_transform' in d:
            d['array_transform'] = ArrayTransform.from_dict(d['array_transform'])

        # PhotoscanRegistrations are split out of TT results as of the multi-registration refactor.
        # Old session JSONs lack this key; if absent we start with an empty list and may populate it
        # below when migrating legacy TT entries that still carry an embedded photoscan_to_volume_transform.
        if 'photoscan_registrations' in d:
            d['photoscan_registrations'] = [
                PhotoscanRegistration(
                    photoscan_id=p['photoscan_id'],
                    transform=ArrayTransform.from_dict(p['transform']),
                    approval=p.get('approval', False),
                    id=p.get('id'),
                )
                for p in d['photoscan_registrations']
            ]
        else:
            d['photoscan_registrations'] = []

        if 'transducer_tracking_results' in d:
            # Per-photoscan counter for any registrations we synthesize during legacy migration;
            # continues past the count of registrations already present so we don't collide.
            pr_count_by_photoscan: Dict[str, int] = {}
            for pr in d['photoscan_registrations']:
                pr_count_by_photoscan[pr.photoscan_id] = pr_count_by_photoscan.get(pr.photoscan_id, 0) + 1

            migrated_tt: List[TransducerTrackingResult] = []
            for t in d['transducer_tracking_results']:
                if 'photoscan_to_volume_transform' in t:
                    # Legacy entry: split the embedded PV transform out into its own registration.
                    pid = t['photoscan_id']
                    n = pr_count_by_photoscan.get(pid, 0)
                    pr_count_by_photoscan[pid] = n + 1
                    synthesized_pr_id = f"{pid}__pr__{n:02d}"
                    d['photoscan_registrations'].append(PhotoscanRegistration(
                        photoscan_id=pid,
                        transform=ArrayTransform.from_dict(t['photoscan_to_volume_transform']),
                        approval=t.get('photoscan_to_volume_tracking_approved', False),
                        id=synthesized_pr_id,
                    ))
                    migrated_tt.append(TransducerTrackingResult(
                        photoscan_id=pid,
                        transducer_to_volume_transform=ArrayTransform.from_dict(t['transducer_to_volume_transform']),
                        photoscan_registration_id=synthesized_pr_id,
                        approval=t.get('transducer_to_volume_tracking_approved', t.get('approval', False)),
                        id=t.get('id'),
                        target_id=t.get('target_id'),
                    ))
                else:
                    migrated_tt.append(TransducerTrackingResult(
                        photoscan_id=t['photoscan_id'],
                        transducer_to_volume_transform=ArrayTransform.from_dict(t['transducer_to_volume_transform']),
                        photoscan_registration_id=t.get('photoscan_registration_id'),
                        approval=t.get('approval', t.get('transducer_to_volume_tracking_approved', False)),
                        id=t.get('id'),
                        target_id=t.get('target_id'),
                    ))
            d['transducer_tracking_results'] = migrated_tt
        if isinstance(d['targets'], list):
            if len(d['targets'])>0 and isinstance(d['targets'][0], dict):
                d['targets'] = [Point.from_dict(p) for p in d['targets']]
        elif isinstance(d['targets'], dict):
            d['targets'] = [Point.from_dict(d['targets'])]
        elif isinstance(d['targets'], Point):
            d['targets'] = [d['targets']]
        if 'virtual_fit_results' in d:
            for target_id,list_of_transforms in d['virtual_fit_results'].items():
                d['virtual_fit_results'][target_id] = [
                    (approval, ArrayTransform.from_dict(t_dict))  for approval, t_dict in list_of_transforms
                ]
        if isinstance(d['markers'], list):
            if len(d['markers'])>0 and isinstance(d['markers'][0], dict):
                d['markers'] = [Point.from_dict(p) for p in d['markers']]
        elif isinstance(d['markers'], dict):
            d['markers'] = [Point.from_dict(d['markers'])]
        elif isinstance(d['markers'], Point):
            d['markers'] = [d['markers']]
        return Session(**d)

    def to_dict(self):
        """
        Convert the session to a dictionary

        :returns: Dictionary of session parameters
        """
        d = copy.deepcopy(self.__dict__) # Deep copy needed so that we don't modify the internals of self below
        d['date_created'] = d['date_created'].isoformat()
        d['date_modified'] = d['date_modified'].isoformat()
        d['targets'] = [p.to_dict() for p in d['targets']]
        d['markers'] = [p.to_dict() for p in d['markers']]

        d['array_transform'] = asdict(d['array_transform'])
        for target_id, list_of_transforms in d['virtual_fit_results'].items():
            d['virtual_fit_results'][target_id] = [
                (approval, asdict(t)) for [approval, t] in list_of_transforms
            ]

        d['transducer_tracking_results'] = [asdict(t) for t in d['transducer_tracking_results']]
        d['photoscan_registrations'] = [asdict(r) for r in d['photoscan_registrations']]

        return d

    @staticmethod
    def from_json(json_string : str) -> Session:
        """Load a Session from a json string"""
        return Session.from_dict(json.loads(json_string))

    def to_json(self, compact:bool) -> str:
        """Serialize a Session to a json string

        Args:
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Session object.
        """
        if compact:
            return json.dumps(self.to_dict(), separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(self.to_dict(), indent=4, cls=PYFUSEncoder)

    def to_file(self, filename):
        """
        Save the session to a file

        :param filename: Name of the file
        """
        Path(filename).parent.parent.mkdir(exist_ok=True) #sessions directory
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w') as file:
            file.write(self.to_json(compact=False))

    def update_modified_time(self, time: datetime | None = None):
        if time is None:
            time = datetime.now()
        self.date_modified = time
