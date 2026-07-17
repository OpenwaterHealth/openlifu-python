from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class OpenLIFUFieldData:
    """
    Lightweight metadata attached to a dataclass field via :class:`typing.Annotated`,
    primarily consumed by GUI editors (e.g. SlicerOpenLIFU) to render fields with
    human-friendly labels, units, and tooltips.

    Example::

        class Pulse:
            frequency: Annotated[
                float,
                OpenLIFUFieldData(
                    name="Frequency",
                    description="Frequency of the pulse",
                    units="Hz",
                    display_units="kHz",
                    precision=1,
                ),
            ] = 400e3

    The presence of ``Annotated[]`` does not affect runtime behavior or type
    compatibility, and these fields are *not* serialized -- they describe how to
    *display* the underlying value. The stored value remains in ``units``.

    Backwards compatibility: callers that historically constructed
    ``OpenLIFUFieldData("Name", "Description")`` positionally continue to work
    because ``name`` and ``description`` remain the first two fields and all
    new fields are optional.

    Attributes:
        name: Display label shown next to the field in editors. ``None`` falls
            back to the dataclass field's attribute name.
        description: Tooltip text shown on hover. ``None`` falls back to a generic
            placeholder.
        units: Storage units. The unit in which the underlying dataclass value
            is stored (e.g. ``"Hz"``, ``"s"``, ``"Pa"``, ``"m"``, ``"deg"``).
            ``None`` means no unit semantics are attached.
        display_units: Preferred units for human display (e.g. ``"kHz"``,
            ``"ms"``, ``"kPa"``, ``"mm"``). When set, editors should convert
            from ``units`` to ``display_units`` for display, and back when
            saving. ``None`` (default) means display in ``units``.
        unit_options: Optional tuple of unit symbols a user may switch between
            for display. Reserved for future use (units dropdown).
        precision: Number of decimal places to display. ``None`` means use the
            editor's default.
        units_field: Optional name of a sibling dataclass field on the same
            instance whose value provides the storage unit dynamically (e.g.
            ``"distance_units"``). When set, editors should NOT auto-convert;
            instead they should display the value as-is and label it with the
            sibling's unit symbol. Mutually exclusive with ``units``.
    """

    name: str | None = None
    description: str | None = None
    units: str | None = None
    display_units: str | None = None
    unit_options: Tuple[str, ...] = field(default_factory=tuple)
    precision: int | None = None
    units_field: str | None = None
