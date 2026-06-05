from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Optional, Tuple, get_args, get_origin

try:
    # Python 3.10+ provides typing.get_type_hints with include_extras, but the
    # rest of the codebase already uses ``get_type_hints`` from ``typing``.
    from typing import get_type_hints as _get_type_hints
except ImportError:  # pragma: no cover - defensive
    from typing_extensions import get_type_hints as _get_type_hints  # type: ignore

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion


def get_field_metadata(cls: type, field_name: str) -> Optional[OpenLIFUFieldData]:
    """Return the :class:`OpenLIFUFieldData` annotation attached to ``cls.field_name``.

    Returns ``None`` if the field has no such annotation, if it has no
    ``Annotated[]`` wrapping, or if the class is not a dataclass.
    """
    if not is_dataclass(cls):
        return None
    try:
        hints = _get_type_hints(cls, include_extras=True)
    except Exception:
        return None
    annotated_type = hints.get(field_name)
    if annotated_type is None:
        return None
    args = get_args(annotated_type)
    if get_origin(annotated_type) is None or len(args) < 2:
        return None
    for meta in args[1:]:
        if isinstance(meta, OpenLIFUFieldData):
            return meta
    return None


def resolve_units(meta: OpenLIFUFieldData, instance: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(storage_units, display_units)`` for ``meta`` applied to ``instance``.

    * Storage unit: ``instance.<meta.units_field>`` when ``units_field`` is set,
      otherwise ``meta.units``.
    * Display unit: ``meta.display_units`` when set, otherwise the storage unit.

    This means ``units_field`` and ``display_units`` may be combined: the value
    is *stored* in whatever unit the sibling field reports, but the editor
    *displays* it in the fixed ``display_units`` (e.g. always ``"mm"`` for a
    distance field, regardless of the protocol's ``distance_units`` setting).
    """
    if meta.units_field:
        sibling = getattr(instance, meta.units_field, None) if instance is not None else None
    else:
        sibling = None
    storage = sibling if meta.units_field else meta.units
    display = meta.display_units or storage
    return storage, display


def to_display(value: float, meta: OpenLIFUFieldData, instance: Optional[Any] = None) -> float:
    """Convert ``value`` from storage units to ``meta``'s display units.

    Falls back to the input value if conversion is not possible (no units
    declared, or a unit-conversion failure)."""
    if value is None:
        return value
    storage_unit, display_unit = resolve_units(meta, instance)
    if storage_unit is None or display_unit is None or storage_unit == display_unit:
        return value
    try:
        return float(value) * getunitconversion(storage_unit, display_unit)
    except Exception:
        return value


def from_display(value: float, meta: OpenLIFUFieldData, instance: Optional[Any] = None) -> float:
    """Inverse of :func:`to_display`: convert from display units to storage units."""
    if value is None:
        return value
    storage_unit, display_unit = resolve_units(meta, instance)
    if storage_unit is None or display_unit is None or storage_unit == display_unit:
        return value
    try:
        return float(value) * getunitconversion(display_unit, storage_unit)
    except Exception:
        return value


def format_value(
    value: Any,
    meta: Optional[OpenLIFUFieldData] = None,
    instance: Optional[Any] = None,
) -> str:
    """Format ``value`` using the precision/units in ``meta``.

    Numeric values are converted to ``meta.display_units`` (when known),
    rounded to ``meta.precision`` (default 2 decimals for floats, no rounding
    for ints), with trailing zeros trimmed, and the unit symbol appended.
    Non-numeric values are passed through ``str()``.
    """
    if value is None:
        return ""
    if meta is None:
        if isinstance(value, float):
            return _format_number(value, None)
        return str(value)

    # Tuple/list: format element-wise, share unit suffix at the end
    if isinstance(value, (tuple, list)):
        formatted = [format_value(v, meta, instance) for v in value]
        # Strip per-element unit so we don't repeat it; we'll add once at the end.
        unit_suffix = _display_unit_suffix(meta, instance)
        if unit_suffix:
            stripped = [s[: -len(unit_suffix)].rstrip() if s.endswith(unit_suffix) else s for s in formatted]
            return ", ".join(stripped) + " " + unit_suffix
        return ", ".join(formatted)

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, (int, float)):
        # Convert to the display unit. We always go through float so that, for
        # an int value with display-unit conversion (e.g. an integer count of
        # microns displayed in mm), we still get the proper scaled number.
        display_value = to_display(float(value), meta, instance)
        if isinstance(value, int) and (meta.display_units is None or meta.display_units == meta.units):
            # No conversion needed for an int field; keep it integral.
            display_value = value
        precision = meta.precision
        text = _format_number(display_value, precision)
        suffix = _display_unit_suffix(meta, instance)
        return f"{text} {suffix}" if suffix else text

    return str(value)


def _display_unit_suffix(meta: OpenLIFUFieldData, instance: Optional[Any]) -> str:
    _, display_unit = (
        resolve_units(meta, instance) if instance is not None else (meta.units, meta.display_units or meta.units)
    )
    return display_unit or ""


def _strip_trailing_zeros(text: str) -> str:
    """Remove trailing zeros (and a trailing dot) from a fixed-precision float string."""
    if "." not in text or "e" in text or "E" in text:
        return text
    stripped = text.rstrip("0").rstrip(".")
    return stripped if stripped not in ("", "-") else "0"


def _format_number(value: Any, precision: Optional[int]) -> str:
    """Format a number with ``precision`` decimal places, stripping trailing zeros.

    Falls back to ``%g`` formatting whenever a fixed-precision render would
    clip a non-zero value to ``"0"`` (for example, ``0.0025`` with
    ``precision=1``). This mirrors the intent: don't print
    ``300.0000000``, but also don't lose the entire value just because the
    declared precision is too coarse for an unusually small number.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if precision is None:
        return _strip_trailing_zeros(f"{value:.6g}")
    fixed = f"{value:.{precision}f}"
    fixed = _strip_trailing_zeros(fixed)
    if fixed in ("0", "-0") and value != 0.0:
        # The declared precision would erase the value entirely; use %g so
        # the reader can still see the magnitude.
        return _strip_trailing_zeros(f"{value:.6g}")
    return fixed


def field_summary(instance: Any, field_name: str, label: Optional[str] = None) -> Optional[str]:
    """Return ``"<label>: <formatted value>"`` for one field, suitable for joining
    into a section summary, or ``None`` if the field doesn't exist on the
    instance."""
    if not hasattr(instance, field_name):
        return None
    value = getattr(instance, field_name)
    meta = get_field_metadata(type(instance), field_name)
    label = label or (meta.name if (meta is not None and meta.name) else field_name)
    return f"{label}: {format_value(value, meta, instance)}"


def join_summary(parts: Any) -> str:
    """Join non-empty summary fragments with ``", "``."""
    return ", ".join(p for p in parts if p)


def summarize_fields(instance: Any, field_names: Tuple[str, ...]) -> str:
    """Build a comma-separated summary of the given fields on ``instance``.

    Used by ``get_summary()`` implementations on Protocol-tree classes to
    generate one-liner section headers.
    """
    return join_summary(field_summary(instance, name) for name in field_names)


__all__ = [
    "field_summary",
    "format_value",
    "from_display",
    "get_field_metadata",
    "join_summary",
    "resolve_units",
    "summarize_fields",
    "to_display",
]
