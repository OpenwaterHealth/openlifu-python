from __future__ import annotations

import re

import numpy as np
from pint import UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError
from xarray import Dataset

ureg = UnitRegistry()
Q_ = ureg.Quantity

_ANGLE_UNITS = {
    "rad",
    "radian",
    "radians",
    "deg",
    "degree",
    "degrees",
    "\u00b0",
    "\u00c2\u00b0",
}

_UNIT_ALIASES = {
    "micron": "micrometer",
    "microns": "micrometer",
    "um": "micrometer",
    "\u00b5m": "micrometer",
    "\u03bcm": "micrometer",
    "sec": "second",
    "secs": "second",
    "min": "minute",
    "mins": "minute",
    "hr": "hour",
    "hrs": "hour",
    "\u00b0": "degree",
    "\u00c2\u00b0": "degree",
    "cc": "cm^3",
    "kgram": "kilogram",
    "kgrams": "kilograms",
    "amps": "ampere",
    "amp": "ampere",
}

_BASE_UNITS_BY_TYPE = {
    "distance": "m",
    "area": "m^2",
    "volume": "m^3",
    "time": "s",
    "angle": "rad",
    "frequency": "Hz",
    "pressure": "Pa",
    "watt": "W",
}


def _normalize_unit(unit: str) -> str:
    unit = unit.strip()

    unit = unit.replace("\u00b2", "^2").replace("\u00b3", "^3")
    unit = unit.replace("\u00b5", "u").replace("\u03bc", "u")

    # Fix common typos
    unit = re.sub(r"sec(s)?\b", "second", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bmili", "milli", unit, flags=re.IGNORECASE)
    unit = re.sub(r"grams?\b", "gram", unit, flags=re.IGNORECASE)
    unit = re.sub(r"meters?\b", "meter", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\s+", " ", unit)

    normalized_parts = []
    for part in re.split(r"([/*])", unit):
        stripped_part = part.strip()
        if stripped_part in {"/", "*", ""}:
            normalized_parts.append(stripped_part)
            continue

        part_key = stripped_part.lower()
        normalized_parts.append(_UNIT_ALIASES.get(part_key, _normalize_unit_symbol(stripped_part)))

    normalized = "".join(normalized_parts)

    normalized = re.sub(r"\b([a-zA-Z]+)([23])\b", r"\1^\2", normalized)
    return normalized


def _normalize_unit_symbol(unit: str) -> str:
    unit = re.sub(r"\b([a-zA-Z]+)([23])\b", r"\1^\2", unit)

    for suffix, canonical_suffix in (("hz", "Hz"), ("pa", "Pa")):
        suffix_match = re.fullmatch(rf"([A-Za-z]*){suffix}(\^\d+)?", unit, flags=re.IGNORECASE)
        if suffix_match:
            prefix, power = suffix_match.groups()
            return f"{prefix}{canonical_suffix}{power or ''}"

    watt_match = re.fullmatch(r"([A-Za-z]*)w(\^\d+)?", unit, flags=re.IGNORECASE)
    if watt_match:
        prefix, power = watt_match.groups()
        return f"{prefix}W{power or ''}"

    return unit


def _quantity(unit: str):
    return Q_(1, _normalize_unit(unit))


def getunittype(unit):
    normalized_unit = _normalize_unit(unit)

    if normalized_unit.lower() in _ANGLE_UNITS:
        return "angle"

    try:
        dim = Q_(1, normalized_unit).dimensionality
    except (TypeError, UndefinedUnitError):
        return "other"

    if dim == ureg.meter.dimensionality:
        return "distance"
    if dim == (ureg.meter**2).dimensionality:
        return "area"
    if dim == (ureg.meter**3).dimensionality:
        return "volume"
    if dim == ureg.second.dimensionality:
        return "time"
    if dim == (1 / ureg.second).dimensionality:
        return "frequency"
    if dim == ureg.pascal.dimensionality:
        return "pressure"
    if dim == ureg.watt.dimensionality:
        return "watt"

    return "other"


def getunitconversion(from_unit, to_unit, unitratio=None, constant=None):
    if not from_unit:
        return 1.0

    if unitratio is not None and constant is not None:
        if "/" not in unitratio:
            raise ValueError("Conversion unit ratio must have a '/' symbol")

        unitn, unitd = unitratio.split("/")
        type0 = getunittype(from_unit)
        type1 = getunittype(to_unit)
        typen = getunittype(unitn)
        typed = getunittype(unitd)

        if type0 == typed and type1 == typen:
            scl = getunitconversion(from_unit, unitd) * constant * getunitconversion(unitn, to_unit)
        elif type0 == typen and type1 == typed:
            scl = getunitconversion(from_unit, unitn) * 1 / constant * getunitconversion(unitd, to_unit)
        elif type0 == type1:
            scl = getunitconversion(from_unit, to_unit)
        else:
            raise ValueError(f"Unit type mismatch {type0} -> ({typen}/{typed}) -> {type1}")
    else:
        try:
            scl = _quantity(from_unit).to(_normalize_unit(to_unit)).magnitude
        except DimensionalityError as exc:
            type0 = getunittype(from_unit)
            type1 = getunittype(to_unit)
            raise ValueError(f"Unit type mismatch ({type0}) vs ({type1})") from exc
        except UndefinedUnitError as exc:
            raise ValueError(f"Cannot convert {from_unit} to {to_unit}") from exc

    return scl


def getsiscale(unit, type):
    type = type.lower()

    if type not in _BASE_UNITS_BY_TYPE:
        raise ValueError(f"Unknown unit type {type}")

    try:
        return getunitconversion(unit, _BASE_UNITS_BY_TYPE[type])
    except ValueError as exc:
        raise ValueError(f"Unknown prefix {unit}") from exc


def rescale_data_arr(data_arr: Dataset, units: str) -> Dataset:
    """
    Rescales the Dataset to the specified units.

    Args:
        data_arr : xarray.Dataset
        units: str

    Returns:
        rescaled: The rescaled xarray to new units.
    """
    rescaled = data_arr.copy(deep=True)
    scale = getunitconversion(data_arr.attrs["units"], units)
    rescaled.data *= scale
    rescaled.attrs["units"] = units

    return rescaled


def rescale_coords(data_arr: Dataset, units: str) -> Dataset:
    """
    Rescales the Dataset coordinates to the specified units.

    Args:
        data_arr : xarray.Dataset
        units: str

    Returns:
        rescaled: The rescaled data_arr coords to new units.
    """
    rescaled = data_arr.copy(deep=True)
    for coord_key in data_arr.coords:
        curr_coord_attrs = rescaled[coord_key].attrs
        if "units" in curr_coord_attrs:
            curr_coord_units = curr_coord_attrs["units"]
            scale = getunitconversion(curr_coord_units, units)
            curr_coord_rescaled = scale * rescaled[coord_key].data
            rescaled = rescaled.assign_coords({coord_key: (coord_key, curr_coord_rescaled, curr_coord_attrs)})
            rescaled[coord_key].attrs["units"] = units

    return rescaled


def get_ndgrid_from_arr(data_arr: Dataset) -> np.ndarray:
    """
    Creates a ndgrid from xarray.Dataset coordinates.

    Args:
        coords : xarray.Coordinates

    Returns:
        ndgrid: The ndgrid from the Coordinates.
    """
    # First need to get correct coordinates for the ndgrid
    first_data_key = next(iter(data_arr.keys()))
    ordered_key = data_arr[first_data_key].dims
    all_coord = []
    for coord_key in ordered_key:
        if "units" in data_arr[coord_key].attrs:
            all_coord += [data_arr.coords[coord_key].data]
    ndgrid = np.stack(np.meshgrid(*all_coord, indexing="ij"), axis=-1)

    return ndgrid
