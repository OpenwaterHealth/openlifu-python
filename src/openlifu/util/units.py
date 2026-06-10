from __future__ import annotations

import numpy as np
from xarray import DataArray, Dataset


def getunittype(unit):
    unit = unit.lower()
    if unit in ['micron', 'microns']:
        return 'distance'
    elif unit in ['minute', 'minutes', 'min', 'mins', 'hour', 'hours', 'hr', 'hrs', 'day', 'days', 'd']:
        return 'time'
    elif unit in ['rad', 'deg', 'radian', 'radians', 'degree', 'degrees', '°']:
        return 'angle'
    elif 'sec' in unit:
        return 'time'
    elif 'meter' in unit or 'micron' in unit:
        return 'distance'
    elif unit.endswith('s'):
        return 'time'
    elif unit.endswith('m'):
        return 'distance'
    elif unit.endswith(('m2', 'm^2')):
        return 'area'
    elif unit.endswith(('m3', 'm^3')):
        return 'volume'
    elif unit.endswith('hz'):
        return 'frequency'
    elif unit.endswith('pa'):
        return 'pressure'
    elif unit.endswith('w'):
        return 'watt'
    else:
        return 'other'

def getunitconversion(from_unit, to_unit, unitratio=None, constant=None):
    if not from_unit:
        return 1.0

    if unitratio is not None and constant is not None:
        if '/' not in unitratio:
            raise ValueError('Conversion unit ratio must have a \'/\' symbol')

        unitn, unitd = unitratio.split('/')
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
            raise ValueError(f'Unit type mismatch {type0} -> ({typen}/{typed}) -> {type1}')
    else:
        slash0 = from_unit.find('/')
        slash1 = to_unit.find('/')

        if slash0 != -1 and slash1 != -1:
            num0 = from_unit[:slash0]
            denom0 = from_unit[slash0+1:]
            num1 = to_unit[:slash1]
            denom1 = to_unit[slash1+1:]
            scl = getunitconversion(num0, num1) / getunitconversion(denom0, denom1)
        elif slash0 == -1 and slash1 == -1:
            type0 = getunittype(from_unit)
            type1 = getunittype(to_unit)

            if type0 != type1:
                raise ValueError(f'Unit type mismatch ({type0}) vs ({type1})')

            if type0 == 'other':
                if from_unit[-1] != to_unit[-1]:
                    raise ValueError(f'Cannot convert {from_unit} to {to_unit}')

                i = 0
                while i < min(len(from_unit), len(to_unit)) and from_unit[-i:] == to_unit[-i:]:
                    type = from_unit[-i:]
                    i += 1

                scl0 = getsiscale(from_unit, type)
                scl1 = getsiscale(to_unit, type)
                scl = scl0 / scl1
            else:
                scl0 = getsiscale(from_unit, type0)
                scl1 = getsiscale(to_unit, type0)
                scl = scl0 / scl1
        else:
            raise ValueError(f'Unit ratio mismatch ({from_unit} vs {to_unit})')

    return scl

def getsiscale(unit, type):
    type = type.lower()

    if type in ['distance', 'area', 'volume']:
        idx = unit.find('meters')
        if idx == -1:
            idx = unit.find('meter')
            if idx == -1:
                if unit.lower() == 'micron':
                    idx = 6
                else:
                    idx = unit.rfind('m')
                    if idx == -1:
                        idx = len(unit)

    elif type == 'time':
        idx = unit.find('seconds')
        if idx == -1:
            idx = unit.find('second')
            if idx == -1:
                idx = unit.find('sec')
                if idx == -1:
                    idx = unit.rfind('s')
                    if idx == -1:
                        idx = len(unit)

    elif type == 'angle':
        idx = len(unit)

    elif type == 'frequency' or type == "pressure":
        idx = len(unit) - 2

    elif type == "watt":
        idx = len(unit) - 1

    else:
        idx = len(unit) - len(type) + 1

    prefix = unit[:idx]

    if not prefix:
        scl = 1.0
    else:
        scl = 1.0

        if prefix == 'pico' or prefix == 'p':
            scl = 1.0e-12
        elif prefix == 'nano' or prefix == 'n':
            scl = 1.0e-9
        elif prefix == 'micro' or prefix == 'u' or prefix == '\u00b5' or prefix == '\u03bc':
            scl = 1.0e-6
        elif prefix == 'milli' or prefix == 'm':
            scl = 1.0e-3
        elif prefix == 'centi' or prefix == 'c':
            scl = 1.0e-2
        elif prefix == '':
            scl = 1.0
        elif prefix == 'kilo' or prefix == 'k':
            scl = 1.0e3
        elif prefix == 'mega' or prefix == 'M':
            scl = 1.0e6
        elif prefix == 'giga' or prefix == 'G':
            scl = 1.0e9
        elif prefix == 'tera' or prefix == 'T':
            scl = 1.0e12
        elif prefix == 'min' or prefix == 'minute':
            scl = 60.0
        elif prefix == 'hour' or prefix == 'hr':
            scl = 60.0 * 60.0
        elif prefix == 'day' or prefix == 'd':
            scl = 60.0 * 60.0 * 24.0
        elif prefix == 'rad' or prefix == 'radian' or prefix == 'radians':
            scl = 1.0
        elif prefix == 'deg' or prefix == 'degree' or prefix == 'degrees' or prefix == '\u00b0':
            scl = 2 * 3.14159265358979323846 / 360
        elif prefix:
            raise ValueError(f'Unknown prefix {prefix}')

    if type == 'area':
        scl = scl ** 2.0
    elif type == 'volume':
        scl = scl ** 3.0

    return scl


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
    scale = getunitconversion(data_arr.attrs['units'], units)
    rescaled.data *= scale
    rescaled.attrs['units'] = units

    return rescaled


def rescale_coords(data_arr: Dataset | DataArray, units: str) -> Dataset:
    """
    Rescales the Dataset or DataArray coordinates to the specified units.

    Args:
        data_arr : xarray.Dataset or xarray.DataArray
        units: str

    Returns:
        rescaled: The rescaled data_arr coords to new units.
    """
    rescaled = data_arr.copy(deep=True)
    for coord_key in data_arr.coords:
        curr_coord_attrs = rescaled[coord_key].attrs
        if 'units' in curr_coord_attrs:
            curr_coord_units = curr_coord_attrs['units']
            scale = getunitconversion(curr_coord_units, units)
            curr_coord_rescaled = scale*rescaled[coord_key].data
            rescaled = rescaled.assign_coords({coord_key: (coord_key, curr_coord_rescaled, curr_coord_attrs)})
            rescaled[coord_key].attrs['units'] = units

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
        if 'units' in data_arr[coord_key].attrs:
            all_coord += [data_arr.coords[coord_key].data]
    ndgrid = np.stack(np.meshgrid(*all_coord, indexing="ij"), axis=-1)

    return ndgrid
