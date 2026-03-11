from __future__ import annotations

import logging
from typing import List

import numpy as np
import xarray as xa

from openlifu import xdc
from openlifu.util.units import getunitconversion


def get_kgrid(coords: xa.Coordinates, t_end = 0, dt = 0, sound_speed_ref=1500, cfl=0.5):
    from kwave.kgrid import kWaveGrid
    units = [coords[dim].attrs['units'] for dim in coords.dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All coordinates must have the same units")
    scl = getunitconversion(units[0], 'm')
    sz = [len(coord) for coord in coords.values()]
    dx = [np.diff(coord)[0]*scl for coord in coords.values()]
    kgrid = kWaveGrid(sz, dx)
    if dt == 0 or t_end == 0:
        kgrid.makeTime(sound_speed_ref, cfl)
    else:
        Nt = round(t_end / dt)
        kgrid.setTime(Nt, dt)
    return kgrid

def get_karray(arr: xdc.Transducer,
               bli_tolerance: float = 0.05,
               upsampling_rate: int = 5,
               translation: List[float] = [0.,0.,0.],
               rotation: List[float] = [0.,0.,0.]):
    import kwave
    import kwave.data
    from kwave.utils.kwave_array import kWaveArray
    karray = kWaveArray(bli_tolerance=bli_tolerance, upsampling_rate=upsampling_rate,
                        single_precision=True)
    for el in arr.elements:
        ele_pos = list(el.get_position(units="m"))
        ele_w, ele_l = el.get_size(units="m")
        ele_angle = list(el.get_angle(units="deg"))
        karray.add_rect_element(ele_pos, ele_w, ele_l, ele_angle)
    translation = kwave.data.Vector(translation)
    rotation = kwave.data.Vector(rotation)
    karray.set_array_position(translation, rotation)
    return karray

def get_medium(params: xa.Dataset, ref_values_only: bool = False):
    from kwave.kmedium import kWaveMedium
    if ref_values_only:
        medium = kWaveMedium(sound_speed=params['sound_speed'].attrs['ref_value'],
                             density=params['density'].attrs['ref_value'],
                             alpha_coeff=params['attenuation'].attrs['ref_value'],
                             alpha_power=0.9,
                             alpha_mode='no_dispersion')
    else:
        medium= kWaveMedium(sound_speed=params['sound_speed'].data,
                        density=params['density'].data,
                        alpha_coeff=params['attenuation'].data,
                        alpha_power=0.9,
                        alpha_mode='no_dispersion')
    return medium

def get_sensor(kgrid, record=['p_max','p_min']):
    from kwave.ksensor import kSensor
    sensor_mask = np.ones([kgrid.Nx, kgrid.Ny, kgrid.Nz])
    sensor = kSensor(sensor_mask, record=record)
    return sensor

def get_source(kgrid, karray, source_sig):
    from kwave.ksource import kSource
    source = kSource()
    logging.info("Getting binary mask")
    source.p_mask = karray.get_array_binary_mask(kgrid)
    logging.info("Getting distributed source signal")
    source.p = karray.get_distributed_source_signal(kgrid, source_sig)
    return source

def run_simulation(arr: xdc.Transducer,
                   params: xa.Dataset,
                   delays: np.ndarray | None = None,
                   apod: np.ndarray | None = None,
                   freq: float = 1e6,
                   cycles: float = 20,
                   amplitude: float = 1,
                   dt: float = 0,
                   t_end: float = 0,
                   cfl: float = 0.5,
                   bli_tolerance: float = 0.05,
                   upsampling_rate: int = 5,
                   gpu: bool = True,
                   ref_values_only: bool = False
):
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())
    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
    t = np.arange(0, np.min([cycles / freq, (kgrid.Nt-np.ceil(max(delays)/kgrid.dt))*kgrid.dt]), kgrid.dt)
    input_signal = amplitude * np.sin(2 * np.pi * freq * t)
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)
    units = [params[dim].attrs['units'] for dim in params.dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All dimensions must have the same units")
    scl = getunitconversion(units[0], 'm')
    array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]
    karray = get_karray(arr,
                        translation=array_offset,
                        bli_tolerance=bli_tolerance,
                        upsampling_rate=upsampling_rate)
    medium = get_medium(params, ref_values_only=ref_values_only)
    sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
    source = get_source(kgrid, karray, source_mat)
    logging.info("Running simulation")
    simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    output = kspaceFirstOrder3D(kgrid=kgrid,
                                source=source,
                                sensor=sensor,
                                medium=medium,
                                simulation_options=simulation_options,
                                execution_options=execution_options)
    logging.info('Simulation Complete')
    sz = list(params.coords.sizes.values())
    p_max = xa.DataArray(output['p_max'].reshape(sz, order='F'),
                         coords=params.coords,
                         name='p_max',
                         attrs={'units':'Pa', 'long_name':'PPP'})
    p_min = xa.DataArray(-1*output['p_min'].reshape(sz, order='F'),
                         coords=params.coords,
                         name='p_min',
                         attrs={'units':'Pa', 'long_name':'PNP'})
    Z = params['density'].data*params['sound_speed'].data
    intensity = xa.DataArray(1e-4*output['p_min'].reshape(sz, order='F')**2/(2*Z),
                         coords=params.coords,
                         name='I',
                         attrs={'units':'W/cm^2', 'long_name':'Intensity'})
    ds = xa.Dataset({'p_max':p_max, 'p_min':p_min, 'intensity':intensity})
    return ds, output
