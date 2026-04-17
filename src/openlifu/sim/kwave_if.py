from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
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
                   ref_values_only: bool = False,
                   return_kwave_outputs: bool = False,
                   return_kwave_inputs: bool = False,
                   sensor_record: List[str] = ['p_max', 'p_min'],
                   _source = None,
                   _sensor = None
):
    """ Run a k-wave simulation for the given transducer array and parameters.
    Args:
        arr: The transducer array to simulate.
        params: The simulation parameters as an xarray Dataset. Must include 'sound_speed', '
        density', and 'attenuation' variables with appropriate units.
        delays: Optional array of time delays for each element in the transducer array, in seconds. If None, no delays will be applied.
        apod: Optional array of apodization values for each
            element in the transducer array. If None, no apodization will be applied.
        freq: The frequency of the input signal in Hz. Default is 1 MHz.
        cycles: The number of cycles in the input signal. Default is 20.
        amplitude: The amplitude of the input signal. Default is 1.
        dt: The time step for the simulation in seconds. If 0, it will be automatically calculated based on the CFL condition.
        t_end: The total time for the simulation in seconds. If 0, it will be automatically calculated based on the input signal duration and the CFL condition.
        cfl: The Courant-Friedrichs-Lewy (CFL) number for the simulation. Default is 0.5.
        bli_tolerance: The tolerance for the boundary layer integral method used in k-wave. Default
            is 0.05.
        upsampling_rate: The upsampling rate for the boundary layer integral method used in k-wave
            Default is 5.
        gpu: Whether to use GPU for the simulation. Default is True. If False, the simulation will run on the CPU. Note that running on CPU may be very slow for large simulations.
        ref_values_only: Whether to use only the reference values for the medium properties (sound speed, density, attenuation) instead of the full spatial maps. Default is False. Setting this to True can significantly speed up the simulation, but will not capture any spatial variations in the medium properties.
        return_kwave_outputs: Whether to return the raw outputs from k-wave in addition to the processed xarray Dataset. Default is False.
        return_kwave_inputs: Whether to return the inputs to k-wave (kgrid, source, sensor, medium) in addition to the processed xarray Dataset. Default is False.
        sensor_record: List of strings specifying which k-wave sensor outputs to record. Can include '
        p_max', 'p_min', and 'p'. Default is ['p_max', 'p_min'].

    Additional Args:
        _source: Optional kSource object to use for the simulation. If None, a source will be created based on the transducer array and input signal.
        _sensor: Optional kSensor object to use for the simulation. If None, a sensor

    Returns:
        An xarray Dataset containing the simulation results, with variables corresponding to the requested sensor outputs and
        coordinates corresponding to the spatial dimensions of the simulation. If return_kwave_outputs is True, also returns a dictionary containing the raw outputs from k-wave. If return_kwave_inputs is True, also returns a dictionary containing the inputs to k-wave.

            """
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    delays = delays if delays is not None else np.zeros(arr.numelements())
    apod = apod if apod is not None else np.ones(arr.numelements())
    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end, cfl=cfl)
    t = np.arange(0, np.min([cycles / freq, (kgrid.Nt-np.ceil(max(delays)/kgrid.dt))*kgrid.dt]), kgrid.dt)
    if cycles/freq > t[-1]:
        cycles = np.ceil(t[-1]*freq)
    units = [params[dim].attrs['units'] for dim in params.dims]
    if not all(unit == units[0] for unit in units):
        raise ValueError("All dimensions must have the same units")
    scl = getunitconversion(units[0], 'm')
    array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]

    medium = get_medium(params, ref_values_only=ref_values_only)
    if _sensor is not None:
        sensor = _sensor
    else:
        sensor = get_sensor(kgrid, sensor_record)
    if 'p_min' not in sensor_record:
        raise ValueError("p_min must be included in sensor_record")
    if _source is not None:
        source = _source
    else:
        source_mat = arr.calc_output(amplitude=amplitude, cycles=cycles, frequency=freq, dt=kgrid.dt, delays=delays, apod=apod)
        source_mat = source_mat[:, :kgrid.Nt]
    if arr.crosstalk_frac != 0:
        # Simulate crosstalk by accumulating fraction of the source signal from each element into
        # its neighbors within a certain distance. This is a simple model of crosstalk and may
        # not capture all the complexities of real crosstalk, but it can be useful for testing
        # and simulation purposes.
        crosstalk_mat = source_mat.copy()
        positions = arr.get_positions(units="m")
        for src_idx in range(arr.numelements()):
            for dst_idx in range(arr.numelements()):
                if src_idx == dst_idx:
                    continue
                src_pos = np.array(positions[src_idx])
                dst_pos = np.array(positions[dst_idx])
                dist = np.linalg.norm(src_pos - dst_pos)
                if dist <= arr.crosstalk_dist:
                    crosstalk_mat[dst_idx,:] += arr.crosstalk_frac*source_mat[src_idx,:]
        source_mat = crosstalk_mat
    karray = get_karray(arr,
                        translation=array_offset,
                        bli_tolerance=bli_tolerance,
                        upsampling_rate=upsampling_rate)
    source = get_source(kgrid, karray, source_mat)
    logging.info("Running simulation")
    simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=gpu)
    inputs = {'kgrid':kgrid, 'source':source, 'sensor':sensor, 'medium':medium,
              'simulation_options':simulation_options, 'execution_options':execution_options}
    output = kspaceFirstOrder3D(**deepcopy(inputs))
    logging.info('Simulation Complete')
    sz = list(params.coords.sizes.values())
    ds_dict = {}
    for record in sensor.record:
        if record == 'p_max':
            ds_dict['p_max'] = xa.DataArray(output['p_max'].reshape(sz, order='F'),
                                coords=params.coords,
                                name='p_max',
                                attrs={'units':'Pa', 'long_name':'PPP'})
        elif record == 'p_min':
            ds_dict['p_min'] = xa.DataArray(-1*output['p_min'].reshape(sz, order='F'),
                            coords=params.coords,
                            name='p_min',
                            attrs={'units':'Pa', 'long_name':'PNP'})
            Z = params['density'].data*params['sound_speed'].data
            ds_dict['intensity'] = xa.DataArray(1e-4*output['p_min'].reshape(sz, order='F')**2/(2*Z),
                         coords=params.coords,
                         name='I',
                         attrs={'units':'W/cm^2', 'long_name':'Intensity'})
        elif record == 'p':
            pcoords = params.coords.copy()
            pcoords['t'] = np.arange(0, output['Nt']*kgrid.dt, kgrid.dt)
            ds_dict['p'] = xa.DataArray(output['p'].reshape([output['Nt'], *sz], order='F'),
                         coords=[pcoords[dim] for dim in ['t','x','y','z']],
                         attrs={'units':'Pa', 'long_name':'Pressure'})

    # clean up temporary files created by k-wave
    for filename in [simulation_options.input_filename, simulation_options.output_filename]:
        Path(filename).unlink(missing_ok=True)


    ds = xa.Dataset(ds_dict)
    if return_kwave_outputs and return_kwave_inputs:
        return ds, output, inputs
    elif return_kwave_outputs:
        return ds, output
    elif return_kwave_inputs:
        return ds, inputs
    return ds
