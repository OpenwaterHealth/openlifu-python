from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion
from openlifu.xdc import Transducer

logger = logging.getLogger(__name__)


@dataclass
class SimulationCorrected(DelayMethod):
    """Delay method using k-wave simulation with reciprocity for phase correction.

    Places a virtual point source at the target and records pressure time series
    at all transducer element positions. The arrival time at each element encodes
    the true acoustic path through the heterogeneous skull model. Delays are
    computed as max(arrival_time) - arrival_time for each element.

    Uses a single k-wave simulation (via reciprocity) instead of one per element.
    """

    c0: Annotated[
        float,
        OpenLIFUFieldData("Speed of Sound (m/s)", "Reference speed of sound in the medium (m/s)"),
    ] = 1500.0
    """Reference speed of sound in the medium (m/s)"""

    cfl: Annotated[float, OpenLIFUFieldData("CFL Number", "Courant-Friedrichs-Lewy number for time stepping")] = 0.3
    """Courant-Friedrichs-Lewy number for time stepping"""

    n_cycles: Annotated[int, OpenLIFUFieldData("Source Cycles", "Number of cycles in the source pulse")] = 3
    """Number of cycles in the source pulse"""

    gpu: Annotated[bool, OpenLIFUFieldData("Use GPU", "Whether to attempt GPU-accelerated simulation")] = True
    """Whether to attempt GPU-accelerated simulation"""

    def __post_init__(self):
        if not isinstance(self.c0, int | float):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)

        if not isinstance(self.cfl, int | float):
            raise TypeError("CFL must be a number")
        if self.cfl <= 0 or self.cfl >= 1:
            raise ValueError("CFL must be between 0 and 1 (exclusive)")
        self.cfl = float(self.cfl)

        if not isinstance(self.n_cycles, int):
            if isinstance(self.n_cycles, float) and self.n_cycles == int(self.n_cycles):
                self.n_cycles = int(self.n_cycles)
            else:
                raise TypeError("n_cycles must be an integer")
        if self.n_cycles < 1:
            raise ValueError("n_cycles must be at least 1")

        if not isinstance(self.gpu, bool):
            raise TypeError("gpu must be a boolean")

    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: np.ndarray | None = None):
        """Calculate delays using k-wave simulation with reciprocity.

        Fires a virtual point source at the target position and records the pressure
        time series at each transducer element location. Arrival times are extracted
        via the Hilbert envelope peak, and delays are computed so that all elements
        fire in phase at the target.

        Falls back to Direct (geometric) delays if k-wave is not available or
        if the simulation fails for any reason.

        Args:
        :param arr: The transducer array.
        :param target: The focal target point.
        :param params: Simulation grid dataset with sound_speed, density, attenuation fields.
        :param transform: Optional 4x4 affine transform for element positions.
        :returns: 1D numpy array of per-element delays in seconds.
        """
        try:
            import importlib
            if importlib.util.find_spec("kwave") is None:
                raise ImportError("k-wave not installed")
        except ImportError:
            logger.warning("k-wave not available. Falling back to Direct delay method.")
            return self._fallback_delays(arr, target, params, transform)

        try:
            arrival_times = self._run_reciprocal_simulation(arr, target, params, transform)
            delays = np.max(arrival_times) - arrival_times
            return delays
        except (RuntimeError, ValueError, IndexError, OSError):
            logger.exception("Simulation-corrected delay calculation failed. Falling back to Direct method.")
            return self._fallback_delays(arr, target, params, transform)

    def _run_reciprocal_simulation(
        self,
        arr: Transducer,
        target: Point,
        params: xa.Dataset,
        transform: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the reciprocal k-wave simulation and extract arrival times.

        Args:
            arr: The transducer array.
            target: The focal target point.
            params: Simulation grid dataset.
            transform: Optional 4x4 affine transform.

        Returns:
            arrival_times: 1D array of arrival times (seconds) per element.
        """
        from scipy.signal import hilbert

        from openlifu.sim.kwave_if import run_point_source_simulation

        # Get the reference sound speed from params if available
        if 'sound_speed' in params and 'ref_value' in params['sound_speed'].attrs:
            sound_speed_ref = params['sound_speed'].attrs['ref_value']
        else:
            sound_speed_ref = self.c0

        # Get frequency from the transducer
        freq = arr.frequency

        # Compute element positions in the simulation coordinate frame.
        # The simulation grid uses params.coords (typically in mm).
        # Element positions come from the transducer in its native units (typically m).
        # We need to convert element positions to the same units as the sim grid,
        # then find the nearest grid voxel for each element.
        coord_dims = list(params.coords.dims)
        coord_units = params[coord_dims[0]].attrs.get('units', 'mm')
        scl_to_grid = getunitconversion('m', coord_units)

        matrix = transform if transform is not None else np.eye(4)
        element_positions_m = np.array([
            el.get_position(units="m", matrix=matrix)
            for el in arr.elements
        ])
        # Convert to grid units
        element_positions_grid = element_positions_m * scl_to_grid

        # Get target position in grid units
        target_pos_grid = target.get_position(units=coord_units)

        # Build the sensor mask: find nearest grid indices for each element
        coord_arrays = [params.coords[dim].to_numpy() for dim in coord_dims]
        grid_shape = tuple(len(c) for c in coord_arrays)

        sensor_indices = []
        out_of_grid = set()
        for el_i, epos in enumerate(element_positions_grid):
            idx = []
            inside = True
            for dim_i, coord_vals in enumerate(coord_arrays):
                cmin, cmax = float(coord_vals[0]), float(coord_vals[-1])
                half_step = abs(float(coord_vals[1] - coord_vals[0])) / 2 if len(coord_vals) > 1 else 0
                if epos[dim_i] < cmin - half_step or epos[dim_i] > cmax + half_step:
                    inside = False
                nearest_idx = int(np.argmin(np.abs(coord_vals - epos[dim_i])))
                idx.append(nearest_idx)
            if not inside:
                out_of_grid.add(el_i)
                logger.warning(
                    f"Element {el_i} at position {epos} is outside the simulation grid. "
                    "Using geometric time-of-flight estimate for this element."
                )
            sensor_indices.append(tuple(idx))

        # Build sensor mask (3D binary)
        sensor_mask = np.zeros(grid_shape, dtype=int)
        for idx in sensor_indices:
            sensor_mask[idx] = 1

        # Find the target voxel index
        target_idx = []
        for dim_i, coord_vals in enumerate(coord_arrays):
            nearest_idx = int(np.argmin(np.abs(coord_vals - target_pos_grid[dim_i])))
            target_idx.append(nearest_idx)
        target_idx = tuple(target_idx)

        # Build source mask (single voxel at target)
        source_mask = np.zeros(grid_shape, dtype=int)
        source_mask[target_idx] = 1

        # Run the point source simulation
        sensor_data, dt = run_point_source_simulation(
            params=params,
            source_mask=source_mask,
            sensor_mask=sensor_mask,
            freq=freq,
            n_cycles=self.n_cycles,
            sound_speed_ref=sound_speed_ref,
            cfl=self.cfl,
            gpu=self.gpu,
        )

        # sensor_data is (n_sensor_points, n_timesteps).
        # Multiple elements may map to the same voxel if the grid is coarse.
        # We need to map sensor data rows back to elements.

        # Build a lookup: grid index -> sensor_data row index.
        # The sensor mask was constructed by setting 1 at unique voxel locations.
        # k-wave returns data for each nonzero voxel in Fortran (column-major) order.
        nonzero_indices = list(zip(*np.nonzero(sensor_mask)))
        # k-wave returns sensor data in Fortran (column-major) order of the mask:
        # x varies fastest, then y, then z. np.nonzero returns C order (row-major),
        # so we sort by the Fortran linear index to match k-wave's output ordering.
        def fortran_linear_index(idx, shape):
            # For Fortran order: idx[0] + idx[1]*shape[0] + idx[2]*shape[0]*shape[1]
            lin = idx[0]
            stride = shape[0]
            for d in range(1, len(shape)):
                lin += idx[d] * stride
                stride *= shape[d]
            return lin

        nonzero_with_fortran = [(fortran_linear_index(idx, grid_shape), idx) for idx in nonzero_indices]
        nonzero_with_fortran.sort(key=lambda x: x[0])
        sorted_nonzero = [item[1] for item in nonzero_with_fortran]

        voxel_to_row = {idx: row for row, idx in enumerate(sorted_nonzero)}

        # Extract arrival time for each element
        n_elements = len(arr.elements)
        arrival_times = np.zeros(n_elements)

        for el_i, sensor_idx in enumerate(sensor_indices):
            if el_i in out_of_grid:
                # Element is outside the simulation grid; use geometric fallback
                dist = np.linalg.norm(element_positions_m[el_i] - target.get_position(units="m"))
                arrival_times[el_i] = dist / self.c0
                continue

            row = voxel_to_row[sensor_idx]
            time_series = sensor_data[row, :]
            # Compute the analytic signal envelope via the Hilbert transform
            analytic = hilbert(time_series)
            envelope = np.abs(analytic)
            # The arrival time is the time of the envelope peak
            peak_sample = int(np.argmax(envelope))
            arrival_times[el_i] = peak_sample * dt

        return arrival_times

    def _fallback_delays(self, arr: Transducer, target: Point, params: xa.Dataset, transform: np.ndarray | None = None) -> np.ndarray:
        """Compute delays using the Direct (geometric) method as a fallback."""
        from openlifu.bf.delay_methods.direct import Direct
        direct = Direct(c0=self.c0)
        return direct.calc_delays(arr, target, params, transform)

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the delay method parameters

        :returns: Pandas DataFrame of the delay method parameters
        """
        records = [
            {"Name": "Type", "Value": "SimulationCorrected", "Unit": ""},
            {"Name": "Default Sound Speed", "Value": self.c0, "Unit": "m/s"},
            {"Name": "CFL Number", "Value": self.cfl, "Unit": ""},
            {"Name": "Source Cycles", "Value": self.n_cycles, "Unit": ""},
            {"Name": "Use GPU", "Value": self.gpu, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
