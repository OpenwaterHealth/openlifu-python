from __future__ import annotations

import base64
import heapq
import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

import numpy as np
import xarray as xa

from openlifu.bf import Pulse, Sequence
from openlifu.bf.focal_patterns import FocalPattern
from openlifu.geo import Point
from openlifu.plan.param_constraint import ParameterConstraint
from openlifu.plan.solution_analysis import (
    SolutionAnalysis,
    SolutionAnalysisOptions,
    find_centroid,
    get_beamwidth,
    get_mask,
    model_tx_temperature_rise,
)
from openlifu.sim import SimSetup, run_simulation
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.checkgpu import gpu_available
from openlifu.util.json import PYFUSEncoder
from openlifu.util.units import getunitconversion, rescale_coords, rescale_data_arr
from openlifu.xdc import Transducer

logger = logging.getLogger(__name__)

def _construct_nc_filepath_from_json_filepath(json_filepath:Path) -> Path:
    """Construct a default filepath to netCDF file given filepath to associated solution json file."""
    nc_filename = json_filepath.name.split(".")[0] + ".nc"
    nc_filepath = json_filepath.parent / nc_filename
    return nc_filepath


@dataclass
class Solution:
    """
    A sonication solution resulting from beamforming and running a simulation.
    """

    id: Annotated[str, OpenLIFUFieldData("Solution ID", "ID of this solution")] = "solution"  # the *solution* id, a concept that did not exist in the matlab software
    """ID of this solution"""

    name: Annotated[str, OpenLIFUFieldData("Solution name", "Name of this solution")] = "Solution"
    """Name of this solution"""

    protocol_id: Annotated[str | None, OpenLIFUFieldData("Protocol ID", "ID of the protocol that was used when generating this solution")] = None  # this used to be called plan_id in the matlab code
    """ID of the protocol that was used when generating this solution"""

    transducer: Annotated[Transducer | None, OpenLIFUFieldData("Transducer", "Transducer used when generating this solution")] = None
    """Transducer used when generating this solution"""

    date_created: Annotated[datetime, OpenLIFUFieldData("Creation date", "Solution creation time")] = field(default_factory=datetime.now)
    """Solution creation time"""

    description: Annotated[str, OpenLIFUFieldData("Description", "Description of this solution")] = ""
    """Description of this solution"""

    delays: Annotated[np.ndarray | None, OpenLIFUFieldData("Delays", "Vectors of time delays to steer the beam. Shape is (number of foci, number of transducer elements).")] = None
    """Vectors of time delays to steer the beam. Shape is (number of foci, number of transducer elements)."""

    apodizations: Annotated[np.ndarray | None, OpenLIFUFieldData("Apodizations", "Vectors of apodizations to steer the beam. Shape is (number of foci, number of transducer elements).")] = None
    """Vectors of apodizations to steer the beam. Shape is (number of foci, number of transducer elements)."""

    pulse: Annotated[Pulse, OpenLIFUFieldData("Pulse", "Pulse to send to the transducer when running sonication")] = field(default_factory=Pulse)
    """Pulse to send to the transducer when running sonication"""

    voltage: Annotated[float, OpenLIFUFieldData("Voltage", "Voltage to use when running sonication. This is the voltage that will be set on the HV power supply.")] = 1.0
    """Voltage to use when running sonication. This is the voltage that will be set on the HV power supply."""

    sequence: Annotated[Sequence, OpenLIFUFieldData("Pulse sequence", "Pulse sequence to use when running sonication")] = field(default_factory=Sequence)
    """Pulse sequence to use when running sonication"""

    foci: Annotated[List[Point], OpenLIFUFieldData("Foci", "Points that are focused on in this Solution due to the focal pattern around the target. Each item in this list is a unique point from the focal pattern, and the pulse sequence is what determines how many times each point will be used.")] = field(default_factory=list)
    """Points that are focused on in this Solution due to the focal pattern around the target.
    Each item in this list is a unique point from the focal pattern, and the pulse sequence is
    what determines how many times each point will be used.
    """

    # there was "target_id" in the matlab software, but here we do not have the concept of a target ID.
    # I believe this was only needed in the matlab software because solutions were organized by target rather
    # than having their own unique solution ID. We do have unique solution IDs so it's possible we don't need
    # this target attribute at all here. Keeping it here for now just in case.
    target: Annotated[Point | None, OpenLIFUFieldData("Target point", "The ultimate target of this sonication. This sonication solution is focused on one focal point in a pattern that is centered on this target.")] = None
    """The ultimate target of this sonication. This sonication solution is focused on one focal point
    in a pattern that is centered on this target."""

    # In the matlab code the simulation result was saved as a separate .mat file.
    # Here we include it as an xarray dataset.
    simulation_result: Annotated[xa.Dataset, OpenLIFUFieldData("Simulation result", "The xarray Dataset of simulation results")] = field(default_factory=xa.Dataset)
    """The xarray Dataset of simulation results"""

    approved: Annotated[bool, OpenLIFUFieldData("Approved?", "Approval state of this solution as a sonication plan. `True` means the user has provided some kind of confirmation that the solution is safe and acceptable to be executed.")] = False
    """Approval state of this solution as a sonication plan. `True` means the user has provided some
    kind of confirmation that the solution is safe and acceptable to be executed."""

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        if self.delays is not None:
            self.delays = np.array(self.delays, ndmin=2)
        if self.apodizations is not None:
            self.apodizations = np.array(self.apodizations, ndmin=2)

        if self.pulse.frequency <= 0:
            raise ValueError("Pulse frequency must be positive")
        if self.voltage <= 0:
            raise ValueError("Voltage must be positive")
        if self.sequence.pulse_interval <= 0:
            raise ValueError("Pulse interval must be positive")
        if self.sequence.pulse_count <= 0:
            raise ValueError("Pulse count must be positive")
        if self.sequence.pulse_train_interval < 0:
            raise ValueError("Pulse train interval must be non-negative")
        elif (self.sequence.pulse_train_interval > 0) and (self.sequence.pulse_train_interval < (self.sequence.pulse_interval * self.sequence.pulse_count)):
            raise ValueError("Pulse train interval must be greater than or equal to the total pulse interval")
        if self.sequence.pulse_train_count <= 0:
            raise ValueError("Pulse train count must be positive")
        if (self.sequence.focus_order is not None and len(self.foci) > 0 and max(self.sequence.focus_order) > len(self.foci)):
            raise ValueError(f"Focus order index {max(self.sequence.focus_order)} exceeds number of foci ({len(self.foci)})")
        if len(self.foci)>0 and self.delays is not None and self.delays.shape[0] != len(self.foci):
            raise ValueError(f"Delays number of foci ({self.delays.shape[0]}) does not match number of foci ({len(self.foci)})")
        if len(self.foci)>0 and self.apodizations is not None and self.apodizations.shape[0] != len(self.foci):
            raise ValueError(f"Apodizations number of foci ({self.apodizations.shape[0]}) does not match number of foci ({len(self.foci)})")
        if self.delays is not None and self.apodizations is not None:
            if self.apodizations.shape[0] != self.delays.shape[0]:
                raise ValueError(f"Apodizations number of foci ({self.apodizations.shape[0]}) does not match delays number of foci ({self.delays.shape[0]})")
            if self.apodizations.shape[1] != self.delays.shape[1]:
                raise ValueError(f"Apodizations number of elements {self.apodizations.shape[1]} does not match delays shape ({self.delays.shape[1]})")


    def num_foci(self) -> int:
        """Get the number of foci"""
        return len(self.foci)

    def get_focus_order(self) -> np.ndarray:
        """Get the focus index order for each pulse."""
        if self.sequence.focus_order is not None:
            return np.array(self.sequence.focus_order)
        return (np.arange(self.sequence.pulse_count) - 1) % self.num_foci() + 1

    def get_focus_counts(self) -> np.ndarray:
        """Get the number of pulses assigned to each focus."""
        focus_order = self.get_focus_order()
        return np.array([
            np.sum(focus_order == (focus_index + 1))
            for focus_index in range(self.num_foci())
        ])

    def compute_balanced_focus_counts(self, balance_metric_values: np.ndarray, pulse_count: int) -> np.ndarray:
        """Compute per-focus pulse counts that balance a positive per-focus metric."""
        balance_metric_values = np.asarray(balance_metric_values, dtype=float)
        if balance_metric_values.shape != (self.num_foci(),):
            raise ValueError(f"Balance metric must have one value per focus ({self.num_foci()})")
        if pulse_count < self.num_foci():
            raise ValueError(f"Pulse count ({pulse_count}) must be greater than or equal to number of foci ({self.num_foci()})")
        if np.any(~np.isfinite(balance_metric_values)) or np.any(balance_metric_values <= 0):
            raise ValueError("Balance metric values must be finite and positive")

        remaining_pulses = pulse_count - self.num_foci()
        counts = np.ones(self.num_foci(), dtype=int)
        if remaining_pulses == 0:
            return counts

        weights = 1 / balance_metric_values
        ideal_extra_counts = weights / np.sum(weights) * remaining_pulses
        extra_counts = np.floor(ideal_extra_counts).astype(int)
        counts += extra_counts

        leftover_pulses = remaining_pulses - int(np.sum(extra_counts))
        remainders = ideal_extra_counts - extra_counts
        for focus_index in np.argsort(remainders)[::-1][:leftover_pulses]:
            counts[focus_index] += 1
        return counts

    def build_focus_order(self, focus_counts: np.ndarray, ordering: str = "minimize_repeats") -> list[int]:
        """Build a focus order from per-focus pulse counts."""
        if ordering != "minimize_repeats":
            raise ValueError(f"Unsupported focus ordering '{ordering}'")
        focus_counts = np.asarray(focus_counts, dtype=int)
        if focus_counts.shape != (self.num_foci(),):
            raise ValueError(f"Focus counts must have one value per focus ({self.num_foci()})")
        if np.any(focus_counts < 0):
            raise ValueError("Focus counts must be non-negative")

        heap = []
        for focus_index, focus_count in enumerate(focus_counts):
            if focus_count > 0:
                heap.append((-focus_count, focus_index + 1))

        heapq.heapify(heap)
        focus_order = []
        previous_count = 0
        previous_focus_index = None

        while heap or previous_count < 0:
            if not heap:
                focus_order.append(previous_focus_index)
                previous_count += 1
                continue

            focus_count, focus_index = heapq.heappop(heap)
            focus_order.append(focus_index)
            focus_count += 1

            if previous_count < 0:
                heapq.heappush(heap, (previous_count, previous_focus_index))

            previous_count = focus_count
            previous_focus_index = focus_index
        return focus_order

    def simulate(self,
        params: xa.Dataset,
        sim_options: SimSetup | None = None,
        _force_cpu: bool = False) -> xa.Dataset:
        """Run a simulation for this solution and store the result in the `simulation_result` attribute.

        Args:
            params: The simulation parameters as an xarray Dataset.
            sim_options: Optional simulation setup options.
            _force_cpu: Whether to force the simulation to run on the CPU. If True, the GPU will not be used even if available.

        Returns:
            The simulation result as an xarray Dataset.
        """

        if _force_cpu:
            use_gpu = False
        else:
            use_gpu = gpu_available()

        if sim_options is None:
            sim_options = SimSetup()
        simulation_outputs_to_stack: List[xa.Dataset] = []
        simulation_cycles = np.round(self.pulse.duration * self.pulse.frequency)
        for focidx, focus in enumerate(self.foci):
            delays = self.delays[focidx, :]
            apodization = self.apodizations[focidx, :]
            simulation_output_xarray = None
            self.logger.info(f"Running k-Wave simulation for focus {focus}...")
            run_simulation_kwargs = {
                    "arr": self.transducer,
                    "params": params,
                    "delays": delays,
                    "apod": apodization,
                    "freq": self.pulse.frequency,
                    "cycles": simulation_cycles,
                    "dt": sim_options.dt,
                    "t_end": sim_options.t_end,
                    "cfl": sim_options.cfl,
                    "amplitude": self.pulse.amplitude * self.voltage,
                    "gpu": use_gpu
                }
            run_simulation_kwargs.update(sim_options.options)
            simulation_output_xarray = run_simulation(
                **run_simulation_kwargs)
            simulation_outputs_to_stack.append(simulation_output_xarray)
        return xa.concat(
            [
                sim.assign_coords(focal_point_index=i)
                for i, sim in enumerate(simulation_outputs_to_stack)
            ],
            dim='focal_point_index',
        )

    def analyze(self,
                simulation_result: xa.Dataset | None = None,
                options: SolutionAnalysisOptions = SolutionAnalysisOptions(),
                param_constraints: Dict[str,ParameterConstraint] | None = None,
                focus_counts: np.ndarray | None = None) -> SolutionAnalysis:
        """Analyzes the treatment solution.

        Args:
            simulation_result: The simulation result dataset to analyze. If None, uses self.simulation_result.
            options: A struct for solution analysis options.
            param_constraints: A dictionary of parameter constraints to apply to the analysis.
                The keys are the parameter names and the values are the ParameterConstraint objects.
            focus_counts: Optional per-focus pulse counts to use for ITA calculations.

        Returns: A struct containing the results of the analysis.
        """
        if param_constraints is None:
            param_constraints = {}
        solution_analysis = SolutionAnalysis()

        dt = 1 / (self.pulse.frequency * 20)
        t = self.pulse.calc_time(dt)
        input_signal_V = self.pulse.calc_pulse(t) * self.voltage

        if simulation_result is None:
            if self.simulation_result is None or len(self.simulation_result)==0:
                raise ValueError("No simulation result provided for analysis, and no simulation result found in the Solution.")
            simulation_result = self.simulation_result

        pnp_MPa_all = rescale_data_arr(rescale_coords(simulation_result['p_min'], options.distance_units),"MPa")
        ipa_Wcm2_all = rescale_data_arr(rescale_coords(simulation_result['intensity'], options.distance_units), "W/cm^2")

        if options.sidelobe_radius is np.nan:
            options.sidelobe_radius = options.mainlobe_radius


        standoff_Z = options.standoff_density * 1500
        c_tic = 40e-3  # W cm-1
        A_cm = self.transducer.get_area("cm")
        d_eq_cm = np.sqrt(4*A_cm / np.pi)
        ele_sizes_cm2 = np.array([elem.get_area("cm") for elem in self.transducer.elements])

        # xyz = np.stack(np.meshgrid(*coords, indexing="xy"), axis=-1)  #TODO: if fus.Axis is defined, coords.ndgrid(dim="z")
        # z_mask = xyz[..., -1] >= options.sidelobe_zmin  #TODO: probably wrong here, should be z{1}>=options.sidelobe_zmin;

        solution_analysis.duty_cycle_pulse_train_pct = self.get_pulsetrain_dutycycle()*100
        solution_analysis.duty_cycle_sequence_pct = self.get_sequence_dutycycle()*100
        if self.sequence.pulse_train_interval == 0:
            solution_analysis.sequence_duration_s = float(self.sequence.pulse_interval * self.sequence.pulse_count * self.sequence.pulse_train_count)
        else:
            solution_analysis.sequence_duration_s = float(self.sequence.pulse_train_interval * self.sequence.pulse_train_count)
        ita_mWcm2 = rescale_coords(self.get_ita(intensity=simulation_result['intensity'], units="mW/cm^2", focus_counts=focus_counts), options.distance_units)

        power_W = np.zeros(self.num_foci())
        TIC = np.zeros(self.num_foci())
        for focus_index in range(self.num_foci()):
            pnp_MPa = pnp_MPa_all.isel(focal_point_index=focus_index)
            ipa_Wcm2 = ipa_Wcm2_all.isel(focal_point_index=focus_index)
            focus = self.foci[focus_index].get_position(units=options.distance_units)
            focus_mm = self.foci[focus_index].get_position(units="mm")
            solution_analysis.target_position_lat_mm += [focus_mm[0]]
            solution_analysis.target_position_ele_mm += [focus_mm[1]]
            solution_analysis.target_position_ax_mm += [focus_mm[2]]  # TODO: this should be a list, not a single value
            apodization = self.apodizations[focus_index]
            origin = self.transducer.get_effective_origin(apodizations=apodization, units=options.distance_units)

            p0_Pa = np.max(self.transducer.calc_output(
                cycles=1,
                frequency=self.pulse.frequency,
                dt=dt,
                delays=self.delays[focus_index, :],
                apod=self.apodizations[focus_index, :],
                amplitude=self.pulse.amplitude * self.voltage,
            ), axis=1)

            mainlobe_mask = get_mask(
                pnp_MPa,
                focus = focus,
                origin = origin,
                distance = options.mainlobe_radius,
                operator = '<',
                aspect_ratio = options.mainlobe_aspect_ratio
            )

            sidelobe_mask = get_mask(
                pnp_MPa,
                focus = focus,
                origin = origin,
                distance = options.sidelobe_radius,
                operator = '>',
                aspect_ratio=options.mainlobe_aspect_ratio
            )
            z_dim = pnp_MPa.dims[2]
            z_mask = pnp_MPa.coords[z_dim] > options.sidelobe_zmin
            sidelobe_mask = sidelobe_mask.where(z_mask, False)

            pnp_mainlobe = pnp_MPa.where(mainlobe_mask)
            pk = float(pnp_mainlobe.max())
            mainlobe_focus = find_centroid(pnp_mainlobe, pk*10**(-3/20), "mm")
            solution_analysis.focal_centroid_lat_mm += [mainlobe_focus[0]]
            solution_analysis.focal_centroid_ele_mm += [mainlobe_focus[1]]
            solution_analysis.focal_centroid_ax_mm += [mainlobe_focus[2]]

            solution_analysis.mainlobe_pnp_MPa += [pk]
            solution_analysis.focal_gain += [pk*1e6/np.max(p0_Pa)]

            for dim, named_dim, scale in zip(pnp_MPa.dims, ("lat","ele","ax"), options.mainlobe_aspect_ratio):
                for threshdB in [3, 6]:
                    attr_name = f'beamwidth_{named_dim}_{threshdB}dB_mm'
                    bw0 = getattr(solution_analysis, attr_name)
                    cutoff = pk*10**(-threshdB / 20)
                    bw = get_beamwidth(
                        pnp_MPa,
                        focus=focus,
                        dim=dim,
                        cutoff=cutoff,
                        origin=origin,
                        min_offset=-scale*options.beamwidth_radius,
                        max_offset=scale*options.beamwidth_radius)
                    bw = getunitconversion(options.distance_units, "mm") * bw
                    bw = [*bw0, bw]
                    setattr(solution_analysis, attr_name, bw)

            solution_analysis.mainlobe_isppa_Wcm2 += [float(ipa_Wcm2.where(mainlobe_mask).max())]
            solution_analysis.mainlobe_ispta_mWcm2 += [float(ita_mWcm2.where(mainlobe_mask).max())]

            sidelobe_pnp = float(pnp_MPa.where(sidelobe_mask).max())
            sidelobe_isppa = float(ipa_Wcm2.where(sidelobe_mask).max())

            solution_analysis.sidelobe_pnp_MPa += [sidelobe_pnp]
            solution_analysis.sidelobe_isppa_Wcm2 += [sidelobe_isppa]

            # Calculate and store ratios (sidelobe / mainlobe)
            mainlobe_pnp = solution_analysis.mainlobe_pnp_MPa[-1]
            mainlobe_isppa = solution_analysis.mainlobe_isppa_Wcm2[-1]

            if mainlobe_pnp == 0:
                pressure_ratio = np.inf if sidelobe_pnp != 0 else np.nan
            else:
                pressure_ratio = sidelobe_pnp / mainlobe_pnp
            solution_analysis.sidelobe_to_mainlobe_pressure_ratio += [pressure_ratio]

            if mainlobe_isppa == 0:
                intensity_ratio = np.inf if sidelobe_isppa != 0 else np.nan
            else:
                intensity_ratio = sidelobe_isppa / mainlobe_isppa
            solution_analysis.sidelobe_to_mainlobe_intensity_ratio += [intensity_ratio]

            solution_analysis.global_pnp_MPa += [float(pnp_MPa.where(z_mask).max())]
            solution_analysis.global_isppa_Wcm2 += [float(ipa_Wcm2.where(z_mask).max())]
            i0_Wcm2 = (p0_Pa**2 / (2*standoff_Z)) * 1e-4
            i0ta_Wcm2 = i0_Wcm2 * solution_analysis.duty_cycle_sequence_pct/100
            power_W[focus_index] = np.mean(np.sum(i0ta_Wcm2 * ele_sizes_cm2 * self.apodizations[focus_index, :]))
            TIC[focus_index] = power_W[focus_index] / (d_eq_cm * c_tic)
            solution_analysis.p0_MPa += [1e-6*np.max(p0_Pa)]
        solution_analysis.global_ispta_mWcm2 = float((ita_mWcm2*z_mask).max())
        solution_analysis.MI = (np.max(solution_analysis.mainlobe_pnp_MPa)/np.sqrt(self.pulse.frequency*1e-6))
        solution_analysis.TIC = np.mean(TIC)
        solution_analysis.voltage_V = self.voltage
        solution_analysis.power_W = np.mean(power_W)
        solution_analysis.estimated_tx_temperature_rise_C = self.estimate_tx_temperature_rise(
            t_sec=solution_analysis.sequence_duration_s,
        )
        solution_analysis.param_constraints = param_constraints
        return solution_analysis

    def estimate_tx_temperature_rise(self,
                                  t_sec: float,
                                  T0_degC: float = 30.0):

        """
        Standalone temperature prediction function for thermal modeling.

        Based on physics-based power decay gradient model fitted from experimental data.
        Model: dT/dt = (t + t_shift)^(-n) + C

        Parameters:
        -----------
        self: Solution
            The treatment solution containing voltage and pulse information.
        t_sec: float
            Time in seconds for which to predict temperature rise.
        T0_degC: float
            Initial temperature in Celsius. Default is 30.0°C.

        Returns:
        --------
        float
            Predicted temperature rise in Celsius
        """
        return model_tx_temperature_rise(
            voltage=self.voltage,
            t_sec=t_sec,
            duty_cycle=self.get_sequence_dutycycle(),
            apodization_fraction=np.mean(self.apodizations),
            frequency_kHz=self.pulse.frequency / 1e3,
            T0_degC=T0_degC,
        )

    def compute_scaling_factors(
            self,
            focal_pattern: FocalPattern,
            analysis: SolutionAnalysis
        ) -> Tuple[np.ndarray, float, float]:
        """

        Compute the scaling factors used to re-scale the apodizations, simulation results and pulse amplitude.

        Args:
            focal_pattern: FocalPattern
            analysis: SolutionAnalysis

        Returns:
            apod_factors: A np.ndarray apodization factors
            v0: A float representing the original pulse amplitude
            v1: A float representing the new pulse amplitude
        """
        scaling_factors = np.zeros(self.num_foci())

        for i in range(self.num_foci()):
            focal_pattern_pressure_in_MPa = focal_pattern.target_pressure * getunitconversion(focal_pattern.units, "MPa")
            scaling_factors[i] = focal_pattern_pressure_in_MPa / analysis.mainlobe_pnp_MPa[i]
        max_scaling = np.max(scaling_factors)
        v0 = self.voltage
        v1 = v0 * max_scaling
        apod_factors = scaling_factors / max_scaling

        return apod_factors, v0, v1

    def scale(
            self,
            focal_pattern: FocalPattern,
            analysis_options: SolutionAnalysisOptions = SolutionAnalysisOptions(),
            balance_method: str | None = None,
            balance_metric: str = "mainlobe_ispta_mWcm2",
            ordering: str = "minimize_repeats",
    ) -> None:
        """
        Scale the solution in-place to match the target pressure.

        Args:
            focal_pattern: FocalPattern
            analysis_options: plan.solution.SolutionAnalysisOptions
            balance_method: Optional method for balancing scaled delivery. Supported: "ispta_repeats".
            balance_metric: The per-focus analysis metric used for balancing.
            ordering: How to order balanced repeats. Supported: "minimize_repeats".

        Returns:
            analysis_scaled: the resulting plan.solution.SolutionAnalysis from scaled solution
        """
        analysis = self.analyze(options=analysis_options)

        apod_factors, v0, v1 = self.compute_scaling_factors(focal_pattern, analysis)

        for i in range(self.num_foci()):
            scaling = v1/v0*apod_factors[i]
            self.simulation_result['p_min'][i].data *= scaling
            self.simulation_result['p_max'][i].data *= scaling
            self.simulation_result['intensity'][i].data *= scaling**2
            self.apodizations[i] = self.apodizations[i]*apod_factors[i]
        self.voltage = v1

        if balance_method is None:
            return
        if balance_method != "ispta_repeats":
            raise ValueError(f"Unsupported balance method '{balance_method}'")
        baseline_focus_counts = np.ones(self.num_foci(), dtype=int)
        scaled_analysis = self.analyze(options=analysis_options, focus_counts=baseline_focus_counts)
        if not hasattr(scaled_analysis, balance_metric):
            raise ValueError(f"Unknown balance metric '{balance_metric}'")
        balance_metric_values = np.array(getattr(scaled_analysis, balance_metric))
        focus_counts = self.compute_balanced_focus_counts(balance_metric_values, self.sequence.pulse_count)
        self.sequence.focus_order = self.build_focus_order(focus_counts, ordering=ordering)

    def get_pulsetrain_dutycycle(self) -> float:
        """
        Compute the pulse train dutycycle given a sequence.

        Returns:
            A float.
        """
        pulsetrain_dutycycle = min(1., self.pulse.duration / self.sequence.pulse_interval)

        return pulsetrain_dutycycle

    def get_sequence_dutycycle(self) -> float:
        """
        Compute the overall sequence dutycycle given a sequence.

        Returns:
            A float.
        """
        if self.sequence.pulse_train_interval == 0:
            between_pulsetrain_duty_cycle = 1
        else:
            between_pulsetrain_duty_cycle = (self.sequence.pulse_count * self.sequence.pulse_interval) / self.sequence.pulse_train_interval
        sequence_duty_cycle = self.get_pulsetrain_dutycycle() * between_pulsetrain_duty_cycle
        return sequence_duty_cycle

    def get_ita(
            self,
            intensity: xa.DataArray | None = None,
            units: str = "mW/cm^2",
            focus_counts: np.ndarray | None = None
    ) -> xa.DataArray:
        """
        Calculate the intensity-time-area product for a treatment solution.

        Args:
            intensity: xa.DataArray | None
                If provided, use this intensity data array instead of the one from the simulation result.
            units: str
                Target units. Default "mW/cm^2".
            focus_counts: Optional per-focus pulse counts. If not provided, use the sequence focus order.

        Returns:
            xa.DataArray
                The intensity-time-area product as an xarray DataArray.
        """
        if intensity is not None:
            intensity_scaled = rescale_data_arr(intensity, units)
        else:
            intensity_scaled = rescale_data_arr(self.simulation_result['intensity'], units)
        pulsetrain_dutycycle = self.get_pulsetrain_dutycycle()
        treatment_dutycycle = self.get_sequence_dutycycle()
        if focus_counts is None:
            focus_counts = self.get_focus_counts()
        focus_counts = np.asarray(focus_counts)
        if focus_counts.shape != (self.num_foci(),):
            raise ValueError(f"Focus counts must have one value per focus ({self.num_foci()})")
        if np.any(focus_counts < 0):
            raise ValueError("Focus counts must be non-negative")
        counts = focus_counts.reshape((1, 1, 1, self.num_foci()))
        intensity = intensity_scaled.copy(deep=True)
        isppa_avg = np.sum(np.expand_dims(intensity.data, axis=-1) * counts, axis=-1) / np.sum(counts)
        intensity.data = isppa_avg * pulsetrain_dutycycle * treatment_dutycycle

        return intensity

    def to_dict(self, include_simulation_data: bool = False) -> dict:
        """Serialize a Solution to a dictionary

        Args:
            include_simulation_data: if enabled then large simulation data arrays are included in the dict,
                otherwise they are excluded.

        Returns: A dictionary representing the complete Solution object.
        """
        solution_dict = asdict(self)
        if 'logger' in solution_dict:
            solution_dict.pop('logger')

        if not include_simulation_data:
            solution_dict.pop('simulation_result')

        return solution_dict

    def to_json(self, include_simulation_data: bool, compact: bool) -> str:
        """Serialize a Solution to a json string

        Args:
            include_array_data: if enabled then large simulation data arrays are serialized somehow into the json,
                so that they can be recovered via `from_json` alone. otherwise they are excluded.
            compact: if enabled then the string is compact (not pretty). Disable for pretty.

        Returns: A json string representing the complete Solution object.
        """
        solution_dict = asdict(self)

        if include_simulation_data:
            # Serialize xarray dataset into a string
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self.simulation_result.to_netcdf(tmp_path, engine='scipy')
                # (to_netcdf can write to general byte stream instead of a file, but its behavior is changing
                # in a way that will likely break how we do it in only some python versions,
                # so writing to a tempfile is a fool proof solution across versions until to_netcdf stabilizes in its behavior)
                raw_bytes = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)
            solution_dict['simulation_result'] = base64.b64encode(raw_bytes).decode('utf-8')
        else:
            solution_dict.pop('simulation_result')

        if compact:
            return json.dumps(solution_dict, separators=(',', ':'), cls=PYFUSEncoder)
        else:
            return json.dumps(solution_dict, indent=4, cls=PYFUSEncoder)

    @staticmethod
    def from_dict(solution_dict: dict) -> Solution:
        """Load a Solution from a dictionary.

        Args:
            solution_dict: The dictionary containing the solution data.

        Returns: The new Solution object.
        """
        # Convert the dictionary back into a Solution object
        solution_dict["date_created"] = datetime.fromisoformat(solution_dict["date_created"])
        if solution_dict["delays"] is not None:
            solution_dict["delays"] = np.array(solution_dict["delays"])
        if solution_dict["apodizations"] is not None:
            solution_dict["apodizations"] = np.array(solution_dict["apodizations"], ndmin=2)
            solution_dict["apodizations"] = np.array(solution_dict["apodizations"], ndmin=2)
        if solution_dict["transducer"] is not None:
            solution_dict["transducer"] = Transducer.from_dict(solution_dict["transducer"])
        solution_dict["pulse"] = Pulse.from_dict(solution_dict["pulse"])
        solution_dict["sequence"] = Sequence.from_dict(solution_dict["sequence"])
        solution_dict["foci"] = [
            Point.from_dict(focus_dict)
            for focus_dict in solution_dict["foci"]
        ]
        if solution_dict["target"] is not None:
            solution_dict["target"] = Point.from_dict(solution_dict["target"])

        if "simulation_result" in solution_dict and isinstance(solution_dict["simulation_result"], str):
            solution_dict["simulation_result"] = xa.open_dataset(
                base64.b64decode(
                    solution_dict["simulation_result"].encode('utf-8')
                ),
                engine='scipy',
            )

        return Solution(**solution_dict)


    @staticmethod
    def from_json(json_string : str, simulation_result: xa.Dataset | None=None) -> Solution:
        """Load a Solution from a json string.

        Args:
            json_string: the json string defining the Solution
            simulation_result: the simulation result arrays to use. If the json string has this then it will
                be read from the json string and it should not be provided in this argument.

        Returns: The new Solution object.
        """
        solution_dict = json.loads(json_string)
        if simulation_result is not None:
            if "simulation_result" in solution_dict:
                raise ValueError(
                    "A simulation result was provided while the json string already contains `simulation_result`. "
                    "Unclear which to use!"
                )
            solution_dict["simulation_result"] = simulation_result

        return Solution.from_dict(solution_dict)

    def to_files(self, json_filepath:Path, nc_filepath:Path | None=None) -> None:
        """Save the solution to json and netCDF files.

        json_filepath: where to save the json file with all data except the simulation results dataset
        nc_filepath: where to save the netCDF file containing the simulation results.
            If None then it will be saved in the same directory as the json file, with the same name but with
            the extension *.nc
        """
        if nc_filepath is None:
            nc_filepath = _construct_nc_filepath_from_json_filepath(json_filepath)
        json_filepath.parent.mkdir(parents=True, exist_ok=True)
        nc_filepath.parent.mkdir(parents=True, exist_ok=True)
        with json_filepath.open("w") as json_file:
            json_file.write(
                self.to_json(include_simulation_data=False, compact=False)
            )
        self.simulation_result.to_netcdf(nc_filepath, engine='h5netcdf')

    @staticmethod
    def from_files(json_filepath:Path, nc_filepath:Path | None=None):
        """Read solution from json and netCDF files.

        json_filepath: solution json file location, containing all data except the simulation results dataset
        nc_filepath: netCDF file location, containing the simulation results.
            If None then it will be assumed to be saved in the same directory as the json file, with the same name but with
            the extension *.nc. This is the default saving behavior of Solution.to_files.
        """
        if nc_filepath is None:
            nc_filepath = _construct_nc_filepath_from_json_filepath(json_filepath)
        simulation_result = xa.open_dataset(nc_filepath, engine='h5netcdf').load()
        simulation_result.close() # this is needed to release the lock on the file so that it can be written to again
        return Solution.from_json(
            json_string = json_filepath.read_text(),
            simulation_result = simulation_result,
        )
