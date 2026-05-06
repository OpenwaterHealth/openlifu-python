from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd

from openlifu.io.LIFUHVController import HVController
from openlifu.io.LIFUSignal import LIFUSignal
from openlifu.io.LIFUTXDevice import TriggerModeOpts, TxDevice
from openlifu.io.LIFUUart import LIFUUart
from openlifu.plan.solution import Solution

REF_MAX_SEQUENCE_TIMES = [2*60, 5*60, 20*60]
REF_MAX_DUTY_CYCLES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MAX_VOLTAGE_BY_DUTY_CYCLE_AND_SEQUENCE_TIME = [
    [65, 65, 65], # 0.05
    [65, 65, 50], # 0.1
    [50, 40, 35], # 0.2
    [45, 35, 30], # 0.3
    [35, 30, 25], # 0.4
    [30, 25, 20] # 0.5
    ]


class LIFUInterfaceStatus(Enum):
    STATUS_COMMS_ERROR = -1
    STATUS_SYS_OFF = 0
    STATUS_SYS_POWERUP = 1
    STATUS_SYS_ON = 2
    STATUS_PROGRAMMING = 3
    STATUS_READY = 4
    STATUS_NOT_READY = 5
    STATUS_RUNNING = 6
    STATUS_FINISHED = 7
    STATUS_ERROR = 8

logger = logging.getLogger(__name__)

class LIFUInterface:
    signal_connect: LIFUSignal = LIFUSignal()
    signal_disconnect: LIFUSignal = LIFUSignal()
    signal_data_received: LIFUSignal = LIFUSignal()
    hvcontroller: HVController = None
    txdevice: TxDevice = None

    def __init__(self,
                 vid: int = 0x0483,
                 tx_pid: int = 0x57AF,
                 con_pid: int = 0x57A0,
                 baudrate: int = 921600,
                 timeout: int = 10,
                 TX_test_mode: bool = False,
                 HV_test_mode: bool = False,
                 run_async: bool = False,
                 ext_power_supply: bool = False,
                 module_invert: bool | List[bool] = False) -> None:
        """
        Initialize the LIFUInterface with given parameters and store them in the class.

        Args:
            vid (int): Vendor ID of the USB device.
            tx_pid (int): Product ID for TX device.
            con_pid (int): Product ID for console device.
            baudrate (int): Communication baud rate.
            timeout (int): Read timeout in seconds.
            test_mode (bool): Enable test mode.
            run_async (bool): Enable asynchronous operation.
        """
        # Store parameters in instance variables
        self.vid = vid
        self.tx_pid = tx_pid
        self.con_pid = con_pid
        self.baudrate = baudrate
        self.timeout = timeout
        self._test_mode = TX_test_mode
        self._async_mode = run_async
        self._tx_uart = None
        self._hv_uart = None
        self.status = LIFUInterfaceStatus.STATUS_SYS_OFF

        # Create a TXDevice instance as part of the interface
        logger.debug("Initializing TX Module of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, tx_pid, baudrate, timeout)
        self._tx_uart = LIFUUart(vid=vid, pid=tx_pid, baudrate=baudrate, timeout=timeout, desc="TX", demo_mode=TX_test_mode, async_mode=run_async)
        self.txdevice = TxDevice(uart=self._tx_uart, module_invert=module_invert)

        if ext_power_supply:
            logger.debug("External power supply selected, skipping HVController initialization.")
            self.hvcontroller = None
        else:
            # Create a LIFUHVController instance as part of the interface
            logger.debug("Initializing Console of LIFUInterface with VID: %s, PID: %s, baudrate: %s, timeout: %s", vid, con_pid, baudrate, timeout)
            self._hv_uart = LIFUUart(vid=vid, pid=con_pid, baudrate=baudrate, timeout=timeout, desc="HV", demo_mode=HV_test_mode, async_mode=run_async)
            self.hvcontroller = HVController(uart=self._hv_uart)


        # Connect signals to internal handlers
        if self._async_mode:
            self._tx_uart.signal_connect.connect(self.signal_connect.emit)
            self._tx_uart.signal_disconnect.connect(self.signal_disconnect.emit)
            self._tx_uart.signal_data_received.connect(self.signal_data_received.emit)
            self._hv_uart.signal_connect.connect(self.signal_connect.emit)
            self._hv_uart.signal_disconnect.connect(self.signal_disconnect.emit)
            self._hv_uart.signal_data_received.connect(self.signal_data_received.emit)

    async def start_monitoring(self, interval: int = 1) -> None:
        """Start monitoring for USB device connections."""
        try:
            await asyncio.gather(
                self._tx_uart.monitor_usb_status(interval),
                self._hv_uart.monitor_usb_status(interval)
            )

        except Exception as e:
            logger.error("Error starting monitoring: %s", e)
            raise e

    def stop_monitoring(self) -> None:
        """Stop monitoring for USB device connections."""
        try:
            self._tx_uart.stop_monitoring()
            self._hv_uart.stop_monitoring()
        except Exception as e:
            logger.error("Error stopping monitoring: %s", e)
            raise e

    def is_device_connected(self) -> tuple:
        """
        Check if the device is currently connected.

        Returns:
            tuple: (tx_connected, hv_connected)
        """
        tx_connected = self.txdevice.is_connected()
        if self.hvcontroller is None:
            hv_connected = False
        else:
            hv_connected = self.hvcontroller.is_connected()
        return tx_connected, hv_connected

    def get_max_voltage(self, solution: Solution | Dict) -> float:
        """
        Get the maximum voltage for a given solution.

        Args:
            solution (Solution | Dict): The solution to check.

        Returns:
            float: The maximum voltage for the solution.
        """
        if isinstance(solution, Solution):
            solution = solution.to_dict()

        sequence_duty_cycle = self.get_sequence_duty_cycle(solution)
        sequence_duration = self.get_sequence_duration(solution)

        # Find the index of the duty cycle in the reference list
        duty_cycles_limits = np.array(REF_MAX_DUTY_CYCLES)
        duty_cycle_index = np.where(duty_cycles_limits >= sequence_duty_cycle)[0][0]

        # Find the index of the duration in the reference list
        duration_limits = np.array(REF_MAX_SEQUENCE_TIMES)
        duration_index = np.where(duration_limits >= sequence_duration)[0][0]

        # Return the maximum voltage for the given duty cycle and duration
        return MAX_VOLTAGE_BY_DUTY_CYCLE_AND_SEQUENCE_TIME[duty_cycle_index][duration_index]

    def get_max_voltage_table(self) -> pd.DataFrame:
        """
        Get a table of the maximum voltages for different duty cycles and sequence times.

        Returns:
            pd.DataFrame: A DataFrame containing the maximum voltages.
        """
        data = {
            "Duty Cycle (%)": [f"<={100 * dc:0.1f}%" for dc in REF_MAX_DUTY_CYCLES],
            }
        for i, duration in enumerate(REF_MAX_SEQUENCE_TIMES):
            col_name = f"<={duration // 60} min"
            data[col_name] = [
                MAX_VOLTAGE_BY_DUTY_CYCLE_AND_SEQUENCE_TIME[j][i] for j in range(len(REF_MAX_DUTY_CYCLES))
            ]
        max_voltage =  pd.DataFrame(data).set_index("Duty Cycle (%)")
        max_voltage.Name = "Maximum Voltage (V)"
        max_voltage.Description = "This table shows the maximum voltage for different duty cycles and sequence times."
        return max_voltage

    def check_solution(self, solution: Solution | Dict) -> None:
        """
        Check if the solution is valid.
        Args:
            solution (Solution | Dict): The solution to check.
        Raises:
            ValueError: If the solution is invalid.
        """
        if isinstance(solution, Solution):
            solution = solution.to_dict()

        sequence_duty_cycle = self.get_sequence_duty_cycle(solution)
        duty_cycles_limits = np.array(REF_MAX_DUTY_CYCLES)
        if sequence_duty_cycle > duty_cycles_limits.max():
            raise ValueError(f"Sequence duty cycle ({100*sequence_duty_cycle:0.1f} %) exceeds maximum allowed duty cycle ({100*duty_cycles_limits.max():0.1f} %).")
        duty_cycle_index = np.where(duty_cycles_limits >= sequence_duty_cycle)[0][0]

        sequence_duration = self.get_sequence_duration(solution)
        duration_limits = np.array(REF_MAX_SEQUENCE_TIMES)
        if sequence_duration > duration_limits.max():
            raise ValueError(f"Sequence duration ({sequence_duration:0.0f} s) exceeds maximum allowed duration ({duration_limits.max()} s).")
        duration_index = np.where(duration_limits >= sequence_duration)[0][0]

        max_voltage = MAX_VOLTAGE_BY_DUTY_CYCLE_AND_SEQUENCE_TIME[duty_cycle_index][duration_index]
        if solution['voltage'] > max_voltage:
            raise ValueError(f"Voltage ({solution['voltage']:0.1f}V) exceeds maximum allowed voltage ({max_voltage:0.1f}V) for duty cycle ({100*sequence_duty_cycle:0.1f} <= {100*duty_cycles_limits[duty_cycle_index]}%) and sequence time ({sequence_duration:0.0f} <= {duration_limits[duration_index]}s).")

    def get_sequence_duty_cycle(self, solution: Solution | Dict) -> float:
        """
        Get the duty cycle of the sequence in the solution.

        Args:
            solution (Solution | Dict): The solution to check.

        Returns:
            float: The duty cycle of the sequence.
        """
        if isinstance(solution, Solution):
            solution = solution.to_dict()

        if solution['sequence']['pulse_train_interval'] == 0:
            return solution['pulse']['duration'] / solution['sequence']['pulse_interval']
        else:
            return (solution['pulse']['duration'] * solution['sequence']['pulse_count']) / solution['sequence']['pulse_train_interval']

    def get_sequence_duration(self, solution: Solution | Dict) -> float:
        """
        Get the duration of the sequence in the solution.

        Args:
            solution (Solution | Dict): The solution to check.

        Returns:
            float: The duration of the sequence.
        """
        if isinstance(solution, Solution):
            solution = solution.to_dict()

        if solution['sequence']['pulse_train_interval'] == 0:
            return solution['sequence']['pulse_interval'] * solution['sequence']['pulse_count'] * solution['sequence']['pulse_train_count']
        else:
            return solution['sequence']['pulse_train_interval'] * solution['sequence']['pulse_train_count']

    def set_module_invert(self, module_invert: bool | List[bool]) -> None:
        if self.txdevice is not None:
            self.txdevice.set_module_invert(module_invert)

    def set_solution(self,
                     solution: Solution | Dict,
                     profile_index:int=1,
                     profile_increment:bool=True,
                     trigger_mode: TriggerModeOpts = "sequence",
                     ) -> None:
        """
        Load a solution to the device.

        Args:
            solution (Solution): The solution to load.
            profile_index (int): The profile index to load the solution to (defaults to 0)
            profile_increment (bool): Increment the profile index
            trigger_mode (TriggerModeOpts): The trigger mode to use (defaults to "sequence")
            module_invert (List[bool]|bool): Invert the signal on all modules (singleton) or specific modules (list) (defaults to False)
        """
        if isinstance(solution, Solution):
            solution = solution.to_dict()

        self.check_solution(solution)

        if "transducer" in solution and solution["transducer"] is not None and "module_invert" in solution["transducer"]:
            self.txdevice.set_module_invert(solution["transducer"]["module_invert"])
        else:
            self.txdevice.set_module_invert(False)

        self.set_status(LIFUInterfaceStatus.STATUS_PROGRAMMING)

        if "name" in solution:
            solution_name = solution["name"]
            solution_name = f'Solution "{solution_name}"'
        else:
            solution_name = "Solution"

        voltage = solution['voltage']
        logger.info("Loading %s...", solution_name)
        # Convert solution data and send to the device
        self.txdevice.set_solution(
                pulse = solution['pulse'],
                delays = solution['delays'],
                apodizations= solution['apodizations'],
                sequence= solution['sequence'],
                profile_index=profile_index,
                profile_increment=profile_increment,
                trigger_mode=trigger_mode
        )
        self.set_status(LIFUInterfaceStatus.STATUS_READY)

        if self.hvcontroller is not None:
            logger.info(f"Setting HV to {voltage} V...")
            self.hvcontroller.set_voltage(voltage)

        logger.info("%s loaded successfully.", solution_name)

    def start_sonication(self) -> bool:
        """
        Start sonication.

        Sets the device to a running state and sends a start command if necessary.
        """
        try:
            if self._test_mode:
                return True

            if self.hvcontroller is not None:
                logger.info("Turn ON HV")
                bHvOn = self.hvcontroller.turn_hv_on()
            else:
                logger.info("Using external power supply, HV will not be turned ON.")
                bHvOn = True

            if self._async_mode:
                self.txdevice.async_mode(True)

            logger.info("Start Sonication")
            # Send the solution data to the device
            bTriggerOn = self.txdevice.start_trigger()

            if bTriggerOn and bHvOn:
                logger.info("Sonication started successfully.")
                self.set_status(LIFUInterfaceStatus.STATUS_RUNNING)
                return True
            else:
                logger.error("Failed to start sonication.")
                return False

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Error Starting sonication: %s", e)
            raise e

    def set_status(self, status: LIFUInterfaceStatus) -> None:
        """
        Set the device status.

        Args:
            status (LIFUInterfaceStatus): The status to set.
        """
        logger.info("Setting device status to %s", status.name)
        self.status = status

    def get_status(self) -> LIFUInterfaceStatus:
        """
        Query the device status.

        Returns:
            int: The device status.
        """
        if self._test_mode:
            return LIFUInterfaceStatus.STATUS_READY

        return self.status

    def stop_sonication(self) -> bool:
        """
        Stop sonication.

        Stops the current sonication process.
        """
        try:
            if self._test_mode:
                return True

            logger.info("Stop Sonication")
            # Send the solution data to the device
            bTriggerOff = self.txdevice.stop_trigger()

            if self.hvcontroller is not None:
                logger.info("Turn OFF HV")
                bHvOff = self.hvcontroller.turn_hv_off()
            else:
                logger.info("Using external power supply, HV will not be turned OFF.")
                bHvOff = True

            if self._async_mode:
                self.txdevice.async_mode(False)

            if bTriggerOff and bHvOff:
                logger.info("Sonication stopped successfully.")
                return True
            else:
                logger.error("Failed to stop sonication.")
                return False

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Error Stopping sonication: %s", e)
            raise e

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_monitoring()
        if self.txdevice:
            self.txdevice.close()
        if self.hvcontroller:
            self.hvcontroller.close()
