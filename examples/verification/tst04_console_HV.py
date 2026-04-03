#!/usr/bin/env python3
"""
LIFU Voltage Sweep Test Script

A professional command-line tool for automated voltage sweep testing of LIFU devices.
This script connects to the device, configures sweep parameters, monitors temperatures,
and logs data with automatic safety shutoffs.

Author: OpenLIFU Team
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from serial.serialutil import SerialException

import numpy as np

import openlifu
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu_sdk.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

__version__ = "1.0.0"
TEST_ID = "tst04_voltage_sweep"
TEST_NAME = "Voltage Sweep Test"

# ------------------- Profile Definitions -------------------
# These correspond to your original profile_choices dict.
PROFILES = {
    1: {
        "id": "low_duty_cycle",
        "name": "Low Duty Cycle",
        "description": "Low duty cycle, high voltage range",
        "pulse_duration_msec": 5,
        "pulse_interval_msec": 100,
        "voltage_end": 65,
    },
    2: {
        "id": "medium_duty_cycle",
        "name": "Medium Duty Cycle",
        "description": "Medium duty cycle, medium voltage range",
        "pulse_duration_msec": 25,
        "pulse_interval_msec": 100,
        "voltage_end": 45,
    },
    3: {
        "id": "high_duty_cycle",
        "name": "High Duty Cycle",
        "description": "High duty cycle, lower voltage range",
        "pulse_duration_msec": 50,
        "pulse_interval_msec": 100,
        "voltage_end": 30,
    },
}

# Frequency choices (kHz)
FREQUENCIES_KHZ = {1: 150, 2: 400}

# Defaults taken from your original script
DEFAULT_VOLTAGE_START = 5.0     # per rail
DEFAULT_VOLTAGE_STEP = 5.0      # per rail
DEFAULT_NUM_MODULES = 2         # "number_of_boards" in original script

# Default safety / logging timing parameters
CONSOLE_SHUTOFF_TEMP_C_DEFAULT = 70.0
TX_SHUTOFF_TEMP_C_DEFAULT = 70.0
AMBIENT_SHUTOFF_TEMP_C_DEFAULT = 70.0
TEMPERATURE_CHECK_INTERVAL_DEFAULT = 1.0
TEMPERATURE_LOG_INTERVAL_DEFAULT = 1.0


class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode characters safely on Windows."""

    def format(self, record):
        """Format the log record, removing unsupported Unicode characters."""
        try:
            msg = super().format(record)
            # Remove or replace emojis/non-ASCII characters for Windows compatibility
            return msg.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return super().format(record)


def format_hhmmss(seconds: float) -> str:
    """Format a number of seconds into HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TestVoltageSweep:
    """Main class for Voltage Sweep Test."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Derived paths
        self.openlifu_dir = Path(openlifu.__file__).parent.parent.parent.resolve()
        self.log_dir = Path(self.args.log_dir or (self.openlifu_dir / "logs")).resolve()

        # Runtime attributes
        self.interface: LIFUInterface | None = None
        self.shutdown_event = threading.Event()
        self.temperature_shutdown_event = threading.Event()
        self.sequence_complete_event = threading.Event()  # used when sweep completes normally

        # Threading locks
        self.mutex = threading.Lock()

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_handler_attached = False

        # Test configuration – set later via interactive selection / arguments
        self.frequency_khz: float | None = None

        self.profile_num: int | None = None
        self.profile: dict | None = None
        self.profile_description: str | None = None
        self.profile_id: str | None = None

        self.start_voltage: float | None = None
        self.end_voltage: float | None = None
        self.voltage_step: float | None = None

        self.pulse_duration_msec: float | None = None
        self.pulse_interval_msec: float | None = None
        self.duty_cycle: float | None = None

        # For delays/apodizations
        self.num_modules: int = self.args.num_modules
        self.numelements: int | None = None

        # Flags from args
        self.use_external_power = self.args.external_power
        self.hw_simulate = self.args.simulate

        # Safety parameters from args
        self.console_shutoff_temp_C = self.args.console_shutoff_temp
        self.tx_shutoff_temp_C = self.args.tx_shutoff_temp
        self.ambient_shutoff_temp_C = self.args.ambient_shutoff_temp
        self.temperature_check_interval = self.args.temperature_check_interval
        self.temperature_log_interval = self.args.temperature_log_interval

        # Logging
        self.logger = self._setup_logging()
        self.logger.debug("TestVoltageSweep initialized with arguments: %s", self.args)

    # ------------------- Logging ------------------- #

    def _setup_logging(self) -> logging.Logger:
        """Configure root logger with console output; file handler added later."""
        logger = logging.getLogger(__name__)

        # Set log level based on verbosity
        if self.args.verbose:
            logger.setLevel(logging.DEBUG)
        elif self.args.quiet:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)

        # Prevent duplicate handlers when re-run
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = SafeFormatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.propagate = False
        return logger

    def _attach_file_handler(self) -> None:
        """Attach a file handler for this run once profile is known."""
        if self._file_handler_attached:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.profile_id is None:
            filename = f"{TEST_ID}_{self.run_timestamp}.log"
        else:
            filename = f"{TEST_ID}_{self.profile_id}_{self.run_timestamp}.log"

        log_path = self.log_dir / filename

        formatter = SafeFormatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logging.getLogger(__name__).addHandler(file_handler)

        self._file_handler_attached = True
        self.logger.info("Run log will be saved to: %s", log_path)

    # ------------------- User Input & Profile Selection ------------------- #

    def _select_frequency(self) -> None:
        """Interactively select frequency if not provided."""
        if self.args.frequency:
            self.frequency_khz = self.args.frequency
            return

        self.logger.info("Choose Frequency:")
        for idx, freq in FREQUENCIES_KHZ.items():
            self.logger.info("  %d. %d kHz", idx, freq)

        while True:
            choice = input(f"Select frequency by number {list(FREQUENCIES_KHZ.keys())}: ").strip()
            if choice.isdigit() and int(choice) in FREQUENCIES_KHZ:
                self.frequency_khz = FREQUENCIES_KHZ[int(choice)]
                break
            self.logger.info("Invalid selection. Please try again.")

    def _select_profile(self) -> None:
        """Select voltage sweep profile via CLI or interactive prompt."""
        if self.args.profile is not None:
            self.profile_num = self.args.profile
            self.profile = PROFILES[self.profile_num]
            return

        # Interactive display
        max_name_len = max(len(p["name"]) for p in PROFILES.values())
        max_desc_len = max(len(p["description"]) for p in PROFILES.values())

        self.logger.info("\nAvailable Voltage Sweep Profiles:")
        for idx, profile in PROFILES.items():
            duty = round(profile["pulse_duration_msec"] / profile["pulse_interval_msec"] * 100.0, 2)
            self.logger.info(
                f"{idx}. {profile['name']:<{max_name_len}} | "
                f"{profile['description']:<{max_desc_len}} | "
                f"Pulse: {profile['pulse_duration_msec']:>3} ms / "
                f"{profile['pulse_interval_msec']:>3} ms "
                f"({duty:>5.2f}%% duty), "
                f"Voltage End: {profile['voltage_end']:>3} V"
            )

        while True:
            choice = input(f"Select a profile by number {list(PROFILES.keys())}: ").strip()
            if choice.isdigit() and int(choice) in PROFILES:
                self.profile_num = int(choice)
                self.profile = PROFILES[self.profile_num]
                break
            self.logger.info("Invalid selection. Please try again.")

    def _derive_profile_parameters(self) -> None:
        """Derive profile-specific parameters and attach file handler."""
        if self.profile is None:
            raise RuntimeError("Profile not selected before deriving parameters.")
        if self.frequency_khz is None:
            raise RuntimeError("Frequency not selected before deriving parameters.")

        self.pulse_duration_msec = float(self.profile["pulse_duration_msec"])
        self.pulse_interval_msec = float(self.profile["pulse_interval_msec"])
        self.duty_cycle = round(self.pulse_duration_msec / self.pulse_interval_msec * 100.0, 2)

        # Voltage configuration
        self.start_voltage = float(self.args.start_voltage)
        # If user did not provide end-voltage, use profile default
        if self.args.end_voltage is not None:
            self.end_voltage = float(self.args.end_voltage)
        else:
            self.end_voltage = float(self.profile["voltage_end"])
        self.voltage_step = float(self.args.voltage_step)

        # Profile description / id for logging + filenames
        self.profile_description = (
            f"{self.frequency_khz}kHz, Profile {self.profile_num} "
            f"({self.profile['name']}): "
            f"{self.start_voltage:.1f}V to {self.end_voltage:.1f}V, "
            f"step size {self.voltage_step:.1f}V, "
            f"{self.duty_cycle:.2f}% duty cycle"
        )
        self.profile_id = (
            f"{self.frequency_khz}kHz_{self.profile['id']}_"
            f"{int(self.start_voltage)}_to_{int(self.end_voltage)}V"
        )

        self._attach_file_handler()

        if self.hw_simulate:
            self.logger.info("Beginning Voltage Sweep %s (TEST MODE)", self.profile_description)
            self.logger.info("TEST MODE: This is a simulated test run.")
            self.logger.info("No actual hardware interactions will occur.")
        else:
            self.logger.info("Selected Frequency: %d kHz", self.frequency_khz)
            self.logger.info("Selected Profile: %d (%s)", self.profile_num, self.profile["name"])
            self.logger.info("Beginning Voltage Sweep %s", self.profile_description)

        # Duty cycle warning as in original script
        if self.duty_cycle > 50:
            self.logger.warning("Duty cycle is above 50%%")

        # Log final user settings summary (similar style to original)
        self.logger.info(
            f"""\n
======================================== User Settings ========================================
{"Voltage Per Rail Starting Value:":33} {self.start_voltage:>5.1f}V (Peak to Peak: {self.start_voltage*2:>5.1f}V)
{"Voltage Per Rail Ending Value:":33} {self.end_voltage:>5.1f}V (Peak to Peak: {self.end_voltage*2:>5.1f}V)
Voltage Step Size: {self.voltage_step:.1f}V

Number of Modules (Boards): {self.num_modules}
Frequency: {self.frequency_khz} kHz
Pulse Duration: {self.pulse_duration_msec:.1f} ms
Pulse Interval: {self.pulse_interval_msec:.1f} ms
Duty Cycle: {self.duty_cycle:.2f} %
===============================================================================================\n
"""
        )

    # ------------------- Device Connection & Setup ------------------- #

    def connect_device(self) -> None:
        """Connect to the LIFU device and verify connection."""
        self.logger.info("Starting %s...", TEST_NAME)
        self.interface = LIFUInterface(
            ext_power_supply=self.use_external_power,
            TX_test_mode=self.hw_simulate,
            HV_test_mode=self.hw_simulate,
            voltage_table_selection="evt0",
            sequence_time_selection="stress_test"
        )
        tx_connected, hv_connected = self.interface.is_device_connected()

        if not self.use_external_power and not tx_connected:
            self.logger.warning("TX device not connected. Attempting to turn on 12V...")
            try:
                self.interface.hvcontroller.turn_12v_on()
            except Exception as e:
                self.logger.error("Error turning on 12V: %s", e)
            time.sleep(2)

            # Reinitialize interface after powering 12V
            try:
                self.interface.stop_monitoring()
            except Exception as e:
                self.logger.warning("Error stopping monitoring during reinit: %s", e)

            with contextlib.suppress(Exception):
                del self.interface

            time.sleep(1)
            self.logger.info("Reinitializing LIFU interface after powering 12V...")
            self.interface = LIFUInterface(
                ext_power_supply=self.use_external_power,
                TX_test_mode=self.hw_simulate,
                HV_test_mode=self.hw_simulate,
                voltage_table_selection="evt0",
                sequence_time_selection="stress_test"
            )
            tx_connected, hv_connected = self.interface.is_device_connected()

        if not self.use_external_power:
            if hv_connected:
                self.logger.info("  HV Connected: %s", hv_connected)
            else:
                self.logger.error("HV NOT fully connected.")
                sys.exit(1)
        else:
            self.logger.info("  Using external power supply")

        if tx_connected:
            self.logger.info("  TX Connected: %s", tx_connected)
            self.logger.info("LIFU Device fully connected.")
        else:
            self.logger.error("TX NOT fully connected.")
            sys.exit(1)

    def verify_communication(self) -> bool:
        """Verify communication with the LIFU device."""
        if self.interface is None:
            self.logger.error("Interface not connected for communication verification.")
            return False

        try:
            if not self.args.external_power and not self.interface.hvcontroller.ping():
                self.logger.error("Failed to ping the console device.")
            return True
        except Exception as e:
            self.logger.error("Console Communication verification failed: %s", e)
            return False

        try:
            if not self.interface.txdevice.ping():
                self.logger.error("Failed to ping the transmitter device.")
            return True
        except Exception as e:
            self.logger.error("TX Device Communication verification failed: %s", e)
            return False

    def get_firmware_versions(self) -> None:
        """Retrieve and log firmware versions from the LIFU device."""
        if self.interface is None:
            self.logger.error("Interface not connected for firmware version retrieval.")
            return

        try:
            if not self.use_external_power:
                console_fw = self.interface.hvcontroller.get_version()
                self.logger.info("Console Firmware Version: %s", console_fw)
        except Exception as e:
            self.logger.error("Error retrieving console firmware version: %s", e)

        try:
            tx_fw = self.interface.txdevice.get_version()
            self.logger.info("TX Device Firmware Version: %s", tx_fw)
        except Exception as e:
            self.logger.error("Error retrieving TX device firmware version: %s", e)

    def enumerate_devices(self) -> None:
        """Enumerate TX7332 devices and verify count; set numelements."""
        if self.interface is None:
            raise RuntimeError("Interface not connected for device enumeration.")

        self.logger.info("Enumerate TX7332 chips")
        num_tx_devices = self.interface.txdevice.enum_tx7332_devices()

        if num_tx_devices == 0:
            raise ValueError("No TX7332 devices found.")
        elif num_tx_devices == self.num_modules * 2:
            self.logger.info("Number of TX7332 devices found: %d", num_tx_devices)
            # 32 channels per TX7332
            self.numelements = 32 * num_tx_devices
        else:
            raise Exception(
                f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.num_modules}"
            )

    def configure_solution(self) -> None:
        """Configure a simple uniform solution and load it into the device."""
        if self.interface is None:
            raise RuntimeError("Interface not connected.")
        if self.numelements is None:
            raise RuntimeError("Devices not enumerated before configure_solution.")
        if self.frequency_khz is None or self.pulse_duration_msec is None or self.pulse_interval_msec is None:
            raise RuntimeError("Profile/frequency not fully configured before configure_solution.")

        # Simple uniform delays/apodizations as in your original script
        delays = np.zeros((1, self.numelements))
        apodizations = np.ones((1, self.numelements))
        pin_order = np.arange(0, self.numelements)

        pulse = Pulse(
            frequency=self.frequency_khz * 1e3,
            duration=self.pulse_duration_msec * 1e-3,
        )

        # Original script used pulse_count for 60s worth of pulses:
        pulse_interval_sec = self.pulse_interval_msec * 1e-3
        pulse_count = int(60.0 / pulse_interval_sec)

        sequence = Sequence(
            pulse_interval=pulse_interval_sec,
            pulse_count=pulse_count,
            pulse_train_interval=0,
            pulse_train_count=1,
        )

        solution = Solution(
            delays=delays[:, pin_order],
            apodizations=apodizations[:, pin_order],
            pulse=pulse,
            voltage=self.start_voltage,
            sequence=sequence,
        )

        profile_index = 1
        profile_increment = True
        trigger_mode = "continuous"

        self.interface.set_solution(
            solution=solution,
            profile_index=profile_index,
            profile_increment=profile_increment,
            trigger_mode=trigger_mode,
        )

        self.logger.info("Solution configured for Voltage Sweep profile %s.", self.profile_id)

    # ------------------- Temperature Monitoring ------------------- #

    def monitor_temperature(self) -> None:
        """Thread target: monitor temperatures and trigger shutdown on safety violations."""
        if self.hw_simulate:
            self.logger.info("Temperature monitoring skipped in hardware simulation mode.")
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
            return

        if self.interface is None:
            self.logger.error("Interface is not initialized in monitor_temperature.")
            return

        start_time = time.time()
        last_log_time = 0.0

        prev_tx_temp = None
        prev_amb_temp = None
        prev_con_temp = None

        while True:
            if self.shutdown_event.is_set():
                return

            time_elapsed = time.time() - start_time

            # Read temperatures
            try:
                if not self.use_external_power:
                    if prev_con_temp is None:
                        with self.mutex:
                            prev_con_temp = self.interface.hvcontroller.get_temperature1()
                    with self.mutex:
                        con_temp = self.interface.hvcontroller.get_temperature1()
                else:
                    con_temp = None

                if prev_tx_temp is None:
                    with self.mutex:
                        prev_tx_temp = self.interface.txdevice.get_temperature()
                with self.mutex:
                    tx_temp = self.interface.txdevice.get_temperature()

                if prev_amb_temp is None:
                    with self.mutex:
                        prev_amb_temp = self.interface.txdevice.get_ambient_temperature()
                with self.mutex:
                    amb_temp = self.interface.txdevice.get_ambient_temperature()

            except SerialException as e:
                self.logger.error("SerialException encountered while reading temperatures: %s", e)
                break
            except Exception as e:
                self.logger.error("Unexpected error while reading temperatures: %s", e)
                break

            # Periodic logging
            time_since_last_log = time_elapsed - last_log_time
            if time_since_last_log >= self.temperature_log_interval:
                last_log_time = time_elapsed
                if not self.use_external_power and con_temp is not None:
                    self.logger.info(
                        "  Console Temp: %.2f°C, TX Temp: %.2f°C, Ambient Temp: %.2f°C",
                        con_temp,
                        tx_temp,
                        amb_temp,
                    )
                else:
                    self.logger.info(
                        "  TX Temp: %.2f°C, Ambient Temp: %.2f°C",
                        tx_temp,
                        amb_temp,
                    )

            # Absolute temperature thresholds
            if (not self.use_external_power and con_temp is not None and
                    con_temp > self.console_shutoff_temp_C):
                self.logger.warning(
                    "Console temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    con_temp,
                    self.console_shutoff_temp_C,
                )
                break

            if tx_temp > self.tx_shutoff_temp_C:
                self.logger.warning(
                    "TX device temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    tx_temp,
                    self.tx_shutoff_temp_C,
                )
                break

            if amb_temp > self.ambient_shutoff_temp_C:
                self.logger.warning(
                    "Ambient temperature %.2f°C exceeds shutoff threshold %.2f°C.",
                    amb_temp,
                    self.ambient_shutoff_temp_C,
                )
                break

            time.sleep(self.temperature_check_interval)

        self.logger.warning("Temperature shutdown triggered.")
        self.shutdown_event.set()
        self.temperature_shutdown_event.set()

    # ------------------- Power-down & Cleanup ------------------- #

    def turn_off_console_and_tx(self) -> None:
        """Safely turn off HV and 12V if console is used."""
        if self.interface is None:
            return
        if self.use_external_power:
            return

        try:
            self.logger.info("Turning off HV and 12V...")
            with contextlib.suppress(Exception):
                with self.mutex:
                    self.interface.hvcontroller.turn_hv_off()
            with contextlib.suppress(Exception):
                with self.mutex:
                    self.interface.hvcontroller.turn_12v_off()
            self.logger.info("HV and 12V turned off.")
        except Exception as e:
            self.logger.warning("Error turning off HV/12V: %s", e)

    def cleanup_interface(self) -> None:
        """Safely cleanup the LIFU interface."""
        if self.interface is None:
            return
        try:
            self.logger.info("Closing device interface...")
            with contextlib.suppress(Exception):
                self.interface.stop_monitoring()
            time.sleep(0.2)
            del self.interface
        except Exception as e:
            self.logger.warning("Issue closing LIFU interface: %s", e)
        finally:
            self.interface = None

    # ------------------- Main Run Logic (Voltage Sweep) ------------------- #

    def run(self) -> None:
        """Execute the voltage sweep test with graceful shutdown."""
        test_status = "not started"

        try:
            # Selection & derivation
            self._select_frequency()
            self._select_profile()
            self._derive_profile_parameters()

            # Device connection & configuration
            if not self.hw_simulate:
                self.connect_device()
                self.verify_communication()
                self.get_firmware_versions()
                self.enumerate_devices()
                self.configure_solution()
            else:
                self.logger.info("Hardware simulation enabled; skipping device configuration.")

            # Optional start prompt
            if not self.args.no_prompt:
                self.logger.info("Press enter to START Voltage Sweep: %s", self.profile_description)
                input()

            # Start sonication
            if not self.hw_simulate:
                self.logger.info("Starting Trigger...")
                with self.mutex:
                    if not self.interface.start_sonication():
                        self.logger.error("Failed to start trigger.")
                        test_status = "error"
                        return
            else:
                self.logger.info("Simulated Trigger start... (no hardware)")

            self.logger.info("Trigger Running... (Press CTRL-C to stop early)")
            test_status = "running"

            # Start temperature monitoring thread
            temp_thread = threading.Thread(
                target=self.monitor_temperature,
                name="TemperatureMonitorThread",
                daemon=True,
            )
            temp_thread.start()

            # Voltage sweep loop
            current_voltage = self.start_voltage
            self.logger.info("Starting Voltage Sweep...")
            self.logger.info("Initial Voltage (per rail): %.1f V", current_voltage)

            # This loop mirrors your original voltage stepping behavior
            while True:
                if self.shutdown_event.is_set():
                    break

                # Determine next voltage step
                next_voltage = current_voltage + self.voltage_step

                # If we have already reached or exceeded end_voltage, final prompt & exit
                if next_voltage > self.end_voltage + 1e-6:
                    input("Reached final voltage step. Press ENTER to end the test: ")
                    self.sequence_complete_event.set()
                    break

                # Prompt user to continue to next step or quit
                user_input = input(
                    f"\nPress ENTER to continue to next voltage step of {next_voltage:.1f} V "
                    "(or press 'q' to quit): \n"
                ).strip().lower()

                if user_input == "q":
                    self.logger.info("User selected 'q', exiting voltage sweep...")
                    break
                if self.shutdown_event.is_set():
                    break

                # Perform HV step (if not in simulate mode and console is used)
                if not self.hw_simulate and not self.use_external_power:
                    try:
                        self.logger.info("Turning HV off before setting new voltage...")
                        with self.mutex:
                            self.interface.hvcontroller.turn_hv_off()
                        time.sleep(2.0)

                        self.logger.info("Setting Voltage to %.1f V...", next_voltage)
                        with self.mutex:
                            if not self.interface.hvcontroller.set_voltage(next_voltage):
                                self.logger.error("Failed to set voltage to %.1f V", next_voltage)
                                test_status = "error"
                                break
                        with self.mutex:
                            self.interface.hvcontroller.turn_hv_on()
                        time.sleep(5.0)
                        self.logger.info("Voltage set to %.1f V", next_voltage)
                    except Exception as e:
                        self.logger.error("Error during voltage step: %s", e)
                        test_status = "error"
                        break
                else:
                    self.logger.info(
                        "Simulated or external-power mode: logically stepping to %.1f V (no HV change).",
                        next_voltage,
                    )

                current_voltage = next_voltage

                # Check if temperature shutdown occurred after this step
                if self.temperature_shutdown_event.is_set():
                    self.logger.warning("Temperature shutdown event detected during sweep.")
                    break

            # Test finished
            if not self.shutdown_event.is_set():
                # If we exited normally (no temp thread forcing shutdown), mark shutdown
                self.shutdown_event.set()

            # Stop sonication
            if not self.hw_simulate and self.interface is not None:
                try:
                    with self.mutex:
                        if self.interface.stop_sonication():
                            self.logger.info("Trigger stopped successfully.")
                        else:
                            self.logger.error("Failed to stop trigger.")
                except Exception as e:
                    self.logger.error("Error stopping trigger: %s", e)

            # Wait for temperature thread to exit gracefully
            temp_thread.join(timeout=2.0)

            # Determine final status
            if test_status not in ("error",):
                if self.temperature_shutdown_event.is_set():
                    test_status = "temperature shutdown"
                elif self.sequence_complete_event.is_set():
                    test_status = "passed"
                else:
                    test_status = "aborted by user"

        finally:
            # Power down and cleanup
            if not self.hw_simulate:
                with contextlib.suppress(Exception):
                    self.turn_off_console_and_tx()
                self.cleanup_interface()

            # Final status log
            if test_status == "passed":
                self.logger.info("TEST PASSED: Voltage Sweep completed successfully.")
            elif test_status == "temperature shutdown":
                self.logger.info("TEST FAILED: Voltage Sweep failed due to temperature shutdown.")
            elif test_status == "aborted by user":
                self.logger.info("TEST ABORTED: Voltage Sweep aborted by user.")
            else:
                self.logger.info("TEST FAILED: Voltage Sweep failed due to unexpected error.")


# ------------------- Argument Parsing & Main ------------------- #

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LIFU Voltage Sweep Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
This script interactively prompts for:
  * TX frequency (kHz)
  * Voltage sweep profile (duty cycle + default end voltage)
  * Then runs a step-wise voltage sweep between start and end voltages.

Examples:
  # Run with default settings and interactive prompts
  %(prog)s

  # Run 400 kHz, medium duty cycle profile, from 5V to 45V in 5V steps
  %(prog)s --frequency 400 --profile 2 --start-voltage 5 --end-voltage 45

  # Run with more aggressive console shutoff
  %(prog)s --console-shutoff-temp 65

  # Run in quiet mode writing logs to ./logs
  %(prog)s --quiet --log-dir ./logs

  # Run non-interactively with no start prompt (still requires ENTER between steps)
  %(prog)s --frequency 400 --profile 1 --no-prompt
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Power & behavior
    behavior_group = parser.add_argument_group("Power & Behavior")
    behavior_group.add_argument(
        "--external-power",
        action="store_true",
        help="Use external power supply instead of console 12V/HV.",
    )
    behavior_group.add_argument(
        "--simulate",
        action="store_true",
        help="Hardware simulation mode (no actual device I/O or HV changes).",
    )
    behavior_group.add_argument(
        "--num-modules",
        type=int,
        default=DEFAULT_NUM_MODULES,
        metavar="N",
        help=f"Number of modules in the system (default: {DEFAULT_NUM_MODULES}).",
    )
    behavior_group.add_argument(
        "--frequency",
        type=int,
        choices=FREQUENCIES_KHZ.values(),
        default=None,
        metavar="KHZ",
        help="TX frequency in kHz (overrides interactive selection).",
    )
    behavior_group.add_argument(
        "--profile",
        type=int,
        choices=list(PROFILES.keys()),
        default=None,
        metavar="N",
        help="Voltage sweep profile number (overrides interactive selection).",
    )

    # Sweep configuration
    sweep_group = parser.add_argument_group("Sweep Configuration")
    sweep_group.add_argument(
        "--start-voltage",
        type=float,
        default=DEFAULT_VOLTAGE_START,
        metavar="V",
        help=f"Starting voltage per rail (default: {DEFAULT_VOLTAGE_START} V).",
    )
    sweep_group.add_argument(
        "--end-voltage",
        type=float,
        default=None,
        metavar="V",
        help=(
            "Ending voltage per rail. If not provided, each profile uses its own "
            "default voltage_end."
        ),
    )
    sweep_group.add_argument(
        "--voltage-step",
        type=float,
        default=DEFAULT_VOLTAGE_STEP,
        metavar="V",
        help=f"Voltage step size per rail (default: {DEFAULT_VOLTAGE_STEP} V).",
    )

    # Safety thresholds
    safety_group = parser.add_argument_group("Safety Thresholds")
    safety_group.add_argument(
        "--console-shutoff-temp",
        type=float,
        default=CONSOLE_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"Console shutoff temperature in Celsius (default: {CONSOLE_SHUTOFF_TEMP_C_DEFAULT}).",
    )
    safety_group.add_argument(
        "--tx-shutoff-temp",
        type=float,
        default=TX_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"TX device shutoff temperature in Celsius (default: {TX_SHUTOFF_TEMP_C_DEFAULT}).",
    )
    safety_group.add_argument(
        "--ambient-shutoff-temp",
        type=float,
        default=AMBIENT_SHUTOFF_TEMP_C_DEFAULT,
        metavar="C",
        help=f"Ambient shutoff temperature in Celsius (default: {AMBIENT_SHUTOFF_TEMP_C_DEFAULT}).",
    )

    # Timing / logging
    timing_group = parser.add_argument_group("Timing & Logging")
    timing_group.add_argument(
        "--temperature-check-interval",
        type=float,
        default=TEMPERATURE_CHECK_INTERVAL_DEFAULT,
        metavar="S",
        help=(
            "Temperature check interval in seconds "
            f"(default: {TEMPERATURE_CHECK_INTERVAL_DEFAULT})."
        ),
    )
    timing_group.add_argument(
        "--temperature-log-interval",
        type=float,
        default=TEMPERATURE_LOG_INTERVAL_DEFAULT,
        metavar="S",
        help=(
            "Temperature log interval in seconds "
            f"(default: {TEMPERATURE_LOG_INTERVAL_DEFAULT})."
        ),
    )
    timing_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for log files (default: <openlifu_root>/logs).",
    )

    # Verbosity
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level).",
    )
    verbosity_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational messages (WARNING level only).",
    )

    # Automation
    automation_group = parser.add_argument_group("Automation")
    automation_group.add_argument(
        "--no-prompt",
        action="store_true",
        help='Skip initial "Press ENTER to start" prompt. Per-step prompts still apply.',
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    test = TestVoltageSweep(args)
    logger = test.logger  # for use in exception handlers

    try:
        test.run()
    except KeyboardInterrupt:
        logger.info("User interrupted. Shutting down...")
        test.shutdown_event.set()
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e)
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(1)


if __name__ == "__main__":
    main()
