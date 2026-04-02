#!/usr/bin/env python3
"""
LIFU Thermal Stress Test Script

A professional command-line tool for automated thermal stress testing of LIFU devices.
This script connects to the device, configures test parameters, monitors temperatures,
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
from openlifu.db import Database
from openlifu.geo import Point
from openlifu_sdk.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

"""
Thermal Stress Test Script
- User selects a test case.
- Test runs for a fixed total duration or until a thermal shutdown occurs.
- Logs temperature and device status.
"""

__version__ = "1.0.0"
TEST_ID = "tst05_thermal_stress"
TEST_NAME = "Thermal Stress Test"

# ------------------- Test Case Definitions -------------------
TEST_CASES = {
    0: {
        'id': 'dryrun',
        'description': 'Dry Run Testing',
        'voltage': 30,
        'duty_cycle_pct': 5,
        'sequence_duration': 10  # seconds
    },
    1: {
        'id': '30v_2min',
        'description': 'Medium Voltage, High Duty Cycle, 2min',
        'voltage': 30,
        'duty_cycle_pct': 50,
        'sequence_duration': 2 * 60  # seconds
    },
    2: {
        'id': '25v_5min',
        'description': 'Low Voltage, High Duty Cycle, 5min',
        'voltage': 25,
        'duty_cycle_pct': 50,
        'sequence_duration': 5 * 60  # seconds
    },
    3: {
        'id':'20v_15min',
        'description': 'Low Voltage, High Duty Cycle, 15min',
        'voltage': 20,
        'duty_cycle_pct': 50,
        'sequence_duration': 15 * 60  # seconds
    },
    4: {
        'id': '60V_15min',
        'description': 'High Voltage, Low Duty Cycle, 15min',
        'voltage': 60,
        'duty_cycle_pct': 5,
        'sequence_duration': 15 * 60  # seconds
    },
    5: {
        'id': '60V_5min',
        'description': 'High Voltage, Low Duty Cycle, 5min',
        'voltage': 60,
        'duty_cycle_pct': 10,
        'sequence_duration': 5 * 60  # seconds
    },
}

# Frequency choices (kHz)
FREQUENCIES_KHZ = {1: 150, 2: 400}

# Pulse/sequence timing
INTERVAL_MSEC_DEFAULT = 100
NUM_MODULES_DEFAULT = 2

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



class TestThermalStress:
    """Main class for Thermal Stress Test 5."""

    def __init__(self, args):
        self.args = args

        # Derived paths
        self.openlifu_dir = Path(openlifu.__file__).parent.parent.parent.resolve()
        self.log_dir = Path(self.args.log_dir or (self.openlifu_dir / "logs")).resolve()
        
        # Runtime attributes
        self.interface: LIFUInterface | None = None
        self.shutdown_event = threading.Event()
        self.sequence_complete_event = threading.Event()
        self.temperature_shutdown_event = threading.Event()
        
        # Threading locks
        self.mutex = threading.Lock()

        self.stop_logging = False

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test configuration – set later via interactive selection
        self.frequency_khz: float | None = None
        self.test_case_num: int | None = None
        self.test_case: dict | None = None
        self.test_case_description: str | None = None
        self.test_case_long_description: str | None = None
        self.test_case_id: str | None = None
        self.voltage: float | None = None
        self.interval_msec: float = INTERVAL_MSEC_DEFAULT
        self.duration_msec: int | None = None
        self.sequence_duration: float | None = None

        # Flags from args
        self.use_external_power = self.args.external_power
        self.hw_simulate = self.args.simulate

        # Safety parameters from args
        self.console_shutoff_temp_C = self.args.console_shutoff_temp
        self.tx_shutoff_temp_C = self.args.tx_shutoff_temp
        self.ambient_shutoff_temp_C = self.args.ambient_shutoff_temp
        self.temperature_check_interval = self.args.temperature_check_interval
        self.temperature_log_interval = self.args.temperature_log_interval

        # Logger
        self.logger = self._setup_logging()
        self._file_handler_attached = False

        self.logger.debug("ThermalStressTest initialized with arguments: %s", self.args)


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
        """Attach a file handler for this run once test case is known."""
        if self._file_handler_attached:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.test_case_id is None:
            filename = f"{TEST_ID}_{self.run_timestamp}.log"
        else:
            filename = f"{TEST_ID}_{self.test_case_id}_{self.run_timestamp}.log"

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

    # ------------------- User Input Section ------------------- #
    def _select_frequency(self) -> None:
        """Interactively select frequency and predefined test case."""
        # Frequency selection
        if self.args.frequency:
            self.frequency_khz = self.args.frequency
        else:
            self.logger.info("Choose Frequency:")
            for idx, freq in FREQUENCIES_KHZ.items():
                self.logger.info("  %d. %d kHz", idx, freq)

            while True:
                choice = input(f"Select frequency by number {list(FREQUENCIES_KHZ.keys())}: ").strip()
                if choice.isdigit() and int(choice) in FREQUENCIES_KHZ:
                    self.frequency_khz = FREQUENCIES_KHZ[int(choice)]
                    break
                self.logger.info("Invalid selection. Please try again.")

    def _select_test_case(self) -> None:
        # Test case selection
        if self.args.test_case is not None:
            self.test_case_num = self.args.test_case
            self.test_case = TEST_CASES[self.test_case_num]
        else:
            self.logger.info("\nAvailable Thermal Stress Test Cases:")
            for idx, case in TEST_CASES.items():
                total = format_hhmmss(case["sequence_duration"])
                self.logger.info(
                    f"{idx}. {case['voltage']}V, "
                    f"{case['duty_cycle_pct']}% Duty Cycle, {total} total"
                )

            while True:
                choice = input(f"Select a test case by number {list(TEST_CASES.keys())}: ").strip()
                if choice.isdigit() and int(choice) in TEST_CASES:
                    self.test_case_num = int(choice)
                    self.test_case = TEST_CASES[self.test_case_num]
                    break
                self.logger.info("Invalid selection. Please try again.")

    def _derive_test_case_parameters(self) -> None:
        # Derive test-case-specific parameters
        self.test_case_description = self.test_case["description"]
        self.test_case_long_description = (
            f"{self.frequency_khz}kHz, Case {self.test_case_num}: "
            f"{self.test_case['voltage']}V, "
            f"{self.test_case['duty_cycle_pct']}%, "
            f"{format_hhmmss(self.test_case['sequence_duration'])}"
        )
        self.test_case_id = f"{self.frequency_khz}kHz_{self.test_case['id']}"
        self.voltage = float(self.test_case["voltage"])
        self.duration_msec = int(self.test_case["duty_cycle_pct"] / 100 * self.interval_msec)
        self.sequence_duration = float(self.test_case["sequence_duration"])

        self._attach_file_handler()

        if self.hw_simulate:
            self.logger.info("Beginning Test %s (TEST MODE)", self.test_case_description)
            self.logger.info("TEST MODE: This is a simulated test run.")
            self.logger.info("No actual hardware interactions will occur.")
        else:
            self.logger.info("Selected Frequency: %d kHz", self.frequency_khz)
            self.logger.info("Selected Test Case: %d", self.test_case_num)
            self.logger.info("Beginning Test %s", self.test_case_description)

        self.logger.info("%s", self.test_case_long_description)   

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
            if not self.args.external_power:
                console_fw = self.interface.hvcontroller.get_version()
                self.logger.info("Console Firmware Version: %s", console_fw)
        except Exception as e:
            self.logger.error("Error retrieving console firmware version: %s", e)

        try:
            tx_fw = self.interface.txdevice.get_version()
            self.logger.info("TX Device Firmware Version: %s", tx_fw)
        except Exception as e:
            self.logger.error("Error retrieving TX device firmware version: %s", e)

    def enumerate_devices(self):
        """Enumerate TX7332 devices and verify count."""
        self.logger.info("Enumerate TX7332 chips")
        num_tx_devices = self.interface.txdevice.enum_tx7332_devices()

        if num_tx_devices == 0:
            raise ValueError("No TX7332 devices found.")
        elif num_tx_devices == self.args.num_modules * 2:
            self.logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
            return 32 * num_tx_devices
        else:
            raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.args.num_modules}")

    def configure_solution(self) -> None:
        """Configure the beamforming solution and load it into the device."""
        if self.interface is None:
            raise RuntimeError("Interface not connected.")

        db_path = self.openlifu_dir / "db_dvc"
        db = Database(db_path)
        arr = db.load_transducer(f"openlifu_{self.args.num_modules}x400_evt1")
        arr.sort_by_pin()

        # Focus at (0, 0, 50 mm)
        x_input, y_input, z_input = 0, 0, 50
        target = Point(position=(x_input, y_input, z_input), units="mm")
        focus = target.get_position(units="mm")

        distances = np.sqrt(
            np.sum((focus - arr.get_positions(units="mm")) ** 2, axis=1)
        ).reshape(1, -1)
        tof = distances * 1e-3 / 1500  # mm to m, divide by 1500 m/s
        delays = tof.max() - tof
        apodizations = np.ones((1, arr.numelements()))

        pulse = Pulse(
            frequency=self.frequency_khz * 1e3,
            duration=self.duration_msec * 1e-3,
        )

        sequence = Sequence(
            pulse_interval=self.interval_msec * 1e-3,
            pulse_count=int(self.sequence_duration / (self.interval_msec * 1e-3)),
            pulse_train_interval=0,
            pulse_train_count=1,
        )

        pin_order = np.argsort([el.pin for el in arr.elements])
        solution = Solution(
            delays=delays[:, pin_order],
            apodizations=apodizations[:, pin_order],
            transducer=arr,
            pulse=pulse,
            voltage=self.voltage,
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

        self.logger.info("Solution configured for Test Case %s.", self.test_case_id)


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

        serial_failures = 0
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

    def exit_on_time_complete(self, total_test_time: float) -> None:
        """Thread target: stop test when total test time is reached."""
        start = time.time()
        last_log_time = 0.0

        while True:
            if self.shutdown_event.is_set():
                return

            time.sleep(.1)
            elapsed_time = time.time() - start
            time_since_last_log = elapsed_time - last_log_time

            if elapsed_time >= total_test_time:
                self.logger.info(
                    "  Sequence complete: %s reached.",
                    format_hhmmss(total_test_time),
                )
                self.shutdown_event.set()
                self.sequence_complete_event.set()
                return

    # ------------------------------------------------------------------ #
    # Hardware shutdown & cleanup
    # ------------------------------------------------------------------ #

    def turn_off_console_and_tx(self) -> None:
        """Safely turn off HV and 12V if console is used."""
        if self.interface is None:
            return
        if self.use_external_power:
            return

        try:
            self.logger.info("Turning off HV and 12V...")
            with contextlib.suppress(Exception):
                self.interface.hvcontroller.turn_hv_off()
            with contextlib.suppress(Exception):
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


    def run(self) -> None:
        """Execute the thermal stress test with graceful shutdown."""
        test_status = "not started"

        try:
            # Interactive selection
            self._select_frequency()
            self._select_test_case()
            self._derive_test_case_parameters()

            # Connect and configure
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
                self.logger.info("Press enter to START %s: ", self.test_case_description)
                input()

            # Start sonication
            if not self.hw_simulate:
                self.logger.info("Starting Trigger...")
                if not self.interface.start_sonication():
                    self.logger.error("Failed to start trigger.")
                    test_status = "error"
                    return
            else:
                self.logger.info("Simulated Trigger start... (no hardware)")

            self.logger.info("Trigger Running... (Press CTRL-C to stop early)")
            test_status = "running"

            # Start monitoring threads
            temp_thread = threading.Thread(
                target=self.monitor_temperature,
                name="TemperatureMonitorThread",
                daemon=True,
            )
            completion_thread = threading.Thread(
                target=self.exit_on_time_complete,
                args=(self.sequence_duration,),
                name="SequenceCompletionThread",
                daemon=True,
            )

            temp_thread.start()
            completion_thread.start()

            # Wait for threads or user interrupt
            try:
                while temp_thread.is_alive() and completion_thread.is_alive() and not self.shutdown_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.logger.warning("Test aborted by user KeyboardInterrupt.")
                test_status = "aborted by user"
                self.shutdown_event.set()

            # Ensure shutdown event set
            if not self.shutdown_event.is_set():
                self.logger.warning("A thread exited without setting shutdown event; forcing shutdown.")
                self.shutdown_event.set()

            # Stop sonication
            if not self.hw_simulate and self.interface is not None:
                try:
                    if self.interface.stop_sonication():
                        self.logger.info("Trigger stopped successfully.")
                    else:
                        self.logger.error("Failed to stop trigger.")
                except Exception as e:
                    self.logger.error("Error stopping trigger: %s", e)

            # Wait for threads to exit gracefully
            temp_thread.join(timeout=2.0)
            completion_thread.join(timeout=2.0)

            # Determine final status
            if test_status not in ("aborted by user", "error"):
                if self.sequence_complete_event.is_set():
                    test_status = "passed"
                elif self.temperature_shutdown_event.is_set():
                    test_status = "temperature shutdown"
                else:
                    test_status = "error"

        finally:
            # Power down and cleanup
            if not self.hw_simulate:
                with contextlib.suppress(Exception):
                    self.turn_off_console_and_tx()
                self.cleanup_interface()

            # Final status log
            if test_status == "passed":
                self.logger.info("TEST PASSED: %s completed successfully.", self.test_case_description)
            elif test_status == "temperature shutdown":
                self.logger.info("TEST FAILED: %s failed due to temperature shutdown.", self.test_case_description)
            elif test_status == "aborted by user":
                self.logger.info("TEST ABORTED: %s aborted by user.", self.test_case_description)
            else:
                self.logger.info("TEST FAILED: %s failed due to unexpected error.", self.test_case_description)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LIFU Thermal Stress Burn-in Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
This script interactively prompts for:
  * TX frequency (kHz)
  * Predefined burn-in test case (voltage, duty, duration)

Examples:
  # Run with default settings
  %(prog)s

  # Run using external power supply
  %(prog)s --external-power

  # Run with more aggressive console shutoff
  %(prog)s --console-shutoff-temp 65

  # Run in quiet mode writing logs to ./logs
  %(prog)s --quiet --log-dir ./logs

  # Run without "Press ENTER to start" prompt
  %(prog)s --no-prompt

  # Run a specific test case and frequency non-interactively
    %(prog)s --frequency 400 --test-case 2 --no-prompt
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
        default=NUM_MODULES_DEFAULT,
        metavar="N",
        help=f"Number of modules in the system (default: {NUM_MODULES_DEFAULT}).",
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
        "--test-case",
        type=int,
        choices=list(TEST_CASES.keys()),
        default=None,
        metavar="N",
        help="Predefined test case number (overrides interactive selection).",
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
        help='Skip "Press ENTER to start" prompt and begin immediately.',
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_arguments()
    test = TestThermalStress(args)

    try:
        test.run()
    except KeyboardInterrupt:
        print("\nUser interrupted. Shutting down...")
        test.shutdown_event.set()
        test.stop_logging = True
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(1)


if __name__ == "__main__":
    main()


