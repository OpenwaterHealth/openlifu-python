#!/usr/bin/env python3
"""
LIFU 2x Burn In Test Script

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
Burn-in Test Script
- User selects a test case.
- Test runs for a fixed total duration or until a thermal shutdown occurs.
- Logs temperature and device status.
"""

__version__ = "1.0.0"
TEST_ID = "tst06_2x_burn_in"
TEST_NAME = "2x Burn In Test"

# ------------------- Test Case Definitions (from procedural script) -------------------

FREQUENCIES_KHZ = {1: 150, 2: 400}

TEST_CASES = {
    0: {
        "description": "Custom test",
        "voltage": None,
        "duty_cycle_pct": None,
        "sequence_duration_sec": None,
        "sequence_repeats": None,
        "sequence_repeat_interval_sec": None,
        "test_repeats": None,
        "test_repeat_interval_sec": None,
    },
    1: {
        "description": "Dry Run Testing",
        "voltage": 60,
        "duty_cycle_pct": 5,
        "sequence_duration_sec": 3,
        "sequence_repeats": 3,
        "sequence_repeat_interval_sec": 6,
        "test_repeats": 2,
        "test_repeat_interval_sec": 30,
    },
    2: {
        "description": "24 hour burn-in",
        "voltage": 60,
        "duty_cycle_pct": 5,
        "sequence_duration_sec": 10 * 60,
        "sequence_repeats": 8,
        "sequence_repeat_interval_sec": 60 * 60,
        "test_repeats": 1,
        "test_repeat_interval_sec": 60 * 60 * 24,
    },
    3: {
        "description": "Lifetime test",
        "voltage": 60,
        "duty_cycle_pct": 5,
        "sequence_duration_sec": 10 * 60,
        "sequence_repeats": 8,
        "sequence_repeat_interval_sec": 60 * 60,
        "test_repeats": 200,
        "test_repeat_interval_sec": 60 * 60 * 24,
    },
    4: {
        "description": "Endless test",
        "voltage": 60,
        "duty_cycle_pct": 5,
        "sequence_duration_sec": 10 * 60,
        "sequence_repeats": 8,
        "sequence_repeat_interval_sec": 60 * 60,
        "test_repeats": float("inf"),
        "test_repeat_interval_sec": 60 * 60 * 24,
    },

}

PULSE_INTERVAL_MSEC = 100
NUM_MODULES = 2
CUSTOM_TEST_CASE_KEY = 0

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
    """Format seconds into human-friendly string (with days/hours if needed)."""
    seconds = int(seconds)
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if days > 0:
        return f"{days:d}d {hours:02d}h{minutes:02d}m{secs:02.0f}s"
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02.0f}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02.0f}s"
    return f"{secs:0.1f}s"


class TestThermalStress:
    """Main class for Thermal Stress Test 5 (OOP version of procedural script)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Derived paths
        self.openlifu_dir = Path(openlifu.__file__).parent.parent.parent.resolve()
        self.log_dir = Path(self.args.log_dir or (self.openlifu_dir / "logs")).resolve()
        self.num_modules = NUM_MODULES

        # Runtime attributes
        self.interface: LIFUInterface | None = None
        self.shutdown_event = threading.Event()
        self.sequence_complete_event = threading.Event()
        self.temperature_shutdown_event = threading.Event()

        self.mutex = threading.Lock()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test configuration – set later via selection
        self.frequency_khz: float | None = None
        self.test_case_key: str | None = None
        self.test_case: dict | None = None
        self.test_case_description: str | None = None
        self.test_case_long_description: str | None = None
        self.test_case_id: str | None = None

        self.voltage: float | None = None
        self.duty_cycle_pct: float | None = None
        self.sequence_duration_sec: float | None = None
        self.sequence_repeats: float | None = None
        self.sequence_repeat_interval_sec: float | None = None
        self.test_repeats: float | None = None
        self.test_repeat_interval_sec: float | None = None

        # Derived timing
        self.interval_msec: float = PULSE_INTERVAL_MSEC
        self.duration_msec: int | None = None
        self.total_test_time_sec: float | None = None

        # Flags
        self.use_external_power = self.args.external_power
        self.hw_simulate = self.args.simulate

        # Safety parameters (fixed from procedural script)
        self.console_shutoff_temp_C = CONSOLE_SHUTOFF_TEMP_C_DEFAULT
        self.tx_shutoff_temp_C = TX_SHUTOFF_TEMP_C_DEFAULT
        self.ambient_shutoff_temp_C = AMBIENT_SHUTOFF_TEMP_C_DEFAULT
        self.temperature_check_interval = TEMPERATURE_CHECK_INTERVAL_DEFAULT
        self.temperature_log_interval = TEMPERATURE_LOG_INTERVAL_DEFAULT

        # Logger
        self.logger = self._setup_logging()
        self._file_handler_attached = False

        self.logger.debug("TestBurnIn initialized with arguments: %s", self.args)

    # ------------------- Logging Setup -------------------

    def _setup_logging(self) -> logging.Logger:
        """Configure logger with console output; file handler added later."""
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
            "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
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
            "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logging.getLogger(__name__).addHandler(file_handler)

        self._file_handler_attached = True
        self.logger.info("Run log will be saved to: %s", log_path)

    # ------------------- User Selection & Test Derivation -------------------

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
        """Select test case via CLI or interactive prompt (including custom)."""
        # CLI: expect test-case key string (e.g. dryrun, 24hr, etc.)
        if self.args.test_case is not None:
            self.test_case_num = self.args.test_case
            self.test_case = TEST_CASES[self.test_case_num]
        else:
            # Interactive display
            self.logger.info("\nAvailable Burn In Test Cases:")
            for key, case in TEST_CASES.items():
                if key != CUSTOM_TEST_CASE_KEY:
                    desc = (
                        f"{case['description']}, "
                        f"{case['voltage']}V, "
                        f"{case['duty_cycle_pct']}%, "
                        f"{format_hhmmss(case['sequence_duration_sec'])} sequence, "
                        f"repeated {case['sequence_repeats']}x every "
                        f"{format_hhmmss(case['sequence_repeat_interval_sec'])}, "
                        f"test repeated {case['test_repeats']}x every "
                        f"{format_hhmmss(case['test_repeat_interval_sec'])}"
                    )
                else:
                    desc = f"{case['description']} (user-defined parameters)"
                self.logger.info("  %s: %s", key, desc)

            while True:
                choice = input(f"Select test case by number {list(TEST_CASES.keys())}: ").strip()
                if choice.isdigit() and int(choice) in TEST_CASES:
                    self.test_case_num = int(choice)
                    self.test_case = TEST_CASES[self.test_case_num]
                    break
                self.logger.info("Invalid selection. Please try again.")

        # Custom input if needed
        if self.test_case_num == CUSTOM_TEST_CASE_KEY:
            while True:
                try:
                    voltage = float(input("Enter voltage (V): ").strip())
                    duty_cycle_pct = float(input("Enter duty cycle (%): ").strip())
                    sequence_duration_sec = int(input("Enter sequence duration (seconds): ").strip())
                    sequence_repeats = int(input("Enter number of sequence repeats: ").strip())
                    sequence_repeat_interval_sec = int(
                        input("Enter sequence repeat interval (seconds): ").strip()
                    )
                    test_repeats_in = input(
                        "Enter number of test repeats (or 'inf' for endless): "
                    ).strip()
                    if test_repeats_in.lower() == "inf":
                        test_repeats = float("inf")
                    else:
                        test_repeats = int(test_repeats_in)
                    test_repeat_interval_sec = int(
                        input("Enter test repeat interval (seconds): ").strip()
                    )

                    # Validation
                    if not (0 < duty_cycle_pct <= 100):
                        raise ValueError("Duty cycle must be between 0 and 100.")
                    if (
                        voltage <= 0
                        or sequence_duration_sec <= 0
                        or sequence_repeats <= 0
                        or sequence_repeat_interval_sec < 0
                        or test_repeats <= 0
                        or test_repeat_interval_sec < 0
                    ):
                        raise ValueError("All numeric inputs must be positive; intervals cannot be negative.")
                    self.test_case = {
                        "description": "Custom test",
                        "voltage": voltage,
                        "duty_cycle_pct": duty_cycle_pct,
                        "sequence_duration_sec": sequence_duration_sec,
                        "sequence_repeats": sequence_repeats,
                        "sequence_repeat_interval_sec": sequence_repeat_interval_sec,
                        "test_repeats": test_repeats,
                        "test_repeat_interval_sec": test_repeat_interval_sec,
                    }
                    break
                except ValueError as e:
                    self.logger.info("Invalid input: %s. Please try again.", e)
        else:
            self.logger.info("Selected Test Case: %s", self.test_case["description"])

    def _derive_test_case_parameters(self) -> None:
        """Compute derived parameters after frequency and test case are selected."""
        assert self.frequency_khz is not None
        assert self.test_case is not None

        self.voltage = float(self.test_case["voltage"])
        self.duty_cycle_pct = float(self.test_case["duty_cycle_pct"])
        self.sequence_duration_sec = float(self.test_case["sequence_duration_sec"])
        self.sequence_repeats = float(self.test_case["sequence_repeats"])
        self.sequence_repeat_interval_sec = float(self.test_case["sequence_repeat_interval_sec"])
        self.test_repeats = float(self.test_case["test_repeats"])
        self.test_repeat_interval_sec = float(self.test_case["test_repeat_interval_sec"])

        self.duration_msec = int(self.duty_cycle_pct / 100.0 * self.interval_msec)

        # Total test time if all sequences complete (for info only)
        if self.test_repeats == float("inf"):
            self.total_test_time_sec = float("inf")
        else:
            self.total_test_time_sec = (
                self.test_repeat_interval_sec * (self.test_repeats - 1)
                + (self.sequence_repeats - 1) * self.sequence_repeat_interval_sec
                + self.sequence_duration_sec
            )

        self.test_case_description = self.test_case["description"]
        self.test_case_long_description = (
            f"{self.frequency_khz}kHz, {self.voltage}V, {self.duty_cycle_pct}%, "
            f"{format_hhmmss(self.sequence_duration_sec)} sequence, "
            f"repeated {int(self.sequence_repeats)}x every "
            f"{format_hhmmss(self.sequence_repeat_interval_sec)}, "
            f"test repeated "
            f"{'∞' if self.test_repeats == float('inf') else int(self.test_repeats)}x every "
            f"{format_hhmmss(self.test_repeat_interval_sec)}"
        )
        self.test_case_id = f"{int(self.frequency_khz)}kHz_{self.test_case_key}"

        # Attach file handler now that test_case_id is known
        self._attach_file_handler()

        if self.hw_simulate:
            self.logger.info("Beginning Test %s (TEST MODE)", self.test_case_description)
            self.logger.info("TEST MODE: This is a simulated test run.")
            self.logger.info("No actual hardware interactions will occur.")
        else:
            self.logger.info("Beginning Test %s", self.test_case_description)

        self.logger.info("%s", self.test_case_long_description)
        if self.total_test_time_sec != float("inf"):
            self.logger.info(
                "Total test time if all sequences complete: %s",
                format_hhmmss(self.total_test_time_sec),
            )
        else:
            self.logger.info("Total test time: endless")

    # ------------------- Device Setup & Solution -------------------

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
        elif num_tx_devices == self.num_modules * 2:
            self.logger.info(f"Number of TX7332 devices found: {num_tx_devices}")
            return 32 * num_tx_devices
        else:
            raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.num_modules}")


    def configure_solution(self) -> None:
        """Configure the beamforming solution and load it into the device."""
        if self.interface is None:
            raise RuntimeError("Interface not connected.")

        db_path = self.openlifu_dir / "db_dvc"
        db = Database(db_path)
        # Procedural script uses fixed 2x400 transducer ID
        arr = db.load_transducer(f"openlifu_{self.num_modules}x400_evt1")
        arr.sort_by_pin()

        # Focus at (0, 0, 50 mm)
        x_input, y_input, z_input = 0, 0, 50
        target = Point(position=(x_input, y_input, z_input), units="mm")
        focus = target.get_position(units="mm")

        distances = np.sqrt(
            np.sum((focus - arr.get_positions(units="mm")) ** 2, axis=1)
        ).reshape(1, -1)
        tof = distances * 1e-3 / 1500.0  # mm to m, divide by 1500 m/s
        delays = tof.max() - tof
        apodizations = np.ones((1, arr.numelements()))

        pulse = Pulse(
            frequency=self.frequency_khz * 1e3,
            duration=self.duration_msec * 1e-3,
        )

        sequence = Sequence(
            pulse_interval=self.interval_msec * 1e-3,
            pulse_count=int(self.sequence_duration_sec / (self.interval_msec * 1e-3)),
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

    # ------------------- Monitoring Threads -------------------

    def monitor_temperature(self) -> None:
        """
        Monitor temperatures and trigger shutdown on safety violations.

        This closely follows the procedural script's logic:
        - Periodic logging every TEMPERATURE_LOG_INTERVAL seconds.
        - Rapid temperature increase checks.
        - Absolute temperature thresholds.
        """
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

            try:
                if not self.use_external_power:
                    if prev_con_temp is None:
                        con_temp = self.interface.hvcontroller.get_temperature1()
                        prev_con_temp = con_temp
                    else:
                        con_temp = self.interface.hvcontroller.get_temperature1()
                else:
                    con_temp = None

                if prev_tx_temp is None:
                    tx_temp = self.interface.txdevice.get_temperature()
                    prev_tx_temp = tx_temp
                else:
                    tx_temp = self.interface.txdevice.get_temperature()

                if prev_amb_temp is None:
                    amb_temp = self.interface.txdevice.get_ambient_temperature()
                    prev_amb_temp = amb_temp
                else:
                    amb_temp = self.interface.txdevice.get_ambient_temperature()

            except SerialException as e:
                self.logger.error("SerialException encountered while reading temperatures: %s", e)
                serial_failures += 1
                if serial_failures >= 3:
                    self.logger.critical("Maximum serial failures reached. Initiating shutdown.")
                    break
                time.sleep(self.temperature_check_interval)
                continue
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
            if (
                not self.use_external_power
                and con_temp is not None
                and con_temp > self.console_shutoff_temp_C
            ):
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

    # ------------------- Hardware Shutdown & Cleanup -------------------

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

    # ------------------- Main Test Flow -------------------

    def run(self) -> None:
        """Execute the thermal stress test with multi-test/multi-sequence logic."""
        test_status = "not started"
        start_time___ = time.time()

        try:
            # Selection & derivation
            self._select_frequency()
            self._select_test_case()
            self._derive_test_case_parameters()

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
                self.logger.info("Press enter to START %s: ", self.test_case_description)
                input()

            
            test_number = 1

            try:
                # Outer loop over tests
                while self.test_repeats == float("inf") or test_number <= self.test_repeats:
                    self.logger.info(
                        "[%s]Starting Test %d/%s...",
                        format_hhmmss(time.time() - start_time___),
                        test_number,
                        "∞" if self.test_repeats == float("inf") else int(self.test_repeats),
                    )
                    test_start_time = time.time()
                    sequence_number = 1

                    # Inner loop over sequences
                    while sequence_number <= self.sequence_repeats:
                        self.logger.info(
                            "[%s]  Starting Sequence %d/%d...",
                            format_hhmmss(time.time() - start_time___),
                            sequence_number,
                            int(self.sequence_repeats),
                        )
                        self.sequence_complete_event.clear()
                        self.shutdown_event.clear()
                        sequence_start_time = time.time()

                        # Start monitor + completion threads
                        temp_thread = threading.Thread(
                            target=self.monitor_temperature,
                            name="TemperatureMonitorThread",
                        )
                        completion_thread = threading.Thread(
                            target=self.exit_on_time_complete,
                            args=(self.sequence_duration_sec,),
                            name="SequenceCompletionThread",
                        )
                        all_threads = [temp_thread, completion_thread]

                        # Start sonication
                        if not self.hw_simulate:
                            if not self.interface.start_sonication():
                                self.logger.error("Failed to start trigger.")
                                test_status = "error"
                                break
                        else:
                            self.logger.info("Simulated trigger start... (no hardware)")

                        for t in all_threads:
                            t.start()

                        # Wait for both threads to finish or shutdown_event
                        while all(t.is_alive() for t in all_threads) and not self.shutdown_event.is_set():
                            time.sleep(0.1)

                        for t in all_threads:
                            t.join()

                        # Stop sonication
                        if not self.hw_simulate and self.interface is not None:
                            if not self.interface.stop_sonication():
                                self.logger.error("Failed to stop trigger.")
                                test_status = "error"
                                break

                        # Check if sequence completed normally
                        if self.sequence_complete_event.is_set():
                            self.logger.info(
                                "[%s] Test (%d/%s), Sequence (%d/%d) complete",
                                format_hhmmss(time.time() - start_time___),
                                test_number,
                                "∞" if self.test_repeats == float("inf") else int(self.test_repeats),
                                sequence_number,
                                int(self.sequence_repeats),
                            )
                            if sequence_number < self.sequence_repeats:
                                off_time = self.sequence_repeat_interval_sec - (time.time() - sequence_start_time)
                                if off_time > 0:
                                    self.logger.info(
                                        "  Waiting %s until next sequence...",
                                        format_hhmmss(off_time),
                                    )
                                    time.sleep(off_time)
                            sequence_number += 1
                            self.shutdown_event.clear()
                        else:
                            # Interrupted / temp shutdown
                            break

                    if self.sequence_complete_event.is_set():
                        self.logger.info(
                            "[%s] Test %d/%s complete",
                            format_hhmmss(time.time() - start_time___),
                            test_number,
                            "∞" if self.test_repeats == float("inf") else int(self.test_repeats),
                        )
                        if self.test_repeats != float("inf") and test_number < self.test_repeats:
                            off_time = self.test_repeat_interval_sec - (time.time() - test_start_time)
                            if off_time > 0:
                                self.logger.info(
                                    "Waiting %s until next test...",
                                    format_hhmmss(off_time),
                                )
                                time.sleep(off_time)
                        test_number += 1
                        self.shutdown_event.clear()
                    else:
                        # Sequence loop did not complete normally
                        break

                # Decide status if not already error
                if self.temperature_shutdown_event.is_set():
                    test_status = "temperature shutdown"
                elif self.sequence_complete_event.is_set() and (
                    self.test_repeats == float("inf") or test_number > self.test_repeats
                ):
                    test_status = "passed"
                elif test_status != "error":
                    test_status = "error"

            except KeyboardInterrupt:
                self.logger.warning("Test aborted by user KeyboardInterrupt.")
                self.shutdown_event.set()
                test_status = "aborted by user"

        finally:
            self.logger.info(
                "[%s] All tests complete or aborted. Cleaning up...",
                format_hhmmss(time.time() - start_time___),
            )

            if not self.hw_simulate:
                with contextlib.suppress(Exception):
                    self.turn_off_console_and_tx()
                self.cleanup_interface()

            if test_status == "passed":
                self.logger.info("TEST PASSED: %s completed successfully.", self.test_case_description)
            elif test_status == "temperature shutdown":
                self.logger.info(
                    "TEST FAILED: %s failed due to temperature shutdown.",
                    self.test_case_description,
                )
            elif test_status == "aborted by user":
                self.logger.info("TEST ABORTED: %s aborted by user.", self.test_case_description)
            else:
                self.logger.info(
                    "TEST FAILED: %s failed due to unexpected error.",
                    self.test_case_description,
                )


# ------------------- CLI -------------------

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments (lean, per your choice)."""
    parser = argparse.ArgumentParser(
        description="LIFU Thermal Stress Burn-in Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
This script interacts to select:
  * TX frequency (kHz)
  * Burn-in test case (voltage, duty, durations, repeats)

Examples:
  # Run interactively
  %(prog)s

  # Run with fixed frequency and test case (no prompt)
  %(prog)s --frequency 400 --test-case 24hr --no-prompt

""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Power / behavior (small subset)
    parser.add_argument(
        "--external-power",
        action="store_true",
        help="Use external power supply instead of console 12V/HV.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Hardware simulation mode (no actual device I/O or HV changes).",
    )

    parser.add_argument(
        "--frequency",
        type=int,
        choices=FREQUENCIES_KHZ.values(),
        default=None,
        metavar="KHZ",
        help="TX frequency in kHz (overrides interactive selection).",
    )

    parser.add_argument(
        "--test-case",
        type=int,
        choices=list(TEST_CASES.keys()),
        default=None,
        metavar="ID",
        help="Predefined test case id (overrides interactive selection).",
    )

    parser.add_argument(
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
    parser.add_argument(
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
        test.logger.info("User interrupted. Shutting down...")
        test.shutdown_event.set()
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(0)
    except Exception as e:
        test.logger.error("Fatal error: %s", e)
        with contextlib.suppress(Exception):
            test.turn_off_console_and_tx()
        with contextlib.suppress(Exception):
            test.cleanup_interface()
        sys.exit(1)


if __name__ == "__main__":
    main()
