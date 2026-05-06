from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Dict, List, Literal

import numpy as np

from openlifu.io.LIFUUart import LIFUUart
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.units import getunitconversion

DEFAULT_NUM_TRANSMITTERS = 2
TRANSMITTERS_PER_MODULE = 2
ADDRESS_GLOBAL_MODE = 0x0
ADDRESS_STANDBY = 0x1
ADDRESS_DYNPWR_2 = 0x6
ADDRESS_LDO_PWR_1 = 0xB
ADDRESS_TRSW_TURNOFF = 0xC
ADDRESS_DYNPWR_1 = 0xF
ADDRESS_LDO_PWR_2 = 0x14
ADDRESS_TRSW_TURNON = 0x15
ADDRESS_DELAY_SEL = 0x16
ADDRESS_PATTERN_MODE = 0x18
ADDRESS_PATTERN_REPEAT = 0x19
ADDRESS_PATTERN_SEL_G2 = 0x1E
ADDRESS_PATTERN_SEL_G1 = 0x1F
ADDRESS_TRSW = 0x1A
ADDRESS_APODIZATION = 0x1B
ADDRESSES_GLOBAL = [ADDRESS_GLOBAL_MODE,
                    ADDRESS_STANDBY,
                    ADDRESS_DYNPWR_2,
                    ADDRESS_LDO_PWR_1,
                    ADDRESS_TRSW_TURNOFF,
                    ADDRESS_DYNPWR_1,
                    ADDRESS_LDO_PWR_2,
                    ADDRESS_TRSW_TURNON,
                    ADDRESS_DELAY_SEL,
                    ADDRESS_PATTERN_MODE,
                    ADDRESS_PATTERN_REPEAT,
                    ADDRESS_PATTERN_SEL_G1,
                    ADDRESS_PATTERN_SEL_G2,
                    ADDRESS_TRSW,
                    ADDRESS_APODIZATION]
ADDRESSES_DELAY_DATA = list(range(0x20, 0x11F+1))
ADDRESSES_PATTERN_DATA = list(range(0x120, 0x19F+1))
ADDRESSES = ADDRESSES_GLOBAL + ADDRESSES_DELAY_DATA + ADDRESSES_PATTERN_DATA
NUM_CHANNELS = 32
MAX_REGISTER = 0x19F
REGISTER_BYTES = 4
REGISTER_WIDTH = REGISTER_BYTES*8
DELAY_ORDER = [[32, 30],
               [28, 26],
               [24, 22],
               [20, 18],
               [31, 29],
               [27, 25],
               [23, 21],
               [19, 17],
               [16, 14],
               [12, 10],
               [8, 6],
               [4, 2],
               [15, 13],
               [11, 9],
               [7, 5],
               [3, 1]]
DELAY_ORDER_REVERSED = [[33 - c for c in row] for row in DELAY_ORDER]
DELAY_CHANNEL_MAP = {}
for row, channels in enumerate(DELAY_ORDER_REVERSED):
    for i, channel in enumerate(channels):
        DELAY_CHANNEL_MAP[channel] = {'row': row, 'lsb': 16*(1-i)}
DELAY_PROFILE_OFFSET = 16
VALID_DELAY_PROFILES = list(range(1, 17))
DELAY_WIDTH = 13
APODIZATION_CHANNEL_ORDER = [17, 19, 21, 23, 25, 27, 29, 31, 18, 20, 22, 24, 26, 28, 30, 32, 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16]
APODIZATION_CHANNEL_ORDER_REVERSED = [33 - c for c in APODIZATION_CHANNEL_ORDER]
DEFAULT_PATTERN_DUTY_CYCLE = 0.66
PATTERN_PROFILE_OFFSET = 4
NUM_PATTERN_PROFILES = 32
VALID_PATTERN_PROFILES = list(range(1, NUM_PATTERN_PROFILES+1))
MAX_PATTERN_PERIODS = 16
PATTERN_PERIOD_ORDER = [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]]
PATTERN_LENGTH_WIDTH = 5
MAX_PATTERN_PERIOD_LENGTH = 30
PATTERN_LEVEL_WIDTH = 3
PATTERN_MAP = {}
for row, periods in enumerate(PATTERN_PERIOD_ORDER):
    for i, period in enumerate(periods):
        PATTERN_MAP[period] = {'row': row, 'lsb_lvl': i*(PATTERN_LEVEL_WIDTH+PATTERN_LENGTH_WIDTH), 'lsb_period': i*(PATTERN_LENGTH_WIDTH+PATTERN_LEVEL_WIDTH)+PATTERN_LEVEL_WIDTH}
MAX_REPEAT = 2**5-1
MAX_ELASTIC_REPEAT = 2**16-1
DEFAULT_TAIL_COUNT = 29
DEFAULT_CLK_FREQ = 10e6
ELASTIC_MODE_PULSE_LENGTH_ADJUST = 125e-6
ProfileOpts = Literal['active', 'configured', 'all']
TriggerModeOpts = Literal['sequence', 'continuous','single']
DEFAULT_PULSE_WIDTH_US = 20
HW_ID_DATA_LENGTH = 12
TEMPERATURE_DATA_LENGTH = 4

from openlifu.io.LIFUConfig import (
    OW_CMD_ASYNC,
    OW_CMD_DFU,
    OW_CMD_ECHO,
    OW_CMD_GET_AMBIENT,
    OW_CMD_GET_TEMP,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_RESET,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_CONTROLLER,
    OW_CTRL_GET_SWTRIG,
    OW_CTRL_SET_SWTRIG,
    OW_CTRL_START_SWTRIG,
    OW_CTRL_STOP_SWTRIG,
    OW_ERROR,
    OW_TX7332,
    OW_TX7332_DEMO,
    OW_TX7332_ENUM,
    OW_TX7332_RREG,
    OW_TX7332_VWBLOCK,
    OW_TX7332_VWREG,
    OW_TX7332_WBLOCK,
    OW_TX7332_WREG,
    TRIGGER_MODE_CONTINUOUS,
    TRIGGER_MODE_SEQUENCE,
    TRIGGER_MODE_SINGLE,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger("TXDevice")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TxDevice:
    def __init__(self, uart: LIFUUart, module_invert: bool | list[bool] = False):
        """
        Initialize the TxDevice.

        Args:
            uart (LIFUUart): The LIFUUart instance for communication.
        """
        self._tx_instances = []
        self.tx_registers = None
        self.uart = uart
        self.module_invert = module_invert
        if self.uart and not self.uart.asyncMode:
            self.uart.check_usb_status()
            if self.uart.is_connected():
                logger.info("TX Device connected.")
            else:
                logger.info("TX Device NOT Connected.")

    def __parse_ti_cfg_file(self, file_path: str) -> list[tuple[str, int, int]]:
        """Parses the given configuration file and extracts all register groups, addresses, and values."""
        parsed_data = []
        pattern = re.compile(r"([\w\d\-]+)\|0x([0-9A-Fa-f]+)\t0x([0-9A-Fa-f]+)")

        with open(file_path) as file:
            for line in file:
                match = pattern.match(line.strip())
                if match:
                    group_name = match.group(1)  # Capture register group name
                    register_address = int(match.group(2), 16)  # Convert hex address to integer
                    register_value = int(match.group(3), 16)  # Convert hex value to integer
                    parsed_data.append((group_name, register_address, register_value))

        return parsed_data

    def is_connected(self) -> bool:
        """
        Check if the TX device is connected.

        Returns:
            bool: True if the device is connected, False otherwise.
        """
        if self.uart:
            return self.uart.is_connected()
        return False

    def close(self):
        """
        Close Uart
        """
        if self.uart and self.uart.is_connected():
            self.uart.disconnect()

    def ping(self, module:int=1) -> bool:
        """
        Send a ping command to the TX device to verify connectivity.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the ping process.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return False

            logger.info("Send Ping to Device.")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_PING, addr=module)
            self.uart.clear_buffer()

            if r.packet_type == OW_ERROR:
                logger.error("Error sending ping")
                return False
            else:
                return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_version(self, module:int=1) -> str:
        """
        Retrieve the firmware version of the TX device.

        Returns:
            str: Firmware version in the format 'vX.Y.Z'.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while fetching the version.
        """
        try:
            if self.uart.demo_mode:
                return 'v0.1.1'

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return 'v0.0.0'

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_VERSION, addr=module)
            self.uart.clear_buffer()
            r.print_packet()
            if r.data_len == 3:
                ver = f'v{r.data[0]}.{r.data[1]}.{r.data[2]}'
            else:
                ver = 'v0.0.0'
            logger.info(ver)
            return ver
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def echo(self, echo_data = None) -> tuple[bytes, int]:
        """
        Send an echo command to the device with data and receive the same data in response.

        Args:
            echo_data (bytes): The data to send (must be a byte array).

        Returns:
            tuple[bytes, int]: The echoed data and its length.

        Raises:
            ValueError: If the UART is not connected.
            TypeError: If the `echo_data` is not a byte array.
            Exception: If an error occurs during the echo process.
        """
        try:
            if self.uart.demo_mode:
                data = b"Hello LIFU!"
                return data, len(data)

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return None, None

            # Check if echo_data is a byte array
            if echo_data is not None and not isinstance(echo_data, (bytes, bytearray)):
                raise TypeError("echo_data must be a byte array")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_ECHO, data=echo_data)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len > 0:
                return r.data, r.data_len
            else:
                return None, None

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except TypeError as t:
            logger.error("TypeError: %s", t)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during echo process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def toggle_led(self, module:int=1) -> bool:
        """
        Toggle the LED on the TX device.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while toggling the LED.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return False

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_TOGGLE_LED, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_hardware_id(self, module:int=1) -> str:
        """
        Retrieve the hardware ID of the TX device.

        Returns:
            str: Hardware ID in hexadecimal format.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while retrieving the hardware ID.
        """
        try:
            if self.uart.demo_mode:
                return bytes.fromhex("deadbeefcafebabe1122334455667788")

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return None

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_HWID, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == HW_ID_DATA_LENGTH:
                return r.data.hex()
            else:
                return None
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_temperature(self, module:int=1) -> float:
        """
        Retrieve the temperature reading from the TX device.

        Returns:
            float: Temperature value in Celsius.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs or the received data length is invalid.
        """
        try:
            if self.uart.demo_mode:
                return 32.4

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return 0

            # Send the GET_TEMP command
            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_GET_TEMP, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()

            # Check if the data length matches a float (4 bytes)
            if r.data_len == TEMPERATURE_DATA_LENGTH:
                # Unpack the float value from the received data (assuming little-endian)
                temperature = struct.unpack('<f', r.data)[0]
                # Truncate the temperature to 2 decimal places
                truncated_temperature = round(temperature, 2)
                return truncated_temperature
            else:
                raise ValueError("Invalid data length received for temperature")
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_ambient_temperature(self, module:int=1) -> float:
        """
        Retrieve the ambient temperature reading from the TX device.

        Returns:
            float: Temperature value in Celsius.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs or the received data length is invalid.
        """
        try:
            if self.uart.demo_mode:
                return 28.9

            if not self.uart.is_connected():
                logger.error("TX Device not connected")
                return 0

            # Send the GET_TEMP command
            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_GET_AMBIENT, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()

            # Check if the data length matches a float (4 bytes)
            if r.data_len == TEMPERATURE_DATA_LENGTH:
                # Unpack the float value from the received data (assuming little-endian)
                temperature = struct.unpack('<f', r.data)[0]
                # Truncate the temperature to 2 decimal places
                truncated_temperature = round(temperature, 2)
                return truncated_temperature
            else:
                logger.error("Invalid data length received for ambient temperature")
                return 0
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle
            return 0

    def set_trigger(self,
                    pulse_interval: float,
                    pulse_count: int = 1,
                    pulse_width: int = DEFAULT_PULSE_WIDTH_US,
                    pulse_train_interval: float = 0.0,
                    pulse_train_count: int = 1,
                    trigger_mode: TriggerModeOpts = "sequence",
                    profile_index: int = 0,
                    profile_increment: bool = True) -> dict:
        """
        Set the trigger configuration on the TX device.

        Args:
            pulse_interval (float): The time interval between pulses in seconds.
            pulse_count (int): The number of pulses to generate.
            pulse_width (int): The pulse width in microseconds.
            pulse_train_interval (float): The time interval between pulse trains in seconds.
            pulse_train_count (int): The number of pulse trains to generate.
            mode (TriggerModeOpts): The trigger mode to use.
            profile_index (int): The pulse profile to use.
            profile_increment (bool): Whether to increment the pulse profile.
        """

        trigger_mode = trigger_mode.lower()
        if trigger_mode == "sequence":
            trigger_mode_int = TRIGGER_MODE_SEQUENCE
        elif trigger_mode == "continuous":
            trigger_mode_int = TRIGGER_MODE_CONTINUOUS
        elif trigger_mode == "single":
            trigger_mode_int = TRIGGER_MODE_SINGLE
        else:
            raise ValueError("Invalid trigger mode")

        logger.info(f"Setting trigger with parameters: "
                        f"pulse_interval={pulse_interval}, "
                        f"pulse_count={pulse_count}, "
                        f"pulse_width={pulse_width}, "
                        f"pulse_train_interval={pulse_train_interval}, "
                        f"pulse_train_count={pulse_train_count}, "
                        f"trigger_mode={trigger_mode}")

        trigger_json = {
            "TriggerFrequencyHz": 1/pulse_interval,
            "TriggerPulseCount": pulse_count,
            "TriggerPulseWidthUsec": pulse_width,
            "TriggerPulseTrainInterval": pulse_train_interval * 1000000,
            "TriggerPulseTrainCount": pulse_train_count,
            "TriggerMode": trigger_mode_int,
            "ProfileIndex": 0,
            "ProfileIncrement": 0
        }
        return self.set_trigger_json(data=trigger_json)

    def set_trigger_json(self, data=None) -> dict:
        """
        Set the trigger configuration on the TX device.

        Args:
            data (dict): A dictionary containing the trigger configuration.

        Returns:
            dict: JSON response from the device.

        Raises:
            ValueError: If `data` is None or the UART is not connected.
            Exception: If an error occurs while setting the trigger.
        """
        try:
            if self.uart.demo_mode:
                return None

            # Ensure data is not None and is a valid dictionary
            if data is None:
                logger.error("Data cannot be None.")
                return None

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            try:
                json_string = json.dumps(data)
            except json.JSONDecodeError as e:
                logger.error(f"Data must be valid JSON: {e}")
                return None

            payload = json_string.encode('utf-8')

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_SET_SWTRIG, data=payload)
            self.uart.clear_buffer()

            if r.packet_type != OW_ERROR and r.data_len > 0:
                # Parse response as JSON, if possible
                try:
                    response_json = json.loads(r.data.decode('utf-8'))
                    return response_json
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
                    return None
            else:
                return None
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_trigger_json(self) -> dict:
        """
        Start the trigger on the TX device.

        Returns:
            bool: True if the trigger was started successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while starting the trigger.
        """
        try:
            if self.uart.demo_mode:
                return None

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_GET_SWTRIG, data=None)
            self.uart.clear_buffer()
            data_object = None
            try:
                data_object = json.loads(r.data.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
            return data_object
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_trigger(self):
        """
        Retrieve the current trigger configuration from the TX device.

        Returns:
            dict: The trigger configuration as a dictionary.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while fetching the trigger configuration.
        """
        trigger_json = self.get_trigger_json()
        if trigger_json:
            if trigger_json["TriggerMode"] == TRIGGER_MODE_SEQUENCE:
                mode = "sequence"
            elif trigger_json["TriggerMode"] == TRIGGER_MODE_CONTINUOUS:
                mode = "continuous"
            elif trigger_json["TriggerMode"] == TRIGGER_MODE_SINGLE:
                mode = "single"
            else:
                mode = "unknown"
            trigger_dict = {
                "pulse_interval": 1 / trigger_json["TriggerFrequencyHz"],
                "pulse_count": trigger_json["TriggerPulseCount"],
                "pulse_width": trigger_json["TriggerPulseWidthUsec"],
                "pulse_train_interval": trigger_json["TriggerPulseTrainInterval"],
                "pulse_train_count": trigger_json["TriggerPulseTrainCount"],
                "mode": mode,
                "profile_index": trigger_json["ProfileIndex"],
                "profile_increment": bool(trigger_json["ProfileIncrement"])
            }
            return trigger_dict

    def start_trigger(self) -> bool:
        """
        Start the trigger on the TX device.

        Returns:
            bool: True if the trigger was started successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while starting the trigger.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CTRL_START_SWTRIG, data=None)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error starting trigger")
                return False
            else:
                return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def stop_trigger(self) -> bool:
        """
        Stop the trigger on the TX device.

        This method sends a command to stop the software trigger on the TX device.
        It checks the device's connection status and handles errors appropriately.

        Returns:
            bool: True if the trigger was successfully stopped, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the operation.
        """
        try:
            if self.uart.demo_mode:
                return True

            # Check if the device is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Send the STOP_SWTRIG command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_CONTROLLER,
                command=OW_CTRL_STOP_SWTRIG,
                data=None
            )

            # Clear the UART buffer to prepare for further communication
            self.uart.clear_buffer()

            # Log the received packet for debugging purposes
            # r.print_packet()

            # Check the packet type to determine success
            if r.packet_type == OW_ERROR:
                logger.error("Error stopping trigger")
                return False
            else:
                return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def soft_reset(self, module:int=1) -> bool:
        """
        Perform a soft reset on the TX device.

        Returns:
            bool: True if the reset was successful, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while resetting the device.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_RESET, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error resetting device")
                return False
            else:
                return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def enter_dfu(self, module:int=1) -> bool:
        """
        Perform a soft reset to enter DFU mode on TX device.

        Returns:
            bool: True if the reset was successful, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while resetting the device.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_DFU, addr=module)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error setting DFU mode for device")
                return False
            else:
                return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def async_mode(self, enable: bool | None = None) -> bool:
        """
        Enable or disable asynchronous mode for the TX device.

        Args:
            enable (bool | None): If True, enable async mode; if False, disable it; if None read the current state.

        Returns:
            bool: True if async mode is enabled, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while setting async mode.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            if enable is not None:
                if enable:
                    payload = struct.pack('<B', 1)
                else:
                    payload = struct.pack('<B', 0)
            else:
                payload = None

            r = self.uart.send_packet(id=None, packetType=OW_CONTROLLER, command=OW_CMD_ASYNC, data=payload)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                raise RuntimeError("Error running async mode command for device")
            else:
                return r.reserved == 1  # reserved field indicates async mode status

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise

    def enum_tx7332_devices(self,
                            num_devices: int | None = None) -> int:
        """
        Enumerate TX7332 devices connected to the TX device.

        Args:
            num_transmitters (int): The number of transmitters expected to be enumerated. If None, the number of
                transmitters will be determined from the response. If provided and the number enumerated does not
                match the expected number, an error will be raised. If the UART is in demo mode, this argument is
                used to set the number of transmitters for the demo (or set to a default if omitted/None)

        Returns:
            n_transmitters: number of devices detected.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during enumeration.
        """
        try:
            if self.uart.demo_mode:
                num_detected_devices = num_devices
            else:
                if not self.uart.is_connected():
                    raise ValueError("TX Device not connected")

                r = self.uart.send_packet(id=None, packetType=OW_TX7332, command=OW_TX7332_ENUM)
                self.uart.clear_buffer()
                # r.print_packet()
                if r.packet_type != OW_ERROR and r.reserved > 0:
                    num_detected_devices = r.reserved
                else:
                    logger.info("Error enumerating TX devices.")
                if num_devices is not None and num_detected_devices != num_devices:
                    raise ValueError(f"Expected {num_devices} devices, but detected {num_detected_devices} devices")
            self.tx_registers = TxDeviceRegisters(num_transmitters=num_detected_devices, module_invert=self.module_invert)
            logger.info("TX Device Count: %d", num_detected_devices)
            return num_detected_devices
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_module_invert(self, module_invert: bool | List[bool]) -> None:
        """
        Set the module invert configuration for the TX device.

        Args:
            module_invert (bool | List[bool]): The module invert configuration to set.
        """
        self.module_invert = module_invert
        if self.tx_registers is not None:
            self.tx_registers.module_invert = module_invert

    def demo_tx7332(self, identifier:int) -> bool:
        """
        Sets all TX7332 chip registers with a test waveform.

        Returns:
            bool: True if all chips are programmed successfully, False otherwise.

        Raises:
            ValueError: If the UART is not connected.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            r = self.uart.send_packet(id=None, addr=identifier, packetType=OW_TX7332, command=OW_TX7332_DEMO)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                logger.error("Error demoing TX devices")
                return False

            return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def write_register(self, identifier:int, address: int, value: int) -> bool:
        """
        Write a value to a register in the TX device.

        Args:
            address (int): The register address to write to.
            value (int): The value to write to the register.

        Returns:
            bool: True if the write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, or the identifier is invalid.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            if self.uart.demo_mode:
                return True

            # Check if the UART is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            # Pack the address and value into the required format
            try:
                data = struct.pack('<HI', address, value)
            except struct.error as e:
                logger.error(f"Error packing address and value: {e}")
                raise ValueError("Invalid address or value format") from e

            # Send the write command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_WREG,
                addr=identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()

            # Check the response for errors
            if r.packet_type == OW_ERROR:
                logger.error("Error writing TX register value")
                return False

            logger.info(f"Successfully wrote value 0x{value:08X} to register 0x{address:04X}")
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def read_register(self, identifier:int, address: int) -> int:
        """
        Read a register value from the TX device.

        Args:
            address (int): The register address to read.

        Returns:
            int: The value of the register if successful, or 0 on failure.

        Raises:
            ValueError: If the identifier is not set or is out of range.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            if self.uart.demo_mode:
                return 45

            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            # Pack the address into the required format
            try:
                data = struct.pack('<H', address)
            except struct.error as e:
                logger.error(f"Error packing address {address}: {e}")
                raise ValueError("Invalid address format") from e

            # Send the read command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_RREG,
                addr=identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()
            # r.print_packet()
            # Check for errors in the response
            if r.packet_type == OW_ERROR:
                logger.error("Error reading TX register value")
                return 0

            # Verify data length and unpack the register value
            if r.data_len == 4:
                try:
                    value = struct.unpack('<I', r.data)[0]
                except struct.error as e:
                    logger.error(f"Error unpacking register value: {e}")
                    return 0
            else:
                logger.error(f"Unexpected data length: {r.data_len}")
                return 0

            logger.info(f"Successfully read value 0x{value:08X} from register 0x{address:04X}")
            return value
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def write_block(self, identifier: int, start_address: int, reg_values: List[int]) -> bool:
        """
        Write a block of register values to the TX device.

        Args:
            start_address (int): The starting register address to write to.
            reg_values (List[int]): List of register values to write.

        Returns:
            bool: True if the block write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, the identifier is invalid, or parameters are out of range.
        """
        try:
            if self.uart.demo_mode:
                return True

            # Ensure the UART connection is active
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            # Validate the reg_values list
            if not reg_values or not isinstance(reg_values, list):
                raise ValueError("Invalid register values: Must be a non-empty list of integers")
            if any(not isinstance(value, int) for value in reg_values):
                raise ValueError("Invalid register values: All elements must be integers")

            # Configure chunking for large blocks
            max_regs_per_block = 62  # Maximum registers per block due to payload size
            num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
            logger.debug(f"Write Block: Total chunks = {num_chunks}")

            # Write each chunk
            for i in range(num_chunks):
                chunk_start = i * max_regs_per_block
                chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
                chunk = reg_values[chunk_start:chunk_end]

                # Pack the chunk into the required data format
                try:
                    data_format = '<HBB' + 'I' * len(chunk)  # Start address (H), chunk length (B), reserved (B), values (I...)
                    data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)
                except struct.error as e:
                    logger.error(f"Error packing data for chunk {i}: {e}")
                    return False

                # Send the packet
                r = self.uart.send_packet(
                    id=None,
                    packetType=OW_TX7332,
                    command=OW_TX7332_WBLOCK,
                    addr=identifier,
                    data=data
                )

                # Clear the UART buffer after sending
                self.uart.clear_buffer()
                # r.print_packet()
                # Check for errors in the response
                if r.packet_type == OW_ERROR:
                    logger.error(f"Error writing TX block at chunk {i}")
                    return False

            logger.debug("Block write successful")
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handleected error in write_block: {e}")
            return False

    def write_register_verify(self, address: int, value: int) -> bool:
        """
        Write a value to a register in the TX device with verification.

        Args:
            address (int): The register address to write to.
            value (int): The value to write to the register.

        Returns:
            bool: True if the write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, or the identifier is invalid.
            Exception: If an unexpected error occurs during the operation.
        """
        try:
            if self.uart.demo_mode:
                return True

            # Check if the UART is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if self.identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            # Pack the address and value into the required format
            try:
                data = struct.pack('<HI', address, value)
            except struct.error as e:
                logger.error(f"Error packing address and value: {e}")
                raise ValueError("Invalid address or value format") from e

            # Send the write command to the device
            r = self.uart.send_packet(
                id=None,
                packetType=OW_TX7332,
                command=OW_TX7332_VWREG,
                addr=self.identifier,
                data=data
            )

            # Clear UART buffer after sending the packet
            self.uart.clear_buffer()

            # Check the response for errors
            if r.packet_type == OW_ERROR:
                logger.error("Error verifying writing TX register value")
                return False

            logger.info(f"Successfully wrote value 0x{value:08X} to register 0x{address:04X}")
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def write_block_verify(self, start_address: int, reg_values: List[int]) -> bool:
        """
        Write a block of register values to the TX device with verification.

        Args:
            start_address (int): The starting register address to write to.
            reg_values (List[int]): List of register values to write.

        Returns:
            bool: True if the block write operation was successful, False otherwise.

        Raises:
            ValueError: If the device is not connected, the identifier is invalid, or parameters are out of range.
        """
        try:
            if self.uart.demo_mode:
                return True

            # Ensure the UART connection is active
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Validate the identifier
            if self.identifier < 0:
                raise ValueError("TX Chip address NOT SET")

            # Validate the reg_values list
            if not reg_values or not isinstance(reg_values, list):
                raise ValueError("Invalid register values: Must be a non-empty list of integers")
            if any(not isinstance(value, int) for value in reg_values):
                raise ValueError("Invalid register values: All elements must be integers")

            # Configure chunking for large blocks
            max_regs_per_block = 62  # Maximum registers per block due to payload size
            num_chunks = (len(reg_values) + max_regs_per_block - 1) // max_regs_per_block
            logger.info(f"Write Block: Total chunks = {num_chunks}")

            # Write each chunk
            for i in range(num_chunks):
                chunk_start = i * max_regs_per_block
                chunk_end = min((i + 1) * max_regs_per_block, len(reg_values))
                chunk = reg_values[chunk_start:chunk_end]

                # Pack the chunk into the required data format
                try:
                    data_format = '<HBB' + 'I' * len(chunk)  # Start address (H), chunk length (B), reserved (B), values (I...)
                    data = struct.pack(data_format, start_address + chunk_start, len(chunk), 0, *chunk)
                except struct.error as e:
                    logger.error(f"Error packing data for chunk {i}: {e}")
                    return False

                # Send the packet
                r = self.uart.send_packet(
                    id=None,
                    packetType=OW_TX7332,
                    command=OW_TX7332_VWBLOCK,
                    addr=self.identifier,
                    data=data
                )

                # Clear the UART buffer after sending
                self.uart.clear_buffer()

                # Check for errors in the response
                if r.packet_type == OW_ERROR:
                    logger.error(f"Error verifying writing TX block at chunk {i}")
                    return False

            logger.debug("Block write successful")
            return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_solution(self,
                     pulse: Dict,
                     delays: np.ndarray,
                     apodizations: np.ndarray,
                     sequence: Dict,
                     trigger_mode: TriggerModeOpts = "sequence",
                     profile_index: int = 1,
                     profile_increment: bool = True):
        """
        Set the solution parameters on the TX device.

        Args:
            pulse (Dict): The pulse parameters to set.
            delays (list): The delays to set.
            apodizations (list): The apodizations to set.
            sequence (Dict): The sequence parameters to set.
            mode: The trigger mode to use.
            profile_index (int): The pulse profile to use.
            profile_increment (bool): Whether to increment the pulse profile.
        """
        delays = np.array(delays)
        if delays.ndim == 1:
            delays = delays.reshape(1, -1)
        apodizations = np.array(apodizations)
        if apodizations.ndim == 1:
            apodizations = apodizations.reshape(1, -1)
        n = delays.shape[0]
        n_elements = delays.shape[1]
        n_required_devices = int(n_elements / NUM_CHANNELS)
        n_detected_tx = self.enum_tx7332_devices(num_devices=n_required_devices)
        n_modules = n_detected_tx / TRANSMITTERS_PER_MODULE
        if n_required_devices != n_detected_tx:
            errmsg = f"Number of detected TX devices ({n_detected_tx}) does not match required ({n_required_devices})"
            logger.exception(errmsg)
            raise OSError(errmsg)

        if n != apodizations.shape[0]:
            raise ValueError("Delays and apodizations must have the same number of rows")
        if n > 1:
            raise NotImplementedError("Multiple foci not supported yet")
        for profile in range(n):
            duty_cycle=DEFAULT_PATTERN_DUTY_CYCLE * max(apodizations[profile,:]) * pulse["amplitude"]
            pulse_profile = Tx7332PulseProfile(
                profile=profile+1,
                frequency=pulse["frequency"],
                cycles=int(pulse["duration"] * pulse["frequency"]),
                duty_cycle=duty_cycle
            )
            self.tx_registers.add_pulse_profile(pulse_profile)
            delay_profile = Tx7332DelayProfile(
                profile=profile+1,
                delays=delays[profile,:],
                apodizations=apodizations[profile, :]
            )
            self.tx_registers.add_delay_profile(delay_profile)
        self.set_trigger(
            pulse_interval=sequence["pulse_interval"],
            pulse_count=sequence["pulse_count"],
            pulse_train_interval=sequence["pulse_train_interval"],
            pulse_train_count=sequence["pulse_train_count"],
            trigger_mode=trigger_mode,
            profile_index=profile_index,
            profile_increment=profile_increment
        )
        self.apply_all_registers()

        # Buffer the pulse and delay profiles in the microcontroller(s), so that they can be used to switch profiles on trigger detection
        delay_control_registers = {profile:self.tx_registers.get_delay_control_registers(profile) for profile in self.tx_registers.configured_delay_profiles()}
        pulse_control_registers = {profile:self.tx_registers.get_pulse_control_registers(profile) for profile in self.tx_registers.configured_pulse_profiles()}



    def apply_all_registers(self):
        """
        Apply all registers to the TX device.

        Raises:
            ValueError: If the device is not connected.
        """
        if self.uart.demo_mode:
            return True

        try:
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")
            registers = self.tx_registers.get_registers(pack=True, pack_single=True)
            for txi, txregs in enumerate(registers):
                for addr, reg_values in txregs.items():
                    if not self.write_block(identifier=txi, start_address=addr, reg_values=reg_values):
                        logger.error(f"Error applying TX CHIP ID: {txi} registers")
                        return False
            return True

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def write_ti_config_to_tx_device(self, file_path: str, txchip_id: int) -> bool:
        """
        Parse a TI configuration file and write the register values to the TX device.

        Args:
            file_path (str): Path to the TI configuration file.
            txchip_id (int): The ID of the TX chip to write the registers to.

        Returns:
            bool: True if all registers were written successfully, False otherwise.
        """
        try:
            # Check if UART is connected
            if not self.uart.is_connected():
                raise ValueError("TX Device not connected")

            # Parse the TI configuration file
            parsed_registers = self.__parse_ti_cfg_file(file_path)
            if not parsed_registers:
                logger.error("No registers parsed from the TI configuration file.")
                return False

            # Write each register to the TX device
            for group, addr, value in parsed_registers:
                logger.info(f"Writing to {group:<20} | Address: 0x{addr:02X} | Value: 0x{value:08X}")
                if not self.write_register(identifier=txchip_id, address=addr, value=value):
                    logger.error(
                        f"Failed to write to TX CHIP ID: {txchip_id} | "
                        f"Register: 0x{addr:02X} | Value: 0x{value:08X}"
                    )
                    return False

            logger.info("Successfully wrote all registers to the TX device.")
            return True

        except FileNotFoundError as e:
            logger.error(f"TI configuration file not found: {file_path}. Error: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid input or device state: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while writing TI config to TX Device: {e}")
            raise

    @property
    def print(self) -> None:
        """
        Print TX device information.

        Raises:
            None
        """
        print("TX Device Information") # noqa: T201
        print("  UART Port:") # noqa: T201
        self.uart.print()

def get_delay_location(channel:int, profile:int=1):
    """
    Gets the address and least significant bit of a delay

    :param channel: Channel number
    :param profile: Delay profile number
    :returns: Register address and least significant bit of the delay location
    """
    if channel not in DELAY_CHANNEL_MAP:
        raise ValueError(f"Invalid channel {channel}.")
    channel_map = DELAY_CHANNEL_MAP[channel]
    if profile not in VALID_DELAY_PROFILES:
        raise ValueError(f"Invalid Profile {profile}")
    address = ADDRESSES_DELAY_DATA[0] + (profile-1) * DELAY_PROFILE_OFFSET + channel_map['row']
    lsb = channel_map['lsb']
    return address, lsb

def set_register_value(reg_value:int, value:int, lsb:int=0, width: int | None=None):
    """
    Sets the value of a parameter in a register integer

    :param reg_value: Register value
    :param value: New value of the parameter
    :param lsb: Least significant bit of the parameter
    :param width: Width of the parameter (bits)
    :returns: New register value
    """
    if width is None:
        width = REGISTER_WIDTH - lsb
    mask = (1 << width) - 1
    if value < 0 or value > mask:
        raise ValueError(f"Value {value} does not fit in {width} bits")
    return (reg_value & ~(mask << lsb)) | ((int(value) & mask) << lsb)

def get_register_value(reg_value:int, lsb:int=0, width: int | None=None):
    """
    Extracts the value of a parameter from a register integer

    :param reg_value: Register value
    :param lsb: Least significant bit of the parameter
    :param width: Width of the parameter (bits)
    :returns: Value of the parameter
    """
    if width is None:
        width = REGISTER_WIDTH - lsb
    mask = (1 << width) - 1
    return (reg_value >> lsb) & mask

def calc_pulse_pattern(frequency:float, duty_cycle:float=DEFAULT_PATTERN_DUTY_CYCLE, bf_clk:float=DEFAULT_CLK_FREQ):
    """
    Calculates the pattern for a given frequency and duty cycle

    The pattern is calculated to represent a single cycle of a pulse with the specified frequency and duty cycle.
    If the pattern requires more than 16 periods, the clock divider is increased to reduce the period length.

    :param frequency: Frequency of the pattern in Hz
    :param duty_cycle: Duty cycle of the pattern
    :param bf_clk: Clock frequency of the BF system in Hz
    :returns: Tuple of lists of levels and lengths, and the clock divider setting
    """
    clk_div_n = 0
    while clk_div_n < 6:
        clk_n = bf_clk / (2**clk_div_n)
        period_samples = int(clk_n / frequency)
        first_half_period_samples = int(period_samples / 2)
        second_half_period_samples = period_samples - first_half_period_samples
        first_on_samples = int(first_half_period_samples * duty_cycle)
        if first_on_samples < 2:
            logging.warning("Duty cycle too short. Setting to minimum of 2 samples")
            first_on_samples = 2
        first_off_samples = first_half_period_samples - first_on_samples
        second_on_samples = max(2, int(second_half_period_samples * duty_cycle))
        if second_on_samples < 2:
            logging.warning("Duty cycle too short. Setting to minimum of 2 samples")
            second_on_samples = 2
        second_off_samples = second_half_period_samples - second_on_samples
        if first_off_samples > 0 and first_off_samples < 2:
            logging.warn
            first_off_samples = 0
            first_on_samples = first_half_period_samples
        if second_off_samples > 0 and first_off_samples < 2:
            second_off_samples = 0
            second_on_samples = second_half_period_samples
        levels = [1, 0, -1, 0]
        per_lengths = []
        per_levels = []
        for i, samples in enumerate([first_on_samples, first_off_samples, second_on_samples, second_off_samples]):
            while samples > 0:
                if samples > MAX_PATTERN_PERIOD_LENGTH+2:
                    if samples == MAX_PATTERN_PERIOD_LENGTH+3:
                        per_lengths.append(MAX_PATTERN_PERIOD_LENGTH-1)
                        samples -= (MAX_PATTERN_PERIOD_LENGTH+1)
                    else:
                        per_lengths.append(MAX_PATTERN_PERIOD_LENGTH)
                        samples -= (MAX_PATTERN_PERIOD_LENGTH+2)
                    per_levels.append(levels[i])
                else:
                    per_lengths.append(samples-2)
                    per_levels.append(levels[i])
                    samples = 0
        if len(per_levels) <= MAX_PATTERN_PERIODS:
            t = (np.arange(np.sum(np.array(per_lengths)+2))*(1/clk_n)).tolist()
            y = np.concatenate([[yi]*(ni+2) for yi,ni in zip(per_levels, per_lengths)]).tolist()
            pattern = {'levels': per_levels,
                        'lengths': per_lengths,
                        'clk_div_n': clk_div_n,
                        't': t,
                        'y': y}
            return pattern
        else:
            clk_div_n += 1
    raise ValueError(f"Pattern requires too many periods ({len(per_levels)} > {MAX_PATTERN_PERIODS})")

def get_pattern_location(period:int, profile:int=1):
    """
    Gets the address and least significant bit of a pattern period

    :param period: Pattern period number
    :param profile: Pattern profile number
    :returns: Register address and least significant bit of the pattern period location
    """
    if period not in PATTERN_MAP:
        raise ValueError(f"Invalid period {period}.")
    if profile not in VALID_PATTERN_PROFILES:
        raise ValueError(f"Invalid profile {profile}.")
    address = ADDRESSES_PATTERN_DATA[0] + (profile-1) * PATTERN_PROFILE_OFFSET + PATTERN_MAP[period]['row']
    lsb_lvl = PATTERN_MAP[period]['lsb_lvl']
    lsb_period = PATTERN_MAP[period]['lsb_period']
    return address, lsb_lvl, lsb_period

def print_regs(d):
    for addr, val in sorted(d.items()):
        if isinstance(val, list):
            for i, v in enumerate(val):
                print(f'0x{addr:X}[+{i:d}]:x{v:08X}') # noqa: T201
        else:
            print(f'0x{addr:X}:x{val:08X}') # noqa: T201

def pack_registers(regs, pack_single:bool=False):
    """
    Packs registers into contiguous blocks

    :param regs: Dictionary of registers
    :param pack_single: Pack single registers into arrays. Default True.
    :returns: Dictionary of packed registers.
    """
    addresses = sorted(regs.keys())
    if len(addresses) == 0:
        return {}
    last_addr = -255
    burst_addr = -255
    packed = {}
    for addr in addresses:
        if addr == last_addr+1 and burst_addr in packed:
            packed[burst_addr].append(regs[addr])
        else:
            packed[addr] = [regs[addr]]
            burst_addr = addr
        last_addr = addr
    if not pack_single:
        for addr, val in packed.items():
            if len(val) == 1:
                packed[addr] = val[0]
    return packed

def swap_byte_order(regs):
    """
    Swaps the byte order of the registers

    :param regs: Dictionary of registers
    :returns: Dictionary of registers with swapped byte order
    """
    swapped = {}
    for addr, val in regs.items():
        if isinstance(val, list):
            swapped[addr] = [int.from_bytes(v.to_bytes(REGISTER_BYTES, 'big'), 'little') for v in val]
        else:
            swapped[addr] = int.from_bytes(val.to_bytes(REGISTER_BYTES, 'big'), 'little')
    return swapped

@dataclass
class Tx7332DelayProfile:
    profile: Annotated[int, OpenLIFUFieldData("Profile Index (1-16)", "Index of the delay profile (1-16)")]
    """Index of the delay profile (1-16). The Tx7332 support 16 unique delay profiles."""

    delays: Annotated[List[float], OpenLIFUFieldData("Delay values", "Delay values for transducer elements")]
    """Delay values for transducer elements"""

    apodizations: Annotated[List[int] | None, OpenLIFUFieldData("Apodizations", "Apodization values for transducer elements")] = None
    """Apodization values for transducer elements"""

    units: Annotated[str, OpenLIFUFieldData("Units", "Time units used for delay values")] = 's'
    """Time units used for delay values"""

    def __post_init__(self):
        self.num_elements = len(self.delays)
        if self.apodizations is None:
            self.apodizations = [1]*self.num_elements
        if len(self.apodizations) != self.num_elements:
            raise ValueError(f"Apodizations list must have {self.num_elements} elements")
        if self.profile not in VALID_DELAY_PROFILES:
            raise ValueError(f"Invalid Profile {self.profile}")

@dataclass
class Tx7332PulseProfile:
    profile: Annotated[int, OpenLIFUFieldData("Profile index (1-32)", "Index of the pulse profile (1-32)")]
    """Index of the pulse profile (1-32). The Tx7332 supports 32 unique pulse profiles."""

    frequency: Annotated[float, OpenLIFUFieldData("Frequency (Hz)", "Center frequency of the pulse (Hz)")]
    """Center frequency of the pulse (Hz)"""

    cycles: Annotated[int, OpenLIFUFieldData("Number of cycles", "Number of cycles in the pulse")]
    """Number of cycles in the pulse"""

    duty_cycle: Annotated[float, OpenLIFUFieldData("Duty cycle (0-1)", "Pulse duty cycle for the generated square wave (0-1)")] = DEFAULT_PATTERN_DUTY_CYCLE
    """Pulse duty cycle for the generated square wave (0-1). By default 0.66 is used to approximate a sinusoidal wave."""

    tail_count: Annotated[int, OpenLIFUFieldData("Tail count (cycles)", "Clock cycles to actively drive the pulser to ground after the pulse ends")] = DEFAULT_TAIL_COUNT
    """Clock cycles to actively drive the pulser to ground after the pulse ends. Default 29"""

    invert: Annotated[bool, OpenLIFUFieldData("Invert polarity?", "Flag indicating whether to invert the pulse amplitude")] = False
    """Invert the pulse amplitude. Default False"""

    def __post_init__(self):
        if self.profile not in VALID_PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {self.profile}.")

@dataclass
class Tx7332Registers:
    bf_clk: Annotated[float, OpenLIFUFieldData("Clock Frequency (Hz)", "The beamformer clock frequency in Hz.")] = DEFAULT_CLK_FREQ
    """The beamformer clock frequency in Hz. This much match the hardware clock frequency in order for calculated register values to produce the correct pulse and delay timting. Default is 64 MHz."""

    _delay_profiles_list: Annotated[List[Tx7332DelayProfile], OpenLIFUFieldData("Delay profiles list", "Internal list of available delay profiles")] = field(default_factory=list)
    """Internal list of available delay profiles"""

    _pulse_profiles_list: Annotated[List[Tx7332PulseProfile], OpenLIFUFieldData("Pulse profiles list", "Internal list of available pulse profiles")] = field(default_factory=list)
    """Internal list of available pulse profiles"""

    active_delay_profile: Annotated[int | None, OpenLIFUFieldData("Active delay profile", "Index of the currently active delay profile")] = None
    """Index of the currently active delay profile"""

    active_pulse_profile: Annotated[int | None, OpenLIFUFieldData("Active pulse profile", "Index of the currently active pulse profile")] = None
    """Index of the currently active pulse profile"""

    def __post_init__(self):
        delay_profile_indices = self.configured_delay_profiles()
        if len(delay_profile_indices) != len(set(delay_profile_indices)):
            raise ValueError("Duplicate delay profiles found")
        if self.active_delay_profile is not None and self.active_delay_profile not in delay_profile_indices:
            raise ValueError(f"Delay profile {self.active_delay_profile} not found")
        pulse_profile_indices = self.configured_pulse_profiles()
        if len(pulse_profile_indices) != len(set(pulse_profile_indices)):
            raise ValueError("Duplicate pulse profiles found")
        if self.active_pulse_profile is not None and self.active_pulse_profile not in pulse_profile_indices:
            raise ValueError(f"Pulse profile {self.active_pulse_profile} not found")

    def add_delay_profile(self, p: Tx7332DelayProfile, activate: bool | None=None):
        if p.num_elements != NUM_CHANNELS:
            raise ValueError(f"Delay profile must have {NUM_CHANNELS} elements")
        profile_indices = self.configured_delay_profiles()
        if p.profile in profile_indices:
            i = profile_indices.index(p.profile)
            self._delay_profiles_list[i] = p
        else:
            self._delay_profiles_list.append(p)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = p.profile

    def add_pulse_profile(self, p: Tx7332PulseProfile, activate: bool | None=None):
        profile_indices = self.configured_pulse_profiles()
        if p.profile in profile_indices:
            i = profile_indices.index(p.profile)
            self._pulse_profiles_list[i] = p
        else:
            self._pulse_profiles_list.append(p)
        if activate is None:
            activate = self.active_pulse_profile is None
        if activate:
            self.active_pulse_profile = p.profile

    def remove_delay_profile(self, profile:int):
        profile_indices = self.configured_delay_profiles()
        if profile not in profile_indices:
            raise ValueError(f"Delay profile {profile} not found")
        index = profile_indices.index(profile)
        del self._delay_profiles_list[index]
        if self.active_delay_profile == index:
            self.active_delay_profile = None

    def remove_pulse_profile(self, profile:int):
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        index = profiles.index(profile)
        del self._pulse_profiles_list[index]
        if self.active_pulse_profile == index:
            self.active_pulse_profile = None

    def get_delay_profile(self, profile: int | None=None) -> Tx7332DelayProfile:
        if profile is None:
            profile = self.active_delay_profile
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        index = profiles.index(profile)
        return self._delay_profiles_list[index]

    def configured_delay_profiles(self) -> List[int]:
        return [p.profile for p in self._delay_profiles_list]

    def get_pulse_profile(self, profile: int | None=None) -> Tx7332PulseProfile:
        if profile is None:
            profile = self.active_pulse_profile
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        index = profiles.index(profile)
        return self._pulse_profiles_list[index]

    def configured_pulse_profiles(self) -> List[int]:
        return [p.profile for p in self._pulse_profiles_list]

    def activate_delay_profile(self, profile:int):
        if profile not in self.configured_delay_profiles():
            raise ValueError(f"Delay profile {profile} not configured")
        self.active_delay_profile = profile

    def activate_pulse_profile(self, profile:int):
        if profile not in self.configured_pulse_profiles():
            raise ValueError(f"Pulse profile {profile} not configured")
        self.active_pulse_profile = profile

    def get_delay_control_registers(self, profile: int | None=None) -> Dict[int,int]:
        if profile is None:
            profile = self.active_delay_profile
        delay_profile = self.get_delay_profile(profile)
        apod_register = 0
        for i, apod in enumerate(delay_profile.apodizations):
            apod_register = set_register_value(apod_register, 1-apod, lsb=APODIZATION_CHANNEL_ORDER_REVERSED.index(i+1), width=1)
        delay_sel_register = 0
        delay_sel_register = set_register_value(delay_sel_register, delay_profile.profile-1, lsb=12, width=4)
        delay_sel_register = set_register_value(delay_sel_register, delay_profile.profile-1, lsb=28, width=4)
        return {ADDRESS_DELAY_SEL: delay_sel_register,
                ADDRESS_APODIZATION: apod_register}

    def get_pulse_control_registers(self, profile: int | None=None, pulse_invert: bool = False) -> Dict[int,int]:
        if profile is None:
            profile = self.active_pulse_profile
        pulse_profile = self.get_pulse_profile(profile)
        if pulse_profile.profile not in VALID_PATTERN_PROFILES:
            raise ValueError(f"Invalid profile {pulse_profile.profile}.")
        pattern = calc_pulse_pattern(pulse_profile.frequency, pulse_profile.duty_cycle, bf_clk=self.bf_clk)
        clk_div_n = pattern['clk_div_n']
        clk_div = 2**clk_div_n
        clk_n = self.bf_clk / clk_div
        cycles = int(pulse_profile.cycles)
        if cycles > (MAX_REPEAT+1):
            # Use elastic repeat
            pulse_duration_samples = self.bf_clk * ((cycles  / pulse_profile.frequency) + ELASTIC_MODE_PULSE_LENGTH_ADJUST)
            repeat = 0
            elastic_repeat = int(pulse_duration_samples / 16)
            period_samples = int(clk_n / pulse_profile.frequency)
            cycles = 16*elastic_repeat / period_samples
            y = pattern['y']*int(cycles+1)
            y = y[:(16*elastic_repeat)]
            y = y + ([0]*pulse_profile.tail_count)
            t = np.arange(len(y))*(1/clk_n)
            elastic_mode = 1
            if elastic_repeat > MAX_ELASTIC_REPEAT:
                raise ValueError("Pattern duration too long for elastic repeat")
        else:
            repeat = cycles-1
            elastic_repeat = 0
            elastic_mode = 0
            y = pattern['y']*(repeat+1)
            y = np.array(y + [0]*pulse_profile.tail_count)
        reg_mode =  0x02000003
        reg_mode = set_register_value(reg_mode, clk_div_n, lsb=3, width=3)
        reg_mode = set_register_value(reg_mode, int(pulse_profile.invert^pulse_invert), lsb=6, width=1)
        reg_repeat = 0
        reg_repeat = set_register_value(reg_repeat, repeat, lsb=1, width=5)
        reg_repeat = set_register_value(reg_repeat, pulse_profile.tail_count, lsb=6, width=5)
        reg_repeat = set_register_value(reg_repeat, elastic_mode, lsb=11, width=1)
        reg_repeat = set_register_value(reg_repeat, elastic_repeat, lsb=12, width=16)
        reg_pat_sel = 0
        reg_pat_sel = set_register_value(reg_pat_sel, pulse_profile.profile-1, lsb=0, width=6)
        registers = {ADDRESS_PATTERN_MODE: reg_mode,
                     ADDRESS_PATTERN_REPEAT: reg_repeat,
                     ADDRESS_PATTERN_SEL_G1: reg_pat_sel,
                     ADDRESS_PATTERN_SEL_G2: reg_pat_sel}
        return registers

    def get_delay_data_registers(self, profile: int | None=None, pack: bool=False, pack_single: bool=False) -> Dict[int,int]:
        if profile is None:
            profile = self.active_delay_profile
        delay_profile = self.get_delay_profile(profile)
        data_registers = {}
        for channel in range(1, NUM_CHANNELS+1):
            address, lsb = get_delay_location(channel, delay_profile.profile)
            if address not in data_registers:
                data_registers[address] = 0
            delay_value = int(delay_profile.delays[channel-1] * getunitconversion(delay_profile.units, 's') * self.bf_clk)
            data_registers[address] = set_register_value(data_registers[address], delay_value, lsb=lsb, width=DELAY_WIDTH)
        if pack:
            data_registers = pack_registers(data_registers, pack_single=pack_single)
        return data_registers

    def get_pulse_data_registers(self, profile: int | None=None, pack: bool=False, pack_single: bool=False) -> Dict[int,int]:
        if profile is None:
            profile = self.active_pulse_profile
        profile_index = self.get_pulse_profile(profile)
        data_registers = {}
        pattern = calc_pulse_pattern(profile_index.frequency, profile_index.duty_cycle, bf_clk=self.bf_clk)
        levels = pattern['levels']
        lengths = pattern['lengths']
        nperiods = len(levels)
        level_lut = {-1: 0b01, 0: 0b11, 1: 0b10}  # Map levels to register values 0b11 drive to ground 0b00 high impedance
        for i, (level, length) in enumerate(zip(levels, lengths)):
            address, lsb_lvl, lsb_length = get_pattern_location(i+1, profile_index.profile)
            if address not in data_registers:
                data_registers[address] = 0
            data_registers[address] = set_register_value(data_registers[address], level_lut[level], lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
            data_registers[address] = set_register_value(data_registers[address], length, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        if nperiods< MAX_PATTERN_PERIODS:
            address, lsb_lvl, lsb_length = get_pattern_location(nperiods+1, profile_index.profile)
            if address not in data_registers:
                data_registers[address] = 0
            data_registers[address] = set_register_value(data_registers[address], 0b111, lsb=lsb_lvl, width=PATTERN_LEVEL_WIDTH)
            data_registers[address] = set_register_value(data_registers[address], 0, lsb=lsb_length, width=PATTERN_LENGTH_WIDTH)
        if pack:
            data_registers = pack_registers(data_registers, pack_single=pack_single)
        return data_registers

    def get_registers(self, profiles: ProfileOpts = "configured", pack: bool=False, pack_single: bool=False, pulse_invert: bool=False) -> Dict[int,int]:
        if len(self._delay_profiles_list) == 0:
            raise ValueError("No delay profiles have been configured")
        if len(self._pulse_profiles_list) == 0:
            raise ValueError("No pulse profiles have been configured")
        if self.active_delay_profile is None:
            raise ValueError("No delay profile activated")
        if self.active_pulse_profile is None:
            raise ValueError("No pulse profile activated")
        registers = {addr:0x0 for addr in ADDRESSES_GLOBAL}
        registers.update(self.get_delay_control_registers())
        registers.update(self.get_pulse_control_registers(pulse_invert=pulse_invert))
        if profiles == "active":
            delay_data = self.get_delay_data_registers()
            pulse_data = self.get_pulse_data_registers()
        else:
            if profiles == "all":
                delay_data = {addr:0x0 for addr in ADDRESSES_DELAY_DATA}
                pulse_data = {addr:0x0 for addr in ADDRESSES_PATTERN_DATA}
            else:
                delay_data = {}
                pulse_data = {}
            for delay_profile in self._delay_profiles_list:
                delay_data.update(self.get_delay_data_registers(profile=delay_profile.profile))
            for profile_index in self._pulse_profiles_list:
                pulse_data.update(self.get_pulse_data_registers(profile=profile_index.profile))
        registers.update(delay_data)
        registers.update(pulse_data)
        if pack:
            registers = pack_registers(registers, pack_single=pack_single)
        return registers

@dataclass
class TxDeviceRegisters:
    bf_clk: Annotated[int, OpenLIFUFieldData("Clock Frequency (Hz)", "The beamformer clock frequency in Hz.")] = DEFAULT_CLK_FREQ
    """The beamformer clock frequency in Hz. This much match the hardware clock frequency in order for calculated register values to produce the correct pulse and delay timting. Default is 64 MHz."""

    _delay_profiles_list: Annotated[List[Tx7332DelayProfile], OpenLIFUFieldData("Delay profiles list", "Internal list of available delay profiles")] = field(default_factory=list)
    """Internal list of available delay profiles"""

    _profiles_list: Annotated[List[Tx7332PulseProfile], OpenLIFUFieldData("Pulse profiles list", "Internal list of available pulse profiles")] = field(default_factory=list)
    """Internal list of available pulse profiles"""

    active_delay_profile: Annotated[int | None, OpenLIFUFieldData("Active delay profile", "Index of the currently active delay profile")] = None
    """Index of the currently active delay profile"""

    active_profile: Annotated[int | None, OpenLIFUFieldData("Active pulse profile", "Index of the currently active pulse profile")] = None
    """Index of the currently active pulse profile"""

    num_transmitters: Annotated[int, OpenLIFUFieldData("Number of transmitters", "The number of transmitters available on the device")] = DEFAULT_NUM_TRANSMITTERS
    """The number of transmitters available on the device"""

    module_invert: Annotated[List[bool] | bool, OpenLIFUFieldData("Module Invert", "List of flags indicating whether to invert the pulse amplitude for each module or a single flag for all modules")] = False
    """List of flags indicating whether to invert the pulse amplitude for each module or a single flag for all modules"""

    def __post_init__(self):
        self.transmitters = tuple([Tx7332Registers(bf_clk=self.bf_clk) for _ in range(self.num_transmitters)])

    def add_pulse_profile(self, pulse_profile: Tx7332PulseProfile, activate: bool | None=None):
        """
        Add a pulse profile

        :param p: Pulse profile
        :param activate: Activate the pulse profile
        """
        profiles = self.configured_pulse_profiles()
        if pulse_profile.profile in profiles:
            i = profiles.index(pulse_profile.profile)
            self._profiles_list[i] = pulse_profile
        else:
            self._profiles_list.append(pulse_profile)
        if activate is None:
            activate = self.active_profile is None
        if activate:
            self.active_profile = pulse_profile.profile
        for tx in self.transmitters:
            tx.add_pulse_profile(pulse_profile, activate = activate)

    def add_delay_profile(self, delay_profile: Tx7332DelayProfile, activate: bool | None=None):
        """
        Add a delay profile

        :param p: Delay profile
        :param activate: Activate the delay profile
        """
        if delay_profile.num_elements != NUM_CHANNELS*self.num_transmitters:
            raise ValueError(f"Delay profile must have {NUM_CHANNELS*self.num_transmitters} elements")
        profiles = self.configured_delay_profiles()
        if delay_profile.profile in profiles:
            i = profiles.index(delay_profile.profile)
            self._delay_profiles_list[i] = delay_profile
        else:
            self._delay_profiles_list.append(delay_profile)
        if activate is None:
            activate = self.active_delay_profile is None
        if activate:
            self.active_delay_profile = delay_profile.profile
        for i, tx in enumerate(self.transmitters):
            module = i // 2
            chip = (i+1) % 2
            start_channel = module*NUM_CHANNELS*2 + chip*NUM_CHANNELS
            profiles = np.arange(start_channel, start_channel+NUM_CHANNELS, dtype=int)
            tx_delays = np.array(delay_profile.delays)[profiles].tolist()
            tx_apodizations = np.array(delay_profile.apodizations)[profiles].tolist()
            txp = Tx7332DelayProfile(delay_profile.profile, tx_delays, tx_apodizations, delay_profile.units)
            tx.add_delay_profile(txp, activate = activate)

    def remove_delay_profile(self, profile:int):
        """
        Remove a delay profile

        :param profile: Delay profile number
        """
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        del self._delay_profiles_list[i]
        if self.active_delay_profile == profile:
            self.active_delay_profile = None
        for tx in self.transmitters:
            tx.remove_delay_profile(profile)

    def remove_pulse_profile(self, profile:int):
        """
        Remove a pulse profile

        :param profile: Pulse profile number
        """
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        del self._profiles_list[i]
        if self.active_profile == profile:
            self.active_profile = None
        for tx in self.transmitters:
            tx.remove_pulse_profile(profile)

    def get_delay_profile(self, profile:int | None=None) -> Tx7332DelayProfile:
        """
        Retrieve a delay profile

        :param profile: Delay profile number
        :return: Delay profile
        """
        if profile is None:
            profile = self.active_delay_profile
        profiles = self.configured_delay_profiles()
        if profile not in profiles:
            raise ValueError(f"Delay profile {profile} not found")
        i = profiles.index(profile)
        return self._delay_profiles_list[i]

    def configured_delay_profiles(self) -> List[int]:
        """
        Get the configured delay profiles

        :return: List of delay profiles
        """
        return [p.profile for p in self._delay_profiles_list]

    def get_pulse_profile(self, profile:int | None=None) -> Tx7332PulseProfile:
        """
        Retrieve a pulse profile

        :param profile: Pulse profile number
        :return: Pulse profile
        """
        if profile is None:
            profile = self.active_profile
        profiles = self.configured_pulse_profiles()
        if profile not in profiles:
            raise ValueError(f"Pulse profile {profile} not found")
        i = profiles.index(profile)
        return self._profiles_list[i]

    def configured_pulse_profiles(self) -> List[int]:
        """
        Get the configured pulse profiles

        :return: List of pulse profiles
        """
        return [p.profile for p in self._profiles_list]

    def activate_delay_profile(self, profile:int=1):
        """
        Activates a delay profile

        :param profile: Delay profile number
        """
        for tx in self.transmitters:
            tx.activate_delay_profile(profile)
        self.active_delay_profile = profile

    def activate_pulse_profile(self, profile:int=1):
        """
        Activates a pulse profile

        :param profile: Pulse profile number
        """
        for tx in self.transmitters:
            tx.activate_pulse_profile(profile)
        self.active_profile = profile

    def recompute_delay_profiles(self):
        """
        Recompute the delay profiles
        """
        for tx in self.transmitters:
            profiles = tx.configured_delay_profiles()
            for profile in profiles:
                tx.remove_delay_profile(profile)
        for dp in self._delay_profiles_list:
            self.add_delay_profile(dp, activate = dp.profile == self.active_delay_profile)

    def recompute_pulse_profiles(self):
        """
        Recompute the pulse profiles
        """
        for tx in self.transmitters:
            profiles = tx.configured_pulse_profiles()
            for profile in profiles:
                tx.remove_pulse_profile(profile)
            for pp in self._profiles_list:
                tx.add_pulse_profile(pp, activate = pp.profile == self.active_profile)

    def get_registers(self, profiles: ProfileOpts = "configured", recompute: bool = False, pack: bool=False, pack_single:bool=False) -> List[Dict[int,int]]:
        """
        Get the registers for all transmitters

        :param profiles: Profile options
        :param recompute: Recompute the registers
        :return: List of registers for each transmitter
        """
        if recompute:
            self.recompute_delay_profiles()
            self.recompute_pulse_profiles()
        if isinstance(self.module_invert, bool):
            tx_invert = [self.module_invert] * self.num_transmitters
        else:
            tx_invert = list(np.array(self.module_invert*TRANSMITTERS_PER_MODULE, dtype=bool).reshape(TRANSMITTERS_PER_MODULE, -1).T.reshape(-1))
        return [tx.get_registers(profiles, pack=pack, pack_single=pack_single, pulse_invert=inv) for tx, inv in zip(self.transmitters, tx_invert)]

    def get_delay_control_registers(self, profile:int | None=None) -> List[Dict[int,int]]:
        """
        Get the delay control registers for all transmitters

        :param profile: Delay profile number
        :return: List of delay control registers for each transmitter
        """
        if profile is None:
            profile = self.active_delay_profile
        return [tx.get_delay_control_registers(profile) for tx in self.transmitters]

    def get_pulse_control_registers(self, profile:int | None=None) -> List[Dict[int,int]]:
        """
        Get the pulse control registers for all transmitters

        :param profile: Pulse profile number
        :return: List of pulse control registers for each transmitter
        """
        if profile is None:
            profile = self.active_profile
        if isinstance(self.module_invert, bool):
            tx_invert = [self.module_invert] * self.num_transmitters
        else:
            tx_invert = list(np.array(self.module_invert*TRANSMITTERS_PER_MODULE, dtype=bool).T.reshape(-1))
        return [tx.get_pulse_control_registers(profile, pulse_invert=inv) for tx, inv in zip(self.transmitters, tx_invert)]

    def get_delay_data_registers(self, profile:int | None=None, pack: bool=False, pack_single: bool=False) -> List[Dict[int,int]]:
        """
        Get the delay data registers for all transmitters

        :param profile: Delay profile number
        :return: List of delay data registers for each transmitter
        """
        if profile is None:
            profile = self.active_delay_profile
        return [tx.get_delay_data_registers(profile, pack=pack, pack_single=pack_single) for tx in self.transmitters]

    def get_pulse_data_registers(self, profile:int | None=None, pack: bool=False, pack_single: bool=False) -> List[Dict[int,int]]:
        """
        Get the pulse data registers for all transmitters

        :param profile: Pulse profile number
        :return: List of pulse data registers for each transmitter
        """
        if profile is None:
            profile = self.active_profile
        return [tx.get_pulse_data_registers(profile, pack=pack, pack_single=pack_single) for tx in self.transmitters]
