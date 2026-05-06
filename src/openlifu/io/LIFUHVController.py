from __future__ import annotations

import logging
import struct

from openlifu.io.LIFUConfig import (
    OW_CMD,
    OW_CMD_DFU,
    OW_CMD_ECHO,
    OW_CMD_HWID,
    OW_CMD_PING,
    OW_CMD_RESET,
    OW_CMD_TOGGLE_LED,
    OW_CMD_VERSION,
    OW_ERROR,
    OW_POWER,
    OW_POWER_12V_OFF,
    OW_POWER_12V_ON,
    OW_POWER_GET_12VON,
    OW_POWER_GET_FAN,
    OW_POWER_GET_HV,
    OW_POWER_GET_HVON,
    OW_POWER_GET_RGB,
    OW_POWER_GET_TEMP1,
    OW_POWER_GET_TEMP2,
    OW_POWER_HV_OFF,
    OW_POWER_HV_ON,
    OW_POWER_SET_DACS,
    OW_POWER_SET_FAN,
    OW_POWER_SET_HV,
    OW_POWER_SET_RGB,
)
from openlifu.io.LIFUUart import LIFUUart

logger = logging.getLogger(__name__)


class HVController:
    def __init__(self, uart: LIFUUart = None):
        """
        Initialize the HVController.

        Args:
            uart (LIFUUart): The LIFUUart instance for communication.
        """
        self.uart = uart

        if self.uart and not self.uart.asyncMode:
            self.uart.check_usb_status()
            if self.uart.is_connected():
                logger.info("HV Console connected.")
            else:
                logger.info("HV Console NOT Connected.")

        # Initialize the high voltage state (should get this from device)
        self.output_voltage = 0.0
        self.is_hv_on = False
        self.is_12v_on = False

        self.supply_voltage = None

    def is_connected(self):
        if self.uart:
            return self.uart.is_connected()

    def close(self):
        """
        Close Uart
        """
        if self.uart and self.uart.is_connected():
            self.uart.disconnect()

    def ping(self) -> bool:
        """
        Send a ping command to the Console device to verify connectivity.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs during the ping process.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            logger.info("Send Ping to Device.")
            r = self.uart.send_packet(id=None, packetType=OW_CMD, command=OW_CMD_PING)
            self.uart.clear_buffer()
            logger.info("Received Ping from Device.")
            # r.print_packet()

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

    def get_version(self) -> str:
        """
        Retrieve the firmware version of the Console device.

        Returns:
            str: Firmware version in the format 'vX.Y.Z'.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while fetching the version.
        """
        try:
            if self.uart.demo_mode:
                return "v0.1.1"

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_CMD, command=OW_CMD_VERSION
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 3:
                ver = f"v{r.data[0]}.{r.data[1]}.{r.data[2]}"
            else:
                ver = "v0.0.0"
            logger.info(ver)
            return ver

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def echo(self, echo_data=None) -> tuple[bytes, int]:
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
                raise ValueError("Console Device  not connected")

            # Check if echo_data is a byte array
            if echo_data is not None and not isinstance(echo_data, (bytes, bytearray)):
                raise TypeError("echo_data must be a byte array")

            r = self.uart.send_packet(
                id=None, packetType=OW_CMD, command=OW_CMD_ECHO, data=echo_data
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len > 0:
                return r.data, r.data_len
            else:
                return None, None

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def toggle_led(self) -> bool:
        """
        Toggle the LED on the Console device.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while toggling the LED.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_CMD, command=OW_CMD_TOGGLE_LED
            )
            self.uart.clear_buffer()
            # r.print_packet()
            if r.packet_type == OW_ERROR:
                return False

            return True
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_hardware_id(self) -> str:
        """
        Retrieve the hardware ID of the Console device.

        Returns:
            str: Hardware ID in hexadecimal format.

        Raises:
            ValueError: If the UART is not connected.
            Exception: If an error occurs while retrieving the hardware ID.
        """
        try:
            if self.uart.demo_mode:
                return bytes.fromhex("deadbeefcafebabe5566778811223344")

            if not self.uart.is_connected():
                raise ValueError("Console Device not connected")

            r = self.uart.send_packet(id=None, packetType=OW_CMD, command=OW_CMD_HWID)
            self.uart.clear_buffer()
            # r.print_packet()
            if r.data_len == 16:
                return r.data.hex()
            else:
                return None

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_temperature1(self) -> float:
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
            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_TEMP1
            )
            self.uart.clear_buffer()
            # r.print_packet()

            # Check if the data length matches a float (4 bytes)
            if r.data_len == 4:
                # Unpack the float value from the received data (assuming little-endian)
                temperature = struct.unpack("<f", r.data)[0]
                # Truncate the temperature to 2 decimal places
                truncated_temperature = round(temperature, 2)
                return truncated_temperature
            else:
                raise ValueError("Invalid data length received for temperature")
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

    def get_temperature2(self) -> float:
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
            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_TEMP2
            )
            self.uart.clear_buffer()
            # r.print_packet()

            # Check if the data length matches a float (4 bytes)
            if r.data_len == 4:
                # Unpack the float value from the received data (assuming little-endian)
                temperature = struct.unpack("<f", r.data)[0]
                # Truncate the temperature to 2 decimal places
                truncated_temperature = round(temperature, 2)
                return truncated_temperature
            else:
                raise ValueError("Invalid data length received for temperature")
        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

    def turn_12v_off(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning off 12V.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_12V_OFF
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning off 12V")
                return False
            else:
                self.is_12v_on = False
                logger.info("12V turned off successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_12v_on(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning on 12V.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_12V_ON
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning on 12V")
                return False
            else:
                self.is_12v_on = True
                logger.info("12V turned on successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_12v_status(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Get 12V voltage status.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_12VON
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error retrieving 12V status")
                return False
            else:
                if r.reserved == 1:
                    self.is_12v_on = True
                else:
                    self.is_12v_on = False

                return self.is_12v_on

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_hv_on(self):
        """
        Turn on the high voltage.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning on high voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_HV_ON, timeout=30
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning on HV Supply")
                return False
            else:
                self.is_hv_on = True
                logger.info("HV Supply turned on successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def turn_hv_off(self):
        """
        Turn off the high voltage.
        """
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Turning off high voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_HV_OFF
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error turning off HV Supply")
                return False
            else:
                self.is_hv_on = False
                logger.info("HV Supply turned off successfully.")
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_hv_status(self):
        try:
            if self.uart.demo_mode:
                return True

            if not self.uart.is_connected():
                raise ValueError("Console not connected")

            logger.info("Get high voltage status.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_HVON
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error retrievinging HV Status")
                return False
            else:
                if r.reserved == 1:
                    self.is_hv_on = True
                else:
                    self.is_hv_on = False

                return self.is_hv_on

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_voltage(self, voltage: float) -> bool:
        """
        Set the output voltage.

        Args:
            voltage (float): The desired output voltage.

        Raises:
            ValueError: If the controller is not connected or voltage exceeds supply voltage.
        """
        if self.uart.demo_mode:
            return True

        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        # Validate and process the DAC input
        if voltage is None:
            voltage = 0
        elif not (5.0 <= voltage <= 100.0):
            raise ValueError(
                "Voltage input must be within the valid range 5 to 100 Volts)."
            )

        try:
            dac_input = int(((voltage) / 162) * 4095)
            # logger.info("Setting DAC Value %d.", dac_input)
            # Pack the 12-bit DAC input into two bytes
            data = bytes(
                [
                    (dac_input >> 8) & 0xFF,  # High byte (most significant bits)
                    dac_input & 0xFF,  # Low byte (least significant bits)
                ]
            )

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_SET_HV, data=data
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting HV")
                return False
            else:
                self.supply_voltage = voltage
                logger.info("Output voltage set to %.2fV successfully.", voltage)
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_dacs(self, hvp: int, hvm: int, hrp: int, hrm: int) -> bool:
        """
        Set the output voltage.

        Args:
            voltage (float): The desired output voltage.

        Raises:
            ValueError: If the controller is not connected or voltage exceeds supply voltage.
        """
        if self.uart.demo_mode:
            return True

        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        # Validate and process the DAC input
        if hvp is None:
            hvp = 0
        elif not (0 <= hvp <= 4095):
            raise ValueError("Dac hvp input range is 0 to 4095.")

        if hvm is None:
            hvm = 0
        elif not (0 <= hvm <= 4095):
            raise ValueError("Dac hvm input range is 0 to 4095.")

        if hrp is None:
            hrp = 0
        elif not (0 <= hrp <= 4095):
            raise ValueError("Dac hrp input range is 0 to 4095.")

        if hrm is None:
            hrm = 0
        elif not (0 <= hrm <= 4095):
            raise ValueError("Dac hrm input range is 0 to 4095.")

        try:
            # logger.info("Setting DAC Value %d.", dac_input)
            # Pack the 12-bit DAC input into two bytes
            data = bytes(
                [
                    (hvp >> 8) & 0xFF,  # High byte (most significant bits)
                    hvp & 0xFF,  # Low byte (least significant bits)
                    (hrp >> 8) & 0xFF,  # High byte (most significant bits)
                    hrp & 0xFF,  # Low byte (least significant bits)
                    (hvm >> 8) & 0xFF,  # High byte (most significant bits)
                    hvm & 0xFF,  # Low byte (least significant bits)
                    (hrm >> 8) & 0xFF,  # High byte (most significant bits)
                    hrm & 0xFF,  # Low byte (least significant bits)
                ]
            )

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_SET_DACS, data=data
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting DACS")
                return False
            else:
                return True

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_voltage(self) -> float:
        """
        Get the current output voltage setting.

        Returns:
            float: The current output voltage.

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        try:
            if self.uart.demo_mode:
                return 18.4

            logger.info("Getting current output voltage.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_HV
            )
            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error Getting HV Voltage reading")
                return 0.0
            elif r.data_len == 4:
                # Unpack the float value from the received data (assuming little-endian)
                voltage = struct.unpack("<f", r.data)[0]
                # Truncate the temperature to 2 decimal places
                truncated_voltage = round(voltage, 2)
                return truncated_voltage
            else:
                logger.error("Error getting output voltage from device")
                return 0.0

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_fan_speed(self, fan_id: int = 0, fan_speed: int = 50) -> int:
        """
        Get the current output fan percentage.

        Args:
            fan_id (int): The desired fan to set (default is 0). bottom fans (0), and top fans (1).
            fan_speed (int): The desired fan speed (default is 50).

        Returns:
            int: The current output fan percentage.

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        if fan_id not in [0, 1]:
            raise ValueError("Invalid fan ID. Must be 0 or 1")

        if fan_speed not in range(101):
            raise ValueError("Invalid fan speed. Must be 0 to 100")

        try:
            if self.uart.demo_mode:
                return 40

            logger.info("Getting current output voltage.")

            data = bytes(
                [
                    fan_speed & 0xFF,  # Low byte (least significant bits)
                ]
            )

            r = self.uart.send_packet(
                id=None,
                addr=fan_id,
                packetType=OW_POWER,
                command=OW_POWER_SET_FAN,
                data=data,
            )

            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting Fan Speed")
                return -1

            logger.info(f"Set fan speed to {fan_speed}")
            return fan_speed

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_fan_speed(self, fan_id: int = 0) -> int:
        """
        Get the current output fan percentage.

        Args:
            fan_id (int): The desired fan to read (default is 0). bottom fans (0), and top fans (1).

        Returns:
            int: The current output fan percentage.

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        if fan_id not in [0, 1]:
            raise ValueError("Invalid fan ID. Must be 0 or 1")

        try:
            if self.uart.demo_mode:
                return 40.0

            logger.info("Getting current output voltage.")

            r = self.uart.send_packet(
                id=None, addr=fan_id, packetType=OW_POWER, command=OW_POWER_GET_FAN
            )

            self.uart.clear_buffer()
            # r.print_packet()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting HV")
                return 0.0

            elif r.data_len == 1:
                fan_value = r.data[0]
                logger.info(f"Output fan speed is {fan_value}")
                return fan_value
            else:
                logger.error("Error getting output voltage from device")
                return -1

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def set_rgb_led(self, rgb_state: int) -> int:
        """
        Set the RGB LED state.

        Args:
            rgb_state (int): The desired RGB state (0 = OFF, 1 = RED, 2 = BLUE, 3 = GREEN).

        Returns:
            int: The current RGB state after setting.

        Raises:
            ValueError: If the controller is not connected or the RGB state is invalid.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        if rgb_state not in [0, 1, 2, 3]:
            raise ValueError(
                "Invalid RGB state. Must be 0 (OFF), 1 (RED), 2 (BLUE), or 3 (GREEN)"
            )

        try:
            if self.uart.demo_mode:
                return rgb_state

            logger.info("Setting RGB LED state.")

            # Send the RGB state as the reserved byte in the packet
            r = self.uart.send_packet(
                id=None,
                reserved=rgb_state & 0xFF,  # Send the RGB state as a single byte
                packetType=OW_POWER,
                command=OW_POWER_SET_RGB,
            )

            self.uart.clear_buffer()

            if r.packet_type == OW_ERROR:
                logger.error("Error setting RGB LED state")
                return -1

            logger.info(f"Set RGB LED state to {rgb_state}")
            return rgb_state

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def get_rgb_led(self) -> int:
        """
        Get the current RGB LED state.

        Returns:
            int: The current RGB state (0 = OFF, 1 = RED, 2 = BLUE, 3 = GREEN).

        Raises:
            ValueError: If the controller is not connected.
        """
        if not self.uart.is_connected():
            raise ValueError("High voltage controller not connected")

        try:
            if self.uart.demo_mode:
                return 1  # Default to RED in demo mode

            logger.info("Getting current RGB LED state.")

            r = self.uart.send_packet(
                id=None, packetType=OW_POWER, command=OW_POWER_GET_RGB
            )

            self.uart.clear_buffer()

            if r.packet_type == OW_ERROR:
                logger.error("Error getting RGB LED state")
                return -1

            rgb_state = r.reserved
            logger.info(f"Current RGB LED state is {rgb_state}")
            return rgb_state

        except ValueError as v:
            logger.error("ValueError: %s", v)
            raise  # Re-raise the exception for the caller to handle

        except Exception as e:
            logger.error("Unexpected error during process: %s", e)
            raise  # Re-raise the exception for the caller to handle

    def soft_reset(self) -> bool:
        """
        Perform a soft reset on the Console device.

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
                raise ValueError("Console Device  not connected")

            r = self.uart.send_packet(
                id=None, packetType=OW_CMD, command=OW_CMD_RESET
            )
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

    def enter_dfu(self) -> bool:
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

            r = self.uart.send_packet(id=None, packetType=OW_CMD, command=OW_CMD_DFU)
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
