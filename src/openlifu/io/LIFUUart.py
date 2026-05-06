from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time

import serial
import serial.tools.list_ports

from openlifu.io.LIFUConfig import OW_ACK, OW_CMD_NOP, OW_DATA, OW_ERROR, OW_RESP
from openlifu.io.LIFUSignal import LIFUSignal

# Packet structure constants
OW_START_BYTE = 0xAA
OW_END_BYTE = 0xDD
ID_COUNTER = 0  # Initializing the ID counter

# Set up logging
log = logging.getLogger("UART")
log.setLevel(logging.ERROR)
log.propagate = False

handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s") # Format output with timestamp
handler.setFormatter(formatter)
log.addHandler(handler)

# CRC16-ccitt lookup table
crc16_tab = [
	0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
	0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
	0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
	0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
	0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
	0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
	0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
	0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
	0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
	0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
	0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
	0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
	0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
	0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
	0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
	0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
	0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
	0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
	0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
	0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
	0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
	0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
	0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
	0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
	0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
	0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
	0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
	0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
	0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
	0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
	0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
	0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
]

def util_crc16(buf):
    crc = 0xFFFF

    for byte in buf:
        crc = ((crc << 8) & 0xFFFF) ^ crc16_tab[((crc >> 8) ^ byte) & 0xFF]

    return crc

class UartPacket:
    def __init__(self, id=None, packet_type=None, command=None, addr=None, reserved=None, data=None, buffer=None):
        if buffer:
            self.from_buffer(buffer)
        else:
            self.id = id
            self.packet_type = packet_type
            self.command = command
            self.addr = addr
            self.reserved = reserved
            self.data = data
            self.data_len = len(data)
            self.crc = self.calculate_crc()

    def calculate_crc(self) -> int:
        crc_value = 0xFFFF
        packet = bytearray()
        packet.append(OW_START_BYTE)
        packet.extend(self.id.to_bytes(2, 'big'))
        packet.append(self.packet_type)
        packet.append(self.command)
        packet.append(self.addr)
        packet.append(self.reserved)
        packet.extend(self.data_len.to_bytes(2, 'big'))
        if self.data_len > 0:
            packet.extend(self.data)
        crc_value = util_crc16(packet[1:])
        return crc_value

    def to_bytes(self) -> bytes:
        buffer = bytearray()
        buffer.append(OW_START_BYTE)
        buffer.extend(self.id.to_bytes(2, 'big'))
        buffer.append(self.packet_type)
        buffer.append(self.command)
        buffer.append(self.addr)
        buffer.append(self.reserved)
        buffer.extend(self.data_len.to_bytes(2, 'big'))
        if self.data_len > 0:
            buffer.extend(self.data)
        crc_value = util_crc16(buffer[1:])
        buffer.extend(crc_value.to_bytes(2, 'big'))
        buffer.append(OW_END_BYTE)
        return bytes(buffer)

    def from_buffer(self, buffer: bytes):
        if buffer[0] != OW_START_BYTE or buffer[-1] != OW_END_BYTE:
            raise ValueError("Invalid buffer format")

        self.id = int.from_bytes(buffer[1:3], 'big')
        self.packet_type = buffer[3]
        self.command = buffer[4]
        self.addr = buffer[5]
        self.reserved = buffer[6]
        self.data_len = int.from_bytes(buffer[7:9], 'big')
        self.data = bytearray(buffer[9:9+self.data_len])
        crc_value = util_crc16(buffer[1:9+self.data_len])
        self.crc = int.from_bytes(buffer[9+self.data_len:11+self.data_len], 'big')
        if self.crc != crc_value:
            raise ValueError("CRC mismatch")

    def print_packet(self):
        log.info("UartPacket:")
        log.info(f"  Packet ID: {self.id}")
        log.info(f"  Packet Type: {hex(self.packet_type)}")
        log.info(f"  Command: {hex(self.command)}")
        log.info(f"  Address: {hex(self.addr)}")
        log.info(f"  Reserved: {hex(self.reserved)}")
        log.info(f"  Data Length: {self.data_len}")
        if self.data_len > 0:
            log.info(f"  Data: {self.data.hex()}")
        else:
            log.info("  Data: None")
        log.info(f"  CRC: {hex(self.crc)}")

class LIFUUart:
    def __init__(self, vid, pid, baudrate=921600, timeout=10, align=0, desc="VCP", demo_mode=False, async_mode=False):
        """
        Initialize the UART instance.

        Args:
            vid (int): Vendor ID of the USB device.
            pid (int): Product ID of the USB device.
            baudrate (int): Communication speed.
            timeout (int): Read timeout in seconds.
            align (int): Data alignment parameter.
            desc (str): Descriptor for the device (e.g. "TX" or "HV").
            demo_mode (bool): If True, simulate the connection.
            async_mode (bool): If True, use asynchronous mode.
        """
        self.vid = vid
        self.pid = pid
        self.port = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.align = align
        self.packet_count = 0
        self.serial = None
        self.running = False
        self.asyncMode = async_mode
        self.monitoring_task = None
        self.demo_mode = demo_mode
        self.descriptor = desc
        self.read_thread = None
        self.read_buffer = []
        self.last_rx = time.monotonic()
        self.demo_responses = []  # Predefined responses for testing

        # Signals: each signal emits (descriptor, port or data)
        self.signal_connect = LIFUSignal()
        self.signal_disconnect = LIFUSignal()
        self.signal_data_received = LIFUSignal()

        if async_mode:
            self.loop = asyncio.get_event_loop()
            self.response_queues = {}  # Dictionary to map packet IDs to response queues
            self.response_lock = threading.Lock()  # Lock for thread-safe access to response_queues

    def connect(self):
        """Open the serial port."""
        if self.demo_mode:
            log.info("Demo mode: Simulating UART connection.")
            self.signal_connect.emit(self.descriptor, "demo_mode")
            return
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            log.info("Connected to UART on port %s.", self.port)
            self.signal_connect.emit(self.descriptor, self.port)

            if self.asyncMode:
                log.info("Starting read thread for %s.", self.descriptor)
                self.running = True
                self.read_thread = threading.Thread(target=self._read_data)
                self.read_thread.daemon = True
                self.read_thread.start()
        except serial.SerialException as se:
            log.error("Failed to connect to %s: %s", self.port, se)
            self.running = False
            self.port = None
        except Exception as e:
            raise e

    def disconnect(self):
        """Close the serial port."""
        self.running = False
        if self.demo_mode:
            log.info("Demo mode: Simulating UART disconnection.")
            self.signal_disconnect.emit(self.descriptor, "demo_mode")
            return

        if self.read_thread:
            self.read_thread.join()
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None
        log.info("Disconnected from UART.")
        self.signal_disconnect.emit(self.descriptor, self.port)
        self.port = None

    def is_connected(self) -> bool:
        """
        Check if the device is connected.

        Returns:
            bool: True if connected, False otherwise.
        """
        if self.demo_mode:
            return True
        return self.port is not None and self.serial is not None and self.serial.is_open

    def check_usb_status(self):
        """Check if the USB device is connected or disconnected."""
        device = self.list_vcp_with_vid_pid()
        if device and not self.port:
            log.debug("Device found; trying to connect.")
            self.port = device
            self.connect()
        elif not device and self.port:
            log.debug("Device removed; disconnecting.")
            self.running = False
            self.disconnect()

    async def monitor_usb_status(self, interval=1):
        """Periodically check for USB device connection."""
        if self.demo_mode:
            log.debug("Monitoring in demo mode.")
            self.connect()
            return
        while True:
            self.check_usb_status()
            await asyncio.sleep(interval)

    def start_monitoring(self, interval=1):
        """Start the periodic USB device connection check."""
        if self.demo_mode:
            log.debug("Monitoring in demo mode.")
            return
        if not self.monitoring_task and self.asyncMode:
            self.monitoring_task = asyncio.create_task(self.monitor_usb_status(interval))

    def stop_monitoring(self):
        """Stop the periodic USB device connection check."""
        if self.demo_mode:
            log.info("Monitoring in demo mode.")
            return
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None

    def list_vcp_with_vid_pid(self):
        """Find the USB device by VID and PID."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if hasattr(port, 'vid') and hasattr(port, 'pid') and port.vid == self.vid and port.pid == self.pid:
                return port.device
        return None

    def _read_data(self, timeout=20):
        """Read data from the serial port in a separate thread."""
        log.debug("Starting data read loop for %s.", self.descriptor)
        if self.demo_mode:
            while self.running:
                if self.demo_responses:
                    data = self.demo_responses.pop(0)
                    log.info("Demo mode: Simulated data received: %s", data)
                    self.signal_data_received.emit(self.descriptor, "Demo Response")
                time.sleep(10)  # Simulate delay (10 second)
            return

        # In async mode, run the reading loop in a thread
        while self.running:
            try:
                if self.serial.in_waiting > 0:
                    data = self.serial.read(self.serial.in_waiting)
                    self.read_buffer.extend(data)
                    log.debug("Data received on %s: %s", self.descriptor, data)
                    # Attempt to parse a complete packet from read_buffer.
                    try:
                        # Note: Depending on your protocol, you might need to check for start/end bytes
                        # and possibly handle partial packets.
                        packet = UartPacket(buffer=bytes(self.read_buffer))
                        # Clear the buffer after a successful parse.

                        self.read_buffer = []
                        if self.asyncMode:
                            with self.response_lock:
                                # Check if a queue is waiting for this packet ID.
                                if packet.id in self.response_queues:
                                    self.response_queues[packet.id].put(packet)
                                elif packet.packet_type == OW_DATA and packet.id == 0:
                                    text = packet.data.decode('utf-8', errors='replace')
                                    self.signal_data_received.emit(self.descriptor, text)
                                else:
                                    log.warning("Received an unsolicited packet with ID %d", packet.id)
                        elif packet.packet_type == OW_DATA:
                            self.signal_data_received.emit(self.descriptor, OW_DATA)

                    except ValueError as ve:
                        log.error("Error parsing packet: %s", ve)
                else:
                    time.sleep(0.05)  # Brief sleep to avoid a busy loop
            except serial.SerialException as e:
                log.error("Serial _read_data error on %s: %s", self.descriptor, e)
                self.running = False

    def _tx(self, data: bytes):
        """Send data over UART."""
        if not self.serial or not self.serial.is_open:
            log.error("Serial port is not initialized.")
            return
        if self.demo_mode:
            log.info("Demo mode: Simulating data transmission: %s", data)
            return
        try:
            if self.align > 0:
                while len(data) % self.align != 0:
                    data += bytes([OW_END_BYTE])
            self.serial.write(data)
        except Exception as e:
            log.error("Error during transmission: %s", e)
            raise e

    def read_packet(self, timeout=20) -> UartPacket:
        """
        Read a packet from the UART interface.

        Returns:
            UartPacket: Parsed packet or an error packet if parsing fails.
        """
        start_time = time.monotonic()
        raw_data = b""
        count = 0

        while timeout == -1 or time.monotonic() - start_time < timeout:
            time.sleep(0.05)
            raw_data += self.serial.read_all()
            if raw_data:
                count += 1
                if count > 1:
                    break

        try:
            if not raw_data:
                raise ValueError("No data received from UART within timeout")
            packet = UartPacket(buffer=raw_data)
        except Exception as e:
            log.error("Error parsing packet: %s", e)
            packet = UartPacket(
                id=0,
                packet_type=OW_ERROR,
                command=0,
                addr=0,
                reserved=0,
                data=[]
            )
            raise e

        return packet

    def send_packet(self, id=None, packetType=OW_ACK, command=OW_CMD_NOP, addr=0, reserved=0, data=None, timeout=20):
        """
        Send a packet over UART and, if not running, return a response packet.
        """
        try:
            if not self.serial or not self.serial.is_open:
                log.error("Cannot send packet. Serial port is not connected.")
                return None

            if id is None:
                self.packet_count += 1
                id = self.packet_count

            if data:
                if not isinstance(data, (bytes, bytearray)):
                    raise ValueError("Data must be bytes or bytearray")
                payload = data
                payload_length = len(payload)
            else:
                payload_length = 0
                payload = b''

            packet = bytearray()
            packet.append(OW_START_BYTE)
            packet.extend(id.to_bytes(2, 'big'))
            packet.append(packetType)
            packet.append(command)
            packet.append(addr)
            packet.append(reserved)
            packet.extend(payload_length.to_bytes(2, 'big'))
            if payload_length > 0:
                packet.extend(payload)

            crc_value = util_crc16(packet[1:])  # Exclude start byte
            packet.extend(crc_value.to_bytes(2, 'big'))
            packet.append(OW_END_BYTE)

            self._tx(packet)

            if not self.asyncMode:
                return self.read_packet(timeout=timeout)
            else:
                response_queue = queue.Queue()
                with self.response_lock:
                    self.response_queues[id] = response_queue

                try:
                    # Wait for a response that matches the packet ID.
                    response = response_queue.get(timeout=timeout)
                    # Optionally, check that the response has the expected type and command.
                    if response.packet_type == OW_RESP and response.command == command:
                        return response
                    else:
                        log.error("Received unexpected response: %s", response)
                        return response
                except queue.Empty:
                    log.error("Timeout waiting for response to packet ID %d", id)
                    return None
                finally:
                    with self.response_lock:
                        # Clean up the queue entry regardless of outcome.
                        self.response_queues.pop(id, None)

        except ValueError as ve:
            log.error("Validation error in send_packet: %s", ve)
            raise
        except Exception as e:
            log.error("Unexpected error in send_packet: %s", e)
            raise

    def clear_buffer(self):
        """Clear the read buffer."""
        self.read_buffer = []

    def run_coroutine(self, coro):
        """Run a coroutine using the internal event loop."""
        if not self.loop.is_running():
            return self.loop.run_until_complete(coro)
        else:
            return asyncio.create_task(coro)

    def add_demo_response(self, response: bytes):
        """Add a predefined response for demo mode."""
        if self.demo_mode:
            self.demo_responses.append(response)
        else:
            log.warning("Cannot add demo response when not in demo mode.")

    def print(self):
        """Print the current UART configuration."""
        log.info("    Serial Port: %s", self.port)
        log.info("    Serial Baud: %s", self.baudrate)
