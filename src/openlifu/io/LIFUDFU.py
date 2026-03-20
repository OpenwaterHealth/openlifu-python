"""LIFU Transmitter Firmware Update (DFU) support.

Provides:
  - :func:`stm32_crc32`          — STM32-compatible CRC32
  - :func:`parse_signed_package` — parse/validate a signed firmware package
  - :class:`STM32USBDFU`         — USB DFU client (PyUSB, for module 0)
  - :class:`STM32I2CDFUviaMaster`— I2C DFU via OW UART master passthrough (modules 1+)
  - :class:`LIFUDFUManager`      — high-level firmware update orchestration
"""

from __future__ import annotations

import logging
import struct
import time
from typing import TYPE_CHECKING, Callable

from openlifu.io.LIFUConfig import OW_ERROR, OW_I2C_PASSTHRU

if TYPE_CHECKING:
    from openlifu.io.LIFUUart import LIFUUart

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional USB DFU dependencies (module 0 only)
# ---------------------------------------------------------------------------
try:
    import usb.backend.libusb1 as _usb_libusb1
    import usb.core as _usb_core
    import usb.util as _usb_util
    _USB_DFU_AVAILABLE = True
except ImportError:
    _usb_core = None
    _usb_util = None
    _usb_libusb1 = None
    _USB_DFU_AVAILABLE = False

try:
    import libusb_package as _libusb_package
except ImportError:
    _libusb_package = None

# ---------------------------------------------------------------------------
# DFU protocol constants (shared by USB and I2C paths)
# ---------------------------------------------------------------------------

# USB DFU virtual addresses (must match usbd_dfu_if.c)
USB_DFU_VERSION_VIRT_ADDR = 0xFFFFFF00
USB_DFU_VERSION_READ_LEN  = 64

# I2C DFU command bytes (must match i2c_dfu_if.h)
I2C_DFU_SLAVE_ADDR      = 0x72
I2C_DFU_CMD_DNLOAD      = 0x01
I2C_DFU_CMD_ERASE       = 0x02
I2C_DFU_CMD_GETSTATUS   = 0x03
I2C_DFU_CMD_MANIFEST    = 0x04
I2C_DFU_CMD_RESET       = 0x05
I2C_DFU_CMD_GETVERSION  = 0x06
I2C_DFU_STATUS_OK       = 0x00
I2C_DFU_STATUS_BUSY     = 0x01
I2C_DFU_STATUS_ERROR    = 0x02
I2C_DFU_STATUS_BAD_ADDR = 0x03
I2C_DFU_STATUS_FLASH_ERR= 0x04
I2C_DFU_STATE_DNBUSY    = 0x01
I2C_DFU_STATE_ERROR     = 0x04
# Maximum data bytes per write_block call.  The enclosing OW_I2C_PASSTHRU UART
# packet carries (1 cmd + 4 addr + 2 len) = 7 bytes of I2C-DFU header, so the
# total packet payload is I2C_DFU_MAX_XFER_SIZE + 7.  The master firmware hard-
# rejects any UART packet with data_len > DATA_MAX_SIZE (2048), so this value
# must be ≤ 2041.  Use 512 for a safe, standard I2C block size.
I2C_DFU_MAX_XFER_SIZE   = 512
I2C_DFU_VERSION_STR_MAX = 32

# OW_I2C_PASSTHRU sub-commands (must match firmware if_commands.c handler)
_PASSTHRU_WRITE       = 0x00   # write only
_PASSTHRU_WRITE_READ  = 0x01   # write then delay 5 ms then read

# Signed package format (must match dfu-test.py)
_PKG_MAGIC        = 0x314B4750   # 'PGK1'
_PKG_VERSION      = 1
_PKG_HDR_NOCRC    = "<IHHIIIII"
_PKG_HDR_FULL     = "<IHHIIIIII"


# ---------------------------------------------------------------------------
# Package helpers
# ---------------------------------------------------------------------------

def stm32_crc32(data: bytes, init: int = 0xFFFFFFFF) -> int:
    """Compute CRC32 compatible with the STM32 CRC peripheral (poly=0x04C11DB7)."""
    poly = 0x04C11DB7
    crc = init & 0xFFFFFFFF
    for b in data:
        crc ^= (b & 0xFF) << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def parse_signed_package(pkg: bytes) -> dict:
    """Parse and integrity-check a signed firmware package.

    Returns a dict with keys: ``fw_address``, ``meta_address``, ``fw``, ``meta``.

    Raises:
        ValueError: If the package is malformed or any CRC fails.
    """
    hdr_size = struct.calcsize(_PKG_HDR_FULL)
    if len(pkg) < hdr_size:
        raise ValueError("signed package too small")

    (magic, version, declared_hdr_size,
     fw_address, fw_len,
     meta_address, meta_len,
     payload_crc, header_crc) = struct.unpack(_PKG_HDR_FULL, pkg[:hdr_size])

    if magic != _PKG_MAGIC:
        raise ValueError(f"signed package magic mismatch: 0x{magic:08X}")
    if version != _PKG_VERSION:
        raise ValueError(f"signed package version mismatch: {version}")
    if declared_hdr_size != hdr_size:
        raise ValueError("signed package header size mismatch")

    calc_hdr_crc = stm32_crc32(pkg[:hdr_size - 4])
    if header_crc != calc_hdr_crc:
        raise ValueError(
            f"header CRC mismatch: pkg=0x{header_crc:08X}, calc=0x{calc_hdr_crc:08X}"
        )

    payload_len = fw_len + meta_len
    payload = pkg[hdr_size:]
    if len(payload) != payload_len:
        raise ValueError(
            f"payload size mismatch: expected {payload_len}, got {len(payload)}"
        )

    calc_payload_crc = stm32_crc32(payload)
    if payload_crc != calc_payload_crc:
        raise ValueError(
            f"payload CRC mismatch: pkg=0x{payload_crc:08X}, calc=0x{calc_payload_crc:08X}"
        )

    return {
        "fw_address":   fw_address,
        "meta_address": meta_address,
        "fw":           payload[:fw_len],
        "meta":         payload[fw_len:],
    }


# ---------------------------------------------------------------------------
# USB DFU client  (module 0)
# ---------------------------------------------------------------------------

class STM32USBDFU:
    """Minimal STM32 DfuSe USB client using PyUSB.

    Supports Set-Address-Pointer, page erase, memory write and DFU UPLOAD
    (used to read the bootloader version string).

    Requires: ``pip install pyusb``  plus a libusb-1.0 backend.
    """

    # DFU class requests
    DFU_DNLOAD    = 1
    DFU_UPLOAD    = 2
    DFU_GETSTATUS = 3
    DFU_CLRSTATUS = 4
    DFU_ABORT     = 6

    # DfuSe DNLOAD block 0 sub-commands
    CMD_SET_ADDRESS_POINTER = 0x21
    CMD_ERASE               = 0x41

    # DFU state values
    STATE_DFU_DNLOAD_SYNC         = 3
    STATE_DFU_DNLOAD_BUSY         = 4
    STATE_DFU_DNLOAD_IDLE         = 5
    STATE_DFU_MANIFEST_SYNC       = 6
    STATE_DFU_MANIFEST            = 7
    STATE_DFU_MANIFEST_WAIT_RESET = 8
    STATE_DFU_ERROR               = 10

    def __init__(self, vid: int = 0x0483, pid: int = 0xDF11,
                 transfer_size: int = 1024, timeout_ms: int = 4000,
                 libusb_dll: str | None = None):
        if not _USB_DFU_AVAILABLE:
            raise RuntimeError(
                "PyUSB not available. Install with: pip install pyusb"
            )
        self.vid = vid
        self.pid = pid
        self.transfer_size = transfer_size
        self.timeout_ms = timeout_ms
        self.libusb_dll = libusb_dll
        self.dev = None
        self.intf = None
        self._backend = None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        if self.libusb_dll:
            self._backend = _usb_libusb1.get_backend(
                find_library=lambda _: self.libusb_dll
            )
        elif _libusb_package is not None:
            self._backend = _usb_libusb1.get_backend(
                find_library=_libusb_package.find_library
            )
        else:
            self._backend = _usb_libusb1.get_backend()
        return self._backend

    def get_backend(self):
        """Public accessor for the libusb backend (avoids protected-member access)."""
        return self._get_backend()

    def open(self) -> STM32USBDFU:
        self.dev = _usb_core.find(
            idVendor=self.vid, idProduct=self.pid, backend=self._get_backend()
        )
        if self.dev is None:
            raise RuntimeError(
                f"USB DFU device not found: VID=0x{self.vid:04X}, PID=0x{self.pid:04X}"
            )
        self.dev.set_configuration()
        cfg = self.dev.get_active_configuration()
        for intf in cfg:
            if (intf.bInterfaceClass == 0xFE
                    and intf.bInterfaceSubClass == 0x01
                    and intf.bInterfaceProtocol == 0x02):
                self.intf = intf
                break
        if self.intf is None:
            raise RuntimeError("No DFU interface found on USB device")
        try:
            if self.dev.is_kernel_driver_active(self.intf.bInterfaceNumber):
                self.dev.detach_kernel_driver(self.intf.bInterfaceNumber)
        except (NotImplementedError, _usb_core.USBError) as e:
            logger.debug("kernel driver detach not supported or failed: %s", e)
        _usb_util.claim_interface(self.dev, self.intf.bInterfaceNumber)
        self._clear_error_state()
        return self

    def close(self) -> None:
        if self.dev is not None and self.intf is not None:
            try:
                _usb_util.release_interface(self.dev, self.intf.bInterfaceNumber)
            except _usb_core.USBError as e:
                logger.exception("error closing interface: %s", e)
            _usb_util.dispose_resources(self.dev)
        self.dev = None
        self.intf = None

    def __enter__(self) -> STM32USBDFU:
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # --- low-level USB control transfers ---

    def _ctrl_out(self, req: int, value: int, data: bytes = b"") -> int:
        return self.dev.ctrl_transfer(
            0x21, req, value, self.intf.bInterfaceNumber,
            data, timeout=self.timeout_ms
        )

    def _ctrl_in(self, req: int, value: int, length: int) -> bytes:
        return bytes(self.dev.ctrl_transfer(
            0xA1, req, value, self.intf.bInterfaceNumber,
            length, timeout=self.timeout_ms
        ))

    def get_status(self) -> dict:
        raw = self._ctrl_in(self.DFU_GETSTATUS, 0, 6)
        poll_ms = raw[1] | (raw[2] << 8) | (raw[3] << 16)
        return {"status": raw[0], "poll_timeout_ms": poll_ms, "state": raw[4]}

    def clear_status(self) -> None:
        self._ctrl_out(self.DFU_CLRSTATUS, 0, b"")

    def abort(self) -> None:
        self._ctrl_out(self.DFU_ABORT, 0, b"")

    def _clear_error_state(self) -> None:
        for _ in range(3):
            st = self.get_status()
            if st["state"] != self.STATE_DFU_ERROR:
                break
            self.clear_status()

    def _recover_idle(self) -> None:
        for _ in range(4):
            st = self.get_status()
            if st["state"] in (
                self.STATE_DFU_DNLOAD_IDLE,
                self.STATE_DFU_MANIFEST_WAIT_RESET,
            ):
                self.abort()
            elif st["state"] == self.STATE_DFU_ERROR:
                self.clear_status()
            else:
                break

    def _wait_while_busy(self, timeout_s: float = 30.0) -> dict:
        busy = {
            self.STATE_DFU_DNLOAD_SYNC,
            self.STATE_DFU_DNLOAD_BUSY,
            self.STATE_DFU_MANIFEST_SYNC,
            self.STATE_DFU_MANIFEST,
        }
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = self.get_status()
            if st["state"] not in busy:
                return st
            time.sleep(max(st["poll_timeout_ms"] / 1000.0, 0.005))
        raise TimeoutError(f"USB DFU device stuck in busy state after {timeout_s:.0f} s")

    def _dnload(self, block_num: int, payload: bytes) -> dict:
        self._recover_idle()
        try:
            self._ctrl_out(
                self.DFU_DNLOAD, block_num, bytes(payload) if payload else b""
            )
        except _usb_core.USBError as e:
            # errno 110 = ETIMEDOUT (libusb); swallow only genuine transfer timeouts
            if e.errno != 110:
                raise
            logger.debug("_dnload timeout on block %d, polling status anyway: %s", block_num, e)
        return self._wait_while_busy()

    def _set_address(self, address: int) -> None:
        payload = bytes([self.CMD_SET_ADDRESS_POINTER]) + struct.pack("<I", address)
        self._dnload(0, payload)

    def _erase_page(self, address: int) -> None:
        payload = bytes([self.CMD_ERASE]) + struct.pack("<I", address)
        self._dnload(0, payload)

    def get_version(self) -> str:
        """Read bootloader version string via DFU UPLOAD from the virtual address."""
        self._set_address(USB_DFU_VERSION_VIRT_ADDR)
        self.abort()
        raw = self._ctrl_in(self.DFU_UPLOAD, 2, USB_DFU_VERSION_READ_LEN)
        try:
            self._wait_while_busy()
        except (TimeoutError, RuntimeError) as e:
            logger.warning("busy wait error: %s", e)
        self.abort()
        return raw.rstrip(b"\x00").decode("ascii", errors="replace")

    def write_memory(self, address: int, data: bytes,
                     page_erase: bool = True,
                     progress_callback: Callable | None = None) -> None:
        """Write data to target flash, optionally erasing each 2 KB page first."""
        total = len(data)
        page_size = 2048
        self._recover_idle()
        self._set_address(address)
        block = 2
        written = 0
        for offset in range(0, total, self.transfer_size):
            chunk = data[offset:offset + self.transfer_size]
            if page_erase and (offset % page_size == 0):
                self._erase_page(address + offset)
            self._dnload(block, chunk)
            block += 1
            written += len(chunk)
            if progress_callback:
                progress_callback(written, total, "USB DFU write")
        self.abort()

    def manifest(self) -> None:
        """Send zero-length DNLOAD to trigger DFU manifestation (launches firmware)."""
        self._recover_idle()
        try:
            self._ctrl_out(self.DFU_DNLOAD, 0, b"")
            self._wait_while_busy()
        except (_usb_core.USBError, TimeoutError) as e:
            logger.debug("device disconnect/timeout during manifest: %s", e)


# ---------------------------------------------------------------------------
# I2C DFU client via OW master passthrough  (modules 1+)
# ---------------------------------------------------------------------------

class STM32I2CDFUviaMaster:
    """I2C DFU client that routes all I2C transactions through the USB-master
    module via the ``OW_I2C_PASSTHRU`` UART packet type.

    The master firmware receives the passthrough request and executes the raw
    I2C write (and optional read) on the global I2C bus toward the slave DFU
    bootloader at *i2c_addr* (default 0x72).

    Packet wire format used::

        packetType = OW_I2C_PASSTHRU (0xE9)
        addr       = 7-bit I2C slave address
        command    = 0x00  write-only
                   = 0x01  write, 5 ms delay, read back <reserved> bytes
        reserved   = number of bytes to read back (command 0x01 only, max 255)
        data       = raw bytes to write
    """

    def __init__(self, uart: LIFUUart,
                 i2c_addr: int = I2C_DFU_SLAVE_ADDR,
                 write_read_delay_s: float = 0.005):
        self._uart = uart
        self._addr = i2c_addr
        self._wr_delay = write_read_delay_s

    # --- low-level transport primitives ---

    def _write(self, payload: bytes) -> None:
        """Send a write-only passthrough packet to the I2C slave."""
        r = self._uart.send_packet(
            id=None,
            packetType=OW_I2C_PASSTHRU,
            command=_PASSTHRU_WRITE,
            addr=self._addr,
            reserved=0,
            data=payload,
        )
        self._uart.clear_buffer()
        if r is None or r.packet_type == OW_ERROR:
            raise RuntimeError(
                f"I2C passthrough write failed (addr=0x{self._addr:02X}, "
                f"payload={payload[:8].hex()}...)"
            )

    def _exchange(self, payload: bytes, read_len: int,
                  pre_read_delay_s: float | None = None) -> bytes:
        """Write *payload* to the I2C slave, wait, then read *read_len* bytes back.

        The firmware inserts a fixed 5 ms gap between write and read.
        An optional extra host-side delay can be added via *pre_read_delay_s*
        (not usually needed).
        """
        if pre_read_delay_s and pre_read_delay_s > 0:
            time.sleep(pre_read_delay_s)

        r = self._uart.send_packet(
            id=None,
            packetType=OW_I2C_PASSTHRU,
            command=_PASSTHRU_WRITE_READ,
            addr=self._addr,
            reserved=read_len,
            data=payload,
        )
        self._uart.clear_buffer()
        if r is None or r.packet_type == OW_ERROR:
            raise RuntimeError(
                f"I2C passthrough exchange failed (addr=0x{self._addr:02X}, "
                f"want_rx={read_len})"
            )
        actual = len(r.data) if r.data else 0
        if actual < read_len:
            raise RuntimeError(
                f"I2C passthrough exchange returned too few bytes: "
                f"expected {read_len}, got {actual} "
                f"(addr=0x{self._addr:02X})"
            )
        return bytes(r.data[:read_len])

    # --- DFU protocol commands ---

    def get_status(self) -> dict:
        """Send CMD_GETSTATUS and return status/state."""
        raw = self._exchange(bytes([I2C_DFU_CMD_GETSTATUS]), 2)
        return {"status": raw[0], "state": raw[1]}

    def _wait_while_busy(self, timeout_s: float = 10.0) -> dict:
        _ERROR_STATUSES = (I2C_DFU_STATUS_ERROR, I2C_DFU_STATUS_BAD_ADDR, I2C_DFU_STATUS_FLASH_ERR)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = self.get_status()
            if st["state"] == I2C_DFU_STATE_ERROR or st["status"] in _ERROR_STATUSES:
                raise RuntimeError(
                    f"I2C DFU error: status=0x{st['status']:02X}, "
                    f"state=0x{st['state']:02X}"
                )
            if (st["status"] != I2C_DFU_STATUS_BUSY
                    and st["state"] != I2C_DFU_STATE_DNBUSY):
                return st
            time.sleep(0.020)
        raise TimeoutError(f"I2C DFU timed out after {timeout_s:.0f} s")

    def erase_page(self, address: int) -> None:
        """Erase the flash page containing *address*."""
        self._write(struct.pack("<BI", I2C_DFU_CMD_ERASE, address))
        self._wait_while_busy(timeout_s=10.0)

    def mass_erase(self) -> None:
        """Erase the entire application flash region (sentinel addr = 0xFFFFFFFF)."""
        self._write(struct.pack("<BI", I2C_DFU_CMD_ERASE, 0xFFFFFFFF))
        self._wait_while_busy(timeout_s=120.0)

    def write_block(self, address: int, data: bytes) -> None:
        """Program one block (≤ ``I2C_DFU_MAX_XFER_SIZE`` bytes)."""
        if not data:
            return
        payload = struct.pack("<BIH", I2C_DFU_CMD_DNLOAD, address, len(data)) + data
        self._write(payload)
        self._wait_while_busy(timeout_s=10.0)

    def write_memory(self, address: int, data: bytes,
                     progress_callback: Callable | None = None) -> None:
        """Write arbitrary-length data in ``I2C_DFU_MAX_XFER_SIZE``-byte chunks."""
        total = len(data)
        written = 0
        for offset in range(0, total, I2C_DFU_MAX_XFER_SIZE):
            chunk = data[offset:offset + I2C_DFU_MAX_XFER_SIZE]
            self.write_block(address + offset, chunk)
            written += len(chunk)
            if progress_callback:
                progress_callback(written, total, "I2C DFU write")

    def manifest(self) -> None:
        """Send CMD_MANIFEST to finalise the download and lock flash."""
        self._write(bytes([I2C_DFU_CMD_MANIFEST]))
        self._wait_while_busy(timeout_s=10.0)

    def reset(self) -> None:
        """Send CMD_RESET; the device reboots immediately (no response)."""
        self._write(bytes([I2C_DFU_CMD_RESET]))

    def get_version(self) -> str:
        """Read the null-terminated bootloader version string."""
        read_len = 2 + I2C_DFU_VERSION_STR_MAX
        raw = self._exchange(bytes([I2C_DFU_CMD_GETVERSION]), read_len)
        if raw[0] not in (I2C_DFU_STATUS_OK, I2C_DFU_STATUS_BUSY):
            raise RuntimeError(
                f"I2C DFU GETVERSION failed: status=0x{raw[0]:02X}"
            )
        return raw[2:].split(b"\x00")[0].decode("ascii", errors="replace")


# ---------------------------------------------------------------------------
# High-level firmware update manager
# ---------------------------------------------------------------------------

class LIFUDFUManager:
    """Orchestrates firmware updates for a single LIFU transmitter module.

    Usage::

        from openlifu.io.LIFUDFU import LIFUDFUManager
        mgr = LIFUDFUManager(uart=txdevice.uart)
        mgr.update_module(
            module=1,
            package_file="path/to/lifu-transmitter-fw.bin.signed.bin",
            enter_dfu_fn=txdevice.enter_dfu,
        )
    """

    def __init__(self, uart: LIFUUart):
        self._uart = uart

    # --- per-transport helpers ---

    def get_bootloader_version_usb(self, vid: int = 0x0483, pid: int = 0xDF11,
                                   libusb_dll: str | None = None) -> str:
        """Read bootloader version string from module 0 via USB DFU."""
        with STM32USBDFU(vid=vid, pid=pid, libusb_dll=libusb_dll) as dfu:
            return dfu.get_version()

    def get_bootloader_version_i2c(self, i2c_addr: int = I2C_DFU_SLAVE_ADDR) -> str:
        """Read bootloader version string from a slave module via I2C passthrough."""
        dfu = STM32I2CDFUviaMaster(uart=self._uart, i2c_addr=i2c_addr)
        return dfu.get_version()

    def program_usb(self, package_file: str,
                    vid: int = 0x0483, pid: int = 0xDF11,
                    libusb_dll: str | None = None,
                    progress_callback: Callable | None = None) -> None:
        """Program a signed package to module 0 via USB DFU.

        The module must already be in DFU bootloader mode.
        """
        with open(package_file, "rb") as f:
            pkg_blob = f.read()
        pkg = parse_signed_package(pkg_blob)

        logger.info(
            "USB DFU: fw %d B @ 0x%08X, meta %d B @ 0x%08X",
            len(pkg["fw"]), pkg["fw_address"],
            len(pkg["meta"]), pkg["meta_address"],
        )
        with STM32USBDFU(vid=vid, pid=pid, libusb_dll=libusb_dll) as dfu:
            dfu.write_memory(
                pkg["fw_address"], pkg["fw"],
                page_erase=True, progress_callback=progress_callback
            )
            dfu.write_memory(
                pkg["meta_address"], pkg["meta"],
                page_erase=True, progress_callback=progress_callback
            )
            logger.info("USB DFU: sending manifest...")
            dfu.manifest()
        logger.info("USB DFU: programming complete.")

    def program_i2c(self, package_file: str,
                    i2c_addr: int = I2C_DFU_SLAVE_ADDR,
                    progress_callback: Callable | None = None) -> None:
        """Program a signed package to a slave module via I2C passthrough.

        The slave must already be in DFU bootloader mode at *i2c_addr*.

        Sequence (mirrors dfu-i2c-test.py program-package):
          1. Mass-erase the application flash region.
          2. Erase the metadata page explicitly (it is outside the app region
             and is NOT touched by mass-erase).
          3. Write the firmware payload.
          4. Write the metadata blob.
          5. Send CMD_MANIFEST.
        """
        with open(package_file, "rb") as f:
            pkg_blob = f.read()
        pkg = parse_signed_package(pkg_blob)

        logger.info(
            "I2C DFU: fw %d B @ 0x%08X, meta %d B @ 0x%08X",
            len(pkg["fw"]), pkg["fw_address"],
            len(pkg["meta"]), pkg["meta_address"],
        )
        dfu = STM32I2CDFUviaMaster(uart=self._uart, i2c_addr=i2c_addr)
        logger.info("I2C DFU: mass erasing application region...")
        dfu.mass_erase()
        logger.info("I2C DFU: erasing metadata page @ 0x%08X...", pkg["meta_address"])
        dfu.erase_page(pkg["meta_address"])
        dfu.write_memory(
            pkg["fw_address"], pkg["fw"],
            progress_callback=progress_callback
        )
        logger.info("I2C DFU: writing metadata...")
        dfu.write_memory(
            pkg["meta_address"], pkg["meta"]
        )
        logger.info("I2C DFU: sending manifest...")
        dfu.manifest()
        logger.info("I2C DFU: programming complete.")

    def _wait_for_usb_dfu(self, vid: int, pid: int, libusb_dll: str | None,
                           timeout_s: float = 30.0, poll_interval_s: float = 1.0) -> str:
        """Poll for the USB DFU device until it enumerates or *timeout_s* elapses.

        Returns the bootloader version string once the device is found.
        Raises RuntimeError if the device does not appear within the timeout.
        """
        # Pre-flight: verify the libusb backend can be loaded before entering
        # the poll loop.  If the DLL is missing or the path is wrong this fails
        # immediately with a clear message instead of silently timing out.
        _probe = STM32USBDFU(vid=vid, pid=pid, libusb_dll=libusb_dll)
        backend = _probe.get_backend()
        if backend is None:
            raise RuntimeError(
                "libusb backend not available — install libusb or pass --libusb-dll "
                "pointing to a valid libusb-1.0.dll."
            )

        deadline = time.monotonic() + timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            # Phase 1: check if the DFU device has appeared (no I/O yet).
            try:
                dev = _usb_core.find(idVendor=vid, idProduct=pid, backend=backend)
            except (_usb_core.USBError, OSError) as e:
                logger.warning("USB DFU find error (attempt %d): %s", attempt, e)
                time.sleep(poll_interval_s)
                continue

            if dev is None:
                remaining = deadline - time.monotonic()
                logger.debug(
                    "USB DFU not found yet (attempt %d, %.0f s remaining)...",
                    attempt, max(remaining, 0)
                )
                time.sleep(poll_interval_s)
                continue

            # Phase 2: device is present — open it and read the version string.
            elapsed = timeout_s - (deadline - time.monotonic())
            logger.info(
                "USB DFU device found after %.1f s (attempt %d)", elapsed, attempt
            )
            try:
                with STM32USBDFU(vid=vid, pid=pid, libusb_dll=libusb_dll) as dfu:
                    version = dfu.get_version()
                return version
            except (_usb_core.USBError, RuntimeError, TimeoutError) as e:
                # Device enumerated but version read failed (e.g. DFU state
                # machine not ready yet or bootloader doesn't support virtual
                # version address).  Log visibly and return a placeholder so
                # the update can still proceed.
                logger.warning(
                    "USB DFU device found but version read failed: %s — "
                    "proceeding with version='unknown'", e
                )
                return "unknown"

        raise RuntimeError(
            f"USB DFU device (VID=0x{vid:04X}, PID=0x{pid:04X}) did not "
            f"enumerate within {timeout_s:.0f} s"
        )

    def update_module(self,
                      module: int,
                      package_file: str,
                      enter_dfu_fn: Callable,
                      vid: int = 0x0483,
                      pid: int = 0xDF11,
                      libusb_dll: str | None = None,
                      i2c_addr: int = I2C_DFU_SLAVE_ADDR,
                      dfu_wait_s: float = 3.0,
                      dfu_enum_timeout_s: float = 30.0,
                      progress_callback: Callable | None = None) -> None:
        """High-level firmware update for a single module.

        Steps:
         1. Call *enter_dfu_fn(module=module)* to reboot into the bootloader.
         2. Wait *dfu_wait_s* seconds (initial settling delay).
         3. For module 0: poll for the USB DFU device until it enumerates
            (up to *dfu_enum_timeout_s*) then program.
            For modules 1+: poll the I2C DFU slave via passthrough then program.
         4. Program the signed package.

        Module 0 (USB master) uses USB DFU.
        Modules 1+ use I2C DFU through the master's ``OW_I2C_PASSTHRU`` path,
        writing to *i2c_addr* (default 0x72).

        Args:
            module:              Physical module index (0 = USB master).
            package_file:        Path to the signed firmware package.
            enter_dfu_fn:        Callable that triggers DFU mode, e.g.
                                 ``txdevice.enter_dfu``.
            vid:                 USB VID for module 0 USB DFU.
            pid:                 USB PID for module 0 USB DFU.
            libusb_dll:          Optional path to libusb-1.0.dll (Windows).
            i2c_addr:            I2C DFU slave address for modules 1+.
            dfu_wait_s:          Initial settling delay after DFU-enter (default 3 s).
            dfu_enum_timeout_s:  Total time to wait for the bootloader to appear
                                 (default 30 s).  Includes *dfu_wait_s*.
            progress_callback:   Optional ``(written, total, label)`` callable.

        Raises:
            RuntimeError: If DFU entry cannot be verified or programming fails.
        """
        logger.info("Requesting DFU mode on module %d...", module)
        enter_dfu_fn(module=module)

        if dfu_wait_s > 0:
            logger.info("Initial DFU settling delay: %.1f s...", dfu_wait_s)
            time.sleep(dfu_wait_s)

        if module == 0:
            logger.info(
                "Waiting for USB DFU device (timeout %ds)...", dfu_enum_timeout_s
            )
            try:
                bl_version = self._wait_for_usb_dfu(
                    vid=vid, pid=pid, libusb_dll=libusb_dll,
                    timeout_s=dfu_enum_timeout_s,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Module 0 did not enter USB DFU mode: {e}"
                ) from e
            logger.info("USB DFU bootloader version: %s", bl_version)
            self.program_usb(
                package_file, vid=vid, pid=pid,
                libusb_dll=libusb_dll,
                progress_callback=progress_callback,
            )
        else:
            logger.info(
                "Verifying I2C DFU entry (module %d, addr=0x%02X via master)...",
                module, i2c_addr,
            )
            try:
                bl_version = self.get_bootloader_version_i2c(i2c_addr=i2c_addr)
            except (RuntimeError, TimeoutError) as e:
                raise RuntimeError(
                    f"Module {module} did not enter I2C DFU mode at "
                    f"0x{i2c_addr:02X}: {e}"
                ) from e
            if not bl_version:
                raise RuntimeError(
                    f"Module {module} I2C DFU bootloader returned an empty version string"
                )
            logger.info("I2C DFU bootloader version: %s", bl_version)
            self.program_i2c(
                package_file, i2c_addr=i2c_addr,
                progress_callback=progress_callback,
            )

        logger.info("Firmware update complete for module %d.", module)
