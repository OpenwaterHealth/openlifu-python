from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Constants from C code
LIFU_MAGIC = 0x4C494655  # 'LIFU'
LIFU_VER = 0x00010002    # v1.0.0

@dataclass
class LifuUserConfigHeader:
    """Wire format header for Lifu config"""
    magic: int
    version: int
    seq: int
    crc: int
    json_len: int

    @classmethod
    def from_bytes(cls, data: bytes) -> LifuUserConfigHeader:
        """Parse header from wire format bytes (little-endian)"""
        if len(data) < 16:
            raise ValueError(f"Header data too short: {len(data)} bytes, need 16")

        # Parse: uint32 magic, uint32 version, uint32 seq, uint16 crc, uint16 json_len
        magic, version, seq, crc, json_len = struct.unpack('<IIIHH', data[:16])

        return cls(
            magic=magic,
            version=version,
            seq=seq,
            crc=crc,
            json_len=json_len
        )

    def to_bytes(self) -> bytes:
        """Convert header to wire format bytes (little-endian)"""
        return struct.pack('<IIIHH',
            self.magic,
            self.version,
            self.seq,
            self.crc,
            self.json_len
        )

    def is_valid(self) -> bool:
        """Check if magic and version are valid"""
        return self.magic == LIFU_MAGIC and self.version == LIFU_VER


class LifuUserConfig:
    """
    Encapsulates the Lifu configuration stored in device flash.

    The configuration is stored as a JSON blob with metadata including:
    - magic number for validation
    - version for compatibility
    - sequence number (monotonically increasing)
    - CRC for integrity
    """

    def __init__(self, header: LifuUserConfigHeader | None = None, json_data: Dict[str, Any] | None = None):
        """
        Initialize LifuUserConfig

        Args:
            header: Configuration header metadata
            json_data: Configuration JSON data as a dictionary
        """
        self.header = header if header else LifuUserConfigHeader(
            magic=LIFU_MAGIC,
            version=LIFU_VER,
            seq=0,
            crc=0,
            json_len=0
        )
        self.json_data = json_data if json_data is not None else {}

    @classmethod
    def from_wire_bytes(cls, data: bytes) -> LifuUserConfig:
        """
        Parse configuration from wire format bytes

        Wire format:
            [header: 16 bytes][json: json_len bytes]

        Args:
            data: Raw bytes from device

        Returns:
            LifuUserConfig instance

        Raises:
            ValueError: If data is invalid or malformed
        """
        if len(data) < 16:
            raise ValueError(f"Wire data too short: {len(data)} bytes")

        header = LifuUserConfigHeader.from_bytes(data[:16])

        if not header.is_valid():
            raise ValueError(f"Invalid magic (0x{header.magic:08X}) or version (0x{header.version:08X})")

        # Extract JSON bytes
        json_bytes_end = 16 + header.json_len
        if len(data) < json_bytes_end:
            logger.warning(f"JSON data truncated: expected {header.json_len} bytes, got {len(data) - 16}")
            json_bytes = data[16:]
        else:
            json_bytes = data[16:json_bytes_end]

        # Parse JSON (handle null terminator if present)
        json_str = json_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')

        try:
            json_data = json.loads(json_str) if json_str else {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}. Using empty config.")
            json_data = {}

        return cls(header=header, json_data=json_data)

    def to_wire_bytes(self) -> bytes:
        """
        Convert configuration to wire format for sending to device

        Returns:
            bytes: Wire format [header][json_bytes]
        """
        # Convert JSON to bytes
        json_str = json.dumps(self.json_data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        # Update header with JSON length
        self.header.json_len = len(json_bytes)

        # Build wire format
        return self.header.to_bytes() + json_bytes

    def get_json_str(self) -> str:
        """Get JSON configuration as a formatted string"""
        return json.dumps(self.json_data, indent=2)

    def set_json_str(self, json_str: str):
        """
        Set configuration from JSON string

        Args:
            json_str: JSON string to parse

        Raises:
            json.JSONDecodeError: If JSON is invalid
        """
        self.json_data = json.loads(json_str)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key"""
        return self.json_data.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value by key"""
        self.json_data[key] = value

    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.json_data.update(updates)

    def to_dict(self) -> Dict[str, Any]:
        """Get the configuration as a dictionary"""
        return self.json_data.copy()

    def __repr__(self) -> str:
        return (f"LifuUserConfig(seq={self.header.seq}, crc=0x{self.header.crc:04X}, "
                f"json_len={self.header.json_len}, data={self.json_data})")
