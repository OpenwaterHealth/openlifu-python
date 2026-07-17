from __future__ import annotations

import logging
import os
import sys
import time

if os.name == 'nt':
    pass
else:
    pass


from openlifu.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_watertank.py

"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""

# TO BE USED TO MONITOR TEMPERATURE CURVE TO SEE HOW LONG IT TAKES TO COOL DOWN

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

log_interval = 1  # seconds; you can adjust this variable as needed
num_modules = 2 # Number of modules in the system

use_external_power_supply = False # Select whether to use console or power supply

logger.info("Starting LIFU Test Script...")
interface = LIFUInterface(ext_power_supply=use_external_power_supply)
tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply and not tx_connected:
    logger.warning("TX device not connected. Attempting to turn on 12V...")
    interface.hvcontroller.turn_12v_on()

    # Give time for the TX device to power up and enumerate over USB
    time.sleep(2)

    # Cleanup and recreate interface to reinitialize USB devices
    interface.stop_monitoring()
    del interface
    time.sleep(1)  # Short delay before recreating

    logger.info("Reinitializing LIFU interface after powering 12V...")
    interface =  LIFUInterface(ext_power_supply=use_external_power_supply)

    # Re-check connection
    tx_connected, hv_connected = interface.is_device_connected()

if not use_external_power_supply:
    if hv_connected:
        logger.info(f"  HV Connected: {hv_connected}")
    else:
        logger.error("❌ HV NOT fully connected.")
        sys.exit(1)
else:
    logger.info("  Using external power supply")

if tx_connected:
    logger.info(f"  TX Connected: {tx_connected}")
    logger.info("✅ LIFU Device fully connected.")
else:
    logger.error("❌ TX NOT fully connected.")
    sys.exit(1)

# Verify communication with the devices
if not interface.txdevice.ping():
    logger.error("Failed to ping the transmitter device.")
    sys.exit(1)

if not use_external_power_supply and not interface.hvcontroller.ping():
    logger.error("Failed to ping the console device.")
    sys.exit(1)

print(f"console version: {interface.hvcontroller.get_version()}")

logger.info("Enumerate TX7332 chips")
# num_tx_devices = interface.txdevice.get_tx_module_count()
num_tx_devices = 10

for module in range(num_tx_devices+1):
    try:
        tx_firmware_version = interface.txdevice.get_version(module=module)
        logger.info(f"TX Firmware Version: {tx_firmware_version}")
    except Exception as e:
        logger.error(f"Error querying TX firmware version: {e}")
