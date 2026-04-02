from __future__ import annotations

import sys

import numpy as np

from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.geo import Point
from openlifu_sdk.io.LIFUInterface import LIFUInterface
from openlifu.plan.solution import Solution

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_transmitter2.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")

interface = LIFUInterface()
tx_connected, hv_connected = interface.is_device_connected()

if not tx_connected:
    print("TX device not connected. Attempting to turn on 12V...")
    sys.exit(1)

print("Ping Transmitter device")
interface.txdevice.ping()

print("Get Version")
version = interface.txdevice.get_version()
print(f"Version: {version}")
print("Get Temperature")
temperature = interface.txdevice.get_temperature()
print(f"Temperature: {temperature} °C")

print("Enumerate TX7332 chips")
num_tx_devices = interface.txdevice.enum_tx7332_devices()
if num_tx_devices > 0:
    print(f"Number of TX7332 devices found: {num_tx_devices}")
else:
    raise Exception("No TX7332 devices found.")

# set focus
xInput = 0
yInput = 0
zInput = 50

frequency = 400e3
voltage = 12.0
duration = 2e-5

pulse = Pulse(frequency=frequency, duration=duration)
pt = Point(position=(xInput,yInput,zInput), units="mm")
sequence = Sequence(
    pulse_interval=0.1,
    pulse_count=10,
    pulse_train_interval=1,
    pulse_train_count=1
)

# Calculate delays and apodizations to perform beam forming

solution = Solution(
    delays = np.zeros((1,64)),
    apodizations = np.ones((1,64)),
    pulse = pulse,
    voltage=voltage,
    sequence = sequence
)

sol_dict = solution.to_dict()
profile_index = 1
profile_increment = True
print("Set Solution")
interface.txdevice.set_solution(
    pulse = sol_dict['pulse'],
    delays = sol_dict['delays'],
    apodizations= sol_dict['apodizations'],
    sequence= sol_dict['sequence'],
    trigger_mode = "continuous",
    profile_index=profile_index,
    profile_increment=profile_increment
)

print("Get Trigger")
trigger_setting = interface.txdevice.get_trigger_json()
if trigger_setting:
    print(f"Trigger Setting: {trigger_setting}")
else:
    print("Failed to get trigger setting.")

print("Starting Trigger...")
if interface.start_sonication():
    print("Trigger Running Press enter to STOP:")
    input()  # Wait for the user to press Enter
    if interface.stop_sonication():
        print("Trigger stopped successfully.")
    else:
        print("Failed to stop trigger.")
else:
    print("Failed to get trigger setting.")
