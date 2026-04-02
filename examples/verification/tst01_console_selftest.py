from __future__ import annotations

import sys

import base58

from openlifu_sdk.io.LIFUInterface import LIFUInterface

# set PYTHONPATH=%cd%\src;%PYTHONPATH%
# python notebooks/test_console.py
"""
Test script to automate:
1. Connect to the device.
2. Test HVController: Turn HV on/off and check voltage.
3. Test Device functionality.
"""
print("Starting LIFU Test Script...")
interface = LIFUInterface(TX_test_mode=False)
tx_connected, hv_connected = interface.is_device_connected()
if tx_connected and hv_connected:
    print("LIFU Device Fully connected.")
else:
    print(f'LIFU Device NOT Fully Connected. TX: {tx_connected}, HV: {hv_connected}')

if not hv_connected:
       print("HV Controller not connected.")
       sys.exit()

print("Ping the device")
interface.hvcontroller.ping()

print("Toggle LED")
interface.hvcontroller.toggle_led()

print("Get Version")
version = interface.hvcontroller.get_version()
print(f"Version: {version}")

print("Echo Data")
echo, echo_len = interface.hvcontroller.echo(echo_data=b'Hello LIFU!')
if echo_len > 0:
    print(f"Echo: {echo.decode('utf-8')}")  # Echo: Hello LIFU!
else:
    print("Echo failed.")

print("Get HW ID")
hw_id = interface.hvcontroller.get_hardware_id()
print(f"HW ID: {hw_id}")
encoded_id = base58.b58encode(bytes.fromhex(hw_id)).decode()
print(f"OW-LIFU-CON-{encoded_id}")

print("Get Temperature1")
temp1 = interface.hvcontroller.get_temperature1()
print(f"Temperature1: {temp1}")

print("Get Temperature2")
temp2 = interface.hvcontroller.get_temperature2()
print(f"Temperature2: {temp2}")

print("Set Bottom Fan Speed to 20%")
btfan_speed = interface.hvcontroller.set_fan_speed(fan_id=0, fan_speed=20)
print(f"Bottom Fan Speed: {btfan_speed}")

print("Set Top Fan Speed to 40%")
tpfan_speed = interface.hvcontroller.set_fan_speed(fan_id=1, fan_speed=40)
print(f"Bottom Fan Speed: {tpfan_speed}")

print("Get Bottom Fan Speed")
btfan_speed = interface.hvcontroller.get_fan_speed(fan_id=0)
print(f"Bottom Fan Speed: {btfan_speed}")

print("Get Top Fan Speed")
tpfan_speed = interface.hvcontroller.get_fan_speed(fan_id=1)
print(f"Bottom Fan Speed: {tpfan_speed}")

print("Set RGB LED")
rgb_led = interface.hvcontroller.set_rgb_led(rgb_state=2)
print(f"RGB STATE: {rgb_led}")

print("Get RGB LED")
rgb_led_state = interface.hvcontroller.get_rgb_led()
print(f"RGB STATE: {rgb_led_state}")

print("Test 12V...")
if interface.hvcontroller.turn_12v_on():
    print("12V ON Press enter to TURN OFF:")
    input()  # Wait for the user to press Enter
    if interface.hvcontroller.turn_12v_off():
        print("12V OFF.")
    else:
        print("Failed to turn off 12V")
else:
    print("Failed to turn on 12V.")

# Set High Voltage Level
print("Set HV Power to +/- 85V")
if interface.hvcontroller.set_voltage(voltage=75.0):
    print("Voltage set to 85.0 V.")
else:
    print("Failed to set voltage.")

# Get Set High Voltage Setting
print("Get Current HV Voltage")
read_voltage = interface.hvcontroller.get_voltage()
print(f"HV Voltage {read_voltage} V.")


print("Test HV Supply...")
if interface.hvcontroller.turn_hv_on():
# Get Set High Voltage Setting
    read_voltage = interface.hvcontroller.get_voltage()
    print(f"HV Voltage {read_voltage} V.")
    print("HV ON Press enter to TURN OFF:")
    input()  # Wait for the user to press Enter
    if interface.hvcontroller.turn_hv_off():
        print("HV OFF.")
    else:
        print("Failed to turn off HV")
else:
    print("Failed to turn on HV.")

print("Reset DevConsoleice:")
# Ask the user for confirmation
user_input = input("Do you want to reset the Console? (y/n): ").strip().lower()

if user_input == 'y':
    if interface.hvcontroller.soft_reset():
        print("Reset Successful.")
elif user_input == 'n':
    print("Reset canceled.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")
