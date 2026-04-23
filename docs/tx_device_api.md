**API Documentation** for the Transmitter Device Controller

---

# TxDevice API Documentation

The `TxDevice` class provides a high-level interface for communicating with and
controlling ultrasound transmitter modules using the `LIFUUart` communication
backend.

---

## Initialization

```python
from openlifu_sdk.io.LIFUTXDevice import TxDevice
from openlifu_sdk.io.LIFUInterface import LIFUInterface

interface = LIFUInterface(TX_test_mode=False)
tx_connected, hv_connected = interface.is_device_connected()

if not tx_connected:
    print("TX Device not connected.")
    sys.exit()
```

---

## Core Methods

### Device Communication

| Method              | Description                                        |
| ------------------- | -------------------------------------------------- |
| `is_connected()`    | Check if the TX device is connected                |
| `ping()`            | Send a ping command to check connectivity          |
| `get_version()`     | Get the firmware version (e.g., `v0.1.1`)          |
| `get_hardware_id()` | Return the 16-byte hardware ID in hex format       |
| `echo(data: bytes)` | Send and receive echo data to verify communication |
| `toggle_led()`      | Toggle the onboard status LED                      |
| `soft_reset()`      | Perform a software reset on the TX device          |
| `enter_dfu()`       | Put the device into DFU mode                       |
| `close()`           | Close the UART connection                          |

---

### Trigger Configuration

| Method                               | Description                                |
| ------------------------------------ | ------------------------------------------ |
| `set_trigger(...)`                   | Configure trigger with parameters directly |
| `set_trigger_json(data: dict)`       | Set trigger using JSON config              |
| `get_trigger()`                      | Return current trigger config (parsed)     |
| `get_trigger_json()`                 | Return raw JSON config from device         |
| `start_trigger()` / `stop_trigger()` | Begin or halt the trigger sequence         |

---

### Register Operations

| Method                                        | Description                                 |
| --------------------------------------------- | ------------------------------------------- |
| `read_register(addr)`                         | Read a 16-bit register value                |
| `write_register(identifier, addr, value)`     | Write a value to a specific register        |
| `write_register_verify(addr, value)`          | Write and verify value to a register        |
| `write_block(identifier, start_addr, values)` | Write a block of register values to a chip  |
| `write_block_verify(start_addr, values)`      | Write and verify a block of register values |

---

### Device Setup & Control

| Method                                   | Description                                    |
| ---------------------------------------- | ---------------------------------------------- |
| `enum_tx7332_devices(num)`               | Enumerate and initialize TX7332 devices        |
| `demo_tx7332(identifier)`                | Set demo waveform to a TX7332 chip             |
| `apply_all_registers()`                  | Push all defined register sets to hardware     |
| `write_ti_config_to_tx_device(path, id)` | Load TI register config from `.txt` and apply  |
| `print()`                                | Print internal TX state (overridden `__str__`) |

---

### Pulse/Delay Profiles

| Method                                         | Description                                |
| ---------------------------------------------- | ------------------------------------------ |
| `set_solution(pulse, delays, apods, seq, ...)` | Apply full beamforming config and sequence |
| `add_pulse_profile(profile)`                   | Add or replace pulse waveform config       |
| `add_delay_profile(profile)`                   | Add or replace delay + apodization profile |

---

## Data Classes

- `Tx7332PulseProfile`: Defines TX waveform: frequency, duration, duty cycle,
  inversion, etc.
- `Tx7332DelayProfile`: Defines delay + apodization per channel (32 channels
  supported)
- `TxDeviceRegisters`: Holds per-chip register maps and manages write blocks

---

## Notes

- Each register block is managed per TX chip and may need verification.
- Profiles assume 32-channel TX chip configurations.
- Delay units are in seconds, apodization values from 0–1.
- Trigger configuration must be set before `start_trigger()`.

---

## Example

```python
pulse = {"frequency": 3e6, "duration": 2e-6}
delays = [0.0] * 32
apods = [1.0] * 32
sequence = {
    "pulse_interval": 0.01,
    "pulse_count": 1,
    "pulse_train_interval": 0.0,
    "pulse_train_count": 1,
}

if tx.is_connected():
    tx.set_solution(pulse, delays, apods, sequence)
    tx.start_trigger()
```
