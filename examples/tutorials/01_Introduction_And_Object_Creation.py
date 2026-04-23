# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: env (3.11.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01: Introduction and Core Object Creation
#
# Welcome to OpenLIFU! This series of notebooks will guide you through the functionalities of the OpenLIFU Python library, from basic object creation to controlling hardware and running simulations.
#
# This first notebook focuses on the fundamental data objects used in OpenLIFU to define ultrasound parameters, geometric entities, and treatment plans.

# %% [markdown]
# ## Core OpenLIFU Objects
#
# Most workflows in OpenLIFU will involve some common objects:
#
# *   **`Point`**: Represents a geometric point in 3D space, typically used for defining targets or focal points.
# *   **`Pulse`**: Defines the characteristics of an ultrasound pulse, such as frequency and duration.
# *   **`Sequence`**: Describes how pulses are repeated or patterned over time.
# *   **`Transducer`**: Represents the physical ultrasound transducer array, including its geometry and element properties. (More detail in Notebook 02).
# *   **`Protocol`**: A container object that groups together a `Pulse`, `Sequence`, `FocalPattern`, simulation settings, and other parameters to define a complete sonication plan.
# *   **`Solution`**: The calculated set of parameters (delays, apodizations, voltage) to be sent to the hardware to achieve a desired sonication, typically derived from a `Protocol` and `Target`.

# %%
# Let's start by importing the necessary classes.
from __future__ import annotations
import numpy as np
from openlifu.bf import Pulse, Sequence, focal_patterns
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup
from openlifu.xdc import Transducer  # We'll briefly touch on this

# %% [markdown]
# ## 1. `Point`
#
# A `Point` object defines a location in 3D space. It can also have a radius, making it a sphere.
# Positions are typically specified in millimeters.

# %%
# Create a point at (x=0, y=0, z=50 mm)
target_point = Point(position=np.array([0, 0, 50]), units="mm")
print(f"Target Point: {target_point}")

# %% [markdown]
# ## 2. `Pulse`
#
# The `Pulse` object defines the acoustic properties of a single ultrasound pulse.
# Key parameters include `frequency` (in Hz) and `duration` (in seconds).
# Nominal amplitude can also be set with `amplitude` (arbitrary units)

# %%
pulse1 = Pulse(frequency=500e3, duration=20e-6)
print(f"Pulse 1:\n{pulse1.to_table()}")

pulse2 = Pulse(frequency=1e6, duration=10e-6, amplitude=0.5)
print(f"Pulse 2:\n{pulse2.to_table()}")

# %% [markdown]
# ## 3. `Sequence`
#
# The `Sequence` object defines how pulses are delivered over time. This includes:
# *   `pulse_interval`: Time between the start of consecutive pulses in a train (in seconds). Also known as Pulse Repetition Interval (PRI).
# *   `pulse_count`: Number of pulses in a single train.
# *   `pulse_train_interval`: Time between the start of consecutive pulse trains (in seconds).
# *   `pulse_train_count`: Number of pulse trains. If 0, it implies continuous pulsing (hardware dependent).

# %%
# Example: 10 pulses, with 100 ms interval between pulses, repeated once.
sequence1 = Sequence(
    pulse_interval=0.1,  # 100 ms
    pulse_count=10,
    pulse_train_interval=0, # Only one train, so interval is 0 or irrelevant
    pulse_train_count=1
)
print(f"Sequence 1: {sequence1}")
print(f"Sequence 1 Pulse Train Duration (approx): {sequence1.get_pulse_train_duration():.2f} s")
print(f"Sequence 1 Total Duration (approx): {sequence1.get_sequence_duration():.2f} s")


# Example: 5 pulse trains, each train has 20 pulses at 50ms PRI.
# Trains are repeated every 2 seconds.
sequence2 = Sequence(
    pulse_interval=0.05, # 50 ms
    pulse_count=20,
    pulse_train_interval=2.0, # 2 seconds
    pulse_train_count=5
)
print(f"Sequence 2: {sequence2}")
print(f"Sequence 2 Pulse Train Duration (approx): {sequence2.get_pulse_train_duration():.2f} s")
print(f"Sequence 2 Total Duration (approx): {sequence2.get_sequence_duration():.2f} s")

# %% [markdown]
# ## 4. `Transducer` (Brief Introduction)
#
# A `Transducer` object holds all information about the ultrasound array, including element positions, dimensions, and potentially electromechanical properties.
# These are often loaded from a database (see Notebook 02). For now, we can see how one might be generated programmatically (though this is less common for standard OpenLIFU hardware).

# %%
# This is an example of generating a generic transducer.
# In practice, you'll usually load a predefined one from the database.
try:
    # Define a simple 2x2 planar array for illustrative purposes
    trans_example = Transducer.gen_matrix_array(
        nx=2, ny=2, # 2x2 elements
        pitch=10,   # 10 mm pitch
        kerf=1,     # 1 mm kerf
        id="example_2x2",
        name="Example 2x2 Array"
    )
    print(f"Example Transducer: {trans_example}")
    print(f"Number of elements: {trans_example.numelements()}")
    print(f"Element positions (first few):\n{trans_example.get_positions(units='mm')[:2]}")
except Exception as e:
    print(f"Could not create generic transducer, possibly due to missing default parameters: {e}")
    print("Transducer definition will be covered in detail in Notebook 02 using database loading.")

# %% [markdown]
# ## 5. `FocalPattern`
#
# A `FocalPattern` defines where the ultrasound energy should be focused. It can be a single point, multiple points, or more complex shapes.
# This is used by a `Protocol` to determine how to calculate a `Solution`.

# %%
# Single point focus at the 'target_point' defined earlier
focal_pattern_single = focal_patterns.SinglePoint()
# When used in a protocol, a specific 'target' Point object will be associated.
print(f"Single Point Focal Pattern: {focal_pattern_single}")

# A wheel pattern - useful for creating annular lesions or patterns.
# (center=False means the spokes don't meet at the very center, creating a ring)
focal_pattern_wheel = focal_patterns.Wheel(
    center=False,
    spoke_radius=5, # 5 mm radius for the spokes/ring
    num_spokes=4,   # 4 points on the wheel
    target_pressure=1.0e6 # Example target acoustic pressure in Pa
)
print(f"Wheel Focal Pattern: {focal_pattern_wheel}")

# %% [markdown]
# ## 6. `SimSetup`
#
# `SimSetup` defines the parameters for running an acoustic simulation, such as the dimensions of the simulation grid.

# %%
sim_setup_example = SimSetup(
    x_extent=(-20, 20), # mm
    y_extent=(-20, 20), # mm
    z_extent=(0, 70),   # mm
    spacing=1        # 1 mm grid resolution
)
print(f"Simulation Setup: {sim_setup_example}")

# %% [markdown]
# ## 7. `Protocol`
#
# A `Protocol` brings many of these elements together. It's a recipe for a specific sonication.
# It typically includes a `Pulse`, `Sequence`, `FocalPattern`, `ApodMethod` (beamforming aperture control), and `SimSetup`.

# %%
from openlifu.bf import apod_methods

# Define an apodization method (e.g., limit active elements by angle from focus)
apod_method_example = apod_methods.MaxAngle(max_angle=30)
print(f"Apodization Method: {apod_method_example}")

# Create a protocol
# For the focal_pattern, we pass the class. The actual target Point is given when calculating a solution.
protocol1 = Protocol(
    id='my_first_protocol',
    name='My First Test Protocol',
    pulse=pulse1,
    sequence=sequence1,
    focal_pattern=focal_pattern_single, # Type of pattern
    apod_method=apod_method_example,
    sim_setup=sim_setup_example
)
print(f"\nProtocol 1: {protocol1}")
print(f"Protocol Pulse: {protocol1.pulse}")
print(f"Protocol Sequence: {protocol1.sequence}")
print(f"Protocol Focal Pattern: {protocol1.focal_pattern}")

# You can also create a protocol with a specific focal pattern instance
protocol2 = Protocol(
    id='my_wheel_protocol',
    name='My Wheel Test Protocol',
    pulse=pulse2,
    sequence=sequence2,
    focal_pattern=focal_pattern_wheel, # Instance of a pattern
    apod_method=apod_methods.Uniform(), # No apodization
    sim_setup=sim_setup_example
)
print(f"\nProtocol 2: {protocol2}")
print(f"Protocol Focal Pattern: {protocol2.focal_pattern}")

# %% [markdown]
# ## Next Steps
#
# This notebook introduced the basic building blocks.
#
# In the next notebook (`02_Database_Interaction.py`), we will explore how to load `Transducer` information from the OpenLIFU database, which is crucial for working with specific hardware.
# Following that, `03_Solution_Generation_and_Analysis.py` will show how to use these objects, particularly a `Protocol` and `Transducer`, to calculate a `Solution` and simulate its acoustic field.

# %% [markdown]
# End of Notebook 01
