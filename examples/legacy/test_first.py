# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Once the environment has been set up, this notebook should be able to be run, cell-by-cell. One issue is that I've made modifications to k-wave-python (to cache and reuse gridweights, which saves a significant amount of time if you are running different simulations using the same array in the same position - it's being discussed in https://github.com/waltsims/k-wave-python/issues/342, but hasn't been merged yet, afaik).
#
# If you have my modified version of k-wave-python, install it with `pip install -e .` from the `k-wave-python` directory. If you don't have my modified version, you can install the original version with `pip install k-wave-python`. If you are using the original version, be sure to set `USE_GRIDWEIGHTS` to `False` in order to prevent `open_pyfus` from trying to use a nonexistent interface for loading the gridweights.
#
# Also, if you are using the original version, import `openlifu` takes _way_ longer (45s on my PC), presumably hanging on `import kwave`. For some reason, it wants to re-download the binaries every time, even though they are already present in the the installation directory. I've opened an issue on this: https://github.com/waltsims/k-wave-python/issues/366.

# %%
from __future__ import annotations

import logging
import sys

root = logging.getLogger()
loglevel = logging.DEBUG
root.setLevel(loglevel)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(loglevel)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
import numpy as np

import openlifu

# %% [markdown]
# We'll start by generating a transducer and drawing it using some vtk-based methods

# %%
arr = openlifu.Transducer.gen_matrix_array(nx=8, ny=8, pitch=4, kerf=.5, units="mm", sensitivity=1e5)
arr.draw()

# %% [markdown]
# Now we'll define the pulse and sequence parameters, the simulation setup, and generate a Protocol.

# %%
pulse = openlifu.Pulse(frequency=400e3, duration=3/400e3)
sequence = openlifu.Sequence()
focal_pattern = openlifu.focal_patterns.Wheel(center=True, spoke_radius=5, num_spokes=5)
sim_setup = openlifu.SimSetup(dt=2e-7, t_end=100e-6)
protocol = openlifu.Protocol(
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    sim_setup=sim_setup)

# %% [markdown]
# Now we can define a sonication target(s), set up the simulation, and compute the delays and apodizations needed to steer the sound to it.

# %%
pt = openlifu.Point(position=(0,0,30), units="mm", radius=2)
pts = protocol.focal_pattern.get_targets(pt)
coords = protocol.sim_setup.get_coords()
params = protocol.seg_method.ref_params(coords)
delays, apod = protocol.beamform(arr=arr, target=pts[0], params=params)

# %% [markdown]
# Now we are ready to run the simulation.  Some custom edits to `k-wave-python` allow for caching of the gridweights, which only need to be computed once for a given grid size and source location.  This can speed up the simulation significantly, especially if a coarse grid that won't take the GPU too long to run is used.

# %%
ds = openlifu.sim.run_simulation(arr=arr,
        params=params,
        delays=delays,
        apod= apod,
        freq = pulse.frequency,
        cycles = np.max([np.round(pulse.duration * pulse.frequency), 20]),
        dt=protocol.sim_setup.dt,
        t_end=protocol.sim_setup.t_end,
        amplitude = 1,
        gpu = False)

# %% [markdown]
# We can use all of `xarray`s built-in plotting capabilities to plot the data.

# %%
ds['p_min'].sel(ele=0).plot.imshow()

# %%
ds['p_min'].sel(ele=0).plot.imshow()

# %% [markdown]
# We can examine the output object, which is an `xarray.DataSet` object with 3 data variables: `p_max` (Peak Positive Pressure), `p_min` (Peak Negative Pressure), and `intensity` (Time Averaged Intensity). It's attributes also contain the `source` pulse (an `xarray.DataArray`), and `output`, the raw K-Wave output structure.

# %%
ds

# %% [markdown]
# Using `nibabel`, we can export the DataArray to a NIftI file. This requires a little bit of manipulation of the coordinates to extract the origin and affine matrix as NIftI needs them. This should get wrapped into a function in the future.

# %%
import nibabel as nb

p_min = ds['p_min'].data
coords = ds['p_min'].coords
affine = np.eye(3) * np.array([float(np.diff(coords[x][:2])) for x in coords])
origin = np.array([float(coords[x][0]) for x in coords]).reshape(3,1)
affine = np.concatenate([np.concatenate([affine, origin], axis=1),np.array([0,0,0,1]).reshape(1,4)], axis=0)
nb.Nifti1Image(p_min, affine).to_filename("p_min.nii.gz")

# %% [markdown]
# Finally, we can use some of the intermediate vtk methods to extract Actors from both the array and points objects, and pipe them to a since render:

# %%
import vtk

arr_actor = arr.get_actor(units="mm")
renderWindow = vtk.vtkRenderWindow()
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderer.AddActor(arr_actor)
for pti in pts:
    pt_actor = pti.get_actor()
    renderer.AddActor(pt_actor)
renderWindow.Render()
renderWindowInteractor.Start()
