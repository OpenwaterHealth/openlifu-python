# openlifu

[![Actions Status](https://github.com/OpenwaterHealth/OpenLIFU-python/workflows/CI/badge.svg)](https://github.com/OpenwaterHealth/OpenLIFU-python/actions)
[![Documentation Status](https://readthedocs.org/projects/openlifu/badge/?version=latest)](https://openlifu.readthedocs.io/en/latest/?badge=latest)

[![PyPI version](https://img.shields.io/pypi/v/openlifu)](https://pypi.org/project/openlifu/)
[![PyPI platforms](https://img.shields.io/pypi/pyversions/openlifu)](https://pypi.org/project/openlifu/)

[![GitHub Discussion](https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github)](https://github.com/OpenwaterHealth/OpenLIFU-python/discussions)

<!-- SPHINX-START -->

`openlifu` is a toolbox for planning and controlling focused ultrasound
treatments. It generally replicates the functionality developed in the MATLAB
[open-TFUS toolbox](https://github.com/OpenwaterHealth/opw_neuromod_sw).

## Installation

### Requirements

Python 3.10-3.12.

We recommend installing `openlifu` in a virtual environment.

### Recommended install

For most users, install the application dependency set:

```sh
python -m pip install "openlifu[app]"
```

This installs the core package plus dependencies for database and volume I/O,
simulation, mesh and photogrammetry workflows, cloud sync, and hardware
communication support via `openlifu-sdk`.

### Minimal install

If you only need the lightweight core package:

```sh
python -m pip install openlifu
```

The minimal install does not include optional dependencies for simulation,
DICOM/NIfTI volume loading, mesh/VTK workflows, photogrammetry, cloud sync, or
hardware communication.

### Install selected features

You can also install only the optional dependency groups you need:

```sh
python -m pip install "openlifu[sim,db]"
python -m pip install "openlifu[io]"
python -m pip install "openlifu[jupyter]"
```

Useful extras include:

- `app`: recommended application set
- `sim`: k-Wave simulation support
- `db`: DICOM/NIfTI and database volume support
- `mesh`: mesh and VTK support
- `photogrammetry`: photo-based mesh reconstruction dependencies
- `io`: hardware communication dependencies, including `openlifu-sdk`
- `cloud`: cloud sync dependencies
- `jupyter`: notebook support

## Installing Meshroom

If you are using `openlifu.nav.photoscan` to reconstruct meshes from photo
collections, then you will need to set up **Meshroom 2025.1.0**.

### Ubuntu

#### Download and Extract

1.  Download Meshroom 2025.1.0 for Linux from
    <https://github.com/alicevision/Meshroom/releases/tag/v2025.1.0>.

2.  Extract the downloaded archive:

    ```bash
    tar -xvf Meshroom-2025.1.0-Linux.tar.gz
    ```

#### Add Meshroom to PATH

**Temporary (Current Session)** Run:

```bash
export PATH="<path-to-meshroom>/Meshroom-2025.1.0-Linux:$PATH"
```

Replace `<path-to-meshroom>` with the actual path where Meshroom was extracted.

**Permanent (Persistent Across Sessions)** For Bash users:

```bash
echo 'export PATH="<path-to-meshroom>/Meshroom-2025.1.0-Linux:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Windows

#### Download and Extract

1.  Download Meshroom 2025.1.0 for Windows from
    <https://github.com/alicevision/Meshroom/releases/tag/v2025.1.0>.
2.  Extract the downloaded archive to a directory of your choice.

#### Add Meshroom to PATH

1.  Open **Edit environment variables for your account** from the Start menu.
2.  In the **Environment Variables** window, under **User variables**, select
    **Path** and click **Edit**.
3.  Click **New**, and add the path to the folder containing `Meshroom.exe`.
4.  Click **OK** on every tab to save the changes.

#### Enable GPU Acceleration

To ensure Meshroom uses your NVIDIA GPU:

1.  Open **NVIDIA Control Panel**.
2.  In the left sidebar under **3D Settings**, click **Manage 3D settings**.
3.  Go to the **Program Settings** tab.
4.  Click **Add**, then browse to and select `Meshroom.exe` from the folder
    where you extracted Meshroom.
5.  Under **Select the preferred graphics processor for this program**, choose
    **High-performance NVIDIA processor**.
6.  Click **Apply**.

## Getting Sample Data

A sample database for testing and examples is maintained in the
[openlifu-sample-database](https://github.com/OpenwaterHealth/openlifu-sample-database)
repository. Its files are tracked with Git LFS, so first
[install Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

Then check out the tagged version of the sample database that is compatible with
this version of `openlifu`:

```bash
git clone --depth 1 --branch openlifu-v0.20.0 https://github.com/OpenwaterHealth/openlifu-sample-database.git
cd openlifu-sample-database
git lfs pull
```

## Disclaimer

CAUTION - Investigational device. Limited by Federal (or United States) law to
investigational use. The system described here has _not_ been evaluated by the
FDA and is not designed for the treatment or diagnosis of any disease. It is
provided AS-IS, with no warranties. User assumes all liability and
responsibility for identifying and mitigating risks associated with using this
software.
