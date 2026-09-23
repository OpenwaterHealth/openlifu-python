# Day 1 with OpenLIFU: Getting Started Without Hardware

This guide documents a first-time setup experience for OpenLIFU on Windows. The goal is to clone the repository, create a local development environment, install OpenLIFU, and explore the project without requiring physical hardware.

> **Platform tested:** Windows PowerShell  
> **Python used:** Python 3.12  
> **Hardware required:** None

## 1. Clone the Repository

Clone your fork of the repository:

```powershell
git clone https://github.com/YOUR-USERNAME/openlifu-python.git
cd openlifu-python
```

Replace `YOUR-USERNAME` with your GitHub username.

Verify the repository remote:

```powershell
git remote -v
```

## 2. Check Your Python Version

OpenLIFU supports Python 3.10 through Python 3.12.

Check your default Python version:

```powershell
python --version
```

During this setup, Python 3.13 was initially installed. Because the project requires a supported version, Python 3.12 was installed separately.

Verify Python 3.12:

```powershell
py -3.12 --version
```

If successful, you should see a Python 3.12 version.

## 3. Create a Virtual Environment

Create a virtual environment using Python 3.12:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your PowerShell prompt should now begin with:

```text
(.venv)
```

## 4. Upgrade pip

Upgrade pip inside the virtual environment:

```powershell
python -m pip install --upgrade pip
```

## 5. Install OpenLIFU

Install the local repository in editable mode:

```powershell
python -m pip install -e .
```

Verify that OpenLIFU imports successfully:

```powershell
python -c "import openlifu; print('OpenLIFU imported successfully')"
```

Expected output:

```text
OpenLIFU imported successfully
```

You can also verify that Python is using the local repository:

```powershell
python -c "import openlifu; print(openlifu.__file__)"
```

The displayed path should point to the local repository's `src\openlifu` directory.

## 6. Explore the Included Examples

List the example files included with the repository:

```powershell
Get-ChildItem examples -Recurse -File | Select-Object FullName
```

The repository includes introductory tutorial files such as:

- `01_Introduction_And_Object_Creation.py`
- `02_Database_Interaction.py`
- `03_Solution_Generation_and_Analysis.py`

You can begin exploring these examples without connecting physical hardware.

## 7. Optional Dependencies

Some OpenLIFU functionality uses optional dependencies.

During this setup, missing dependencies produced errors such as:

```text
ModuleNotFoundError: No module named 'vtk'
```

```text
ModuleNotFoundError: No module named 'skimage'
```

```text
ModuleNotFoundError: No module named 'trimesh'
```

The project's `pyproject.toml` defines optional dependency groups for different features, including:

- `mesh`
- `db`
- `io`
- `sim`
- `cloud`
- `jupyter`

Before installing individual packages, check which dependency group is appropriate for the functionality you want to use.

## 8. Troubleshooting Dependency Conflicts

While installing additional dependencies, NumPy was upgraded to version 2.x.

However, OpenLIFU required:

```text
numpy<2
```

To restore a compatible version:

```powershell
python -m pip install "numpy<2"
```

After changing dependencies, verify the installation again:

```powershell
python -c "import openlifu; print('OpenLIFU imported successfully')"
```

## 9. Running Without Hardware

OpenLIFU can be explored without connecting physical hardware.

A no-hardware first-day workflow can include:

1. Cloning the repository.
2. Creating a supported Python environment.
3. Installing OpenLIFU locally.
4. Importing OpenLIFU successfully.
5. Exploring the included tutorial examples.
6. Working with Python objects and supported simulation-related functionality.

This allows a new contributor to become familiar with the repository before working with physical devices.

## Suggested Screenshots

Before submitting this tutorial, capture screenshots showing:

1. The repository successfully cloned in PowerShell.
2. The Python 3.12 virtual environment activated.
3. The successful OpenLIFU import.

Add these screenshots to the documentation in the appropriate location.

## Day 1 Checklist

- [x] Clone the repository
- [x] Install a supported Python version
- [x] Create a virtual environment
- [x] Activate the virtual environment
- [x] Install OpenLIFU locally
- [x] Verify that OpenLIFU imports successfully
- [x] Explore the included examples
- [x] Document setup friction points
- [ ] Add screenshots
- [ ] Test and document a complete simulator/mock workflow
- [ ] Open a draft pull request

## Next Steps

After completing the initial setup:

1. Read the repository's `README.md`.
2. Review `CONTRIBUTING.md`.
3. Explore the introductory examples.
4. Test additional no-hardware functionality.
5. Document any friction points encountered during setup.
6. Open separate issues for significant documentation or setup problems.
7. Add screenshots and open a draft pull request.

## Conclusion

After completing these steps, you should have a local OpenLIFU development environment running on Windows without requiring physical hardware. You can now explore the repository's examples and continue toward the simulator or mock workflow.