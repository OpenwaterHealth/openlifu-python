# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

`openlifu` is the core Python library for Openwater's Low Intensity Focused
Ultrasound (LIFU) platform. It provides beamforming, k-Wave acoustic simulation,
treatment planning, a file-based database, and hardware I/O. Licensed under MIT.

The primary downstream consumer is **SlicerOpenLIFU** (`../SlicerOpenLIFU`), a
3D Slicer extension that wraps `openlifu` objects with a GUI, VTK visualization,
and MRML persistence.

Python 3.10-3.12 on Windows or Linux.

## Build and Development Commands

```bash
# Install (editable, with test deps)
pip install -e '.[test]'

# Install (editable, with dev deps)
pip install -e '.[dev]'

# Run all tests
pytest

# Run a single test file
pytest tests/test_database.py

# Run a single test function
pytest tests/test_database.py::test_function_name -v

# Run tests with coverage
pytest -ra --cov --cov-report=term --durations=20

# Lint (runs pre-commit hooks)
pipx run nox -s lint
# or if nox is installed:
nox -s lint

# Run ruff directly
ruff check src/

# Run pylint
nox -s pylint

# Build docs
nox -s docs
```

## Linting and Code Style

- **Ruff** is the primary linter, configured in `pyproject.toml` with many rule
  sets enabled (bugbear, isort, pylint, etc.).
- **Every file must start with `from __future__ import annotations`** — enforced
  by isort via `isort.required-imports` in `pyproject.toml`.
- **No auto-formatter** is currently enabled (ruff-format is commented out in
  `.pre-commit-config.yaml`).
- **Pre-commit hooks** include: ruff (with `--fix`), codespell, shellcheck,
  prettier (YAML/MD/JSON), blacken-docs, and a custom check disallowing
  `PyBind|Numpy|Cmake|CCache|Github|PyTest` capitalization.
- **PyLint** runs in CI via nox. MyPy is disabled.
- Warnings are treated as errors in pytest, with specific exceptions listed in
  `pyproject.toml`.

## CI

CI runs on push to main and on PRs (`.github/workflows/ci.yml`):

1. **Commit message check** (PRs only): every commit must reference a GitHub
   issue (`#123` or full URL). Exceptions: Bump, Merge, Revert, fixup!, squash!,
   amend! commits.
2. **Pre-commit + PyLint**: linting on ubuntu.
3. **Tests**: matrix of Python {3.10, 3.12} x {ubuntu, windows, macos} with
   coverage uploaded to Codecov.

## Commit Guidelines

Every commit must reference a relevant GitHub issue number in the title or body
(e.g. `Fix target placement crash (#42)` or `Fixes #42` in the body). CI
enforces this on PRs.

## Architecture

### Subpackages

| Package  | Purpose                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------- |
| `plan/`  | `Protocol` and `Solution` — treatment planning, beamforming orchestration, solution analysis    |
| `bf/`    | Beamforming primitives: `Pulse`, `Sequence`, `FocalPattern`, `DelayMethod`, `ApodizationMethod` |
| `sim/`   | k-Wave acoustic simulation integration (`SimSetup`, `run_simulation`)                           |
| `db/`    | File-based JSON database (`Database`, `User`, `Subject`, `Session`, `Run`)                      |
| `xdc/`   | Transducer hardware definitions (`Transducer`, `Element`, `TransducerArray`)                    |
| `seg/`   | Tissue segmentation and material properties (`Material`, `SegmentationMethod`)                  |
| `geo.py` | `Point` class — 3D coordinates with metadata and VTK visualization                              |
| `io/`    | Hardware communication (`LIFUInterface`) — USB, serial, voltage safety                          |
| `nav/`   | Photogrammetry (`Photoscan`), Meshroom pipeline integration, MODNet portrait matting            |
| `virtual_fit.py` | Virtual fit algorithm — transducer positioning optimization (top-level module, not a subpackage) |
| `cloud/` | REST API client for cloud-based session sync, WebSocket support                                 |
| `util/`  | Shared infrastructure: `DictMixin`, `PYFUSEncoder`, unit conversion, volume conversion          |

### Core Domain Model

```
Protocol (treatment plan template)
  ├── Pulse, Sequence, FocalPattern
  ├── DelayMethod, ApodizationMethod
  ├── SegmentationMethod
  └── SimSetup
        │
        ▼  Protocol.calc_solution(target, transducer, volume)
Solution (computed beam parameters: delays[], apodizations[], foci[])
  ├── simulate() → xarray.Dataset (pressure fields)
  ├── analyze() → SolutionAnalysis (pressure/safety metrics)
  └── scale() → adjust voltage to target pressure

Session
  ├── Subject, Protocol, Transducer, Volume
  ├── targets[] (Point), markers[]
  └── virtual_fit_results, transducer_tracking_results
```

### Serialization Patterns

Three patterns are used consistently:

1. **`DictMixin`** (`util/dict_conversion.py`): automatic
   `to_dict()`/`from_dict()` from dataclass fields. Used by simpler classes
   (Sequence, Pulse, SimSetup, Material, VirtualFitOptions, Subject).

2. **Manual `to_dict`/`from_dict`/`to_json`/`from_json`/`to_file`/`from_file`**:
   used by Protocol, Solution, Point, User, Run.

3. **Polymorphic factory with class name lookup**: abstract base classes store
   `{'class': 'ClassName', ...}` in dicts and reconstruct via module
   introspection. Used by DelayMethod, ApodizationMethod, FocalPattern,
   SegmentationMethod.

All JSON encoding uses `PYFUSEncoder` (`util/json.py`) which handles numpy
types, datetime, and domain objects.

### Database File Layout

Each top-level resource has an index JSON listing its members (`users.json`,
`protocols.json`, `subjects.json`, `transducers.json`); nested collections have
their own index files alongside.

```
{db_root}/
  users/users.json
  users/{user_id}/...                                        → User record + password hash
  protocols/protocols.json
  protocols/{protocol_id}/...                                → Protocol JSON
  subjects/subjects.json
  subjects/{subject_id}/{subject_id}.json                    → Subject record
  subjects/{subject_id}/volumes/volumes.json
  subjects/{subject_id}/volumes/{volume_id}/...
  subjects/{subject_id}/sessions/sessions.json
  subjects/{subject_id}/sessions/{session_id}/{session_id}.json
  subjects/{subject_id}/sessions/{session_id}/runs/runs.json
  subjects/{subject_id}/sessions/{session_id}/solutions/solutions.json
  subjects/{subject_id}/sessions/{session_id}/solutions/{solution_id}/...   → Solution JSON + .nc
  subjects/{subject_id}/sessions/{session_id}/photoscans/photoscans.json
  subjects/{subject_id}/sessions/{session_id}/photocollections/photocollections.json
  transducers/transducers.json
  transducers/{transducer_id}/{transducer_id}.json           → Transducer JSON
  transducers/{transducer_id}/{transducer_id}_gridweights_{hash}.h5   → Precomputed grid weights
```

### Parameter Constraints

`ParameterConstraint` provides validation with operators (`<`, `<=`, `>`, `>=`,
`within`, `inside`, `outside`) and warning/error thresholds. `TargetConstraints`
validates spatial bounds per dimension. Both used during
`Protocol.calc_solution()` and `Solution.analyze()`.

### Simulation Pipeline

`Protocol.calc_solution()` flow:

1. Validate target against `TargetConstraints`
2. Create simulation params from volume + `SegmentationMethod`
3. For each focal point: compute delays (`DelayMethod`) and apodizations
   (`ApodizationMethod`)
4. Create `Solution` with stacked delay/apodization arrays
5. `Solution.simulate()` → k-Wave via `run_simulation()` → xarray Dataset
6. `Solution.scale()` → adjust voltage for target pressure
7. `Solution.analyze()` → `SolutionAnalysis` (PNP, ISPPA, ISPTA, MI, TIC,
   beamwidths, temperature)

## Testing

- Tests use **pytest** (not Slicer's test framework — that's SlicerOpenLIFU).
- No shared `conftest.py` — fixtures are local to each test file.
- `tests/helpers.py` provides `dataclasses_are_equal()` for deep equality of
  nested dataclass/numpy structures.
- Test data lives in `tests/resources/` (includes `example_db/` and a DICOM
  file).
- The full sample database is published as its own repo (see "Sample Data"
  below); it is not pulled in by the test suite.

## Sample Data

The official sample database lives in
[`openlifu-sample-database`](https://github.com/OpenwaterHealth/openlifu-sample-database)
and is tracked with Git LFS. Tags on that repo (e.g. `openlifu-v0.20.0`) pin
sample-data versions to specific `openlifu` releases, and downstream releases
(SlicerOpenLIFU, openlifu-app) link to the README section below for the
compatible version.

```bash
git clone --depth 1 --branch openlifu-v0.20.0 https://github.com/OpenwaterHealth/openlifu-sample-database.git
cd openlifu-sample-database
git lfs pull
```

The legacy DVC flow (`db_dvc/`, `db_dvc.dvc`, `dvc[gdrive]` in the `dev` extra,
Google Drive remote with `gdrive_client_secret`) is still present in the repo
but is not the path users or downstream consumers should be pointed to. Direct
customers and downstream projects to the README section "Getting Sample Data"
instead.
