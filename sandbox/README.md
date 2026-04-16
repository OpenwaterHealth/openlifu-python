# Sandbox — Pre-Release & Experimental Code

This directory is the home for experimental features, prototypes, algorithm
explorations, and hardware integration tests that are not yet ready for
production. If you're trying something new, this is where it lives until it's
validated and promoted.

## Why Sandbox Exists

Instead of creating a new repo for every experiment — which leads to orphaned
repos, inconsistent naming, and invisible work — we keep pre-release code in the
platform repos, where it belongs. This makes experiments discoverable,
reviewable, and easy to promote when they're ready.

## Rules

### 1. Isolation

Sandbox code is **never imported by production code**. The main build system
explicitly excludes this directory. No production module may reference anything
in `sandbox/`.

If your experiment depends on production code, import the installed package
(e.g., `import openlifu`, `#include` from installed headers). Never use relative
paths back into `src/` or `Core/`.

### 2. Self-Contained

Each experiment is a standalone folder. It must include everything someone needs
to understand and run it: its own README, dependencies, sample data (or
instructions to get it), and run instructions. Assume the next person has never
seen your code.

### 3. Ownership

Every experiment folder **must** have a `README.md` with the following header:

```markdown
# [Experiment Name]

| Field                 | Value                                               |
| --------------------- | --------------------------------------------------- |
| **Status**            | `prototype` / `testing` / `validating` / `archived` |
| **Owner**             | [Your name]                                         |
| **Created**           | [Date]                                              |
| **Target graduation** | [Quarter/date or "exploratory"]                     |

## Description

[One paragraph: what this is, what problem it solves, why it matters.]
```

### 4. Status Definitions

| Status       | Meaning                                                               | What happens next                   |
| ------------ | --------------------------------------------------------------------- | ----------------------------------- |
| `prototype`  | Early exploration, proof-of-concept. May not run cleanly.             | Continue development or archive.    |
| `testing`    | Core functionality works. Being tested against real data or hardware. | Move to validation or archive.      |
| `validating` | Feature-complete and under review for production promotion.           | Open a graduation PR or archive.    |
| `archived`   | Experiment concluded — either graduated or abandoned.                 | Move to `_archived/` with a reason. |

### 5. Graduation Path

When an experiment is ready for production:

1. Open a PR that moves the code from `sandbox/[experiment]/` into the
   appropriate production directory (`src/`, `Core/Src/`, `processing/`,
   `Modules/`, etc.)
2. Add corresponding tests
3. Remove the sandbox folder in the same PR
4. Reference the original sandbox README in the PR description for context
5. Get a code review from someone who didn't write the experiment

The PR is the gate — sandbox code doesn't sneak into production. It gets a
proper review, tests, and documentation before it graduates.

### 6. Archival

Experiments that don't graduate get moved to `sandbox/_archived/` with an
updated README:

```markdown
| **Status** | `archived` | | **Archived** | [Date] | | **Reason** | [Why —
e.g., "approach superseded by X", "hardware target canceled", "merged into
production as Y"] |
```

Archived experiments are kept for historical reference. They may contain useful
ideas, data, or lessons even if the code itself didn't ship.

### 7. CI Treatment

CI runs a **separate, lightweight job** on `sandbox/` — linting, type checking,
and basic syntax validation where applicable. Sandbox CI failures are **visible
but non-blocking**: they do not gate merges to `main` or production releases.

This keeps quality visible without penalizing experimentation.

### 8. No New Repos for Experiments

If it's a prototype, an algorithm experiment, a new sensor integration, or a
hardware test script — it goes here. New repositories are created only for
genuinely new standalone components: a new firmware target for a new MCU, a new
PCB design, or a new standalone tool with its own release cycle.

When in doubt, start in a sandbox. You can always extract to a new repo later if
it grows into something that justifies its own lifecycle.

## Directory Layout

```
sandbox/
├── README.md                    # This file
├── [experiment-name]/           # One folder per experiment
│   ├── README.md                # Status, owner, description (required)
│   ├── [code, data, configs]    # Whatever the experiment needs
│   └── ...
└── _archived/                   # Concluded experiments (kept for reference)
    └── [old-experiment]/
        └── README.md            # Must include archive reason
```

### Naming Conventions

- Folder names: **lowercase-with-hyphens** (e.g., `multi-element-beamforming`,
  `speckle-contrast-v2`)
- Be descriptive: `new-algo` is bad, `real-time-dcs-classification` is good
- Prefix with the component area if it helps: `ui-3d-overlay`,
  `fw-power-sequencing`

## Questions?

If you're unsure whether something belongs in a sandbox, a new repo, or a
feature branch, ask in the team channel or open a discussion. The goal is to
keep experiments visible and the org clean — not to create bureaucracy.
