# Contributing to Open-LIFU

Open-LIFU is built in the open and welcomes contributions from researchers,
developers, and clinicians. This page is the developer-facing summary for the
software repo (`opw_neuromod_sw`); for the full org-wide policy see the
[Openwater Contributing Guide](https://github.com/OpenwaterHealth/.github/blob/main/CONTRIBUTING.md).

## Code of Conduct

Please read our
[Code of Conduct](https://github.com/OpenwaterHealth/.github/blob/main/CODE_OF_CONDUCT.md)
before contributing. We believe in a respectful and collaborative environment;
the Code of Conduct outlines our expectations for all participants.

## Contribute without a unit

You do not need an Openwater device to contribute. Most of our community will
never receive a physical unit, and that's fine — the work that moves the
platform forward happens in code, documentation, quality, research methodology,
and clinical workflow design. We call this the **Contribute Without a Unit**
track. The live version with featured good-first-issues and lane contacts is
at the
[community hub](https://openwaterhealth.github.io/openwater-community/#contribute-without-a-unit).

### Five lanes

Pick the lane that fits the work you want to do, not the role you happen to
hold. Tags reflect best fit, not exclusivity.

- **Code** *(Best for: Developers)* — Python, Slicer extension, and tooling
  improvements against simulators and fixtures. The MATLAB-to-Python migration
  is the highest-leverage place to start.
  Contact: <community@openwater.health>
- **Documentation** *(Best for: All roles)* — Make the platform legible to the
  next contributor. Tutorials, glossaries, API docs, translations.
  Contact: <community@openwater.health>
- **Quality** *(Best for: Developers)* — Triage, reproduction, accessibility,
  CI. The work that keeps the platform trustworthy as the community grows.
  Contact: <community@openwater.health>
- **Research** *(Best for: Researchers)* — Literature synthesis, schemas,
  methodology, RFCs. Especially good for grad students new to focused
  ultrasound.
  Contact: <community@openwater.health>
- **Clinical** *(Best for: Clinicians)* — IRB templates, plain-language
  summaries, safety reporting, protocol refinement. The bridge between bench
  and clinic.
  Contact: <community@openwater.health>

### Filter open issues by lane

- [All good-first-issues](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
- [Code](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22lane%3Acode%22)
- [Documentation](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22lane%3Adocs%22)
- [Quality](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22lane%3Aquality%22)
- [Research](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22lane%3Aresearch%22)
- [Clinical](https://github.com/OpenwaterHealth/opw_neuromod_sw/issues?q=is%3Aopen+is%3Aissue+label%3A%22lane%3Aclinical%22)

## Getting started

1. **Fork the repository.** Fork `opw_neuromod_sw` to your GitHub account.
2. **Set up your environment.** Follow `README.md` for the canonical setup
   steps. Target is Python 3.10+. The active development branch for the
   migration work is `python-migration`.
3. **For Slicer extension work,** see the
   [Slicer developer documentation](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html);
   the Code lane lead will point you at the right branch.
4. **For Documentation, Research, and Clinical lane work,** no local
   environment setup is required — you can edit on the wiki directly or open
   documentation PRs from a fork.

## State of the project

A few specifics worth knowing before you start:

- We are actively migrating Open-LIFU's software stack from MATLAB to Python.
  Many utilities still have MATLAB originals; ports to Python are the most
  impactful Code-lane work right now.
- The current platform uses Verasonics Vantage 128 for treatment delivery and
  Localite TMSNavigator for neuronavigation. The next generation will use
  fully open-source, miniaturized hardware.
- A 3D Slicer extension provides an advanced user interface to Open-LIFU
  features.
- Hardware manuals, mechanical drawings, and 3D models live in
  [`opw_neuromod_hw`](https://github.com/OpenwaterHealth/opw_neuromod_hw).

## Making contributions

For the full contribution workflow (forking, branching, commit conventions,
style guides), see the
[org-wide Contributing Guide](https://github.com/OpenwaterHealth/.github/blob/main/CONTRIBUTING.md#-contribution-workflow).
Repo-specific notes below.

### Issues

- **Search first.** Before creating a new issue, please check whether the
  issue already exists to avoid duplicates.
- **Be specific.** When creating an issue, provide a clear and detailed
  description, including steps to reproduce, expected vs. observed behavior,
  Python version, OS, and the commit SHA you tested against.
- **Claim before working.** Comment "I'd like to work on this" on an open
  issue before you start. We'll assign it. If you don't see an issue for the
  work you want to do, open one first and propose your approach.

### Pull requests

- **Branching.** Branch off `main` unless the work targets the active
  migration — issues will tell you when to target `python-migration` instead.
- **Style.** Python code is formatted with `black` and linted with `ruff`.
  Run both before opening the PR.
- **Tests.** New code wants new tests. Run `pytest` locally and confirm
  green before requesting review.
- **Description.** Link the issue (`Closes #123`), summarize what changed,
  call out any decisions you made that a reviewer might second-guess.
- **Draft early.** Open a draft PR as soon as you have something to show.
  It's easier to catch direction problems before you've written a thousand
  lines.

## Regulatory note

Open-LIFU is exclusively intended for research purposes and is not cleared or
approved by the FDA for clinical use. If your contribution touches anything
that could be read as a clinical claim — a documentation change, a parameter
description, a summary of trial results — please include or preserve the
short-form disclaimer:

> Openwater's platform is exclusively intended for research purposes and is
> not cleared or approved by the FDA for clinical use.

We will flag this in review if it's missing.

## License and CLA

This repository is currently licensed under **AGPL v3.0**. We are planning a
transition to **Apache 2.0** to support broader adoption. All contributions
are made under our Contributor License Agreement — see the
[org-wide CLA section](https://github.com/OpenwaterHealth/.github/blob/main/CONTRIBUTING.md#-contributor-license-agreement-cla)
for details.

## Asking for help

If you need help or have questions, reach out through:

- **Discord** —
  [discord.gg/openwater](https://discord.gg/openwater), specifically the
  `#contributing` channel
- **GitHub** — open a Discussion or Issue in this repo
- **Email** —
  [community@openwater.health](mailto:community@openwater.health)
- **Office hours** — Weekly on Discord; see the
  [community hub](https://openwaterhealth.github.io/openwater-community/)
  for the next session

## The Openwater community

Have a question, found a bug, or want to connect and get involved with the
broader Openwater community? Visit the community hub at
[openwaterhealth.github.io/openwater-community](https://openwaterhealth.github.io/openwater-community/),
or join the [Discord server](https://discord.gg/openwater) to connect with
other members, ask for help, and stay updated on project developments.

---

*Last updated: 2026-06-08. If you spot something stale or wrong in this file,
that's a Documentation-lane contribution — open a PR.*
