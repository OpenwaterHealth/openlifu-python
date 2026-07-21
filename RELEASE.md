# openlifu-python Release Process

This document describes how to release the `openlifu` Python package. Publishing
a GitHub release triggers the workflow that publishes the package to PyPI.

## Versioning and Branches

Use git tags of the form `vX.Y.Z`. A new release line begins at `vX.Y.0`, and
patch releases increment `Z`.

The package version is derived from git metadata by `hatch-vcs`; do not maintain
a separate hard-coded package version.

`main` is the development branch. Create `release/X.Y` for each release line and
tag all releases in that line from the branch. New feature work continues on
`main`.

## Release Steps

1. If sample database compatibility changed, publish or identify the matching
   `openlifu-sample-database` tag, update the openlifu-python README, and note
   the change in the release notes.
2. Confirm `main` is ready for release and CI is passing.
3. Create and push the release branch:

   ```bash
   git checkout main
   git pull
   git checkout -b release/X.Y
   git push -u origin release/X.Y
   ```

4. Tag the release from the release branch:

   ```bash
   git tag vX.Y.0
   git push origin vX.Y.0
   ```

5. Draft and publish the GitHub release from `vX.Y.0`.

## Patch Releases

1. Merge the fix to `main`, then cherry-pick it to the release branch. A fix
   specific to the released line may be committed directly to that branch.

   ```bash
   git checkout release/X.Y
   git pull
   git cherry-pick <fix-commit-sha>
   git push origin release/X.Y
   ```

2. Confirm the release branch is ready and CI is passing.
3. Tag the next patch release:

   ```bash
   git tag vX.Y.N
   git push origin vX.Y.N
   ```

4. Draft and publish the GitHub release from `vX.Y.N`.
