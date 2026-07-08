# openlifu-python Release Process

This document describes how to release the `openlifu` Python package. This
repository is the upstream library for SlicerOpenLIFU and openlifu-app.

It comes down to just making a git tag and publishing a GitHub release. A PyPI
publishing workflow is automatically triggered.

## Versioning

Use git tags of the form:

```text
vX.Y.Z
```

The package version is derived from git metadata by `hatch-vcs`; do not maintain
a separate hard-coded package version.

## Release steps

1. Check whether the sample database needs a matching release tag. If the
   database format, bundled example data, protocols, transducers, or
   compatibility expectations changed, publish or identify the corresponding
   `openlifu-sample-database` tag before releasing, and update the
   openlifu-python README to reference the new sample database tag.
2. Confirm `main` is ready for release and CI is passing.
3. Create and push the openlifu-python git tag:

   ```bash
   git checkout main
   git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. Draft a GitHub release from the tag.
5. Publish the GitHub release.

The CD workflow should trigger and the package should automatically become
available on PyPI.

## Release Branches

This repository does not normally need release branches. Prefer the tag-only
flow above.

Create a release branch only when an older library line needs maintenance after
`main` has moved on:

```text
release/0.21
  v0.21.1
  v0.21.2
```

In that case, merge fixes to `main` first when practical, then cherry-pick them
to the maintenance branch.

## Downstream Updates

Downstream repositories can update their pins when ready:

- SlicerOpenLIFU pins `openlifu` in
  `OpenLIFULib/OpenLIFULib/Resources/python-requirements.txt`.
- OpenLIFU-app gets the `openlifu` version indirectly through its pinned
  SlicerOpenLIFU `GIT_TAG`.

If this release changes database compatibility, call that out in the GitHub
release notes.
