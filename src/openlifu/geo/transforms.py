from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Tuple

import numpy as np

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin


@dataclass
class ArrayTransform(DictMixin):
    """An affine transform with a unit string, often intended to represent how a transducer array is positioned in space."""

    matrix: Annotated[np.ndarray, OpenLIFUFieldData("Affine matrix", "4x4 affine transform matrix")]
    """4x4 affine transform matrix"""

    units: Annotated[str, OpenLIFUFieldData("Units", "The units of the space on which to apply the transform matrix , e.g. 'mm' (In order to apply the transform to points, first represent the points in these units.)")]
    """The units of the space on which to apply the transform matrix , e.g. "mm"
    (In order to apply the transform to points, first represent the points in these units.)
    """


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert cartesian coordinates to spherical coordinates

    Args: x, y, z are cartesian coordinates
    Returns: r, theta, phi, where
        r is the radial spherical coordinate, a nonnegative float.
        theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle.
            theta is in the range [0,pi].
        phi is the azimuthal spherical coordinate, in the range [-pi,pi]

    Angles are in radians.
    """
    return (
        np.sqrt(x**2 + y**2 + z**2),
        np.arctan2(np.sqrt(x**2 + y**2), z),
        np.arctan2(y, x),
    )


def spherical_to_cartesian(r: float, th: float, ph: float) -> Tuple[float, float, float]:
    """Convert spherical coordinates to cartesian coordinates

    Args:
        r: the radial spherical coordinate
        th: the polar spherical coordinate theta, aka the angle off the z-axis, aka the non-azimuthal spherical angle
        ph: the azimuthal spherical coordinate phi
    Returns the cartesian coordinates x,y,z

    Angles are in radians.
    """
    return (r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th))


def cartesian_to_spherical_vectorized(p: np.ndarray) -> np.ndarray:
    """Convert cartesian coordinates to spherical coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point cartesian coordinates x,y,z.
    Returns: An array of shape (...,3), where the last axis describes point spherical coordinates r, theta, phi, where
        r is the radial spherical coordinate, a nonnegative float.
        theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle.
            theta is in the range [0,pi].
        phi is the azimuthal spherical coordinate, in the range [-pi,pi]

    Angles are in radians.
    """
    return np.stack(
        [
            np.sqrt((p**2).sum(axis=-1)),
            np.arctan2(np.sqrt((p[..., 0:2] ** 2).sum(axis=-1)), p[..., 2]),
            np.arctan2(p[..., 1], p[..., 0]),
        ],
        axis=-1,
    )


def spherical_to_cartesian_vectorized(p: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to cartesian coordinates

    Args:
        p: an array of shape  (...,3), where the last axis describes point spherical coordinates r, theta, phi, where:
            r is the radial spherical coordinate
            theta is the polar spherical coordinate, aka the angle off the z-axis, aka the non-azimuthal spherical angle
            phi is the azimuthal spherical coordinate
    Returns the cartesian coordinates x,y,z

    Angles are in radians.
    """
    return np.stack(
        [
            p[..., 0] * np.sin(p[..., 1]) * np.cos(p[..., 2]),
            p[..., 0] * np.sin(p[..., 1]) * np.sin(p[..., 2]),
            p[..., 0] * np.cos(p[..., 1]),
        ],
        axis=-1,
    )


def spherical_coordinate_basis(th: float, phi: float) -> np.ndarray:
    """Return normalized spherical coordinate basis at a location with spherical polar and azimuthal coordinates (th, phi).
    The coordinate basis is returned as an array `basis` of shape (3,3), where the rows are the basis vectors,
    in the order r, theta, phi. So `basis[0], basis[1], basis[2]` are the vectors $\\hat{r}$, $\\hat{\\theta}$, $\\hat{\\phi}$.
    Angles are assumed to be provided in radians."""
    return np.array(
        [
            [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)],
            [np.cos(th) * np.cos(phi), np.cos(th) * np.sin(phi), -np.sin(th)],
            [-np.sin(phi), np.cos(phi), 0],
        ]
    )


def create_standoff_transform(z_offset: float, dzdy: float) -> np.ndarray:
    """Create a standoff transform based on a z_offset and a dzdy value.

    A "standoff transform" applies a displacement in transducer space that moves a transducer to where it would
    be situated with the standoff in place. The idea is that if you start with a transform that places a transducer
    directly against skin, then pre-composing that transform by a "standoff transform" serves to nudge the transducer
    such that there is space for the standoff to be between it and the skin.

    This function assumes that the standoff is laterally symmetric, has some thickness, and can raise the bottom of
    the transducer a bit more than the top. The `z_offset` is the thickness in the middle of the standoff,
    while the `dzdy` is the elevational slope.

    Args:
        z_offset: Thickness in the middle of the standoff
        dzdy: Slope of the standoff, as axial displacement per unit elevational displacement. A positive number
            here means that the bottom of the transducer is raised a little bit more than the top.

    Returns a 4x4 matrix representing a rigid transform in whatever units z_offset was provided in.
    """
    angle = np.arctan(dzdy)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), -z_offset],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
