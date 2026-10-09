from __future__ import annotations

import numpy as np

from openlifu.geo.point import Point
from openlifu.geo.transforms import (
    cartesian_to_spherical,
    cartesian_to_spherical_vectorized,
    create_standoff_transform,
    lps_to_spherical,
    lps_to_spherical_vectorized,
    spherical_coordinate_basis,
    spherical_to_cartesian,
    spherical_to_cartesian_vectorized,
    spherical_to_lps,
    spherical_to_lps_vectorized,
)


def test_point_from_dict():
    point = Point.from_dict({'position' : [10,20,30],})
    assert (point.position == np.array([10,20,30], dtype=float)).all()

def test_spherical_coordinate_range():
    """Verify that spherical coordinate output is in the prescribed value ranges"""
    rng = np.random.default_rng(848)
    # try all 8 octants of 3D space
    for sign_x in [-1,1]:
        for sign_y in [-1,1]:
            for sign_z in [-1,1]:
                cartesian_coords = np.array([sign_x, sign_y, sign_z]) * rng.random(size=3)
                r, th, ph = cartesian_to_spherical(*cartesian_coords)
                assert r>=0
                assert 0 <= th <= np.pi
                assert -np.pi <= ph <= np.pi

def test_spherical_coordinate_conversion_inverse():
    """Verify that the spherical coordinate conversion forward and backward functions are inverses of one another"""
    rng = np.random.default_rng(241)
    # try all 8 octants of 3D space
    for sign_x in [-1,1]:
        for sign_y in [-1,1]:
            for sign_z in [-1,1]:
                cartesian_coords = np.array([sign_x, sign_y, sign_z]) * rng.random(size=3)
                np.testing.assert_almost_equal(
                    spherical_to_cartesian(*cartesian_to_spherical(*cartesian_coords)),
                    cartesian_coords
                )
                np.testing.assert_almost_equal(
                    cartesian_to_spherical(*spherical_to_cartesian(*cartesian_to_spherical(*cartesian_coords))),
                    cartesian_to_spherical(*cartesian_coords)
                )

def test_cartesian_to_spherical_vectorized():
    rng = np.random.default_rng(35932)
    points_cartesian = rng.normal(size=(10,3), scale=2) # make 10 random cartesian points
    points_spherical = cartesian_to_spherical_vectorized(points_cartesian)
    # Check individual points against the non-vectorized conversion function:
    for point_cartesian, point_spherical in zip(points_cartesian, points_spherical):
        assert np.allclose(
            point_spherical, # result of vectorized converter
            np.array(cartesian_to_spherical(*point_cartesian)), # non-vectorized converter
        )

def test_spherical_to_cartesian_vectorized():
    rng = np.random.default_rng(85932)

    # make 10 random points in spherical coordinates
    num_pts = 10
    points_spherical = np.zeros(shape=(num_pts,3))
    points_spherical[...,0] = rng.random(num_pts)*5 # random r coordinates
    points_spherical[...,1] = rng.random(num_pts)*np.pi # random theta coordinates
    points_spherical[...,2] = rng.random(num_pts)*2*np.pi-np.pi # random phi coordinates

    points_cartesian = spherical_to_cartesian_vectorized(points_spherical)
    # Check individual points against the non-vectorized conversion function:
    for point_cartesian, point_spherical in zip(points_cartesian, points_spherical):
        assert np.allclose(
            point_cartesian, # result of vectorized converter
            np.array(spherical_to_cartesian(*point_spherical)), # non-vectorized converter
        )

def test_spherical_coordinate_basis():
    rng = np.random.default_rng(35235)
    th = rng.random()*np.pi
    phi = rng.random()*2*np.pi-np.pi
    r  = rng.random()*10
    basis = spherical_coordinate_basis(th,phi)
    assert np.allclose(basis @ basis.T, np.eye(3)) # verify it is an orthonormal basis
    r_hat, theta_hat, phi_hat = basis
    point = np.array(spherical_to_cartesian(r, th, phi))
    assert np.allclose(np.diff(r_hat / point), 0) # verify that r_hat is a scalar multiple of the cartesian coords
    assert cartesian_to_spherical_vectorized(point + 0.01*phi_hat)[2] > phi # verify phi_hat points along increasing phi
    assert cartesian_to_spherical_vectorized(point + 0.01*theta_hat)[1] > th # verify theta_hat points along increasing theta

def test_create_standoff_transform():
    z_offset = 3.2
    dzdy = 0.15
    t = create_standoff_transform(z_offset, dzdy)
    assert np.allclose(t[:3,:3] @ t[:3,:3].T, np.eye(3)) # it's an orthonormal transform
    assert np.allclose(np.linalg.det(t[:3,:3]), 1.0) # orientation preserving
    assert np.allclose(t @ np.array([0,0,0,1]), np.array([0,0,-z_offset,1.])) # translates the origin correctly
    new_x_axis = (t @ np.array([1,0,0,1]) - t @ np.array([0,0,0,1]))[:3]
    new_y_axis = (t @ np.array([0,1,0,1]) - t @ np.array([0,0,0,1]))[:3]
    assert np.allclose(new_x_axis, np.array([1.,0,0]))
    assert new_y_axis[2] > 0 # the y axis was rotated upward, so that the top of the transducer gets closer to the skin


def test_lps_to_spherical_golden():
    """Anchor lps_to_spherical against values computed from the MATLAB reference (fus.seg.lps2sph)."""
    np.testing.assert_almost_equal(lps_to_spherical(0.0, 0.0, 1.0), (90.0, 90.0, 1.0))
    np.testing.assert_almost_equal(lps_to_spherical(1.0, 0.0, 0.0), (90.0, 0.0, 1.0))
    np.testing.assert_almost_equal(lps_to_spherical(0.0, 1.0, 0.0), (180.0, 0.0, 1.0))
    np.testing.assert_almost_equal(lps_to_spherical(0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_almost_equal(lps_to_spherical(3.0, 4.0, 0.0), (143.13010235, 0.0, 5.0))
    np.testing.assert_almost_equal(lps_to_spherical(1.0, 1.0, 1.0), (135.0, 35.26438968, 1.73205081))


def test_spherical_to_lps_golden():
    """Anchor spherical_to_lps against values computed from the MATLAB reference (fus.seg.sph2lps)."""
    np.testing.assert_almost_equal(spherical_to_lps(90.0, 90.0, 1.0), (0.0, 0.0, 1.0))
    np.testing.assert_almost_equal(spherical_to_lps(90.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    np.testing.assert_almost_equal(spherical_to_lps(180.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    np.testing.assert_almost_equal(spherical_to_lps(0.0, 0.0, 1.0), (0.0, -1.0, 0.0))


def test_lps_spherical_inverse():
    """Verify lps_to_spherical and spherical_to_lps invert one another across all eight octants."""
    rng = np.random.default_rng(1234)
    for sign_l in [-1, 1]:
        for sign_p in [-1, 1]:
            for sign_s in [-1, 1]:
                lps = np.array([sign_l, sign_p, sign_s]) * (rng.random(size=3) + 0.1)
                np.testing.assert_almost_equal(spherical_to_lps(*lps_to_spherical(*lps)), lps)


def test_spherical_to_lps_inverse():
    """Verify the round trip the other direction for angles in the range produced by lps_to_spherical."""
    rng = np.random.default_rng(5678)
    for _ in range(20):
        sph = (rng.uniform(-89.0, 269.0), rng.uniform(-89.0, 89.0), rng.uniform(0.1, 10.0))
        np.testing.assert_almost_equal(lps_to_spherical(*spherical_to_lps(*sph)), sph)


def test_lps_spherical_r_zero():
    """Edge case r=0: the origin has zero radius, and any zero-radius spherical point maps back to the origin."""
    th, phi, r = lps_to_spherical(0.0, 0.0, 0.0)
    assert r == 0.0
    assert np.isfinite([th, phi]).all()
    np.testing.assert_almost_equal(spherical_to_lps(37.0, -12.0, 0.0), (0.0, 0.0, 0.0))
    np.testing.assert_almost_equal(spherical_to_lps(180.0, 90.0, 0.0), (0.0, 0.0, 0.0))


def test_lps_to_spherical_vectorized():
    """The vectorized LPS-to-spherical converter must agree with the scalar one point by point."""
    rng = np.random.default_rng(913)
    points_lps = rng.normal(size=(10, 3), scale=2)
    points_spherical = lps_to_spherical_vectorized(points_lps)
    for point_lps, point_spherical in zip(points_lps, points_spherical):
        assert np.allclose(point_spherical, np.array(lps_to_spherical(*point_lps)))


def test_spherical_to_lps_vectorized():
    """The vectorized spherical-to-LPS converter must agree with the scalar one point by point."""
    rng = np.random.default_rng(2024)
    points_spherical = np.stack(
        [
            rng.uniform(-89.0, 269.0, size=10),
            rng.uniform(-89.0, 89.0, size=10),
            rng.uniform(0.1, 10.0, size=10),
        ],
        axis=-1,
    )
    points_lps = spherical_to_lps_vectorized(points_spherical)
    for point_spherical, point_lps in zip(points_spherical, points_lps):
        assert np.allclose(point_lps, np.array(spherical_to_lps(*point_spherical)))


def test_lps_spherical_vectorized_inverse():
    """The vectorized converters must invert one another over arbitrary leading dimensions."""
    rng = np.random.default_rng(77)
    points_lps = rng.normal(size=(8, 5, 3), scale=3)
    np.testing.assert_almost_equal(
        spherical_to_lps_vectorized(lps_to_spherical_vectorized(points_lps)),
        points_lps,
    )
