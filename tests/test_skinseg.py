from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import trimesh
import vtk
from scipy.linalg import expm
from scipy.stats import skew

from openlifu.seg.skinseg import (
    affine_from_vtk_image_data,
    apply_affine_to_polydata,
    cartesian_to_spherical,
    compute_foreground_mask,
    create_closed_surface_from_labelmap,
    spherical_interpolator_from_mesh,
    take_largest_connected_component,
    vtk_img_from_array_and_affine,
)


def add_ball(
    volume:np.ndarray,
    center:Tuple[float,float,float],
    radius:float,
    value:float = 1.
):
    """Add a ball to a given 3D volume. Useful for creating test volumes."""
    z_size, y_size, x_size = volume.shape

    z, y, x = np.ogrid[:z_size, :y_size, :x_size]

    dist_sq = (
        (z - center[0])**2 +
        (y - center[1])**2 +
        (x - center[2])**2
    )

    volume[dist_sq <= radius**2] = value

def test_take_largest_connected_component():
    vol_array = np.zeros((20,20,20))
    add_ball(vol_array, (5,5,5), 4) # ball of radius 4 at (5,5,5)
    expected_output = np.copy(vol_array).astype(bool) # at the end we expect to only get this first ball
    add_ball(vol_array, (15,15,15), 2) # smaller ball over at (15,15,15)
    assert np.all(
        take_largest_connected_component(vol_array) == expected_output
    )

def test_compute_foreground_mask():
    vol_array = np.zeros((20,20,20))
    add_ball(vol_array, (6,6,6), 5)
    expected_output = np.copy(vol_array).astype(bool) # this first ball shall be the expected output
    add_ball(vol_array, (6,5,5), 2, value=0.) # erase a hole inside this first ball
    add_ball(vol_array, (15,15,15), 3) # make a smaller disconnected ball elsewhere
    assert np.all(
        compute_foreground_mask(vol_array) == expected_output
    )

def test_vtk_img_from_array_and_affine():
    rng = np.random.default_rng(241)
    vol_array = rng.random((5,4,3))
    affine = np.eye(4)
    affine[:3,:3] = expm((lambda A: (A - A.T)/2)(rng.normal(size=(3,3)))) # generate a random orthogonal matrix
    affine[:3,3] = rng.random(3) # generate a random origin
    vtk_img = vtk_img_from_array_and_affine(vol_array, affine)

    i,j,k = 2,3,1 # We will test at the point with these indices i,j,k. The image value there should be vol_array[i,j,k].
    x,y,z,_ = affine @ np.array([i,j,k,1]) # the physical coordinates of the test point
    point_id = vtk_img.FindPoint([x,y,z])

    assert vtk_img.GetPointData().GetScalars().GetTuple1(point_id) == pytest.approx(vol_array[i,j,k])

def test_affine_from_vtk_image_data():
    rng = np.random.default_rng(716)
    vol_array = rng.random((5,4,3))
    affine = np.eye(4)
    affine[:3,:3] = expm((lambda A: (A - A.T)/2)(rng.normal(size=(3,3)))) # generate a random orthogonal matrix
    affine[:3,3] = rng.random(3) # generate a random origin
    vtk_img = vtk_img_from_array_and_affine(vol_array, affine)
    affine_reconstructed = affine_from_vtk_image_data(vtk_img)
    assert np.allclose(affine, affine_reconstructed)

def test_create_closed_surface_from_labelmap():
    # create a ball of radius 7 for a labelmap
    labelmap = np.zeros((20,20,20))
    sphere_radius = 7
    sphere_center = np.array([10,10,10])
    add_ball(labelmap, tuple(sphere_center), sphere_radius)
    rng = np.random.default_rng(6548)
    affine = np.eye(4)
    affine[:3,:3] = expm((lambda A: (A - A.T)/2)(rng.normal(size=(3,3)))) # generate a random orthogonal matrix
    affine[:3,3] = rng.random(3) # generate a random origin
    labelmap_vtk = vtk_img_from_array_and_affine(labelmap, affine = affine)

    # run the algorithm to be tested
    surface = create_closed_surface_from_labelmap(labelmap_vtk, decimation_factor=0.1)

    # the mesh is in "physical space" mapped to by `affine`, transform it back to the ijk space of the original `labelmap`
    surface = apply_affine_to_polydata(np.linalg.inv(affine), surface)

    # verify that the points on the generated mesh are not too far off being at distance 7 from the ball center
    points = surface.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        point_position = np.array(points.GetPoint(i))
        point_distance_from_sphere_center = np.linalg.norm(point_position - sphere_center, ord=2)
        assert np.abs(point_distance_from_sphere_center - sphere_radius) < 1.

    # verify that there are no scalars on the point data since these are undesirable in slicer, and the function
    # isn't supposed to add a colormap or anything like that
    assert surface.GetPointData().GetScalars() is None

@pytest.mark.parametrize(
        "use_embree",
        [
            False,
            pytest.param(
                True,
                marks = pytest.mark.skipif(not trimesh.ray.has_embree, reason="Embree not available (needs x86 architecture)")
            )
        ])
def test_spherical_interpolator_from_mesh(use_embree):
    """Check using a torus that the spherical interpolator behaves reasonably"""
    parametric_torus = vtk.vtkParametricTorus()
    parametric_torus.SetRingRadius(12.)
    parametric_torus.SetCrossSectionRadius(5)
    parametric_function_source = vtk.vtkParametricFunctionSource()
    parametric_function_source.SetUResolution(50)
    parametric_function_source.SetVResolution(50)
    parametric_function_source.SetParametricFunction(parametric_torus)
    parametric_function_source.Update()
    torus_polydata = parametric_function_source.GetOutput()

    origin = (12., 0., 0.) # center at a point inside the torus, on the central ring of the torus
    rng = np.random.default_rng(241)
    xyz_direction_columns = expm((lambda A: (A - A.T)/2)(rng.normal(size=(3,3)))) # generate a random orthogonal matrix

    interpolator = spherical_interpolator_from_mesh(
        surface_mesh = torus_polydata,
        origin = origin,
        xyz_direction_columns = xyz_direction_columns,
        use_embree=use_embree
    )

    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(1.0)
    sphere_source.SetThetaResolution(50)  # Set the resolution in the theta direction
    sphere_source.SetPhiResolution(50)  # Set the resolution in the phi direction
    sphere_source.Update()
    sphere_polydata = sphere_source.GetOutput()
    sphere_points = sphere_polydata.GetPoints()
    for i in range(sphere_points.GetNumberOfPoints()):
        point = np.array(sphere_points.GetPoint(i))
        r, theta, phi = cartesian_to_spherical(*point)
        r = interpolator(theta, phi)
        sphere_points.SetPoint(i, r * point)

    xyz_affine = np.eye(4)
    xyz_affine[:3,:3] = xyz_direction_columns
    xyz_affine[:3,3] = origin
    xyz_affine_vtkmat = vtk.vtkMatrix4x4()
    xyz_affine_vtkmat.DeepCopy(xyz_affine.ravel())
    xyz_transform = vtk.vtkTransform()
    xyz_transform.SetMatrix(xyz_affine_vtkmat)
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(xyz_transform)
    transform_filter.SetInputData(sphere_polydata)
    transform_filter.Update()
    sphere_polydata_transformed = transform_filter.GetOutput()

    # These tools will be used to measure the disance between a point and a mesh
    distance_from_torus = vtk.vtkImplicitPolyDataDistance()
    distance_from_torus.SetInput(torus_polydata)
    distance_from_sphere = vtk.vtkImplicitPolyDataDistance()
    distance_from_sphere.SetInput(sphere_polydata_transformed)

    # Now here is the unit test: sphere_polydata_transformed should be a sphere that has been "wrapped" around the torus.
    # To test this we first check that (a) the points on sphere_polydata_transformed are generally close to the torus.
    # Then we check that (b) the points of the torus are generally inside the sphere_polydata_transformed surface.
    # We say "generally" and allow for a tolerance, because
    # (a) some points on sphere_polydata_transformed will be far from the torus, for example as the sphere stretches over the donut hole
    # (b) some points on the torus will slightly stick out of the sphere just because of the interpolation
    # (the torus would be perfectly inside the sphere if the sphere mesh had infinite resolution, but it doesn't)
    # This is why some tolerance is allowed in both checks.

    sphere_transformed_points = sphere_polydata_transformed.GetPoints()
    distances = [
        np.abs(distance_from_torus.FunctionValue(sphere_transformed_points.GetPoint(i)))
        for i in range(sphere_transformed_points.GetNumberOfPoints())
    ]
    assert np.quantile(distances,0.9) < 0.3 # The points on sphere_polydata_transformed are generally close to the torus.

    torus_points = torus_polydata.GetPoints()
    signed_distances = [
        distance_from_sphere.FunctionValue(torus_points.GetPoint(i))
        for i in range(torus_points.GetNumberOfPoints())
    ]

    # A point is *inside* iff the signed distance is negative, so ideally we should check that all signed_distance elements are negative.
    # However we must leave a tolerance.
    assert np.max(signed_distances) < 2 # With this tolerance, all torus points are inside rather than outside
    assert skew(signed_distances) < 0 # The distribution of signed distances should be negatively skewed,
    # the idea is that while most torus points are close to the sphere (signed distance hovering around 0)
    # there is a tail in the distribution of signed distances consisting of points that are well inside the sphere
