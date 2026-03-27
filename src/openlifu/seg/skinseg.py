"""Tools to get a skin surface from an MRI.

Example workflow, starting from

- an MRI volume `vol_array`, as a numpy array
- an associated 4x4 `affine` transform, as a numpy array
- a desired spherical coordinate system `origin` location inside the volume (e.g. the sonication target)

foreground_mask_array = compute_foreground_mask(vol_array)
foreground_mask_vtk_image = vtk_img_from_array_and_affine(foreground_mask_array, affine)
skin_mesh = create_closed_surface_from_labelmap(foreground_mask_vtk_image)
skin_interpolator = spherical_interpolator_from_mesh(skin_mesh, origin)
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import skimage.filters
import skimage.measure
import trimesh
import vtk
from packaging.version import parse
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_closing, generate_binary_structure
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from openlifu.geo import cartesian_to_spherical, spherical_to_cartesian_vectorized


def apply_affine_to_polydata(affine:np.ndarray, polydata:vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Apply an affine transform to a vtkPolyData."""
    affine_vtkmat = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            affine_vtkmat.SetElement(i, j, affine[i, j])
    affine_vtktransform = vtk.vtkTransform()
    affine_vtktransform.SetMatrix(affine_vtkmat)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(affine_vtktransform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()
    return transform_filter.GetOutput()

def take_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Given a boolean image array (or any integer numpy array), return a mask of the largest connected component."""
    mask_labeled = skimage.measure.label(mask)
    connected_component_info = skimage.measure.regionprops(mask_labeled)
    largest_connected_componet_label = connected_component_info[np.argmax([rp.area for rp in connected_component_info])].label
    return (mask_labeled == largest_connected_componet_label)

def compute_foreground_mask(
        vol_array : np.ndarray,
        closing_radius : float = 9.,
        lower_quantile_for_otsu_threshold : float = 0.02,
        upper_quantile_for_otsu_threshold : float = 0.99,
    ) -> np.ndarray:
    """Given a 3D image array, return a boolean mask representing the "foreground."

    Args:
        vol_array: a 3D image array of shape (H,W,D)
        closing_readius: the radius of the ball used in the morphological closing operation
        lower_quantile_for_otsu_threshold: a number from 0 to 1. Before otsu thresholding,
            values below this quantile are omitted from the histogram as outliers.
        upper_quantile_for_otsu_threshold: a number from 0 to 1.  Before otsu thresholding,
            values above this quantile are omitted from the histogram as outliers.

    Returns: a boolean array of shape (H,W,D) representing a foreground mask

    This is essentially a port of the BRAINSTools automated foreground masking algorithm.
    - Original algorithm documentation: https://slicer.readthedocs.io/en/latest/user_guide/modules/brainsroiauto.html
    - Original algorithm code: https://github.com/BRAINSia/BRAINSTools/tree/7c37d9e8c238f66f8a83f997d9c9bb659c494c90/BRAINSROIAuto

    The algorithm roughly works as follows:
    - step 1: otsu thresholding
    - step 2: keep only the largest connected component
    - step 3: morphological closing
    - step 4: hole filling

    The default values of the parameters have been observed to work well for mm-spaced brain MRIs.
    """

    # step 1: otsu-threshold the image to create an initial foreground mask.
    threshold_lower, threshold_upper = np.quantile(
        vol_array,
        [lower_quantile_for_otsu_threshold,upper_quantile_for_otsu_threshold]
    )
    threshold_foreground = skimage.filters.threshold_otsu(
        vol_array[(vol_array >= threshold_lower) & (vol_array <= threshold_upper)]
    )
    foreground_mask = vol_array >= threshold_foreground

    # step 2: keep only the largest connected component to throw out spurious bits.
    foreground_mask = take_largest_connected_component(foreground_mask)

    # step 3: do a morphological closing.
    # while this does fill some holes, that's not the main point since step 4 already fills holes.
    # the point of this step is rather to clean up and smooth out the skin surface of small cavities.
    # Uses binary_closing with a 6-connected structuring element iterated
    # closing_radius times, which is much faster than the original two-pass
    # EDT approach (~15x). The diamond-shaped kernel produces slightly
    # different boundary voxels (<0.2%) versus the EDT's Euclidean ball,
    # but this does not affect downstream segmentation quality.
    closing_r = int(closing_radius)
    if closing_r > 0:
        pad_width = closing_r + 2
        padded = np.pad(foreground_mask, pad_width, mode='constant')
        struct = generate_binary_structure(3, 1)  # 6-connected
        closed_padded = binary_closing(
            padded.astype(np.uint8), structure=struct, iterations=closing_r
        ).astype(bool)
        h, w, d = foreground_mask.shape
        p = pad_width
        foreground_mask = closed_padded[p:p+h, p:p+w, p:p+d]

    # step 4: take the complement of the largest connected component of the current background.
    # the background mask at this point contains the "actual background" and possibly also some
    # holes that are inside the foreground region. the largest connected component of this background
    # mask is considered to be the "actual background."" this step therefore serves to fill
    # any remaining holes in the foreground mask.
    # this step is analogous to the seeded flood fill in the original algorithm:
    # https://github.com/BRAINSia/BRAINSTools/blob/7c37d9e8c238f66f8a83f997d9c9bb659c494c90/BRAINSCommonLib/itkLargestForegroundFilledMaskImageFilter.hxx#L255-L302
    foreground_mask = ~take_largest_connected_component(~foreground_mask)

    return foreground_mask


def vtk_img_from_array_and_affine(vol_array:np.ndarray, affine:np.ndarray) -> vtk.vtkImageData:
    """ Convert a numpy (array, affine) pair into a vtkImageData.

    Args:
        vol_array: a 3D image numpy array with float type data
        affine: a numpy array of shape (4,4) representing the affine matrix of the 3D image.

    Returns: vtkImageData with a copy of vol_array as the underlying image data,
        and with origin, spacing, and direction matrix set according to the affine matrix.

    Since a vtkImageData is intended to represent image data on a structured grid with *orthogonal* axes,
    the upper-left 3x3 submatrix of the affine matrix should be an orthogonal matrix. There will be no error
    if it isn't, since the "direction matrix" of a vtkImageData can be set to be non-orthogonal -- it just isn't
    the intended usage of vtkImageData and could be misinterpreted by downstream vtk filters.

    Maintain a reference to vol_array, so that it is not garbage collected
    (which could leave the vtkImageData pointing to invalid memory -- see vtk.util.numpy_support.numpy_to_vtk documentation).
    """

    matrix_3x3 = affine[:3, :3]
    origin = affine[:3, 3]
    spacing = np.linalg.norm(matrix_3x3, axis=0)
    direction_matrix = matrix_3x3 / spacing[np.newaxis,:]

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(vol_array.shape)
    vtk_img.SetOrigin(origin.tolist())
    vtk_img.SetSpacing(spacing.tolist())

    direction_matrix_vtk = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            direction_matrix_vtk.SetElement(i, j, direction_matrix[i, j])
    vtk_img.SetDirectionMatrix(direction_matrix_vtk)

    vol_array_flat = vol_array.transpose((2,1,0)).ravel(order='C')
    vol_array_vtk = numpy_to_vtk(num_array=vol_array_flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_img.GetPointData().SetScalars(vol_array_vtk)

    return vtk_img

def affine_from_vtk_image_data(vtk_img:vtk.vtkImageData) -> np.ndarray:
    """Get a 4x4 affine matrix out of a vtkImageData, a partial reverse to `vtk_img_from_array_and_affine`"""
    origin = np.array(vtk_img.GetOrigin())
    spacing = np.array(vtk_img.GetSpacing())

    direction_vtk = vtk_img.GetDirectionMatrix()
    direction = np.eye(3)
    for i in range(3):
        for j in range(3):
            direction[i, j] = direction_vtk.GetElement(i, j)

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    return affine

def create_closed_surface_from_labelmap(
    binary_labelmap:vtk.vtkImageData,
    decimation_factor:float=0.,
    smoothing_factor:float=0.5
) -> vtk.vtkPolyData:
    """ Create a surface mesh vtkPolyData from a binary labelmap vtkImageData.

    Args:
        binary_labelmap: input vtkImageData binary labelmap
        decimation_factor: 0.0 for no decimation, 1.0 for maximum reduction.
        smoothing_factor: 0.0 for no smoothing, 1.0 for maximum smoothing.

    Returns:
        vtkPolyData: the resulting surface mesh

    The algorithm here is based on the labelmap-to-closed-surface algorithm in 3D Slicer:
    https://github.com/Slicer/Slicer/blob/677932127c73a6c78654d4afd9458a655a4eef63/Libs/vtkSegmentationCore/vtkBinaryLabelmapToClosedSurfaceConversionRule.cxx#L246-L476
    """

    affine = None # Only needed if vtk version is less than 9.3.0
    if parse(vtk.__version__) < parse("9.3.0"):
        # In these older versions of vtk, the labelmap would not work.
        affine = affine_from_vtk_image_data(binary_labelmap)
        binary_labelmap.SetOrigin([0,0,0])
        binary_labelmap.SetSpacing([1,1,1])
        direction_matrix_vtk = vtk.vtkMatrix3x3()
        direction_matrix_vtk.Identity()
        binary_labelmap.SetDirectionMatrix(direction_matrix_vtk)


    # step 1: pad by 1 pixel all around with 0s, to ensure that the surface is still closed
    # even if the labelmap runs up against the image boundary.
    padder = vtk.vtkImageConstantPad()
    padder.SetInputData(binary_labelmap)
    extent = binary_labelmap.GetExtent()
    padder.SetOutputWholeExtent(
        extent[0] - 1, extent[1] + 1,
        extent[2] - 1, extent[3] + 1,
        extent[4] - 1, extent[5] + 1,
    )
    padder.Update()
    padded_labelmap = padder.GetOutput()

    # step 1: extract surface
    flying_edges = vtk.vtkDiscreteFlyingEdges3D()
    flying_edges.SetInputData(padded_labelmap)
    flying_edges.ComputeGradientsOff()
    flying_edges.ComputeNormalsOff()
    flying_edges.Update()
    surface_mesh = flying_edges.GetOutput()

    # step 2: decimation
    if decimation_factor > 0.0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(surface_mesh)
        decimator.SetFeatureAngle(60)
        decimator.SplittingOff()
        decimator.PreserveTopologyOn()
        decimator.SetMaximumError(1)
        decimator.SetTargetReduction(decimation_factor)
        decimator.Update()
        surface_mesh = decimator.GetOutput()

    # step 3: smoothing
    if smoothing_factor > 0.0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(surface_mesh)

        # map smoothing factor to passband and iterations, copying the approach taken by Slicer
        passband = pow(10.0, -4.0 * smoothing_factor)
        num_iterations = 20 + int(smoothing_factor * 40)

        smoother.SetNumberOfIterations(num_iterations)
        smoother.SetPassBand(passband)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        surface_mesh = smoother.GetOutput()

    # step 4: compute normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface_mesh)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    surface_mesh = normals.GetOutput()

    if parse(vtk.__version__) < parse("9.3.0"):
        # In these older versions of vtk, the labelmap internal affine transform is not used correctly,
        # so we manually apply the transform after the fact
        surface_mesh = apply_affine_to_polydata(
            affine,
            surface_mesh,
        )

    # Some scalars can get tacked on by the above processing for some reason, so remove those in case they are present
    surface_mesh.GetPointData().SetScalars(None)
    return surface_mesh

def spherical_interpolator_from_mesh(
    surface_mesh: vtk.vtkPolyData,
    origin: Tuple[float, float, float] = (0.,0.,0.),
    xyz_direction_columns: np.ndarray | None = None,
    use_embree: bool|None = None,
    dist_tolerance: float = 0.0001,
) -> Callable:
    """Create a spherical interpolator from a vtkPolyData.

    Here a "spherical interpolator" is a function that maps angles from a spherical coordinate system
    to r values (radial spherical coordinate values) by interpolating over a set of known values.
    It's essentially a "spherical plotter."

    Args:
        surface_mesh: The mesh containing the points to be interpolated over
        origin: The origin of the spherical coordinate system
        xyz_direction_columns: A matrix of shape (3,3) the columns of which are unit vectors that describe
            the cartesian x,y,z axis directions on which to base the spherical coordinate system. For example
            the spherical azimuthal angle is the polar angle of the projection of the point into the x-y-plane, etc.
            See the documentation on `spherical_to_cartesian` and `cartesian_to_spherical` for a complete description
            of how the spherical angles relate to the x, y, and z axes. If not provided, the xyz_direction_columns will
            be an identity matrix, which means that the coordinates in which surface_mesh is given will directly be
            interpreted as the x,y,z upon which a spherical coordinate system will be based.
        use_embree: Use an alternative algorithm that uses embree CPU raytracing. Defaults to True only if embree is available;
            it requires x86 architecture.
        dist_tolerance: A vertex of the surface_mesh will only be included if it is the furthest point from the origin
            that is on the mesh along the ray emanating from the origin and passing through the vertex. The
            dist_tolerance is the threshold for determining whether an intersection of the ray with the mesh
            counts as being a distinct further out point from the vertex. This parameter only matters if use_embree is off.

    Returns:
        A spherical interpolator, which is a callable that maps (theta,phi) pairs of spherical coordinates (phi being azimuthal)
        to r values (radial spherical coordinate values). The angles are in radians.
        A spherical interpolator can also run in batch mode, operating on a numpy array of shape (...,2) consisting
        of theta,phi pairs in the last axis.

    Summary of the algorithm:
        - Transform the input mesh based on the desired origin and orientation of the spherical coordinate system.
        - We will gather some points into a set $S$. For each point $P$ on the mesh consider the ray $\\vec{OP}$
          from the origin through $P$ and look at all the intersections of this ray $\\vec{OP}$ with the mesh.
          If none of those intersections are further out from the origin than $P$ is, then we put $P$ into our set $S$.
        - Using the spherical coordinates of the points in $S$, build a `scipy.interpolate.LinearNDInterpolator`
          that interpolates spherical $r$ values from the spherical $(\\theta,\\phi)$ values.
        - Problem: All the gathered $(\\theta,\\phi)$ values are likely strictly inside the square $[0,\\pi]\\times[-\\pi,\\pi]$,
          and `LinearNDInterpolator` does not _extrapolate_, and so angles close to the "seams" of the spherical coordinate
          system (the boundaries of that square) generate NaNs through the interpolator. The solution used here is to first
          clone the gathered points with appropriate angular shifts so as to cover those seams, and then give that larger set
          of points to the interpolator.
        - Return the interpolator.
    """

    if xyz_direction_columns is None:
        xyz_direction_columns = np.eye(3, dtype=float)

    if use_embree is None:
        use_embree = trimesh.ray.has_embree

    xyz_affine = np.eye(4)
    xyz_affine[:3,:3] = xyz_direction_columns
    xyz_affine[:3,3] = origin
    # Now xyz_affine is a coordinate transformation matrix that transforms from the xyz system to the coord system of the vtkPolyData
    # We want to apply the inverse to the vtkPolyData
    xyz_affine_inverse = np.linalg.inv(xyz_affine)
    xyz_affine_inverse_vtkmat = vtk.vtkMatrix4x4()
    xyz_affine_inverse_vtkmat.DeepCopy(xyz_affine_inverse.ravel())
    xyz_inverse_transform = vtk.vtkTransform()
    xyz_inverse_transform.SetMatrix(xyz_affine_inverse_vtkmat)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(xyz_inverse_transform)
    transform_filter.SetInputData(surface_mesh)
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(transform_filter.GetOutputPort())
    triangle_filter.Update()
    surface_mesh_transformed = triangle_filter.GetOutput()

    if use_embree:
        return _spherical_interpolator_from_mesh_embree(surface_mesh_transformed)
    else:
        return _spherical_interpolator_from_mesh_cell_locator(surface_mesh_transformed, dist_tolerance)

def _spherical_interpolator_from_mesh_cell_locator(surface_mesh : vtk.vtkPolyData, dist_tolerance:float) -> Callable:

    spherical_coords_on_mesh : List[Tuple[float,float,float]] = []

    points = surface_mesh.GetPoints()

    # The farthest point from the origin is this far out:
    r_max = np.max([np.sqrt(np.sum(np.array(points.GetPoint(i))**2)) for i in range(points.GetNumberOfPoints())])

    sqdist_tolerance = dist_tolerance**2

    locator = vtk.vtkCellLocator() # Tried vtkOBBTree and it seems vtkCellLocator is much faster for this application
    locator.SetDataSet(surface_mesh)
    locator.BuildLocator()

    for i in range(points.GetNumberOfPoints()):
        point = np.array(points.GetPoint(i))
        point_r_squared = np.sum(point**2)

        # A point that is distance 2*r_max from the origin along the same ray as `point`
        # We will check for intersections along the line segment from `point` to `distant_point_along_same_ray_as_point`
        # The distance 2*r_max is chosen just to ensure that the line segment captures any possible intersection in the infinite
        # ray emanating from `point` outward
        distant_point_along_same_ray_as_point = (2*r_max/np.sqrt(point_r_squared)) * point

        intersection_points = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        locator.IntersectWithLine(
            point, # p1
            distant_point_along_same_ray_as_point, # p2
            0., # tol
            intersection_points, # points
            cell_ids, # cellIds
        )

        point_is_the_furthest_out = True
        for j in range(intersection_points.GetNumberOfPoints()):
            intersection_point = np.array(intersection_points.GetPoint(j))
            sqdist = np.sum((point-intersection_point)**2) # squared distance from point to intersection point
            if sqdist > sqdist_tolerance:
                point_is_the_furthest_out = False
                break

        if point_is_the_furthest_out:
            spherical_coords_on_mesh.append(cartesian_to_spherical(*point)) # append the (r, theta, phi) triple

    spherical_coords_on_mesh = np.array(spherical_coords_on_mesh)

    # We clone the points with a +/- 2pi translation in the phi (azimuthal) coordinate, creating 3 times as many points
    # This will help the LinearNDInterpolator to better handle phi values as they wrap around
    spherical_coords_on_mesh = np.concatenate(
        [
            spherical_coords_on_mesh,
            spherical_coords_on_mesh + np.array([0.,0.,2*np.pi]).reshape((1,3)), # add 2pi to phi coordinate
            spherical_coords_on_mesh - np.array([0.,0.,2*np.pi]).reshape((1,3)), # subtract 2pi from phi coordinate
        ],
        axis=0
    )

    # We clone the points with a pi translation in the phi (azimuthal) coordinate and suitable flips in theta,
    # creating another 3 times as many points. This will help the LinearNDInterpolator to better
    # handle theta values close to the poles (theta=0 and theta=pi).
    spherical_coords_on_mesh = np.concatenate(
        [
            spherical_coords_on_mesh,
            # theta |--> -theta, phi |--> phi+pi, introduces negative theta values
            (spherical_coords_on_mesh * np.array([1.,-1.,1.]).reshape((1,3))) + np.array([0.,0.,np.pi]).reshape((1,3)),
            # theta |--> 2pi-theta, phi |--> phi+pi, introduces theta values greater than pi
            (spherical_coords_on_mesh * np.array([1.,-1.,1.]).reshape((1,3))) + np.array([0.,2*np.pi,np.pi]).reshape((1,3)),
        ],
        axis=0
    )

    interpolator = LinearNDInterpolator(
        points = spherical_coords_on_mesh[:,[1,2]], # The (theta, phi) spherical coordinates
        values = spherical_coords_on_mesh[:,0], # The r spherical coordinates
    )

    return interpolator

def _spherical_interpolator_from_mesh_embree(surface_mesh : vtk.vtkPolyData) -> Callable:
    vtk_points = surface_mesh.GetPoints()
    points_np = vtk_to_numpy(vtk_points.GetData()).astype(np.float64) # (N,3)
    polys = surface_mesh.GetPolys()
    connectivity_np = vtk_to_numpy(polys.GetConnectivityArray())
    if connectivity_np.size == 0:
        raise RuntimeError("Input mesh has no polygons after transformation/triangulation.")
    faces_np = connectivity_np.reshape(-1, 3).astype(np.int64) # (M, 3)

    r_squared = np.sum(points_np**2, axis=1)

    # The farthest point from the origin is this far out:
    r_max = float(np.sqrt(r_squared.max()))

    tm = trimesh.Trimesh(vertices=points_np, faces=faces_np, process=False)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(tm)

    def interpolator(*args):
        if len(args)==2:
            arr = np.array(args)
        elif len(args)==1 and isinstance(args[0], np.ndarray):
            arr = args[0] # expected shape (...,2)
            if arr.shape[-1] != 2:
                msg = f"Interpolator expects array of shape (...,2). Got shape {arr.shape}"
                raise ValueError(msg)
        else:
            raise ValueError("Interpolator expects either two args (theta, phi) or a single numpy array arg shaped (...,2)")

        origins = spherical_to_cartesian_vectorized(
            np.concatenate([np.full(arr.shape[:-1] + (1,), r_max+1),arr], axis=-1) # add r coordinate, giving shape (...,3)
        )

        # intersects_id will expect shape (N,3), but we want to support (...,3), so reshape if needed:
        batch_shape = origins.shape[:-1]
        origins = origins.reshape((-1,3))

        _, _, hit_locations = intersector.intersects_id(origins, -origins, multiple_hits=False, return_locations=True)
        return np.linalg.norm(hit_locations, axis=-1).reshape(batch_shape)

    return interpolator
