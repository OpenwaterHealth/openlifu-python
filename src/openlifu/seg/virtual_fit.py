from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
)

import numpy as np
import vtk

from openlifu.geo.transforms import (
    cartesian_to_spherical,
    cartesian_to_spherical_vectorized,
    spherical_coordinate_basis,
    spherical_to_cartesian,
    spherical_to_cartesian_vectorized,
)
from openlifu.seg.skinseg import (
    apply_affine_to_polydata,
    compute_foreground_mask,
    create_closed_surface_from_labelmap,
    spherical_interpolator_from_mesh,
    vtk_img_from_array_and_affine,
)
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin
from openlifu.util.units import getunitconversion, getunittype

log = logging.getLogger("VirtualFit")

ras2asl_3x3 = np.array([[0,1,0],[0,0,1],[-1,0,0]], dtype=float) # ASL means Anterior-Superior-Left coordinates
asl2ras_3x3 = ras2asl_3x3.transpose()

@dataclass
class VirtualFitOptions(DictMixin):
    """Parameters to configure the `virtual_fit` algorithm.

    The terms 'pitch' and 'yaw' used here refer to the following target-centric angular coordinates in patient space:
        pitch: The angle between the anterior axis through the target and the ray from the target to the projection of
            a given point into the anterior-superior plane.
        yaw: The angle between the anterior-superior plane through the target and the ray from the target to a given point.

    Another way to describe them in terms of standard spherical coordinates centered at the target in ASL (anterior-superior-left) space:
        pitch: The azimuthal spherical coordinate.
        yaw: 90 degrees minus the polar spherical coordinate.
    """

    units: Annotated[str, OpenLIFUFieldData("Length units", "The units of length used in the length attributes of this class")] = "mm"
    """The units of length used in the length attributes of this class"""

    transducer_steering_center_distance: Annotated[float, OpenLIFUFieldData("Steering center distance", "Distance from the transducer origin axially to the center of the steering zone in the units `units`")] = 50.
    """Distance from the transducer origin axially to the center of the steering zone in the units `units`"""

    steering_limits: Annotated[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                               OpenLIFUFieldData("Steering limits", "Steering bounds along each axis from the transducer origin, in the units `units`")] = ((-50, 50), (-50, 50), (-50, 50))
    """Distance from the transducer origin axially to the center of the steering zone in the units `units`"""

    pitch_range: Annotated[Tuple[float, float], OpenLIFUFieldData("Pitch range (deg)", "Range of pitches to include in the transducer fitting search grid, in degrees")] = (-10, 150)
    """Range of pitches to include in the transducer fitting search grid, in degrees"""

    pitch_step: Annotated[float, OpenLIFUFieldData("Pitch step size (deg)", "Pitch step size when forming the transducer fitting search grid, in degrees")] = 5
    """Pitch step size when forming the transducer fitting search grid, in degrees"""

    yaw_range: Annotated[Tuple[float, float], OpenLIFUFieldData("Yaw range (deg)", "Range of yaws to include in the transducer fitting search grid, in degrees")] = (-65, 65)
    """Range of yaws to include in the transducer fitting search grid, in degrees"""

    yaw_step: Annotated[float, OpenLIFUFieldData("Yaw step size (deg)", "Yaw step size when forming the transducer fitting search grid, in degrees")] = 5
    """Yaw step size when forming the transducer fitting search grid, in degrees"""

    planefit_dyaw_extent: Annotated[float, OpenLIFUFieldData("Plane fit yaw extent", "Left and right extents of the point grid to be used for plane fitting along the local yaw axes, in units of `units`")] = 15
    """Left and right extents of the point grid to be used for plane fitting along the local yaw axes,
    in units of `units`. The plane fitting point grid will be twice this size, since this is left
    and right extents. (Note that this has units of length, not angle!)"""

    planefit_dyaw_step: Annotated[float, OpenLIFUFieldData("Plane fit yaw step", "Local yaw axis step size to use when constructing plane fitting grids. In spatial units of `units`")] = 3
    """Local yaw axis step size to use when constructing plane fitting grids. In spatial units of `units`."""

    planefit_dpitch_extent: Annotated[float, OpenLIFUFieldData("Plane fit pitch extent", "Left and right extents of the point grid to be used for plane fitting along the local pitch axes, in spatial units of `units`")] = 15
    """Left and right extents of the point grid to be used for plane fitting along the local pitch axes,
    in spatial units of `units`. The plane fitting point grid will be twice this size, since this is left
    and right extents."""

    planefit_dpitch_step: Annotated[float, OpenLIFUFieldData("Plane fit pitch step", "Local pitch axis step size to use when constructing plane fitting grids. In spatial units of `units`")] = 3
    """Local pitch axis step size to use when constructing plane fitting grids. In spatial units of `units`."""

    top_n_candidates: Annotated[int, OpenLIFUFieldData("No. of candidates returned", "Sets the limit for the number of transducer transform candidates returned by the algorithm.")] = 4
    """Sets the limit for the number of transducer transform candidates returned by the algorithm."""

    def __post_init__(self):
        if not isinstance(self.units, str):
            raise TypeError("Units must be a string")
        if getunittype(self.units) != 'distance':
            raise ValueError(f"Units must be a length unit, got {self.units}")
        if self.transducer_steering_center_distance <= 0:
            raise ValueError("Transducer steering center distance must be greater than 0")
        if not isinstance(self.transducer_steering_center_distance, int | float):
            raise TypeError("Transducer steering center distance must be a number")
        if not isinstance(self.steering_limits, tuple) or len(self.steering_limits) != 3:
            raise ValueError("Steering limits must be a tuple of three tuples, each with two elements")
        if not all(isinstance(sl, tuple) and len(sl) == 2 for sl in self.steering_limits):
            raise ValueError("Each steering limit must be a tuple of two elements")
        if not all(isinstance(sl[0], int | float) and isinstance(sl[1], int | float) for sl in self.steering_limits):
            raise ValueError("Each steering limit must contain two numbers")
        if not isinstance(self.pitch_range, tuple) or len(self.pitch_range) != 2:
            raise ValueError("Pitch range must be a tuple of two elements")
        if not all(isinstance(pr, int | float) for pr in self.pitch_range):
            raise ValueError("Pitch range must contain two numbers")
        if not isinstance(self.pitch_step, int | float):
            raise TypeError("Pitch step must be a number")
        if self.pitch_step <= 0:
            raise ValueError("Pitch step must be greater than 0")
        if not isinstance(self.yaw_range, tuple) or len(self.yaw_range) != 2:
            raise ValueError("Yaw range must be a tuple of two elements")
        if not all(isinstance(yr, int | float) for yr in self.yaw_range):
            raise ValueError("Yaw range must contain two numbers")
        if not isinstance(self.yaw_step, int | float):
            raise TypeError("Yaw step must be a number")
        if self.yaw_step <= 0:
            raise ValueError("Yaw step must be greater than 0")
        if not isinstance(self.planefit_dyaw_extent, int | float):
            raise TypeError("Plane fit yaw extent must be a number")
        if self.planefit_dyaw_extent <= 0:
            raise ValueError("Plane fit yaw extent must be greater than 0")
        if not isinstance(self.planefit_dyaw_step, int | float):
            raise TypeError("Plane fit yaw step must be a number")
        if self.planefit_dyaw_step <= 0:
            raise ValueError("Plane fit yaw step must be greater than 0")
        if not isinstance(self.planefit_dpitch_extent, int | float):
            raise TypeError("Plane fit pitch extent must be a number")
        if self.planefit_dpitch_extent <= 0:
            raise ValueError("Plane fit pitch extent must be greater than 0")
        if not isinstance(self.planefit_dpitch_step, int | float):
            raise TypeError("Plane fit pitch step must be a number")
        if not isinstance(self.top_n_candidates, int ):
            raise TypeError("Number of transducer transform candidates returned must be an integer")
        if self.top_n_candidates <= 0:
            raise ValueError("Number of transducer transform candidates returned must be greater than 0")

    def to_units(self, target_units: str) -> VirtualFitOptions:
        """Do unit conversion and return a version of this VirtualFitOptions that uses
        `target_units` as the units for all attributes that have units of length."""
        conversion_factor = getunitconversion(from_unit = self.units, to_unit=target_units)
        return VirtualFitOptions(
            units = target_units,
            transducer_steering_center_distance = conversion_factor * self.transducer_steering_center_distance,
            steering_limits = tuple(map(tuple,conversion_factor*np.array(self.steering_limits))),
            pitch_range = self.pitch_range,
            pitch_step = self.pitch_step,
            yaw_range = self.yaw_range,
            yaw_step = self.yaw_step,
            planefit_dyaw_extent = conversion_factor * self.planefit_dyaw_extent,
            planefit_dyaw_step = conversion_factor * self.planefit_dyaw_step,
            planefit_dpitch_extent = conversion_factor * self.planefit_dpitch_extent,
            planefit_dpitch_step = conversion_factor * self.planefit_dpitch_step,
            top_n_candidates = self.top_n_candidates,
        )

    @staticmethod
    def from_dict(parameter_dict: Dict[str,Any]) -> VirtualFitOptions: # Override DictMixin here
        parameter_dict["pitch_range"] = tuple(parameter_dict["pitch_range"])
        parameter_dict["yaw_range"] = tuple(parameter_dict["yaw_range"])
        parameter_dict["steering_limits"] = tuple(map(tuple,parameter_dict["steering_limits"]))
        return VirtualFitOptions(**parameter_dict)

def compute_skin_mesh_from_volume(
    volume_array : np.ndarray,
    volume_affine_RAS : np.ndarray,
) -> vtk.vtkPolyData:
    log.info("Computing foreground mask...")
    foreground_mask_array = compute_foreground_mask(volume_array)
    foreground_mask_vtk_image = vtk_img_from_array_and_affine(foreground_mask_array, volume_affine_RAS)
    log.info("Creating closed surface from labelmap...")
    skin_mesh = create_closed_surface_from_labelmap(foreground_mask_vtk_image)
    return skin_mesh

@dataclass
class VirtualFitDebugInfo:
    """Debugging information for the result of running `virtual_fit`."""
    skin_mesh : vtk.vtkPolyData
    """The skin mesh that was used for virtual fitting"""

    spherically_interpolated_mesh : vtk.vtkPolyData
    """A mesh representing the spherical interpolator that was used for virtual fitting"""

    search_points : np.ndarray
    """Array of shape (N,3) containing the coordinates of the points that were tried for virtual fitting"""

    plane_normals : np.ndarray
    """Array of shape (N,3) containing the normal vectors of the planes that were fitted at each of `search_points`"""

    steering_dists : np.ndarray
    """Array of shape (N,) containing the computed steering distance for each point in `search_points`"""

    in_bounds : np.ndarray
    """Boolean array of shape (N,) giving for each point in `search_points` whether the target was determined to be
    in bounds for that candidate transducer placement."""


def sphere_from_interpolator(
        interpolator: Callable[[float, float], float],
        theta_res:int = 50,
        phi_res:int = 50,
    ) -> vtk.vtkPolyData:
    """Create a spherical mesh from a spherical interpolator, to help visualize how the interpolator works.
    This is intended as a debugging utility."""
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(1.0)
    sphere_source.SetThetaResolution(theta_res)
    sphere_source.SetPhiResolution(phi_res)
    sphere_source.Update()
    sphere_polydata = sphere_source.GetOutput()
    sphere_points = sphere_polydata.GetPoints()
    for i in range(sphere_points.GetNumberOfPoints()):
        point = np.array(sphere_points.GetPoint(i))
        r, theta, phi = cartesian_to_spherical(*point)
        r = interpolator(theta, phi)
        sphere_points.SetPoint(i, r * point)
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(sphere_polydata)
    normals_filter.Update()
    return normals_filter.GetOutput()

def run_virtual_fit(
    units: str,
    target_RAS : Sequence[float],
    standoff_transform : np.ndarray,
    options : VirtualFitOptions,
    volume_array : np.ndarray | None = None,
    volume_affine_RAS : np.ndarray | None = None,
    skin_mesh : vtk.vtkPolyData | None = None,
    include_debug_info : bool = False,
    progress_callback : Callable[[int,str],None] | None = None,
) -> List[np.ndarray] | Tuple[List[np.ndarray], VirtualFitDebugInfo]:
    """Run patient-specific "virtual fitting" algorithm, suggesting a series of candidate transducer
    transforms for optimal sonicaiton of a given target.

    Provide either a `volume_array` and `volume_affine_RAS`, or a `skin_mesh`.

    Args:
        units: The spatial units of the RAS space into which volume_affine_RAS maps
        target_RAS: A 3D point, in the coordinates and units of `volume_affine_RAS` (the `units` argument)
        standoff_transform: See the documentation of `create_standoff_transform` or
            `Transducer.standoff_transform` for the meaning of this. Here it should be provided in the
            units `units`. The method `Transducer.get_standoff_transform_in_units` is useful for getting this.
        options : Virtual fitting algorithm configuration. See the `VirtualFitOptions` documentation.
        volume_array: A 3D volume MRI
        volume_affine_RAS: A 4x4 affine transform that maps `volume_array` into RAS space with certain units
        skin_mesh: Optional pre-computed closed surface mesh. If provided, `volume_array` and
            `volume_affine_RAS` can be omitted. The provided skin mesh should be in RAS space, with units
            being the provided `units` arg. The function `compute_skin_mesh_from_volume` can be used to pre-compute
            a skin mesh.
        include_debug_info: Whether to include debugging info in the return value. Disabled by default because some of the debugging
            info takes some time to compute.
        progress_callback: An optional function that will be called to report progress. The function should accept two arguments:
            an integer progress value from 0 to 100 followed by a string message describing the step currently being worked on.

    Returns: A list of transducer transform candidates sorted starting from the best-scoring one. The transforms map transducer space
        into LPS space, and they are in the same units as the RAS space of `volume_affine_RAS` (aka the `units` argument).
    """

    if progress_callback is None:
        def progress_callback(progress_percent : int, step_description : str): # noqa: ARG001
            pass # Define it to be a no-op if no progress_callback was provided.

    progress_callback(0, "Starting virtual fit")

    # Express all virtual fit options in the units of volume_affine_RAS, i.e. the physical space of the volume
    options = options.to_units(units)
    pitch_range = options.pitch_range
    pitch_step = options.pitch_step
    yaw_range = options.yaw_range
    yaw_step = options.yaw_step
    transducer_steering_center_distance = options.transducer_steering_center_distance
    steering_limits = options.steering_limits
    planefit_dyaw_extent = options.planefit_dyaw_extent
    planefit_dyaw_step = options.planefit_dyaw_step
    planefit_dpitch_extent = options.planefit_dpitch_extent
    planefit_dpitch_step = options.planefit_dpitch_step
    top_n_candidates = options.top_n_candidates


    if skin_mesh is None:
        if volume_array is None or volume_affine_RAS is None:
            raise ValueError("Both `volume_array` and `volume_affine_RAS` must be provided if `skin_mesh` is None.")

        log.info("Computing skin mesh...")
        progress_callback(0, "Computing skin mesh")
        skin_mesh = compute_skin_mesh_from_volume(volume_array, volume_affine_RAS)
    else:
        log.info("Using provided skin mesh.")


    log.info("Building skin interpolator...")
    progress_callback(5, "Building skin interpolator")
    skin_interpolator = spherical_interpolator_from_mesh(
        surface_mesh = skin_mesh,
        origin = target_RAS,
        xyz_direction_columns = asl2ras_3x3, # surface mesh was in RAS, so here spherical coordinates are placed on ASL space
    )

    # Useful transforms to and from the skin_interpolator ASL space and between RAS and LPS
    # Note that ASL is a left-handed coordinate system while RAS and LPS are right-handed.
    interpolator2ras = np.eye(4)
    interpolator2ras[:3,:3] = asl2ras_3x3
    interpolator2ras[:3,3] = target_RAS
    ras_lps_swap = np.diag([-1.,-1,1,1])
    interpolator2lps = ras_lps_swap @ interpolator2ras

    # Useful arrays for vectorized comparisons
    steering_mins = np.array([sl[0] for sl in steering_limits], dtype=float) # shape (3,). It is the lat,ele,ax steering min
    steering_maxs = np.array([sl[1] for sl in steering_limits], dtype=float) # shape (3,). It is the lat,ele,ax steering max

    log.info("Searching through candidate transducer poses...")
    progress_callback(50, "Searching through poses")

    # Construct search grid
    theta_sequence = np.arange(90 - yaw_range[-1], 90 - yaw_range[0], yaw_step)
    phi_sequence = np.arange(pitch_range[0], pitch_range[-1], pitch_step)
    theta_grid, phi_grid = np.meshgrid(theta_sequence, phi_sequence, indexing="ij") # each has shape (number of thetas, number of phis)
    num_thetas, num_phis = theta_grid.shape
    num_search_points = num_thetas*num_phis
    thetas = theta_grid.reshape(num_search_points)
    phis = phi_grid.reshape(num_search_points)

    # Things that will be computed over the search grid
    transducer_poses = np.empty((num_search_points,4,4), dtype=float)
    in_bounds = np.zeros(shape=num_search_points, dtype=bool)
    steering_dists = np.zeros(shape=num_search_points, dtype=float)

    # Additional debugging info that will be computed over the search grid
    points_asl = np.zeros((num_search_points,3), dtype=float) # search grid points in ASL coordinates
    normals_asl = np.zeros((num_search_points,3), dtype=float) # normal vector of the plane that is fitted at each point, in ASL coordinates

    # Pre-build plane fitting grid that will be used repeatedly in the search loop below.
    # This grid lives in the spherical coordinate basis theta-phi plane at each point.
    # Below we will project it onto the skin surface for each search point.
    dtheta_sequence = np.arange(-planefit_dyaw_extent, planefit_dyaw_extent + planefit_dyaw_step, planefit_dyaw_step)
    dphi_sequence = np.arange(-planefit_dpitch_extent, planefit_dpitch_extent + planefit_dpitch_step, planefit_dpitch_step)
    dtheta_grid, dphi_grid = np.meshgrid(dtheta_sequence, dphi_sequence, indexing='ij')

    for i in range(num_search_points):
        theta_rad, phi_rad = thetas[i]*np.pi/180, phis[i]*np.pi/180

        # Cartesian coordinate location of the point at which we are fitting a plane
        point = np.array(spherical_to_cartesian(skin_interpolator(theta_rad, phi_rad), theta_rad, phi_rad))

        r_hat, theta_hat, phi_hat = spherical_coordinate_basis(theta_rad,phi_rad)
        planefit_points_unprojected_cartesian = (
            point.reshape((1,1,3))
            + dtheta_grid[...,np.newaxis] * theta_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
            + dphi_grid[...,np.newaxis] * phi_hat.reshape(1,1,3) # shape (num dthetas, num dphis, 3)
        ) # shape (num dthetas, num dphis, 3)

        planefit_points_unprojected_spherical = cartesian_to_spherical_vectorized(
            planefit_points_unprojected_cartesian
        ) # shape (num dthetas, num dphis, 3)
        skin_projected_r_values = skin_interpolator(planefit_points_unprojected_spherical[...,1:]) # shape (num dthetas, num dphis)
        planefit_points_cartesian = spherical_to_cartesian_vectorized( # Could instead renormalize planefit_points_unprojected_cartesian, not sure if it would give a speedup versus this
            np.stack([
                skin_projected_r_values, # New r values after projection to skin
                planefit_points_unprojected_spherical[...,1], # Same old theta values
                planefit_points_unprojected_spherical[...,2], # Same old phi values
            ], axis=-1)
        )

        # Fit the best plane to these points among the planes that pass through `point`. Here we find the normal vector to the plane.
        plane_normal = np.linalg.svd(
            planefit_points_cartesian.reshape(-1,3)-point.reshape(1,3),
            full_matrices=False, # we don't need the left-singular vectors anyway, so this speeds things up
        ).Vh[-1] # The right-singular vector corresponding to the smallest singular value


        # Transducer axial axis: Parallel to plane_normal, but points towards rather than away from the origin.
        plane_normal_norm = np.linalg.norm(plane_normal)
        if plane_normal_norm < 1e-10:
            continue # Bad geometry at this location, so it's not a virtual fit candidate
        transducer_z = - np.sign(np.dot(plane_normal,point)) * plane_normal / plane_normal_norm

        # Transducer elevational axis: Phi-hat, but then with its component along transducer_z eliminated. This orients the transducer "up" if this were forehead, for example.
        transducer_y = phi_hat - np.dot(phi_hat, transducer_z) * transducer_z
        transducer_y_norm = np.linalg.norm(transducer_y)
        if transducer_y_norm < 1e-10:
            continue # Bad geometry at this location, so it's not a virtual fit candidate
        transducer_y = transducer_y / transducer_y_norm

        # Transducer lateral axis, here simply the only remaining choice to keep it a left handed coordinate system
        # (ASL is left-handed, so the transducer axes must be left-handed to make for an orientation-preserving transducer transform)
        transducer_x = np.cross(transducer_z, transducer_y)

        transducer_transform = np.array(
            [
                [*transducer_x, 0],
                [*transducer_y, 0],
                [*transducer_z, 0],
                [*point, 1],
            ],
            dtype=float
        ).transpose()

        # The transform moves the transducer into the ASL skin interpolator space.
        # We want a transform that moves the transducer into LPS space, and we also want to apply the standoff transform
        transducer_transform = interpolator2lps @ transducer_transform @ standoff_transform

        # Target in transducer coordinates (lat, ele, ax)
        target_XYZ = (np.linalg.inv(transducer_transform) @ ras_lps_swap @ np.array([*target_RAS,1.0]))[:3]

        # Target in "steering space", where the origin is the center of the steering zone.
        target_steering_space = target_XYZ - np.array([0.,0.,transducer_steering_center_distance])

        steering_distance : float = float(np.linalg.norm(target_steering_space))

        # Check whether the target is in the steering range
        target_in_bounds : bool = bool(np.all((steering_mins < target_steering_space) & (target_steering_space < steering_maxs)))

        # Finally, fill out the arrays we have been building in this loop
        transducer_poses[i] = transducer_transform
        steering_dists[i] = steering_distance
        in_bounds[i] = target_in_bounds
        points_asl[i] = point
        normals_asl[i] = transducer_z


    sorted_transforms = [
        x[0] for x in sorted(zip(transducer_poses[in_bounds],steering_dists[in_bounds]), key = lambda x : x[1])
    ]

    log.info("Virtual fitting complete.")

    if include_debug_info:
        log.info("Generating debug meshes...")
        progress_callback(80, "Generating debug meshes")
        interpolator_mesh : vtk.vtkPolyData = sphere_from_interpolator(skin_interpolator, theta_res=100, phi_res=100)

        # A few things are in ASL coordinates, so we transform it to RAS space so that they are in the same coordinates as skin_mesh.
        interpolator_mesh = apply_affine_to_polydata(interpolator2ras, interpolator_mesh)
        points_asl_homogenized = np.concatenate([points_asl.T, np.ones((1,num_search_points))], axis=0) # shape (4,num_search_points)
        points_ras = (interpolator2ras @ points_asl_homogenized)[:3].T # back to shape (num_search_points,3)
        normals_ras = (asl2ras_3x3 @ (normals_asl.T)).T

        # After transforming the interpolator_mesh, the normals can end up flipped, so we fix it just in case
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(interpolator_mesh)
        normals_filter.Update()
        interpolator_mesh = normals_filter.GetOutput()

        progress_callback(100, "Complete")
        return (
            sorted_transforms[:top_n_candidates],
            VirtualFitDebugInfo(
                skin_mesh = skin_mesh,
                spherically_interpolated_mesh = interpolator_mesh,
                search_points = points_ras,
                plane_normals = normals_ras,
                steering_dists = steering_dists,
                in_bounds = in_bounds,
            ),
        )

    progress_callback(100, "Complete")
    return sorted_transforms[:top_n_candidates]
