from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa
from scipy.ndimage import distance_transform_edt, gaussian_filter

from openlifu.seg.material import MATERIALS, Material
from openlifu.seg.seg_method import SegmentationMethod
from openlifu.seg.skinseg import compute_foreground_mask
from openlifu.util.annotations import OpenLIFUFieldData

# Literature-backed material definitions for brain tissue subtypes.
#
# Sound speed: Labuda et al. 2022 (Ultrasonics, vol. 124, 106742) found no
# significant difference in speed of sound between gray and white matter,
# reporting 1532-1541 m/s across specimens. We use 1540 m/s for both GM and
# WM, consistent with the Aubry 2022 JASA benchmark (1560 m/s for
# undifferentiated brain) and IT'IS (1546 m/s). CSF speed of sound from
# IT'IS Foundation via Mueller et al. 2016 (J Neural Eng 13(5):056002).
#
# Density: IT'IS Foundation (BfS-2018-3615S82433 report).
#   GM: 1045 kg/m3, WM: 1041 kg/m3, CSF: 1007 kg/m3.
#
# Attenuation: Labuda et al. 2022 measured significantly higher attenuation
# in WM than GM. Values in Np/m from BabelBrain (Pichardo), derived by
# averaging IT'IS (0.25-0.75 MHz) and Labuda 2022 (3.5-10 MHz):
#   GM: 4.40 Np/m @ 1MHz -> 0.38 dB/cm/MHz
#   WM: 10.18 Np/m @ 1MHz -> 0.88 dB/cm/MHz
#   CSF: ~0.099 Np/m -> 0.009 dB/cm/MHz (essentially negligible)
# Conversion: Np/m * (8.686 dB/Np) / (100 cm/m) = dB/cm; at 1 MHz this
# equals dB/cm/MHz directly.
#
# Thermal properties: IT'IS Foundation (specific heat, thermal conductivity).
CSF = Material(
    name="csf",
    sound_speed=1507.0,
    density=1007.0,
    attenuation=0.009,
    specific_heat=4096.0,
    thermal_conductivity=0.57,
)
GRAY_MATTER = Material(
    name="gray_matter",
    sound_speed=1540.0,
    density=1045.0,
    attenuation=0.38,
    specific_heat=3696.0,
    thermal_conductivity=0.55,
)
WHITE_MATTER = Material(
    name="white_matter",
    sound_speed=1540.0,
    density=1041.0,
    attenuation=0.88,
    specific_heat=3583.0,
    thermal_conductivity=0.48,
)

_BASE_MATERIAL_KEYS = frozenset({"water", "skull", "air"})
_BRAIN_UNIFORM_KEYS = frozenset({"tissue"})
_BRAIN_CLASSIFIED_KEYS = frozenset({"csf", "gray_matter", "white_matter"})


def _default_materials_uniform() -> dict[str, Material]:
    return MATERIALS.copy()


def _default_materials_classified() -> dict[str, Material]:
    """Default materials dict when brain tissue classification is enabled."""
    m = MATERIALS.copy()
    # Remove the generic "tissue" and add the three brain subtypes.
    m.pop("tissue", None)
    m["csf"] = CSF
    m["gray_matter"] = GRAY_MATTER
    m["white_matter"] = WHITE_MATTER
    return m


@dataclass
class ThresholdMRI(SegmentationMethod):
    """MRI-based segmentation using intensity thresholding and distance transforms.

    Segments an MRI volume into skull (outer shell), air (low-intensity
    cavities), and brain tissue regions, with optional sub-classification
    of brain tissue into CSF, gray matter, and white matter.

    When ``classify_brain_tissues`` is False (default), the brain interior
    is labeled as a single "tissue" material. The segmentation assigns four
    tissue types: water, skull, tissue, and air. When True, the brain
    interior is split into CSF, gray matter, and white matter using a
    3-component Gaussian Mixture Model (EM-GMM) on T1-weighted intensity,
    with optional spatial regularization. Two regularization modes are
    available: Markov Random Field (MRF) via Iterated Conditional Modes
    (ICM) when ``mrf_beta > 0``, or Gaussian smoothing of posterior
    probability maps when ``spatial_sigma_mm > 0``. Both are disabled by
    default. MRF regularization preserves tissue boundaries better than
    Gaussian smoothing by penalizing label disagreement among 6-connected
    neighbors without blurring probability maps. This yields a total of six assigned tissue types:
    water, skull, csf, gray_matter, white_matter, and air. Additional
    materials in the dict (e.g. standoff) are carried through but not
    assigned by the segmentation algorithm.

    The algorithm assumes standard T1-weighted MRI contrast where the head
    is brighter than background, and within the brain, white matter is
    brightest, gray matter intermediate, and CSF darkest. Volumes with
    different contrast (e.g. T2-weighted) will not classify brain tissues
    correctly.

    Each axis of the input volume must have at least 2 coordinate values
    so that voxel spacing can be computed. Coordinate spacing is assumed
    to be uniform along each axis.
    """

    skull_thickness_mm: Annotated[
        float,
        OpenLIFUFieldData(
            "Skull thickness (mm)",
            "Approximate skull thickness used to define the skull shell via erosion",
        ),
    ] = 12.0
    """Approximate skull thickness in millimeters used to define the skull shell via erosion."""

    air_threshold_quantile: Annotated[
        float,
        OpenLIFUFieldData(
            "Air threshold quantile",
            "Quantile of brain-interior voxel intensities below which voxels are classified as air. "
            "An additional absolute-intensity guard requires the voxel to also be below 10% of the "
            "median brain intensity, preventing dark-but-not-air tissue (e.g. CSF) from being "
            "misclassified as air.",
        ),
    ] = 0.05
    """Quantile of brain-interior voxel intensities below which voxels are classified as air.

    A voxel is only labeled air if its intensity is both below this quantile
    threshold and below 10% of the median brain intensity. This dual condition
    prevents CSF (dark but with meaningful signal) from being classified as air."""

    classify_brain_tissues: Annotated[
        bool,
        OpenLIFUFieldData(
            "Classify brain tissues",
            "If True, classify brain interior into CSF, gray matter, and white matter "
            "using a spatially-regularized EM-GMM on T1-weighted intensity",
        ),
    ] = False
    """If True, sub-classify brain interior into CSF, gray matter, and white matter."""

    bias_correction_sigma_mm: Annotated[
        float,
        OpenLIFUFieldData(
            "Bias correction sigma (mm)",
            "Gaussian smoothing sigma for homomorphic bias field correction. "
            "Set to 0 to disable. Only applied when classify_brain_tissues is True.",
        ),
    ] = 30.0
    """Sigma in mm for bias field estimation. Larger values capture broader gradients.
    Set to 0 to disable bias correction. Only used when classify_brain_tissues is True."""

    use_n4: Annotated[
        bool,
        OpenLIFUFieldData(
            "Use N4 bias correction",
            "If True and SimpleITK is installed, use N4BiasFieldCorrection "
            "instead of homomorphic correction. Requires SimpleITK.",
        ),
    ] = True
    """If True, use SimpleITK N4BiasFieldCorrection for bias correction.
    Falls back to homomorphic if SimpleITK is not available."""

    spatial_sigma_mm: Annotated[
        float,
        OpenLIFUFieldData(
            "Spatial regularization sigma (mm)",
            "Gaussian smoothing sigma for spatial regularization of brain tissue "
            "probability maps. Set to 0 to disable. Only used when classify_brain_tissues is True "
            "and mrf_beta is 0.",
        ),
    ] = 0.0
    """Gaussian smoothing sigma (mm) for spatial regularization.
    Defaults to 0 (disabled). Non-zero values apply masked Gaussian
    smoothing to posterior probability maps before label assignment.
    Smoothing is confined to the parenchyma mask to prevent boundary leakage.
    Ignored when mrf_beta > 0 (MRF regularization takes precedence)."""

    mrf_beta: Annotated[
        float,
        OpenLIFUFieldData(
            "MRF regularization strength",
            "Markov Random Field regularization parameter (beta) for brain tissue "
            "classification. Controls the penalty for label disagreement among "
            "6-connected neighbors via Iterated Conditional Modes (ICM). "
            "Set to 0 to disable MRF and fall back to Gaussian smoothing. "
            "Typical values are 0.1-0.3. Only used when classify_brain_tissues is True.",
        ),
    ] = 0.0
    """MRF regularization strength (beta).
    Defaults to 0 (disabled). When positive, Iterated Conditional Modes (ICM) is used
    for spatial regularization instead of Gaussian smoothing. Higher values
    produce smoother label maps but may over-regularize thin structures.
    Set to 0 to disable MRF and use Gaussian smoothing (spatial_sigma_mm) instead."""

    brain_extraction_margin_mm: Annotated[
        float,
        OpenLIFUFieldData(
            "Brain extraction margin (mm)",
            "Extra erosion margin beyond the skull shell for the GMM classification mask. "
            "Excludes dura, meninges, and cortical boundary tissues from the mixture model "
            "to improve tissue classification accuracy. Labels are expanded back to the full "
            "brain mask via nearest-neighbor assignment. Only used when classify_brain_tissues is True.",
        ),
    ] = 0.0
    """Extra erosion in mm beyond the skull boundary for brain tissue classification.
    On whole-head data, voxels just inside the skull (dura, meninges, cortical CSF)
    have intensities that overlap with GM/WM and can contaminate the GMM. This margin
    defines a tighter mask for the GMM fit; after classification, labels are propagated
    back to the full brain mask using nearest-neighbor assignment. Set to 0 to disable.
    Only used when classify_brain_tissues is True."""

    refine_skull_intensity: Annotated[
        bool,
        OpenLIFUFieldData(
            "Refine skull with intensity",
            "If True, refine the EDT skull shell using Otsu intensity thresholding "
            "to separate dark bone from bright soft tissue within the shell. "
            "Morphological closing bridges the diploe gap, and only the largest "
            "connected component is retained. Non-bone shell voxels sufficiently "
            "far from the surface are reclassified as brain tissue.",
        ),
    ] = True
    """If True, refine the skull shell by applying Otsu thresholding to
    separate dark bone voxels from bright soft tissue within the EDT erosion
    shell. Morphological closing bridges the diploe gap (bright marrow between
    inner and outer cortical tables), and only the largest connected component
    is retained. Non-bone shell voxels that are interior enough are reclassified
    as brain tissue."""

    def __post_init__(self) -> None:
        # Let the base class normalize materials=None -> MATERIALS.copy()
        # before we attempt the auto-swap.
        super().__post_init__()

        # If brain classification is enabled and the materials dict has
        # "tissue" but not the classified keys, copy the dict (to avoid
        # mutating a caller-owned object) and swap tissue for the three
        # brain subtypes. This preserves user customizations to other
        # materials (e.g., custom skull or water properties).
        if self.classify_brain_tissues and "tissue" in self.materials and "csf" not in self.materials:
            self.materials = dict(self.materials)
            self.materials.pop("tissue")
            self.materials.setdefault("csf", CSF)
            self.materials.setdefault("gray_matter", GRAY_MATTER)
            self.materials.setdefault("white_matter", WHITE_MATTER)
            if self.ref_material not in self.materials:
                msg = (
                    f"ref_material '{self.ref_material}' was removed during "
                    f"brain tissue auto-swap. Use 'water' or another key in "
                    f"{set(self.materials.keys())}."
                )
                raise ValueError(msg)

        if self.skull_thickness_mm < 0:
            msg = f"skull_thickness_mm must be non-negative, got {self.skull_thickness_mm}."
            raise ValueError(msg)
        if not 0 <= self.air_threshold_quantile <= 1:
            msg = f"air_threshold_quantile must be between 0 and 1, got {self.air_threshold_quantile}."
            raise ValueError(msg)
        if self.bias_correction_sigma_mm < 0:
            msg = f"bias_correction_sigma_mm must be non-negative, got {self.bias_correction_sigma_mm}."
            raise ValueError(msg)
        if self.spatial_sigma_mm < 0:
            msg = f"spatial_sigma_mm must be non-negative, got {self.spatial_sigma_mm}."
            raise ValueError(msg)
        if self.mrf_beta < 0:
            msg = f"mrf_beta must be non-negative, got {self.mrf_beta}."
            raise ValueError(msg)
        if self.brain_extraction_margin_mm < 0:
            msg = f"brain_extraction_margin_mm must be non-negative, got {self.brain_extraction_margin_mm}."
            raise ValueError(msg)

        required = _BASE_MATERIAL_KEYS | (
            _BRAIN_CLASSIFIED_KEYS if self.classify_brain_tissues else _BRAIN_UNIFORM_KEYS
        )
        missing = required - set(self.materials.keys())
        if missing:
            msg = f"ThresholdMRI requires material keys {required}, missing: {missing}."
            raise ValueError(msg)

    def _segment(self, volume: xa.DataArray) -> xa.DataArray:
        """Segment an MRI volume into tissue regions.

        :param volume: An xarray DataArray containing the MRI volume data
        :returns: An xarray DataArray with integer labels matching the ordering
            from ``self._material_indices()``
        """
        data: np.ndarray = volume.to_numpy()
        material_idx = self._material_indices()
        water_label = material_idx["water"]

        # Replace NaN values with 0 to avoid crashes in thresholding and
        # connected-component analysis within compute_foreground_mask().
        # Only copy the array if modification is actually needed.
        has_nan = np.isnan(data).any()
        if has_nan:
            data = data.copy()
            nan_count = int(np.isnan(data).sum())
            logging.warning(
                f"Volume contains {nan_count} NaN voxels; "
                "replacing with 0 for segmentation.",
            )
            data[np.isnan(data)] = 0.0

        # Each axis must have at least 2 coordinates for spacing computation.
        for dim in volume.dims:
            if len(volume.coords[dim]) < 2:
                msg = (
                    f"Axis '{dim}' has fewer than 2 coordinates; "
                    "cannot compute voxel spacing for segmentation."
                )
                raise ValueError(msg)
        # Extract voxel spacing from the first coordinate interval per axis.
        # Assumes uniform spacing along each dimension.
        spacing = np.array([
            float(np.abs(volume.coords[dim].to_numpy()[1] - volume.coords[dim].to_numpy()[0]))
            for dim in volume.dims
        ])

        # Step 1: Compute foreground (head) mask.
        # If the volume has no intensity variation (e.g. all zeros or uniform),
        # Otsu thresholding will fail. Return all-water in that case.
        if float(data.max() - data.min()) == 0:
            seg = np.full(data.shape, water_label, dtype=int)
            return xa.DataArray(seg, coords=volume.coords, dims=volume.dims)

        # compute_foreground_mask uses Otsu thresholding and connected-component
        # analysis. It can raise ValueError (from threshold_otsu on degenerate
        # histograms) or IndexError (from regionprops on empty label maps) in
        # edge cases not caught by the intensity range check above.
        try:
            foreground: np.ndarray = compute_foreground_mask(data)
        except (ValueError, IndexError):
            logging.warning(
                "Foreground mask computation failed; returning all-water segmentation.",
            )
            seg = np.full(data.shape, water_label, dtype=int)
            return xa.DataArray(seg, coords=volume.coords, dims=volume.dims)

        # Step 2: Determine effective skull thickness.
        # Detect skull-stripped input: if foreground covers most of the
        # non-zero voxels (ratio > 0.80), the volume is likely skull-stripped
        # (brain only, no skull). In that case, override skull thickness to 0
        # to avoid mislabeling cortical gray matter as skull. This heuristic
        # has a wide margin: skull-stripped data clusters at ratio ~1.0 while
        # whole-head data is below 0.50.
        effective_skull_mm = self.skull_thickness_mm
        nonzero_count = int(np.sum(data > 0))
        foreground_count_check = int(np.sum(foreground))
        if (
            effective_skull_mm > 0
            and nonzero_count > 0
            and foreground_count_check / nonzero_count > 0.80
            and data.size > 1000000  # skip heuristic on small/synthetic volumes
        ):
            logging.warning(
                "Input appears to be skull-stripped (foreground covers "
                f"{foreground_count_check / nonzero_count:.0%} of non-zero voxels). "
                "Overriding skull_thickness_mm to 0.",
            )
            effective_skull_mm = 0.0

        # Erode the foreground inward by effective_skull_mm to get brain interior.
        # distance_transform_edt computes the distance of each foreground voxel
        # to the nearest background voxel, accounting for anisotropic spacing.
        foreground_dist: np.ndarray = distance_transform_edt(foreground, sampling=spacing)
        brain_mask: np.ndarray = foreground_dist > effective_skull_mm

        # Warn if the brain mask is suspiciously small.
        foreground_count = int(np.sum(foreground))
        brain_count = int(np.sum(brain_mask))
        if foreground_count > 0 and brain_count < 0.1 * foreground_count:
            logging.warning(
                f"Brain mask contains fewer than 10% of foreground voxels "
                f"({brain_count} / {foreground_count}). "
                f"The skull_thickness_mm value ({self.skull_thickness_mm:.1f}) may be too large.",
            )

        # Step 3: Skull shell = foreground AND NOT brain interior.
        skull_mask: np.ndarray = foreground & ~brain_mask

        # Step 3b: Intensity-based skull refinement within the shell.
        if self.refine_skull_intensity and effective_skull_mm > 0:
            skull_mask, brain_mask = self._refine_skull_shell(
                data, skull_mask, brain_mask, foreground, foreground_dist, effective_skull_mm,
            )

        # Step 4: Detect air cavities within the brain interior.
        # Air cavities (sinuses, mastoid air cells) are near-zero intensity
        # on T1, whereas CSF is dark but retains meaningful signal. We use
        # a dual condition: a voxel is classified as air only if its
        # intensity is below the quantile threshold AND below 10% of the
        # median brain intensity. This prevents CSF from being mislabeled.
        brain_intensities = data[brain_mask]
        if brain_intensities.size > 0:
            air_threshold = float(np.quantile(brain_intensities, self.air_threshold_quantile))
            median_brain = float(np.median(brain_intensities))
            absolute_ceiling = 0.10 * median_brain
            effective_air_threshold = min(air_threshold, absolute_ceiling)
            air_mask: np.ndarray = brain_mask & (data < effective_air_threshold)
        else:
            air_mask = np.zeros_like(brain_mask)

        # The "parenchyma" mask is the brain interior minus air cavities.
        parenchyma_mask: np.ndarray = brain_mask & ~air_mask

        # Step 5: Assemble the label volume.
        seg = np.full(data.shape, material_idx["water"], dtype=int)
        seg[skull_mask] = material_idx["skull"]
        seg[air_mask] = material_idx["air"]

        if self.classify_brain_tissues:
            # Compute a tighter mask for GMM classification that excludes
            # boundary tissues (dura, meninges) just inside the skull.
            if self.brain_extraction_margin_mm > 0:
                gmm_brain_mask: np.ndarray = foreground_dist > (effective_skull_mm + self.brain_extraction_margin_mm)
                gmm_parenchyma_mask: np.ndarray = gmm_brain_mask & ~air_mask
            else:
                gmm_parenchyma_mask = parenchyma_mask
            self._classify_brain(data, gmm_parenchyma_mask, parenchyma_mask, seg, material_idx, spacing)
        else:
            seg[parenchyma_mask] = material_idx["tissue"]

        return xa.DataArray(seg, coords=volume.coords, dims=volume.dims)

    @staticmethod
    def _refine_skull_shell(
        data: np.ndarray,
        skull_mask: np.ndarray,
        brain_mask: np.ndarray,
        foreground: np.ndarray,
        foreground_dist: np.ndarray,
        effective_skull_mm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine the EDT skull shell using intensity thresholding.

        The raw EDT shell includes scalp soft tissue (bright on T1) along
        with actual bone (dark on T1). Otsu thresholding separates them,
        morphological closing bridges the diploe gap, and keeping only the
        largest connected component removes noise.

        Returns the (possibly updated) skull_mask and brain_mask. If the
        shell lacks sufficient intensity variation or bimodality, the
        original masks are returned unchanged.

        :param data: Raw MRI intensity array.
        :param skull_mask: Binary skull shell from EDT erosion.
        :param brain_mask: Binary brain interior mask.
        :param foreground: Binary foreground mask.
        :param foreground_dist: EDT distance from foreground boundary (mm).
        :param effective_skull_mm: Skull thickness used for erosion.
        :returns: Tuple of (refined_skull_mask, refined_brain_mask).
        """
        shell_mask = foreground & ~brain_mask
        shell_intensities = data[shell_mask]
        if shell_intensities.size <= 100:
            return skull_mask, brain_mask

        import skimage.filters
        from scipy.ndimage import binary_closing
        from scipy.ndimage import label as ndlabel

        # Only refine if the shell has meaningful intensity variation.
        # On uniform/synthetic data (CV < 0.1), Otsu cannot distinguish
        # bone from soft tissue, so return unchanged.
        nonzero_shell = shell_intensities[shell_intensities > 0]
        shell_mean = float(np.mean(nonzero_shell)) if nonzero_shell.size > 0 else 0.0
        shell_cv = (
            float(np.std(nonzero_shell)) / shell_mean
            if shell_mean > 0 and nonzero_shell.size > 100
            else 0.0
        )
        if shell_cv <= 0.1:
            return skull_mask, brain_mask

        bone_threshold = skimage.filters.threshold_otsu(nonzero_shell)

        # Skip refinement if Otsu splits the shell near 50/50, which
        # indicates the shell lacks clear bimodality between bone (dark)
        # and soft tissue (bright). A skewed split (e.g., 65/35) means
        # one tissue type dominates and Otsu can separate them reliably.
        below_frac = float(np.sum(nonzero_shell < bone_threshold)) / nonzero_shell.size
        if abs(below_frac - 0.5) < 0.10:
            return skull_mask, brain_mask

        # Dark voxels in the shell are bone.
        bone_mask = shell_mask & (data < bone_threshold)

        # Light morphological closing (1 voxel) to connect fragmented
        # cortical bone without re-adding soft tissue.
        bone_mask = binary_closing(bone_mask, iterations=1)
        bone_mask = bone_mask & shell_mask  # constrain back to shell

        # Keep only the largest connected component.
        labeled_arr, n_features = ndlabel(bone_mask)
        if n_features > 0:
            sizes = np.bincount(labeled_arr.ravel())[1:]
            bone_mask = labeled_arr == (np.argmax(sizes) + 1)

        # Reclassify non-bone shell voxels: those far enough from
        # the outer surface (more than 30% of shell thickness into the
        # head) are likely brain-adjacent tissue, so add them to brain.
        # Closer to the surface stays water (scalp).
        shell_not_bone = shell_mask & ~bone_mask
        brain_mask = brain_mask | (
            shell_not_bone & (foreground_dist > effective_skull_mm * 0.3)
        )

        return bone_mask, brain_mask

    def _correct_bias_field(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        spacing: np.ndarray,
    ) -> np.ndarray:
        """Apply bias field correction and intensity normalization.

        Uses N4BiasFieldCorrection (SimpleITK) if available, otherwise falls
        back to homomorphic correction via Gaussian smoothing in log space.
        After bias correction, normalizes intensities within the mask to
        [0, 1] using robust percentile scaling so that multi-Otsu thresholds
        are consistent across scanners, protocols, and intensity scales.

        :param data: The MRI intensity array
        :param mask: Boolean mask of the region to correct (brain parenchyma)
        :param spacing: Voxel spacing in mm per axis
        :returns: Bias-corrected and intensity-normalized array
        """
        if self.use_n4:
            try:
                corrected = self._correct_bias_n4(data, mask, spacing)
            except (ImportError, RuntimeError, OSError):
                logging.warning(
                    "N4 bias correction failed (SimpleITK may not be installed); "
                    "falling back to homomorphic correction.",
                )
                corrected = self._correct_bias_homomorphic(data, mask, spacing)
        else:
            corrected = self._correct_bias_homomorphic(data, mask, spacing)

        # Intensity normalization: rescale masked region to [0, 1] using
        # robust percentile scaling. This makes multi-Otsu thresholds
        # independent of the original intensity scale.
        masked_vals = corrected[mask]
        if masked_vals.size > 0:
            vmin = float(np.percentile(masked_vals, 1))
            vmax = float(np.percentile(masked_vals, 99))
            if vmax > vmin:
                corrected[mask] = np.clip(
                    (corrected[mask] - vmin) / (vmax - vmin), 0, 1
                )

        return corrected

    def _correct_bias_n4(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        spacing: np.ndarray,
    ) -> np.ndarray:
        """N4 bias field correction via SimpleITK.

        Fits the bias field on a downsampled (shrink_factor=4) version of
        the image for speed, then reconstructs the full-resolution log bias
        field and applies the correction at the original resolution.  Uses
        4 fitting levels with 50 iterations each for robust convergence.

        :param data: The MRI intensity array
        :param mask: Boolean mask of the region to correct
        :param spacing: Voxel spacing in mm per axis
        :returns: Bias-corrected intensity array
        :raises ImportError: If SimpleITK is not installed
        :raises RuntimeError: If N4 correction fails
        """
        import SimpleITK as sitk  # pylint: disable=import-error

        image = sitk.GetImageFromArray(data.astype(np.float32))
        image.SetSpacing([float(s) for s in reversed(spacing)])
        mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_image.SetSpacing(image.GetSpacing())

        # Downsample for faster bias field estimation.
        shrink_factor = 4
        # Clamp shrink factor so no axis shrinks below 1 voxel.
        clamped = [min(shrink_factor, sz) for sz in image.GetSize()]
        shrunk_image = sitk.Shrink(image, clamped)
        shrunk_mask = sitk.Shrink(mask_image, clamped)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)
        corrector.Execute(shrunk_image, shrunk_mask)

        # Reconstruct the log bias field at original resolution and apply.
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(log_bias)

        return sitk.GetArrayFromImage(corrected).astype(np.float64)

    def _correct_bias_homomorphic(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        spacing: np.ndarray,
    ) -> np.ndarray:
        """Homomorphic bias field correction using Gaussian smoothing.

        Fallback when SimpleITK is not available. Estimates the bias field
        as a heavily smoothed version of the log-transformed image, then
        divides it out.

        :param data: The MRI intensity array
        :param mask: Boolean mask of the region to correct
        :param spacing: Voxel spacing in mm per axis
        :returns: Bias-corrected intensity array
        """
        sigma_voxels = self.bias_correction_sigma_mm / spacing

        masked_data = np.where(mask, np.clip(data, 0, None), 0.0)
        log_data = np.log1p(masked_data)

        mask_float = mask.astype(np.float64)
        smoothed_log = gaussian_filter(log_data * mask_float, sigma=sigma_voxels)
        smoothed_mask = gaussian_filter(mask_float, sigma=sigma_voxels)

        safe_mask = smoothed_mask > 1e-6
        bias_field = np.zeros_like(log_data)
        bias_field[safe_mask] = smoothed_log[safe_mask] / smoothed_mask[safe_mask]

        corrected = np.zeros_like(data)
        corrected[mask] = np.expm1(log_data[mask] - bias_field[mask])

        return corrected

    @staticmethod
    def _fit_gmm_1d(
        values: np.ndarray,
        n_components: int = 3,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a 1D Gaussian Mixture Model via Expectation-Maximization.

        Uses only numpy (no scipy dependency in the inner loop). Initializes
        component means from evenly-spaced quantiles of the data.

        For large arrays (> 50,000 elements), the EM iterations are run on a
        random subsample of 50,000 values for speed. The fitted parameters
        are nearly identical to those from the full array and are returned
        for posterior computation on the full dataset by the caller.

        :param values: 1-D array of observed values
        :param n_components: Number of Gaussian components
        :param max_iter: Maximum EM iterations
        :param tol: Convergence threshold on mean change (L-inf norm)
        :returns: Tuple of (means, stds, weights), each of shape (n_components,),
            sorted by ascending mean
        """
        # Subsample for EM if the array is large. Quantile initialization
        # uses the full array so that tail components (e.g. CSF) are
        # captured, but the iterative EM loop runs on the subsample.
        _GMM_SUBSAMPLE_SIZE = 50_000
        n_full = len(values)
        quantiles = np.linspace(0.1, 0.9, n_components)
        means = np.array([float(np.quantile(values, q)) for q in quantiles])

        if n_full > _GMM_SUBSAMPLE_SIZE:
            rng = np.random.default_rng(seed=42)
            fit_values = rng.choice(values, size=_GMM_SUBSAMPLE_SIZE, replace=False)
        else:
            fit_values = values
        n = len(fit_values)

        # If any initial means are identical (common with discrete or
        # heavily skewed data), spread them apart using the data range
        # so that EM can discover distinct modes.
        data_range = float(np.ptp(values))
        if data_range > 0:
            for i in range(1, n_components):
                if means[i] <= means[i - 1]:
                    means[i] = means[i - 1] + data_range * 0.01

        # Initialize stds as a fraction of the data standard deviation.
        data_std = float(np.std(values))
        stds = np.full(n_components, max(data_std / n_components, 1e-10))

        # Equal initial weights.
        weights = np.full(n_components, 1.0 / n_components)

        # Precompute constant for inline logpdf.
        _LOG_2PI_HALF = 0.5 * np.log(2.0 * np.pi)

        for _ in range(max_iter):
            # E-step: compute posterior responsibilities (vectorized across
            # all components in a single broadcast operation).
            safe_stds = np.maximum(stds, 1e-10)
            # shapes: means[:,None] = (K,1), fit_values[None,:] = (1,N)
            z = (fit_values[None, :] - means[:, None]) / safe_stds[:, None]
            log_resp = (
                -0.5 * z * z
                - np.log(safe_stds[:, None])
                - _LOG_2PI_HALF
                + np.log(weights[:, None] + 1e-300)
            )

            # Log-sum-exp for numerical stability.
            log_resp_max = np.max(log_resp, axis=0)
            log_norm = log_resp_max + np.log(
                np.sum(np.exp(log_resp - log_resp_max), axis=0)
            )
            log_resp -= log_norm
            resp = np.exp(log_resp)

            # M-step: update parameters.
            prev_means = means.copy()
            for k in range(n_components):
                nk = np.sum(resp[k])
                if nk < 1e-10:
                    continue
                means[k] = np.dot(resp[k], fit_values) / nk
                diff = fit_values - means[k]
                stds[k] = np.sqrt(np.dot(resp[k], diff * diff) / nk)
                stds[k] = max(stds[k], 1e-10)
                weights[k] = nk / n

            # Check convergence.
            if np.max(np.abs(means - prev_means)) < tol:
                break

        # Sort components by ascending mean.
        order = np.argsort(means)
        return means[order], stds[order], weights[order]

    @staticmethod
    def _icm_iteration(
        seg_arr: np.ndarray,
        log_likelihoods: np.ndarray,
        parenchyma_mask: np.ndarray,
        label_indices: list[int],
        beta: float,
    ) -> np.ndarray:
        """One pass of Iterated Conditional Modes (ICM) for MRF regularization.

        For each voxel in the parenchyma mask, computes a posterior that
        combines the GMM log-likelihood with a spatial prior that counts
        how many of the 6-connected (face) neighbors share the same label.
        The voxel is assigned to the class that maximizes this posterior.

        Uses vectorized numpy operations (np.roll along each axis) rather
        than per-voxel loops for performance.

        :param seg_arr: Current integer label array (modified in-place)
        :param log_likelihoods: Log-likelihood for each class, shape (n_classes, *vol_shape)
        :param parenchyma_mask: Boolean mask restricting ICM updates
        :param label_indices: Integer label values for each class (same order as log_likelihoods)
        :param beta: MRF regularization strength
        :returns: Updated seg_arr (same object, modified in-place)
        """
        n_classes = log_likelihoods.shape[0]

        # For each class k, count how many of the 6 face-neighbors share
        # label k. np.roll wraps at boundaries, but boundary voxels are
        # typically outside the parenchyma mask and do not affect the result.
        neighbor_agreement = np.zeros(
            (n_classes, *seg_arr.shape), dtype=np.float32
        )
        for k, label_val in enumerate(label_indices):
            same_label = (seg_arr == label_val).astype(np.float32)
            agreement = np.zeros_like(same_label)
            for axis in range(3):
                agreement += np.roll(same_label, 1, axis=axis)
                agreement += np.roll(same_label, -1, axis=axis)
            neighbor_agreement[k] = agreement  # Range 0..6

        # Posterior = log_likelihood + beta * neighbor_agreement.
        # Higher agreement means more neighbors share this label, which is
        # rewarded. This is equivalent to penalizing disagreement.
        posterior = log_likelihoods + beta * neighbor_agreement

        # Assign each voxel to the class with the highest posterior.
        best_class = np.argmax(posterior, axis=0)
        for k, label_val in enumerate(label_indices):
            seg_arr[parenchyma_mask & (best_class == k)] = label_val

        return seg_arr

    def _classify_brain(
        self,
        data: np.ndarray,
        gmm_parenchyma_mask: np.ndarray,
        full_parenchyma_mask: np.ndarray,
        seg: np.ndarray,
        material_idx: dict[str, int],
        spacing: np.ndarray,
    ) -> None:
        """Classify brain parenchyma voxels into CSF, gray matter, and white matter.

        Modifies ``seg`` in-place. Optionally applies bias field correction,
        then fits a 3-component Gaussian Mixture Model (EM-GMM) to the
        (possibly tighter) GMM parenchyma mask. Posterior probability maps are
        spatially smoothed with a Gaussian kernel for regularization, and the
        process is iterated twice to refine the classification.

        After GMM classification within the tight mask, labels are expanded
        to cover the full parenchyma mask via nearest-neighbor assignment
        using the Euclidean distance transform.

        On T1-weighted MRI the component ordering (by mean) is:
        CSF (dark) < gray matter (medium) < white matter (bright).

        If the GMM fit fails (e.g. too few parenchyma voxels or degenerate
        distribution), all parenchyma voxels fall back to the gray_matter label.

        :param data: The (possibly NaN-cleaned) MRI intensity array
        :param gmm_parenchyma_mask: Tight boolean mask for GMM fitting (may exclude boundary tissues)
        :param full_parenchyma_mask: Full brain parenchyma mask for final label assignment
        :param seg: Integer label array to modify in-place
        :param material_idx: Mapping from material name to integer label
        :param spacing: Voxel spacing in mm per axis
        """
        if int(np.sum(gmm_parenchyma_mask)) < 10:
            # Too few voxels for meaningful classification.
            seg[full_parenchyma_mask] = material_idx["gray_matter"]
            return

        # Apply bias field correction if enabled.
        if self.bias_correction_sigma_mm > 0:
            classify_data = self._correct_bias_field(data, gmm_parenchyma_mask, spacing)
        else:
            classify_data = data

        try:
            self._classify_brain_gmm(
                classify_data, gmm_parenchyma_mask, seg, material_idx, spacing
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            logging.warning(
                "EM-GMM brain tissue classification failed; "
                "labeling all brain parenchyma as gray matter.",
            )
            seg[full_parenchyma_mask] = material_idx["gray_matter"]
            return

        # Expand GMM labels from the tight mask to the full parenchyma mask
        # using nearest-neighbor assignment via the distance transform.
        gap_mask = full_parenchyma_mask & ~gmm_parenchyma_mask
        if np.any(gap_mask):
            # distance_transform_edt with return_indices gives the coordinates
            # of the nearest labeled (non-zero in the seed) voxel for each
            # background voxel. We seed with the GMM-classified region.
            seed = gmm_parenchyma_mask.astype(np.uint8)
            _, nearest_indices = distance_transform_edt(
                seed == 0, sampling=spacing, return_indices=True
            )
            # Copy labels from the nearest GMM-classified voxel into the gap.
            seg[gap_mask] = seg[tuple(nearest_indices[:, gap_mask])]

    def _classify_brain_gmm(
        self,
        classify_data: np.ndarray,
        parenchyma_mask: np.ndarray,
        seg: np.ndarray,
        material_idx: dict[str, int],
        spacing: np.ndarray,
    ) -> None:
        """Core GMM classification with spatial regularization.

        Fits a 3-component GMM, computes full-volume log-likelihood maps,
        then applies spatial regularization. When ``mrf_beta > 0``, uses
        Markov Random Field regularization via Iterated Conditional Modes
        (ICM), which penalizes label disagreement among 6-connected
        neighbors while preserving tissue boundaries. When ``mrf_beta == 0``
        and ``spatial_sigma_mm > 0``, falls back to Gaussian smoothing of
        posterior probability maps. The outer loop iterates twice: after
        each regularization pass, GMM parameters are re-estimated from the
        current labels/posteriors.

        Called by ``_classify_brain``; raises on failure so the caller can
        apply the gray-matter fallback.

        :param classify_data: Bias-corrected (or raw) intensity array
        :param parenchyma_mask: Boolean mask of brain parenchyma
        :param seg: Integer label array to modify in-place
        :param material_idx: Mapping from material name to integer label
        :param spacing: Voxel spacing in mm per axis
        """
        use_mrf = self.mrf_beta > 0
        sigma_voxels = self.spatial_sigma_mm / spacing
        label_keys = ["csf", "gray_matter", "white_matter"]  # sorted by T1 mean
        label_indices = [material_idx[k] for k in label_keys]
        n_components = len(label_keys)
        n_outer = 2  # total spatial-regularization iterations
        n_icm = 5  # ICM iterations per outer loop (when using MRF)

        # Extract parenchyma intensities for the initial fit.
        parenchyma_vals = classify_data[parenchyma_mask]

        # Initial GMM fit on raw parenchyma intensities.
        means, stds, weights = self._fit_gmm_1d(
            parenchyma_vals, n_components=n_components
        )

        # Precompute constant for inline logpdf (avoids scipy overhead).
        _LOG_2PI_HALF = 0.5 * np.log(2.0 * np.pi)

        # Determine whether we need full 3D probability maps.
        # MRF needs 3D log-likelihoods for ICM; Gaussian smoothing needs 3D
        # prob_maps for the convolution. When neither is active, we can
        # compute likelihoods only at parenchyma voxels (flat arrays).
        needs_3d = use_mrf or self.spatial_sigma_mm > 0

        for _iteration in range(n_outer):
            safe_stds = np.maximum(stds, 1e-10)
            log_weights = np.log(weights + 1e-300)

            if needs_3d:
                # Build full-volume log-likelihood maps (only inside parenchyma).
                log_likelihoods = np.full(
                    (n_components, *classify_data.shape), -np.inf, dtype=np.float64
                )
                prob_maps = np.zeros(
                    (n_components, *classify_data.shape), dtype=np.float64
                )
                for k in range(n_components):
                    z = (classify_data - means[k]) / safe_stds[k]
                    log_prob = (
                        -0.5 * z * z
                        - np.log(safe_stds[k])
                        - _LOG_2PI_HALF
                        + log_weights[k]
                    )
                    log_likelihoods[k][parenchyma_mask] = log_prob[parenchyma_mask]
                    prob_maps[k] = np.exp(log_prob)
                    # Zero out non-parenchyma to avoid leakage.
                    prob_maps[k][~parenchyma_mask] = 0.0
            else:
                # Fast path: compute likelihoods only at parenchyma voxels.
                # shapes: parenchyma_vals = (M,), means = (K,)
                z_flat = (parenchyma_vals[None, :] - means[:, None]) / safe_stds[:, None]
                log_ll_flat = (
                    -0.5 * z_flat * z_flat
                    - np.log(safe_stds[:, None])
                    - _LOG_2PI_HALF
                    + log_weights[:, None]
                )
                prob_flat = np.exp(log_ll_flat)

            if use_mrf:
                # MRF regularization via ICM.
                # Initialize seg labels from the argmax of the raw likelihoods
                # (only within the parenchyma mask) before ICM iterations.
                init_labels = np.argmax(
                    log_likelihoods[:, parenchyma_mask], axis=0
                )
                for k, label_val in enumerate(label_indices):
                    mask_k = parenchyma_mask.copy()
                    mask_k[parenchyma_mask] = init_labels == k
                    seg[mask_k] = label_val

                # Run ICM iterations to refine labels using spatial context.
                for _icm_iter in range(n_icm):
                    self._icm_iteration(
                        seg, log_likelihoods, parenchyma_mask,
                        label_indices, self.mrf_beta,
                    )

                # Build soft responsibilities from the final labels for
                # GMM re-estimation. Use the normalized likelihoods (posteriors)
                # so the M-step is consistent with the E-step.
                prob_sum = np.sum(prob_maps, axis=0)
                prob_sum[prob_sum < 1e-300] = 1e-300
                for k in range(n_components):
                    prob_maps[k] /= prob_sum

                # Extract labels from seg for final output.
                labels_3d = np.zeros(int(np.sum(parenchyma_mask)), dtype=int)
                for k, label_val in enumerate(label_indices):
                    labels_3d[seg[parenchyma_mask] == label_val] = k

            elif self.spatial_sigma_mm > 0:
                # Gaussian smoothing regularization path.
                mask_float = parenchyma_mask.astype(np.float64)
                smoothed_mask = gaussian_filter(mask_float, sigma=sigma_voxels)
                safe = parenchyma_mask & (smoothed_mask > 1e-6)
                for k in range(n_components):
                    smoothed_prob = gaussian_filter(
                        prob_maps[k] * mask_float, sigma=sigma_voxels
                    )
                    result = np.zeros_like(prob_maps[k])
                    result[safe] = smoothed_prob[safe] / smoothed_mask[safe]
                    prob_maps[k] = result

                # Normalize posteriors within parenchyma.
                prob_sum = np.sum(prob_maps, axis=0)
                prob_sum[prob_sum < 1e-300] = 1e-300
                for k in range(n_components):
                    prob_maps[k] /= prob_sum

                # Assign labels via argmax of smoothed posteriors.
                labels_3d = np.argmax(prob_maps[:, parenchyma_mask], axis=0)

            else:
                # No spatial regularization: work entirely with flat arrays.
                # Normalize posteriors at parenchyma voxels only.
                prob_sum_flat = np.sum(prob_flat, axis=0)
                prob_sum_flat[prob_sum_flat < 1e-300] = 1e-300
                for k in range(n_components):
                    prob_flat[k] /= prob_sum_flat

                # Assign labels via argmax of posteriors.
                labels_3d = np.argmax(prob_flat, axis=0)

            # Re-estimate GMM parameters from the current soft assignment
            # (for the next iteration). Use the posteriors as weights.
            parenchyma_intensities = classify_data[parenchyma_mask]
            new_means = np.empty(n_components)
            new_stds = np.empty(n_components)
            new_weights = np.empty(n_components)
            for k in range(n_components):
                if needs_3d:
                    resp_k = prob_maps[k][parenchyma_mask]
                else:
                    resp_k = prob_flat[k]
                nk = np.sum(resp_k)
                if nk < 1e-10:
                    new_means[k] = means[k]
                    new_stds[k] = stds[k]
                    new_weights[k] = weights[k]
                    continue
                new_means[k] = np.dot(resp_k, parenchyma_intensities) / nk
                diff = parenchyma_intensities - new_means[k]
                new_stds[k] = np.sqrt(np.dot(resp_k, diff * diff) / nk)
                new_stds[k] = max(new_stds[k], 1e-10)
                new_weights[k] = nk / len(parenchyma_intensities)

            means, stds, weights = new_means, new_stds, new_weights

        # Write final labels into seg. Components are sorted by mean:
        # index 0 = CSF (lowest), 1 = gray matter, 2 = white matter (highest).
        for k, key in enumerate(label_keys):
            voxel_mask = parenchyma_mask.copy()
            voxel_mask[parenchyma_mask] = labels_3d == k
            seg[voxel_mask] = material_idx[key]

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the segmentation method parameters

        :returns: Pandas DataFrame of the segmentation method parameters
        """
        records = [
            {"Name": "Type", "Value": "Threshold MRI", "Unit": ""},
            {"Name": "Skull Thickness", "Value": self.skull_thickness_mm, "Unit": "mm"},
            {"Name": "Air Threshold Quantile", "Value": self.air_threshold_quantile, "Unit": ""},
            {"Name": "Classify Brain Tissues", "Value": self.classify_brain_tissues, "Unit": ""},
            {"Name": "Bias Correction Sigma", "Value": self.bias_correction_sigma_mm, "Unit": "mm"},
            {"Name": "Use N4", "Value": self.use_n4, "Unit": ""},
            {"Name": "Spatial Regularization Sigma", "Value": self.spatial_sigma_mm, "Unit": "mm"},
            {"Name": "MRF Beta", "Value": self.mrf_beta, "Unit": ""},
            {"Name": "Brain Extraction Margin", "Value": self.brain_extraction_margin_mm, "Unit": "mm"},
            {"Name": "Refine Skull Intensity", "Value": self.refine_skull_intensity, "Unit": ""},
            {"Name": "Reference Material", "Value": self.ref_material, "Unit": ""},
        ]
        return pd.DataFrame.from_records(records)
