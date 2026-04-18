from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xa

from openlifu.seg import SegmentationMethod
from openlifu.seg.material import MATERIALS, TISSUE, WATER, Material
from openlifu.seg.seg_methods.threshold_mri import ThresholdMRI

# --- Helpers ---

def create_synthetic_head_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    spacing: float = 1.0,
    head_radius: float = 25.0,
    intensity: float = 100.0,
) -> xa.DataArray:
    """Sphere with given intensity inside, 0 outside, centered at origin."""
    nz, ny, nx = shape
    x = np.arange(nx) * spacing - (nx - 1) * spacing / 2
    y = np.arange(ny) * spacing - (ny - 1) * spacing / 2
    z = np.arange(nz) * spacing - (nz - 1) * spacing / 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    data = np.where(dist <= head_radius, intensity, 0.0)
    return xa.DataArray(data, dims=["z", "y", "x"], coords={"z": z, "y": y, "x": x})


def create_three_tissue_sphere(
    shape: tuple[int, int, int] = (64, 64, 64),
    spacing: float = 1.0,
    head_radius: float = 25.0,
) -> xa.DataArray:
    """Concentric spheres simulating T1 contrast: CSF=30, GM=70, WM=100."""
    nz, ny, nx = shape
    x = np.arange(nx) * spacing - (nx - 1) * spacing / 2
    y = np.arange(ny) * spacing - (ny - 1) * spacing / 2
    z = np.arange(nz) * spacing - (nz - 1) * spacing / 2
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    data = np.zeros(shape)
    data[dist <= head_radius] = 30.0
    data[dist <= head_radius * 0.7] = 70.0
    data[dist <= head_radius * 0.5] = 100.0
    return xa.DataArray(data, dims=["z", "y", "x"], coords={"z": z, "y": y, "x": x})


@pytest.fixture()
def synthetic_sphere() -> xa.DataArray:
    return create_synthetic_head_volume()


# --- Serialization ---

def test_dict_roundtrip_default() -> None:
    seg = ThresholdMRI()
    reconstructed = SegmentationMethod.from_dict(seg.to_dict())
    assert isinstance(reconstructed, ThresholdMRI)
    assert reconstructed == seg


def test_dict_roundtrip_custom_params() -> None:
    seg = ThresholdMRI(skull_thickness_mm=10.0, air_threshold_quantile=0.1)
    reconstructed = SegmentationMethod.from_dict(seg.to_dict())
    assert reconstructed.skull_thickness_mm == 10.0
    assert reconstructed.air_threshold_quantile == 0.1


def test_dict_roundtrip_classified() -> None:
    seg = ThresholdMRI(classify_brain_tissues=True, skull_thickness_mm=5.0)
    reconstructed = SegmentationMethod.from_dict(seg.to_dict())
    assert reconstructed.classify_brain_tissues is True
    assert "csf" in reconstructed.materials


# --- Input validation ---

def test_negative_skull_thickness_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ThresholdMRI(skull_thickness_mm=-1.0)


def test_invalid_air_quantile_raises() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ThresholdMRI(air_threshold_quantile=1.5)


def test_negative_bias_sigma_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ThresholdMRI(bias_correction_sigma_mm=-5.0)


def test_missing_material_key_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        ThresholdMRI(materials={"water": WATER, "tissue": TISSUE})


def test_classify_with_tissue_ref_material_raises() -> None:
    with pytest.raises(ValueError, match="removed during"):
        ThresholdMRI(classify_brain_tissues=True, ref_material="tissue")


def test_singleton_axis_raises() -> None:
    coords = {"z": [0.0], "y": np.linspace(-10, 10, 21), "x": np.linspace(-10, 10, 21)}
    volume = xa.DataArray(np.zeros((1, 21, 21)), dims=["z", "y", "x"], coords=coords)
    with pytest.raises(ValueError, match="fewer than 2 coordinates"):
        ThresholdMRI()._segment(volume)


# --- Auto-swap behavior ---

def test_materials_none_with_classification() -> None:
    seg = ThresholdMRI(materials=None, classify_brain_tissues=True)
    assert "csf" in seg.materials
    assert "tissue" not in seg.materials


def test_auto_swap_preserves_custom_materials() -> None:
    custom_skull = Material(
        name="skull", sound_speed=2800.0, density=1850.0,
        attenuation=6.0, specific_heat=1100.0, thermal_conductivity=0.3,
    )
    custom = MATERIALS.copy()
    custom["skull"] = custom_skull
    seg = ThresholdMRI(classify_brain_tissues=True, materials=custom)
    assert seg.materials["skull"].sound_speed == 2800.0
    assert seg.materials["skull"].attenuation == 6.0


def test_auto_swap_does_not_mutate_caller_dict() -> None:
    original = MATERIALS.copy()
    original_keys = set(original.keys())
    ThresholdMRI(classify_brain_tissues=True, materials=original)
    assert set(original.keys()) == original_keys


# --- Core segmentation (4-label mode) ---

def test_segment_synthetic_sphere(synthetic_sphere: xa.DataArray) -> None:
    seg = ThresholdMRI(air_threshold_quantile=0.0)._segment(synthetic_sphere)
    idx = ThresholdMRI(air_threshold_quantile=0.0)._material_indices()

    assert seg.to_numpy()[0, 0, 0] == idx["water"]
    assert np.any(seg.to_numpy() == idx["skull"])
    center = tuple(s // 2 for s in seg.shape)
    assert seg.to_numpy()[center] == idx["tissue"]
    assert len(set(np.unique(seg.to_numpy()))) >= 3


def test_segment_with_air_cavity() -> None:
    volume = create_synthetic_head_volume()
    # Explicitly set air_threshold_quantile to test the detection mechanism
    # independent of the default value.
    seg_method = ThresholdMRI(air_threshold_quantile=0.05)
    center = 32
    volume.to_numpy()[center - 3 : center + 3, center - 3 : center + 3, center - 3 : center + 3] = 0.1
    seg = seg_method._segment(volume)
    idx = seg_method._material_indices()
    air_region = seg.to_numpy()[center - 3 : center + 3, center - 3 : center + 3, center - 3 : center + 3]
    assert np.any(air_region == idx["air"])


def test_different_thickness_changes_result(synthetic_sphere: xa.DataArray) -> None:
    skull_idx = ThresholdMRI()._material_indices()["skull"]
    thin = ThresholdMRI(skull_thickness_mm=3.0, air_threshold_quantile=0.0)._segment(synthetic_sphere)
    thick = ThresholdMRI(skull_thickness_mm=12.0, air_threshold_quantile=0.0)._segment(synthetic_sphere)
    assert int(np.sum(thick.to_numpy() == skull_idx)) > int(np.sum(thin.to_numpy() == skull_idx))


# --- seg_params integration ---

def test_seg_params_produces_heterogeneous_maps(synthetic_sphere: xa.DataArray) -> None:
    params = ThresholdMRI(air_threshold_quantile=0.0).seg_params(synthetic_sphere)
    expected_vars = {"sound_speed", "density", "attenuation", "specific_heat", "thermal_conductivity"}
    assert set(params.data_vars) == expected_vars
    for var_name in ("sound_speed", "density"):
        arr = params[var_name].to_numpy()
        assert not np.all(arr == arr.flat[0])


def test_seg_params_correct_material_values(synthetic_sphere: xa.DataArray) -> None:
    seg = ThresholdMRI(air_threshold_quantile=0.0)
    ss = seg.seg_params(synthetic_sphere)["sound_speed"].to_numpy()
    assert ss[0, 0, 0] == seg.materials["water"].sound_speed
    center = tuple(s // 2 for s in ss.shape)
    assert ss[center] == seg.materials["tissue"].sound_speed


# --- Edge cases ---

def test_empty_volume_returns_all_water() -> None:
    volume = create_synthetic_head_volume(shape=(32, 32, 32), intensity=0.0)
    volume.to_numpy()[:] = 0.0
    seg = ThresholdMRI()._segment(volume)
    assert np.all(seg.to_numpy() == ThresholdMRI()._material_indices()["water"])


def test_nan_handling(synthetic_sphere: xa.DataArray, caplog: pytest.LogCaptureFixture) -> None:
    volume = synthetic_sphere.copy(deep=True)
    volume.to_numpy()[16, 16, 16] = np.nan
    with caplog.at_level(logging.WARNING):
        seg = ThresholdMRI()._segment(volume)
    assert seg.shape == volume.shape
    assert any("NaN" in r.message for r in caplog.records)


def test_foreground_mask_failure_fallback() -> None:
    volume = create_synthetic_head_volume(shape=(32, 32, 32))
    water_idx = ThresholdMRI()._material_indices()["water"]
    with patch(
        "openlifu.seg.seg_methods.threshold_mri.compute_foreground_mask",
        side_effect=ValueError("mocked"),
    ):
        seg = ThresholdMRI()._segment(volume)
    assert np.all(seg.to_numpy() == water_idx)


def test_skull_too_thick_warning(synthetic_sphere: xa.DataArray, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        ThresholdMRI(skull_thickness_mm=40.0)._segment(synthetic_sphere)
    assert any("fewer than 10%" in r.message for r in caplog.records)


# --- Spacing awareness ---

def test_spacing_aware_skull_thickness() -> None:
    seg = ThresholdMRI(skull_thickness_mm=7.0, air_threshold_quantile=0.0)
    skull_idx = seg._material_indices()["skull"]

    seg_1mm = seg._segment(create_synthetic_head_volume(shape=(64, 64, 64), spacing=1.0))
    seg_2mm = seg._segment(create_synthetic_head_volume(shape=(32, 32, 32), spacing=2.0))

    c1 = seg_1mm.shape[1] // 2
    c2 = seg_2mm.shape[1] // 2
    t1 = float(np.sum(seg_1mm.to_numpy()[:, c1, c1] == skull_idx)) * 1.0
    t2 = float(np.sum(seg_2mm.to_numpy()[:, c2, c2] == skull_idx)) * 2.0

    assert t2 < t1 * 1.5, f"2mm skull ({t2}mm) should not be 1.5x the 1mm ({t1}mm)"


# --- Brain tissue classification (6-label mode) ---

def test_classify_brain_three_tissues() -> None:
    """Verify that the three-tissue sphere produces all expected labels:
    center should be WM, and CSF, GM, WM, skull, and water should all be present.
    The thin CSF shell (only 2-3 voxels thick after skull erosion) must be
    preserved and not absorbed by adjacent gray matter."""
    volume = create_three_tissue_sphere()
    seg = ThresholdMRI(classify_brain_tissues=True, air_threshold_quantile=0.0, skull_thickness_mm=7.0, brain_extraction_margin_mm=0.0)
    result = seg._segment(volume)
    idx = seg._material_indices()

    center = tuple(s // 2 for s in result.shape)
    assert result.to_numpy()[center] == idx["white_matter"]

    unique = set(np.unique(result.to_numpy()))
    for label in ("csf", "gray_matter", "white_matter", "skull", "water"):
        assert idx[label] in unique, f"{label} should be present"


def test_classify_brain_seg_params_attenuation_varies() -> None:
    """Verify that attenuation values vary across voxels when brain tissues
    are classified, confirming that distinct material properties are assigned."""
    volume = create_three_tissue_sphere()
    params = ThresholdMRI(classify_brain_tissues=True, air_threshold_quantile=0.0).seg_params(volume)
    att = params["attenuation"].to_numpy()
    assert not np.all(att == att.flat[0]), "Attenuation should vary with brain tissue classification"


def test_classify_brain_gmm_failure_fallback() -> None:
    """When the GMM fitting raises an exception, all parenchyma voxels should
    fall back to the gray_matter label (the general exception handler)."""
    volume = create_three_tissue_sphere()
    seg = ThresholdMRI(classify_brain_tissues=True, air_threshold_quantile=0.0)
    idx = seg._material_indices()
    with patch.object(
        ThresholdMRI,
        "_classify_brain_gmm",
        side_effect=RuntimeError("mocked GMM failure"),
    ):
        result = seg._segment(volume)
    brain_labels = {idx["csf"], idx["gray_matter"], idx["white_matter"]}
    brain_voxels = np.isin(result.to_numpy(), list(brain_labels))
    if np.any(brain_voxels):
        assert set(np.unique(result.to_numpy()[brain_voxels])) == {idx["gray_matter"]}


def test_classify_brain_degenerate_constant_intensity() -> None:
    """When the brain interior has near-constant intensity, the GMM cannot
    fit meaningful components. All brain voxels should fall back to
    gray_matter rather than collapsing to CSF (argmax class 0)."""
    volume = create_synthetic_head_volume(shape=(64, 64, 64), intensity=100.0)
    seg = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.0,
        skull_thickness_mm=7.0,
        bias_correction_sigma_mm=0.0,
    )
    result = seg._segment(volume)
    idx = seg._material_indices()
    brain_labels = {idx["csf"], idx["gray_matter"], idx["white_matter"]}
    brain_voxels = np.isin(result.to_numpy(), list(brain_labels))
    if np.any(brain_voxels):
        unique_brain = set(np.unique(result.to_numpy()[brain_voxels]))
        assert idx["gray_matter"] in unique_brain, (
            "Degenerate GMM should fall back to gray_matter"
        )
        assert idx["csf"] not in unique_brain, (
            "Degenerate GMM should not produce all-CSF labels"
        )


def test_gmm_preserves_natural_class_proportions() -> None:
    """EM-GMM should recover class proportions that reflect the actual geometry
    of the concentric spheres, NOT force them to ~33% each (as histogram
    equalization + multi-Otsu tends to do).

    The three-tissue sphere has:
      WM: r <= 0.5R   -> volume fraction = (0.5)^3 = 12.5%
      GM: 0.5R < r <= 0.7R -> fraction = (0.7)^3 - (0.5)^3 = 21.8%
      CSF: 0.7R < r <= R   -> fraction = 1 - (0.7)^3 = 65.7%

    After skull erosion (7mm from 25mm radius), the outer CSF shell is clipped,
    so the observed fractions differ from the raw geometry. We compute them
    dynamically from the actual brain mask to set expectations.
    """
    shape = (64, 64, 64)
    head_radius = 25.0
    skull_thickness = 7.0

    volume = create_three_tissue_sphere(shape=shape, head_radius=head_radius)
    seg = ThresholdMRI(
        classify_brain_tissues=True,
        air_threshold_quantile=0.0,
        bias_correction_sigma_mm=0.0,
        skull_thickness_mm=skull_thickness,
        brain_extraction_margin_mm=0.0,
    )
    result = seg._segment(volume)
    idx = seg._material_indices()

    # Compute expected proportions from the ground-truth geometry after
    # accounting for skull erosion. The brain interior is voxels whose
    # distance from the sphere boundary exceeds skull_thickness.
    x = np.arange(64) * 1.0 - 31.5
    zz, yy, xx = np.meshgrid(x, x, x, indexing="ij")
    dist = np.sqrt(xx**2 + yy**2 + zz**2)

    # The brain mask from the segmentation is: distance from background > skull_thickness.
    # For a sphere, distance from background ~ head_radius - dist (approximately).
    brain_mask = dist <= (head_radius - skull_thickness)
    brain_total = int(np.sum(brain_mask))
    assert brain_total > 0, "Brain mask should not be empty"

    expected_wm_frac = float(np.sum(brain_mask & (dist <= head_radius * 0.5))) / brain_total
    expected_gm_frac = float(np.sum(
        brain_mask & (dist > head_radius * 0.5) & (dist <= head_radius * 0.7)
    )) / brain_total
    expected_csf_frac = float(np.sum(
        brain_mask & (dist > head_radius * 0.7)
    )) / brain_total

    # Observed proportions from the segmentation
    result_data = result.to_numpy()
    brain_labels = {idx["csf"], idx["gray_matter"], idx["white_matter"]}
    total_brain_voxels = sum(
        int(np.sum(result_data == idx[k])) for k in ("csf", "gray_matter", "white_matter")
    )
    assert total_brain_voxels > 0, "Should have brain voxels"

    obs_wm_frac = int(np.sum(result_data == idx["white_matter"])) / total_brain_voxels
    obs_gm_frac = int(np.sum(result_data == idx["gray_matter"])) / total_brain_voxels
    obs_csf_frac = int(np.sum(result_data == idx["csf"])) / total_brain_voxels

    # The GMM should recover proportions within 15 percentage points of the
    # geometric truth. The old equalize_hist approach would force all three
    # classes to roughly 33%, which would fail this test for the CSF and WM
    # fractions (expected CSF ~50%+, expected WM ~20%-).
    tolerance = 0.15
    assert abs(obs_wm_frac - expected_wm_frac) < tolerance, (
        f"WM fraction {obs_wm_frac:.3f} should be within {tolerance} "
        f"of expected {expected_wm_frac:.3f}"
    )
    assert abs(obs_gm_frac - expected_gm_frac) < tolerance, (
        f"GM fraction {obs_gm_frac:.3f} should be within {tolerance} "
        f"of expected {expected_gm_frac:.3f}"
    )
    assert abs(obs_csf_frac - expected_csf_frac) < tolerance, (
        f"CSF fraction {obs_csf_frac:.3f} should be within {tolerance} "
        f"of expected {expected_csf_frac:.3f}"
    )

    # Sanity check: the three fractions should NOT all be close to 0.33
    # (which would indicate forced equal partitioning).
    fracs = [obs_wm_frac, obs_gm_frac, obs_csf_frac]
    assert not all(abs(f - 1.0 / 3.0) < 0.05 for f in fracs), (
        f"All three fractions are near 33% ({fracs}), suggesting forced "
        "equal partitioning rather than natural class proportions"
    )
