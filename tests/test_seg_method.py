from __future__ import annotations

import pytest

from openlifu.seg import MATERIALS, Material, SegmentationMethod, seg_methods
from openlifu.seg.seg_methods.uniform import UniformSegmentation


@pytest.fixture()
def example_seg_method() -> seg_methods.UniformSegmentation:
    return seg_methods.UniformSegmentation(
        materials = {
            'water' : Material(
                name="water",
                sound_speed=1500.0,
                density=1000.0,
                attenuation=0.002,
                specific_heat=4182.0,
                thermal_conductivity=0.598
            ),
            'skull' : Material(
                name="skull",
                sound_speed=4080.0,
                density=1900.0,
                attenuation=8.0,
                specific_heat=1100.0,
                thermal_conductivity=0.3
            ),
        },
        ref_material = 'water',
    )

def test_seg_method_dict_conversion(example_seg_method : seg_methods.UniformSegmentation):
    assert SegmentationMethod.from_dict(example_seg_method.to_dict()) == example_seg_method

def test_seg_method_no_instantiate_abstract_class():
    with pytest.raises(TypeError):
        SegmentationMethod()  # pyright: ignore[reportAbstractUsage]

def test_uniform_seg_method_no_reference_material():
    with pytest.raises(ValueError, match="Reference material non_existent_material not found."):
        UniformSegmentation(
            materials = {
                'water' : Material(
                    name="water",
                    sound_speed=1500.0,
                    density=1000.0,
                    attenuation=0.002,
                    specific_heat=4182.0,
                    thermal_conductivity=0.598
                ),
                'skull' : Material(
                    name="skull",
                    sound_speed=4080.0,
                    density=1900.0,
                    attenuation=8.0,
                    specific_heat=1100.0,
                    thermal_conductivity=0.3
                ),
            },
            ref_material = 'non_existent_material',
        )

def test_uniformwater_errors_when_specify_ref_material():
    with pytest.raises(TypeError):
        seg_methods.UniformWater(
                ref_material = 'water',  # pyright: ignore[reportCallIssue]
            )

def test_materials_as_none_gets_default_materials():
    seg_method = seg_methods.UniformSegmentation(materials=None)  # pyright: ignore[reportArgumentType]
    assert seg_method.materials == MATERIALS.copy()

def test_from_dict_on_keyword_mismatch():
    d = {
        "class": "UniformWater",
        "materials": {
            "water": {
                "name": "water",
                "sound_speed": 1500,
                "density": 1000,
                "attenuation": 0.0022,
                "specific_heat": 4182,
                "thermal_conductivity": 0.598
            }
        },
        "ref_material": "water"
    }

    with pytest.raises(TypeError, match=r"Unexpected keyword arguments for UniformWater: \['ref_material'\]"):
        SegmentationMethod.from_dict(d, on_keyword_mismatch='raise')

    # This should not raise any warning or exception
    SegmentationMethod.from_dict(d, on_keyword_mismatch='ignore')
