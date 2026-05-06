from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from openlifu.util.annotations import OpenLIFUFieldData

PARAM_INFO = {"sound_speed":{"id":"sound_speed",
                               "name": "Speed of Sound",
                               "units": "m/s"},
                "density":{"id":"density",
                           "name": "Density",
                           "units": "kg/m^3"},
                "attenuation":{"id":"attenuation",
                               "name": "Attenuation",
                               "units": "dB/cm/MHz"},
                "specific_heat":{"id":"specific_heat",
                                 "name": "Specific Heat",
                                 "units": "J/kg/K"},
                "thermal_conductivity":{"id":"thermal_conductivity",
                                        "name": "Thermal Conductivity",
                                        "units": "W/m/K"}}
@dataclass
class Material:
    name: Annotated[str, OpenLIFUFieldData("Material name", "Name for the material")] = "Material"
    """Name for the material"""

    sound_speed: Annotated[float, OpenLIFUFieldData("Sound speed (m/s)", "Speed of sound in the material (m/s)")] = 1500.0  # m/s
    """Speed of sound in the material (m/s)"""

    density: Annotated[float, OpenLIFUFieldData("Density (kg/m^3)", "Mass density of the material (kg/m^3)")] = 1000.0  # kg/m^3
    """Mass density of the material (kg/m^3)"""

    attenuation: Annotated[float, OpenLIFUFieldData("Attenuation (dB/cm/MHz)", "Ultrasound attenuation in the material (dB/cm/MHz)")] = 0.0  # dB/cm/MHz
    """Ultrasound attenuation in the material (dB/cm/MHz)"""

    specific_heat: Annotated[float, OpenLIFUFieldData("Specific heat (J/kg/K)", "Specific heat capacity of the material (J/kg/K)")] = 4182.0  # J/kg/K
    """Specific heat capacity of the material (J/kg/K)"""

    thermal_conductivity: Annotated[float, OpenLIFUFieldData("Thermal conductivity (W/m/K)", "Thermal conductivity of the material (W/m/K)")] = 0.598  # W/m/K
    """Thermal conductivity of the material (W/m/K)"""

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Material name must be a string.")
        if not isinstance(self.sound_speed, int | float):
            raise TypeError(f"Sound speed of {self.name} must be a number.")
        if self.sound_speed <= 0:
            raise ValueError(f"Sound speed of {self.name} must be positive.")
        if not isinstance(self.density, int | float):
            raise TypeError(f"Density of {self.name} must be a number.")
        if self.density <= 0:
            raise ValueError(f"Density of {self.name} must be positive.")
        if not isinstance(self.attenuation, int | float):
            raise TypeError(f"Attenuation of {self.name} must be a number.")
        if self.attenuation < 0:
            raise ValueError(f"Attenuation of {self.name} must be non-negative.")
        if not isinstance(self.specific_heat, int | float):
            raise TypeError(f"Specific heat of {self.name} must be a number.")
        if self.specific_heat <= 0:
            raise ValueError(f"Specific heat of {self.name} must be positive.")
        if not isinstance(self.thermal_conductivity, int | float):
            raise TypeError(f"Thermal conductivity of {self.name} must be a number.")
        if self.thermal_conductivity <= 0:
            raise ValueError(f"Thermal conductivity of {self.name} must be positive.")

    def to_dict(self):
        return {
            "name": self.name,
            "sound_speed": self.sound_speed,
            "density": self.density,
            "attenuation": self.attenuation,
            "specific_heat": self.specific_heat,
            "thermal_conductivity": self.thermal_conductivity
        }

    @classmethod
    def param_info(cls, param_id: str):
        if param_id not in PARAM_INFO:
            raise ValueError(f"Parameter {param_id} not found.")
        return PARAM_INFO[param_id]

    def get_param(self, param_id: str):
        if param_id not in PARAM_INFO:
            raise ValueError(f"Parameter {param_id} not found.")
        return self.__getattribute__(param_id)

    @staticmethod
    def from_dict(d: dict[str, Any]):
        return Material(**d)

WATER = Material(name="water",
                 sound_speed=1500.0,
                 density=1000.0,
                 attenuation=0.002,
                 specific_heat=4182.0,
                 thermal_conductivity=0.598)

TISSUE = Material(name="tissue",
                  sound_speed=1540.0,
                  density=1000.0,
                  attenuation=0.6,
                  specific_heat=3600.0,
                  thermal_conductivity=0.5)

SKULL = Material(name="skull",
                 sound_speed=4080.0,
                 density=1900.0,
                 attenuation=8.0,
                 specific_heat=1100.0,
                 thermal_conductivity=0.3)

AIR = Material(name="air",
               sound_speed=344.0,
               density=1.25,
               attenuation=1.64,
               specific_heat=1012.0,
               thermal_conductivity=0.025)

STANDOFF = Material(name="standoff",
                    sound_speed=1420.0,
                    density=1000.0,
                    attenuation=1.0,
                    specific_heat=4182.0,
                    thermal_conductivity=0.598)

MATERIALS = {"water": WATER,
             "tissue": TISSUE,
             "skull": SKULL,
             "air": AIR,
             "standoff": STANDOFF}
