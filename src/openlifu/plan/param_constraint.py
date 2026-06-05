from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Tuple

import pandas as pd

from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.util.dict_conversion import DictMixin

PARAM_STATUS_SYMBOLS = {
    "ok": "✅",
    "warning": "⚠️",
    "error": "⛔",
}

@dataclass
class ParameterConstraint(DictMixin):
    operator: Annotated[Literal['<', '<=', '>', '>=', 'within', 'inside', 'outside', 'outside_inclusive'], OpenLIFUFieldData("Constraint operator", "Constraint operator used to evaluate parameter values")]
    """Constraint operator used to evaluate parameter values"""

    warning_value: Annotated[float | int | Tuple[float | int, float | int] | None, OpenLIFUFieldData("Warning value", "Threshold or range that triggers a warning")] = None
    """Threshold or range that triggers a warning"""

    error_value: Annotated[float | int | Tuple[float | int, float | int] | None, OpenLIFUFieldData("Error value", "Threshold or range that triggers an error")] = None
    """Threshold or range that triggers an error"""

    def __post_init__(self):
        if self.warning_value is None and self.error_value is None:
            raise ValueError("At least one of warning_value or error_value must be set")
        if self.operator in ['within', 'inside', 'outside', 'outside_inclusive']:
            if self.warning_value and (not isinstance(self.warning_value, tuple) or len(self.warning_value) != 2 or self.warning_value[0] >= self.warning_value[1]):
                raise ValueError("Warning value must be a sorted tuple of two numbers")
            if self.error_value and (not isinstance(self.error_value, tuple) or len(self.error_value) != 2 or self.error_value[0] >= self.error_value[1]):
                raise ValueError("Error value must be a sorted tuple of two numbers")
        elif self.operator in ['<', '<=', '>', '>=']:
            if self.warning_value is not None and not isinstance(self.warning_value, int | float):
                raise ValueError("Warning value must be a single value")
            if self.error_value is not None and not isinstance(self.error_value, int | float):
                raise ValueError("Error value must be a single value")

    @staticmethod
    def compare(value, operator, threshold) -> bool:
        if operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == 'within':
            return threshold[0] < value < threshold[1]
        elif operator == 'inside':
            return threshold[0] <= value <= threshold[1]
        elif operator == 'outside':
            return value < threshold[0] or value > threshold[1]
        elif operator == 'outside_inclusive':
            return value <= threshold[0] or value >= threshold[1]
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def is_warning(self, value: float | int) -> bool:
        if self.warning_value is not None:
            return not self.compare(value, self.operator, self.warning_value)
        return False

    def is_error(self, value: float | int) -> bool:
        if self.error_value is not None:
            return not self.compare(value, self.operator, self.error_value)
        return False

    def get_status(self, value: float) -> str:
        if self.is_error(value):
            return "error"
        elif self.is_warning(value):
            return "warning"
        else:
            return "ok"

    def get_status_symbol(self, value: float) -> str:
        return PARAM_STATUS_SYMBOLS[self.get_status(value)]

    def to_table(self) -> pd.DataFrame:
        """Convert the parameter constraint to a table format."""
        records = []
        if self.operator in ['<', '<=', '>', '>='] or self.operator in ['within', 'inside', 'outside', 'outside_inclusive']:
            value_template = f"value {self.operator} {{value}}"
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
        if self.warning_value is not None:
            records.append({"Name": "Warn if not", "Value": value_template.format(value=self.warning_value), "Unit": ""})
        if self.error_value is not None:
            records.append({"Name": "Error if not", "Value": value_template.format(value=self.error_value), "Unit": ""})
        return pd.DataFrame.from_records(records)
