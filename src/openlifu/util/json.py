from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


class PYFUSEncoder(json.JSONEncoder):
    def default(self, obj):
        from openlifu.db.subject import Subject
        from openlifu.geo.point import Point
        from openlifu.plan.solution_analysis import SolutionAnalysisOptions
        from openlifu.seg.material import Material
        from openlifu.xdc.element import Element
        from openlifu.xdc.transducer import Transducer
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Point):
            return obj.to_dict()
        if isinstance(obj, Transducer):
            return obj.to_dict()
        if isinstance(obj, Element):
            return obj.to_dict()
        if isinstance(obj, Material):
            return obj.to_dict()
        if isinstance(obj, Subject):
            return obj.to_dict()
        if isinstance(obj, SolutionAnalysisOptions):
            return obj.to_dict()
        return super().default(obj)

def to_json(obj, filename):
    dirname = Path(filename).parent
    if dirname and not dirname.exists():
        dirname.mkdir(parents=True)
    with open(filename, 'w') as file:
        json.dump(obj, file, cls=PYFUSEncoder, indent=4)
