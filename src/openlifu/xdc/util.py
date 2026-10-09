from __future__ import annotations

import json
import re

import pandas as pd
import xarray as xa

from openlifu.util.types import PathLike
from openlifu.xdc.transducer import Transducer
from openlifu.xdc.transducerarray import TransducerArray


def _load_focal_gain_lut() -> xa.DataArray:
    resource = files("openlifu.xdc.data").joinpath("focal_gain_lut.json")
    with resource.open("r", encoding="utf-8") as file:
        return xa.DataArray.from_dict(json.load(file))


FOCAL_GAIN_LUT = _load_focal_gain_lut()

def load_transducer_from_file(transducer_filepath : PathLike, convert_array:bool = True) -> Transducer|TransducerArray:
    """Load a Transducer or TransducerArray from file, depending on the "type" field in the file.
    Note: the transducer object includes the relative path to the affiliated transducer model data. `get_transducer_absolute_filepaths`, should
    be used to obtain the absolute data filepaths based on the Database directory path.
    Args:
        transducer_filepath: path to the transducer json file
        convert_array: When enabled, if a TransducerArray is encountered then it is converted to a Transducer.
    Returns: a Transducer if the json file defines a Transducer, or if the json file defines a TransducerArray and convert_array is enabled.
        Otherwise a TransducerArray.
    """
    with open(transducer_filepath) as f:
        if not f:
            raise FileNotFoundError(f"Transducer file not found at: {transducer_filepath}")
        d = json.load(f)
    if "type"  in d and d["type"] == "TransducerArray":
        transducer = TransducerArray.from_dict(d)
        if convert_array:
            transducer = transducer.to_transducer()
    else:
        transducer = Transducer.from_file(transducer_filepath)
    return transducer

def read_test_report(filename: PathLike) -> pd.DataFrame:
    sections = [{"name": "info", "start_row": "A"},
                {"name": "txm", "start_row": "B"},
                {"name": "console", "start_row": "C"},
                {"name": "scans", "start_row": "D"},
                {"name": "freq", "start_row": "E"},
                {"name": "voltage", "start_row": "F"}]
    raw = pd.read_excel(filename, sheet_name="Report", header=None, usecols="A").rename({0: "Index"}, axis=1)
    all_data = []
    for section in sections:
        skiprows = raw.loc[raw["Index"] == section["start_row"]].index[0]+1
        nrows = raw['Index'].str.startswith(f'{section["start_row"]}.').sum()
        report_df = pd.read_excel(filename, sheet_name="Report", skiprows=skiprows, nrows=nrows, index_col=0, usecols="A:C")
        report_df["Section"] = section["name"]
        all_data.append(report_df)

    report_df = pd.concat(all_data)
    return report_df

def report_to_matrix_dict(report_df: pd.DataFrame, focal_gain_lut=FOCAL_GAIN_LUT) -> dict:
    ROW_SN = 'B.1'
    ROW_FREQ = 'B.2'
    ROW_VOLTAGE = 'E.1'
    LIFU_400 = {'id': r'txm_400_{sn}', 'name': r'TXM 400kHz (S/N {sn})', 'nx': 8, 'ny': 8, 'pitch': 5, 'frequency': 400e3, 'kerf': 0.3, 'crosstalk_frac': 0.12, 'crosstalk_dist': 5.05e-3}
    LIFU_155 = {'id': r'txm_155_{sn}', 'name': r'TXM 155kHz (S/N {sn})', 'nx': 8, 'ny': 8, 'pitch': 5, 'frequency': 155e3, 'kerf': 0.3, 'crosstalk_frac': 0.12, 'crosstalk_dist': 5.05e-3}
    LIFU_MODULES = {400: LIFU_400, 155: LIFU_155}
    freq_kHz = report_df.loc[ROW_FREQ]["Value"]
    voltage = report_df.loc[ROW_VOLTAGE]["Value"]
    sn = report_df.loc[ROW_SN]["Value"]
    pattern = r'[^a-zA-Z0-9\-\_]'
    replacement = ''
    sn = re.sub(pattern, replacement, sn)
    matrix_dict = LIFU_MODULES[freq_kHz]
    freq_df = report_df[report_df["Section"] == "freq"].copy().drop(columns=["Section"])
    freq_df = freq_df.rename(columns={"Value": "PNP"})
    freq_df = freq_df[freq_df['Item'].str.startswith("PNP")]
    freq_df["Frequency"] = freq_df['Item'].apply(lambda x: float(re.search(r"(?<=^PNP \()\d+(?= kHz\)$)", x).group(0)))
    freq_df['focal_gain'] = freq_df['Frequency'].apply(lambda f: focal_gain_lut.interp(f0=f*1e3, crosstalk=matrix_dict['crosstalk_frac']).item())
    freq_df['Sensitivity'] = freq_df['PNP'].astype(float)*1e6/freq_df['focal_gain']/voltage
    matrix_dict['sensitivity'] = [(f*1e3, sens) for f, sens in zip(freq_df["Frequency"], freq_df['Sensitivity'])]
    matrix_dict['id'] = matrix_dict['id'].format(sn=sn.lower())
    matrix_dict['name'] = matrix_dict['name'].format(sn=sn)
    return matrix_dict
