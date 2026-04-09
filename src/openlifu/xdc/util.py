from __future__ import annotations

import json
import re

import pandas as pd
import xarray as xa

from openlifu.util.types import PathLike
from openlifu.xdc.transducer import Transducer
from openlifu.xdc.transducerarray import TransducerArray

FOCAL_GAIN_LUT = xa.DataArray.from_dict(
    {'dims': ('f0', 'crosstalk'),
 'attrs': {},
 'data': [[2.807589054107666,
   3.2286391258239746,
   3.649686813354492,
   3.8181092739105225,
   4.07073974609375,
   4.491786956787109,
   4.912837505340576,
   5.333885669708252,
   5.754938125610352,
   6.175983428955078,
   6.597033500671387,
   7.01808500289917],
  [2.90433931350708,
   3.3324313163757324,
   3.760524272918701,
   3.931760549545288,
   4.188616752624512,
   4.616710662841797,
   5.044803142547607,
   5.472893714904785,
   5.900986671447754,
   6.3290791511535645,
   6.757172107696533,
   7.185482501983643],
  [2.9909276962280273,
   3.428293466567993,
   3.865659713745117,
   4.040605068206787,
   4.3030242919921875,
   4.740390777587891,
   5.1777544021606445,
   5.615119934082031,
   6.052487373352051,
   6.489851951599121,
   6.927217483520508,
   7.364583969116211],
  [3.0771772861480713,
   3.5201354026794434,
   3.9643561840057373,
   4.142045497894287,
   4.408576965332031,
   4.852799415588379,
   5.297021865844727,
   5.741242408752441,
   6.185462474822998,
   6.629685878753662,
   7.073910713195801,
   7.518129348754883],
  [3.170368194580078,
   3.617199182510376,
   4.064029693603516,
   4.242762565612793,
   4.51104211807251,
   4.961826801300049,
   5.4126152992248535,
   5.863399505615234,
   6.314184665679932,
   6.764969825744629,
   7.215755462646484,
   7.666543960571289],
  [3.242729663848877,
   3.6979565620422363,
   4.153182506561279,
   4.335274696350098,
   4.6084089279174805,
   5.064931869506836,
   5.521984100341797,
   5.979033946990967,
   6.4360833168029785,
   6.89313268661499,
   7.350184440612793,
   7.8072357177734375],
  [3.327850103378296,
   3.783803701400757,
   4.245124340057373,
   4.429731845855713,
   4.706640720367432,
   5.168155193328857,
   5.6296706199646,
   6.091186046600342,
   6.552703380584717,
   7.014214992523193,
   7.475728988647461,
   7.937244415283203],
  [3.415055990219116,
   3.8783867359161377,
   4.341715335845947,
   4.527048110961914,
   4.8050456047058105,
   5.268375396728516,
   5.731706142425537,
   6.195037841796875,
   6.658364295959473,
   7.12183952331543,
   7.587955474853516,
   8.054073333740234],
  [5.705652713775635,
   5.966911792755127,
   6.2332234382629395,
   6.343565940856934,
   6.509076118469238,
   6.784928798675537,
   7.060781478881836,
   7.336633682250977,
   7.612486362457275,
   7.888339519500732,
   8.164192199707031,
   8.447998046875],
  [5.73893404006958,
   5.998416423797607,
   6.260423183441162,
   6.365228652954102,
   6.522432327270508,
   6.784440994262695,
   7.046449184417725,
   7.310236930847168,
   7.583878517150879,
   7.8575215339660645,
   8.131163597106934,
   8.404805183410645],
  [5.780664920806885,
   6.028132915496826,
   6.275601387023926,
   6.3745880126953125,
   6.523068904876709,
   6.777853965759277,
   7.03900146484375,
   7.300150394439697,
   7.561298370361328,
   7.822445869445801,
   8.083595275878906,
   8.350235939025879],
  [5.814091205596924,
   6.0464091300964355,
   6.284290313720703,
   6.383488178253174,
   6.532283306121826,
   6.780277252197266,
   7.028269290924072,
   7.27626371383667,
   7.524255752563477,
   7.772250175476074,
   8.040055274963379,
   8.318523406982422],
  [5.836524486541748,
   6.067535400390625,
   6.301788806915283,
   6.3954901695251465,
   6.536042213439941,
   6.770293712615967,
   7.00454568862915,
   7.238796710968018,
   7.482921600341797,
   7.740076541900635,
   8.005291938781738,
   8.270505905151367],
  [5.86801815032959,
   6.0880255699157715,
   6.308032989501953,
   6.396038055419922,
   6.528041839599609,
   6.749823093414307,
   6.9840264320373535,
   7.218224048614502,
   7.4528584480285645,
   7.70409631729126,
   7.955334663391113,
   8.206572532653809],
  [5.892360210418701,
   6.097702980041504,
   6.303044319152832,
   6.390904903411865,
   6.523708343505859,
   6.7450480461120605,
   6.966385841369629,
   7.187726020812988,
   7.417387962341309,
   7.661881923675537,
   7.91318941116333,
   8.164498329162598],
  [5.90617036819458,
   6.104805946350098,
   6.312875747680664,
   6.396102428436279,
   6.520944595336914,
   6.729012966156006,
   6.937082290649414,
   7.15734338760376,
   7.39542818069458,
   7.6335129737854,
   7.8715996742248535,
   8.1096830368042]],
 'coords': {'f0': {'dims': ('f0',),
   'attrs': {},
   'data': [130000.0,
    135000.0,
    140000.0,
    145000.0,
    150000.0,
    155000.0,
    160000.0,
    165000.0,
    375000.0,
    380000.0,
    385000.0,
    390000.0,
    395000.0,
    400000.0,
    405000.0,
    410000.0]},
  'crosstalk': {'dims': ('crosstalk',),
   'attrs': {},
   'data': [0.0,
    0.05,
    0.1,
    0.12,
    0.15000000000000002,
    0.2,
    0.25,
    0.30000000000000004,
    0.35000000000000003,
    0.4,
    0.45,
    0.5]}},
 'name': 'focal_gain'})

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
