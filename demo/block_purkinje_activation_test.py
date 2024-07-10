import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pylab as plt
import sys

sys.path.append("/mnt/Research")

from heArt.src.sim_protocols.run_block_purkinje_act import run_block_purkinje_act as run_block_purkinje_act 

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {
    "casename": "Block",
    "casename_pj": "PJ",
    "outputfolder": "./outputs_block_purkinje_act/",
    "folderName": "",
    "caseID": "isotonic",
    "directory_ep": "../Block_Purkinje_mesh/",
    "directory_pj": "../Block_Purkinje_mesh/",
}

SimDetails = {
    "isdispctrl": True,
    "length": 1,
    "width": 1,
    "Passive model": {"Name": "HolzapfelOgden"},
    "Passive params": {
        "a": 50.059,
        "b": 8.023,
        "a_f": 18.472,
        "b_f": 16.026,
        "a_s": 12.481,
        "b_s": 11.120,
        "a_fs": 0.216,
        "b_fs": 11.436,
    },
    "Active model": {"Name": "Time-varying"},
    "Active params": {
        "tau": 25,
        "t_trans": 300,
        "B": 4.75,
        "t0": 250,
        "deg": 4,
        "l0": 1.58,
        "Tmax": 1e3,
        "Ca0": 4.35,
        "Ca0max": 4.35,
        "lr": 1.9,
    },
    "nload": 100,
    "load": 5000,
    "dt": 1.0,
    "ntpt": 800,
    "nelem_ep": 4,
    "nelem_me": 1,
    "length": 5,
    "width": 1,
    "isLV": False,
    "HeartBeatLength": 800,
    "Ischemia": False,
    #"ploc": [[0.1, 0.5, 0.5, 1.0, 1], [4.9, 0.5, 0.5, 1.0, 2]],
    "tnode": [[0.0, 10.0, 0.0], [0.0,0.0,10.0]],
    "ploc": [[10.0, 0.0, 0.0]],
    "pacing_timing": [[10.0, 10.0]],
    #"pacing_timing": [[0.0, 5], [10, 5]],
    "d_iso": 0.01,
    "d_iso_pj": 10.0,
    "d_ani_factor": 4.0,
    "writeStep": 1,
    "HomogenousActivation": False,
}

plt.figure(1)

run_block_purkinje_act(IODet=IODetails, SimDet=SimDetails)
