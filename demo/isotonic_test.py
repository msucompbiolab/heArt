import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pylab as plt
import sys

sys.path.append("/mnt/Research")

from heArt.src.sim_protocols.run_isotonic import run_isotonic as run_isotonic

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - -
IODetails = {"outputfolder": "./outputs/", "folderName": "", "caseID": "ustretch1"}

SimDetails = {
    "isdispctrl": True,
    "length": 5,
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
    "applied load": 5000,
    "dt": 2.0,
    "ntpt": 400,
}

load_array = [0]  # , 1000, 2000, 3000, 4000, 5000]
plt.figure(1)
for load in load_array:
    SimDetails.update({"applied load": load})
    (
        tpt_array,
        active_load_array,
        total_load_array,
        lbda_array,
        activematparams,
        passivematparams,
    ) = run_isotonic(IODet=IODetails, SimDet=SimDetails)

    plt.plot(
        tpt_array,
        lbda_array,
        "-",
        label="Model" + " lbda = " + "%10.5f" % (SimDetails["applied load"]),
    )

plt.xlabel("tpt")
plt.ylabel("lbda")
plt.legend()
outfilename = (
    IODetails["outputfolder"]
    + IODetails["folderName"]
    + IODetails["caseID"]
    + "/PK11_vs_lbda.png"
)
plt.savefig(outfilename)
