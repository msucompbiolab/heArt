# # To calibrate LVP with experimental data

import sys, os, subprocess
import numpy as np
import math as mt
from matplotlib import pyplot as plt
import scipy.io
import scipy.optimize
import re
import pdb

# # reading & preparing data
data = scipy.io.loadmat(
    os.path.join(os.curdir, "raw_data", "Rat_23019_constQbpm_hemodynamics.mat")
)


# cleaning up
if "__header__" in data:
    del data["__header__"]
if "__version__" in data:
    del data["__version__"]
if "__globals__" in data:
    del data["__globals__"]


# exclude baseline (& t)
if "t" in data:
    del data["t"]
if "baseline" in data:
    del data["baseline"]

baseline_volume = 0.18
d_lvv = {key: baseline_volume for key in data.keys()}

# d_dlvv = {'twoc': 0.0, 'threec': 0.02, 'fourc': 0.04, 'fivec': 0.06, 'sixc': 0.08, 'sevenc': 0.1, 'eightc': 0.12, 'ninec': 0.14, 'tenc': 0.16, 'elevenc': 0.18, 'twelvec': 0.2}

d_dlvv = {
    "twoc": 0.0,
    "threec": 0.02,
    "fourc": 0.04,
    "fivec": 0.06,
    "sixc": 0.08,
    "sixcfourg": 0.09,
    "sevenc": 0.1,
    "sevencfourg": 0.11,
    "eightc": 0.12,
    "eightcfourg": 0.13,
    "ninec": 0.14,
    "ninecfourg": 0.15,
    "tenc": 0.16,
}


for key in d_lvv.keys():
    d_lvv[key] += d_dlvv[key]

# exp vs sim
d_lvp = {}

for key in d_lvv.keys():
    d_lvp[key] = np.zeros(4)
    d_lvp[key][0] = d_lvv[key]  # preload (vol) -- cm3
    d_lvp[key][1] = data[key][0][0]  # EDP -- mmHg
    d_lvp[key][3] = max(data[key][0])
#    print('For preload ' + str(key) + 'endsys is: ' + str(d_lvp[key][3]))
# pdb.set_trace()

# import builtins
# # prepare script file & run jobs

# read previous version of selected variables from the script | terms only used once
init_case = None
init_lvv = None
init_Cparam = None
init_bff = None
init_bfx = None
init_bxx = None

script_f = "run_beta.py"
try:
    with open(os.path.join(os.curdir, script_f), "r") as ifile:
        lines = ifile.readlines()
except FileNotFoundError:
    print("file " + script_f + " not found!")

for line in lines:
    if "caseID" in line:
        for key in d_lvp.keys():
            if key in line:
                init_case = key  # ACHT: init_case may not be assigned
            elif "baseline" in line:
                init_case = "baseline"

    if "V_LV" in line:
        init_lvv = re.findall("[-+]?(?:\d*\.*\d+)", line)

    if "Passive params" in line:
        init_Cparam = re.findall("[-+]?(?:\d*\.*\d+)", line)  # or "\d+\.\d+"
    if "bff" in line:
        init_bff = re.findall("[-+]?(?:\d*\.*\d+)", line)
    if "bfx" in line:
        init_bfx = re.findall("[-+]?(?:\d*\.*\d+)", line)
    if "bxx" in line:
        init_bxx = re.findall("[-+]?(?:\d*\.*\d+)", line)

    if "contRactility =" in line:
        init_contRactility = re.findall("[-+]?(?:\d*\.*\d+)", line)

np_pload = np.array([])
np_edp_exp = np.array([])

sel_key_C = "sixc"
sel_key_b = "tenc"

# curve fitting for exp data
for key in d_lvp.keys():
    np_pload = np.append(np_pload, d_lvp[key][0])
    np_edp_exp = np.append(np_edp_exp, d_lvp[key][1])
    if key == sel_key_C:
        bench_exp = np_edp_exp[-1]
    elif key == sel_key_b:
        benchb_exp = np_edp_exp[-1]


def func(x, a, b, c):
    return a * np.exp(b * x) + c


popt_exp, pcov_exp = scipy.optimize.curve_fit(func, np_pload, np_edp_exp)
np_edp_exp = popt_exp[0] * np.exp(popt_exp[1] * np_pload) + popt_exp[2]

# initial values for material parameters
prev_Cparam = 11.7
prev_bff = 49.63
prev_bfx = 22.76
prev_bxx = 45.85

# determine defaults parameters for C, bxx
with open(os.path.join(os.curdir, script_f), "r") as ifile:
    content = ifile.read()  # read all content of the file

if content.find("Passive params") != -1:
    content = re.sub(
        r"\b" + re.escape(str(init_Cparam[0])) + r"\b", str(prev_Cparam), content
    )

if content.find("bff") != -1:
    content = re.sub(
        r"\b" + re.escape(str(init_bff[0])) + r"\b", str(prev_bff), content
    )

if content.find("bfx") != -1:
    content = re.sub(
        r"\b" + re.escape(str(init_bfx[0])) + r"\b", str(prev_bfx), content
    )

if content.find("bxx") != -1:
    content = re.sub(
        r"\b" + re.escape(str(init_bxx[0])) + r"\b", str(prev_bxx), content
    )

with open(os.path.join(os.curdir, script_f), "w") as ofile:
    ofile.write(content)

# run the script for all/few cases
# import pdb

np_pload = np.array([])
np_edp_sim = np.array([])

## to run for sel_key_C iteratively to determine C
preload = sel_key_C

if content.find("caseID") != -1:  # update caseID
    content = re.sub(r"\b" + re.escape(init_case) + r"\b", preload, content)
init_case = sel_key_C

lvv = d_lvp[sel_key_C][0]
if content.find("V_LV") != -1:  # update lvv
    content = re.sub(r"\b" + re.escape(str(init_lvv[0])) + r"\b", str(lvv), content)
init_lvv[0] = lvv

while True:
    with open(
        os.path.join(os.curdir, script_f), "w"
    ) as ofile:  # finalize script_f to run
        ofile.write(content)
    # pdb.set_trace()

    cmd = "mpirun.mpich -np 8 python" + " " + os.path.join(os.curdir, script_f)

    failure = os.system(cmd)
    if failure:
        print("Cannot run for preload ", preload)

    for line in lines:
        if "outputfolder" in line:
            match = re.search(r"'(.*?)'", line)

    try:
        with open(os.path.join(match.group(1), preload, "BiV_P.txt"), "r") as ifile:
            d_lvp[preload][2] = float(ifile.readline().split()[2]) * 7.5 / 1.0e3
    except FileNotFoundError:
        print("File doesn't exist for preload", preload)

    bench_sim = d_lvp[preload][2]

    adj_coeff = bench_exp / bench_sim
    print("Adjusting Coefficient is " + str(adj_coeff))

    if 0.95 < adj_coeff < 1.05:
        print("This is the final C: " + str(prev_Cparam))
        break

    Cparam = prev_Cparam * adj_coeff
    print("Next time we try this for C: " + str(Cparam))

    if content.find("Passive params") != -1:
        content = re.sub(
            r"\b" + re.escape(str(prev_Cparam)) + r"\b", str(Cparam), content
        )
    prev_Cparam = Cparam
#    pdb.set_trace()

# pdb.set_trace()

preload = sel_key_b
if content.find("caseID") != -1:  # update caseID
    content = re.sub(r"\b" + re.escape(init_case) + r"\b", preload, content)
init_case = preload

lvv = d_lvp[sel_key_b][0]
if content.find("V_LV") != -1:  # update lvv
    content = re.sub(r"\b" + re.escape(str(init_lvv[0])) + r"\b", str(lvv), content)
init_lvv[0] = lvv

while True:
    with open(os.path.join(os.curdir, script_f), "w") as ofile:
        ofile.write(content)

    cmd = "mpirun.mpich -np 8 python" + " " + os.path.join(os.curdir, script_f)

    failure = os.system(cmd)
    if failure:
        print("Cannot run for preload ", preload)

    for line in lines:
        if "outputfolder" in line:
            match = re.search(r"'(.*?)'", line)
    try:
        with open(os.path.join(match.group(1), preload, "BiV_P.txt"), "r") as ifile:
            edp_sim = float(ifile.readline().split()[2]) * 7.5 / 1.0e3
    except FileNotFoundError:
        print("File doesn't exist for preload", preload)

    b_coeff = edp_sim / benchb_exp
    print("b_coeff is: ", b_coeff)
    if 0.8 < b_coeff < 1.2:
        print(
            "bff is: "
            + str(prev_bff)
            + " and bfx is: "
            + str(prev_bfx)
            + " and bxx is: "
            + str(prev_bxx)
        )
        break

    bff = mt.log(benchb_exp / edp_sim) / d_lvp[preload][0] + prev_bff
    bfx = bff / prev_bff * prev_bfx
    bxx = bff / prev_bff * prev_bxx

    print(
        "next bff is: "
        + str(bff)
        + " next bfx is: "
        + str(bfx)
        + " and next bxx is: "
        + str(bxx)
    )

    if content.find("bff") != -1:
        content = re.sub(r"\b" + re.escape(str(prev_bff)) + r"\b", str(bff), content)

    if content.find("bfx") != -1:
        content = re.sub(r"\b" + re.escape(str(prev_bfx)) + r"\b", str(bfx), content)

    if content.find("bxx") != -1:
        content = re.sub(r"\b" + re.escape(str(prev_bxx)) + r"\b", str(bxx), content)

    prev_bff = bff
    prev_bfx = bfx
    prev_bxx = bxx

#    pdb.set_trace()

# pdb.set_trace()

# read contRactility and save it in prev_contRactility
# prev_contRactility = 22000.0
#
# if content.find('contRactility') != -1:
#    content = re.sub(r'\b' + re.escape(str(init_contRactility[0])) + r'\b', str(prev_contRactility), content)
#
# dict_contRa = {}
#
#
#
# lst_contRa = ['sixc', 'sixcfourg', 'sevenc', 'sevencfourg', 'eightc', 'ninecfourg']
#
## d_dlvv = {'twoc': 0.0, 'threec': 0.02, 'fourc': 0.04, 'fivec': 0.06, 'sixc': 0.08, 'sixcfourg': 0.09, 'sevenc': 0.1, 'sevencfourg': 0.11, 'eightc': 0.12, 'eightcfourg': 0.13, 'ninec': 0.14, 'ninecfourg': 0.15}
#
#
## for preload in d_lvp.keys():
# for preload in lst_contRa:
#
#    if content.find('caseID') != -1:  # update caseID
#        content = re.sub(r'\b' + re.escape(init_case) + r'\b', preload, content) #change
#    init_case = preload
#
#    lvv = d_lvp[preload][0]
#    if content.find('V_LV') != -1:  # update lvv
#        content = re.sub(r'\b' + re.escape(str(init_lvv[0])) + r'\b', str(lvv), content)
#    init_lvv[0] = lvv
#
#    while True:
#
#        with open(os.path.join(os.curdir, script_f), 'w') as ofile:
#            ofile.write(content)
#
#        cmd = 'mpirun.mpich -np 8 python' + ' ' +  os.path.join(os.curdir, script_f)
#
#        failure = os.system(cmd)
#        if failure:
#            print('Cannot run for preload ', preload)
#
#        try:
#            with open(os.path.join(match.group(1), preload, 'BiV_P.txt'), 'r') as ifile:
#                lines = ifile.readlines()
#        except FileNotFoundError:
#            print('File doesn\'t exist for preload', preload)
#
#        l_lvp = [line.split()[2] for line in lines]
#        endsys = max(l_lvp)
#        endsys_sim = float(endsys) * 7.5 / 1.e3
#        contRa_coeff = d_lvp[preload][3] / endsys_sim
#
#        print('endsys for sim is: ' + str(endsys_sim) + ' and for exp is: ' + str(d_lvp[preload][3]))
#
#        if (0.95 < contRa_coeff < 1.05):
#            print('fitting contRactility achieved: coeff = ', contRa_coeff)
#            break
#
#        contRactility = contRa_coeff * prev_contRactility
#        print('next time we try this for contRactility:', contRactility)
#
#        if content.find('contRactility') != -1:
#            content = re.sub(r'\b' + re.escape(str(prev_contRactility)) + r'\b', str(contRactility), content)
#        prev_contRactility = contRactility
#
#    dict_contRa[preload] = contRactility
#    # pdb.set_trace()
#
# pdb.set_trace()
# for key in dict_contRa.keys():
#    print('for preload ' + str(key) + ' contRa is: ' + dict_contRa[key][0])
#
# import json
#
## Specify the file path
# file_path = "dictionary.json"
#
## Write the dictionary to the file
# with open(file_path, "w") as json_file:
#    json.dump(data, json_file)
#
# print("Dictionary written to", file_path)
