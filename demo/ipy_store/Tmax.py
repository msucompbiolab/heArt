
# coding: utf-8

# In[2]:


#!/usr/bin/python
import sys, os
import numpy as np
import math as mt
from matplotlib import pyplot as plt


# In[3]:


import scipy.io
data = scipy.io.loadmat(os.path.join(os.curdir, 'Rat_23016_1bpm_hemodynamics.mat'))

# to build a dictionary of preload and its EDLVP
if '__header__' in data:
    del data['__header__']
if '__version__' in data:
    del data['__version__']
if '__globals__' in data:
    del data['__globals__']

dict_LVV = {'twoc': 0.0, 'threec': 0.02, 'fourc': 0.04, 'fivec': 0.06,             'sixc': 0.08, 'sixcfourg': 0.09, 'sevenc': 0.1, 'sevencfourg': 0.11, 'eightc': 0.12,             'eightcfourg': 0.13, 'ninec': 0.14, 'ninecfourg': 0.15, 'baseline': 0.08}

init_LV = 0.18

# dict_LVV = {'sixc': 0.08, 'sevenc': 0.1}
for key in dict_LVV.keys():
    dict_LVV[key] += init_LV
# print(dict_LVV)

dict_LVP = {} # LV Pressure (exp-sim) and Volume
for key in dict_LVV.keys():
    dict_LVP[key] = np.zeros(5)
    dict_LVP[key][0] = data[key][0][0] # End-diastolic pressure (exp)

    dict_LVP[key][2] = max(data[key][0]) # End-systolic pressure (exp)
    dict_LVP[key][4] = dict_LVV[key] # Volume (exp)
# print(dict_LVP.items())


# In[ ]:


# Steps to find Cparam based on EDLVP
import pdb

scriptfile = 'pre_ninec.py'

try: # check if the file exists
    with open(os.path.join(os.curdir, scriptfile), 'r') as file:
        content = file.read() # read all content of the file
except:
    print('file ' + scriptfile + ' does not exist'); exit(1)
# pdb. set_trace()


contRactility_p = 15100.0
preload = 'ninec'
contRactility = contRactility_p

while True:

    cmd = 'mpirun.mpich -np 12 python' + ' ' + os.path.join(os.curdir, scriptfile)
    failure = os.system(cmd) # run the script file
    if failure:
        print('cannot run for preload ', preload); exit(1)

    ifile = open(os.path.join(os.curdir, 'outputs', preload, 'BiV_P.txt'), 'r') # postprocess

    lines = ifile.readlines()
    lst_LVP = [float(line.split()[2]) for line in lines]
    dict_LVP[preload][3] = max(lst_LVP) # end-systolic pressure (sim)

    LVP_sim = dict_LVP[preload][3] * 7.5 / 1.e3

    print("correction: ", dict_LVP[preload][2] / LVP_sim); pdb.set_trace()

    if (dict_LVP[preload][2] / LVP_sim < 1.05) & (dict_LVP[preload][2] / LVP_sim > 0.95):
        break

    contRactility = contRactility_p * dict_LVP[preload][2] / LVP_sim

    if content.find('contRactility =') != -1: # update Cparam (a material parameter) to its 'current' value
        content = content.replace(str(contRactility_p), str(contRactility))

    with open(scriptfile, 'w') as file: # write changes to the script file
        file.write(content)
    contRactility_p = contRactility


print('Current contractility is: ', contRactility)
# pdb.set_trace()
