
# coding: utf-8

# # To calibrate LVP with experimental data

# In[1]:


#!/usr/bin/python
import sys, os
import numpy as np
import math as mt
from matplotlib import pyplot as plt


# In[2]:


#to delete keys that do not provide data
import scipy.io
data = scipy.io.loadmat(os.path.join(os.curdir, 'LVP_Langendorff', 'Rat_23016_1bpm_hemodynamics.mat'))

if '__header__' in data:
    del data['__header__']
if '__version__' in data:
    del data['__version__']
if '__globals__' in data:
    del data['__globals__']


# In[3]:


#to initialize the dictionary 'dict_LVP' with preload and EDP(exp)
if 't' in data:
    del data['t']
if 'baseline' in data:
    del data['baseline']

dict_LVP = {} #preload/EDP(exp)/EDP(sim)
#preload
# dict_LVV = {'twoc': 0.0, 'threec': 0.02, 'fourc': 0.04, 'fivec': 0.06, \
#             'sixc': 0.08, 'sixcfourg': 0.09, 'sevenc': 0.1, 'sevencfourg': 0.11, \
#             'eightc': 0.12, 'eightcfourg': 0.13, 'ninec': 0.14, 'ninecfourg': 0.15}

# dict_LVV = {'sevenc': 0.1, 'sevencfourg': 0.11, \
#             'eightc': 0.12, 'eightcfourg': 0.13, 'ninec': 0.14, 'ninecfourg': 0.15}
dict_LVV = {'fourc': 0.04, 'fivec': 0.06,             'sixc': 0.08, 'sixcfourg': 0.09, 'sevenc': 0.1, 'sevencfourg': 0.11,             'eightc': 0.12, 'eightcfourg': 0.13, 'ninec': 0.14, 'ninecfourg': 0.15}

v_base = 0.18
for key in dict_LVV.keys():
    dict_LVP[key] = np.zeros(3)
    dict_LVP[key][0] = dict_LVV[key] + v_base #preload
    dict_LVP[key][1] = data[key][0][0] #end diastolic pressure (mmHg)


# In[ ]:


#It reads script file and store its current values for further changes
#Assumption: the terms 'Passive params', 'caseID', 'V_LV', 'bff' , 'bfx', 'bxx' only used oncce
import re

scriptfile = 'tst_sCsb.py'
ifile = open(os.path.join(os.curdir, scriptfile), 'r')
lines = ifile.readlines()

for line in lines:
    if 'Passive params' in line:
        Cparam_p = re.findall("[-+]?(?:\d*\.*\d+)", line) # or "\d+\.\d+"
    if 'caseID' in line: # the aim is to find case_p instead of initializing it ...
        for key in dict_LVP.keys():
            if key in line:
                case_p = key
            elif 'baseline' in line:
                case_p = 'baseline'
    if 'V_LV' in line:
        V_LV_p = re.findall("[-+]?(?:\d*\.*\d+)", line)
    if 'bff' in line:
        bff_p = re.findall("[-+]?(?:\d*\.*\d+)", line)
    if 'bfx' in line:
        bfx_p = re.findall("[-+]?(?:\d*\.*\d+)", line)
    if 'bxx' in line:
        bxx_p = re.findall("[-+]?(?:\d*\.*\d+)", line)

ifile.close()


# In[ ]:


# initial guess for material parameters
Cparam = 11.7
bff = 49.63;
bfx = 22.76; bxx = 45.85


# In[ ]:


#to find the script file, change its target parameters to 'current', and execute it 'once' for several preload
scriptfile = "tst_sCsb.py"
try: # check if the file exists
    with open(os.path.join(os.curdir, scriptfile), 'r') as file:
        content = file.read() # read all content of the file
except:
    print('file ' + scriptfile + ' does not exist'); exit(1)

# todo: This should be inside a while loop later
if content.find('Passive params') != -1: # update Cparam
    content = content.replace(str(Cparam_p[0]), str(Cparam)); Cparam_p[0] = Cparam

if content.find('bff') != -1:
    content = content.replace(str(bff_p[0]), str(bff)); bff_p[0] = bff

if content.find('bfx') != -1:
    content = content.replace(str(bfx_p[0]), str(bfx)); bfx_p[0] = bfx

if content.find('bxx') != -1:
    content = content.replace(str(bxx_p[0]), str(bxx)); bxx_p[0] = bxx


# In[ ]:


# todo: run the script for fewer cases/preloads ...
for preload in dict_LVP.keys():
    if content.find('caseID') != -1: # update caseID
        content = content.replace(case_p, preload)
    case_p = preload

    V_LV = dict_LVP[preload][0]
    if content.find('V_LV') != -1: # update V_LV with the corresponding value for preload
        content = content.replace('"V_LV" : ' + str(V_LV_p[0]), '"V_LV" : ' + str(V_LV))
    V_LV_p[0] = V_LV

    with open(scriptfile, 'w') as file: # write changes to the script file
        file.write(content)

    cmd = 'mpirun.mpich -np 12 python' + ' ' + os.path.join(os.curdir, scriptfile)
    failure = os.system(cmd) # run the script file
    if failure:
        print('cannot run for preload ', preload); exit(1)

#     print(case_p)
#     import pdb; pdb.set_trace()
