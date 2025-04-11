import sys, pdb
from dolfin import *
import csv
from numpy import genfromtxt

sys.path.append("/mnt/Research")

from heArt.src.preprocessing import (
    preprocess_PJ as preprocess_PJ
)

PJinputfilename = "endoRV_pj_mid2.vtu"
PJoutputfilename = "PJRV"
preprocess_PJ(PJinputfilename, PJoutputfilename)


tnodes = genfromtxt('PJRV.csv', delimiter=',')
print(tnodes)
