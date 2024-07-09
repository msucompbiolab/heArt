import sys, pdb
from dolfin import *
import csv
from numpy import genfromtxt

sys.path.append("/mnt/Research")

from heArt.src.preprocessing import (
    preprocess_PJ as preprocess_PJ
)

PJinputfilename = "paraview_line_LC.vtu"
PJoutputfilename = "PJ"
preprocess_PJ(PJinputfilename, PJoutputfilename)


tnodes = genfromtxt('PJ.csv', delimiter=',')
print(tnodes)



