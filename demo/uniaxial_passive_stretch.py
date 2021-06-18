#from dolfin import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import sys

sys.path.append("/mnt/Research")

from heArt.src.sim_protocols.uniaxial_passive_stretch import run_uniaxial_test as run_uniaxial_test 
from heArt.src.bmark_analytical.Holzapfel_uniaxial_stretching import Holzapfel_analytical_uniaxial as analytical

#  - - - - - - - - - - - -- - - - - - - - - - - - - - - -- - - - - - - 
IODetails = {             
	     "outputfolder" : './outputs/',
             "folderName" : '',
             "caseID" : 'ustretch1'
};

SimDetails = {
	     "isdispctrl": True,
	     "length": 1,
	     "width": 1,
             "Passive model": {"Name": "HolzapfelOgden"},
             "Passive params": {"a": 50.059, 
				"b": 8.023, 
			        "a_f": 18.472, 
		                 "b_f": 16.026, 
				 "a_s": 12.481, 
			         "b_s": 11.120, 
			         "a_fs": 0.216, 
				 "b_fs": 11.436,
				},
	    "nload": 50,
            "maxdisp": 0.3,
	}

pload_array, lbda_array, matparams = run_uniaxial_test(IODet=IODetails, SimDet=SimDetails)
pload_analytical = [analytical(lbda, matparams)[0] for lbda in lbda_array]

plt.figure(1)
plt.xlabel("lbda ")
plt.ylabel("PK1 (kPa)")
plt.plot(lbda_array, pload_array, '-*', label = "Model")
plt.plot(lbda_array, pload_analytical, '-', label="Analytical")
plt.legend()
outfilename = IODetails["outputfolder"] + IODetails["folderName"] + IODetails["caseID"] + "/Uniaxialtest.png"
plt.savefig(outfilename)


