import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pylab as plt
import sys

sys.path.append("/home/vahid")

from heArt.src.sim_protocols.run_isometric import run_isometric as run_isometric
from heArt.src.bmark_analytical.Time_varying import Time_varying as TV_analytical

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
	    "Active model": {"Name": "Time-varying"},
     	    "Active params": {"tau" : 25, "t_trans" : 300, "B" : 4.75,  "t0" : 250,  "deg" : 4, "l0" : 1.58, "Tmax" : 1e3, \
				    "Ca0" : 4.35, "Ca0max" : 4.35, "lr" : 1.9},

	    "nload": 100,
            "maxdisp": 0.2,
	    "dt": 2.0,
	    "ntpt": 400,
	    
	}

ux_array = [-0.4, -0.2, 0.0, 0.2, 0.35]
plt.figure(1)
for ux in ux_array:

	SimDetails.update({"maxdisp": ux})
	tpt_array, active_load_array, total_load_array, lbda, activematparams, passivematparams = run_isometric(IODet=IODetails, SimDet=SimDetails)
	active_load_analytical = [TV_analytical(lbda, t_a, activematparams)*lbda for t_a in tpt_array]
	 
	plt.plot(tpt_array, active_load_array, '-', label = "Model" + " lbda = " + '%10.5f'%(lbda)) 
	plt.plot(tpt_array, active_load_analytical, '*', label="Analytical" + " lbda = " + '%10.5f'%(lbda))

plt.xlabel("tpt")
plt.ylabel("PK1 active (kPa)")
plt.legend()
outfilename = IODetails["outputfolder"] + IODetails["folderName"] + IODetails["caseID"] + "/PK11_vs_lbda.png"
plt.savefig(outfilename)


