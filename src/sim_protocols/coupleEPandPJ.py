from dolfin import *
import sys
sys.path.append("/mnt/Research")
from fenicstools import *
import numpy as np

class coupleEPandPJ(object):

    def __init__(self, EPmodel, PJmodel, EPparams, PJparams, state_obj):
        self.EPmodel = EPmodel
        self.EPparams = EPparams
        self.PJmodel = PJmodel
        self.PJparams = PJparams
        self.state_obj = state_obj

        self.pj_t_nodes = np.stack(EPparams["ploc"])
        self.pj_intensity = EPparams["current_intensity"]
        self.tstart_arr = [-10]*len(self.pj_t_nodes)
        try:
            self.probesPJ = Probes(self.pj_t_nodes.flatten(), PJmodel.w_ep.function_space().sub(0))
        except ValueError:
            self.probesPJ = Probes(self.pj_t_nodes.flatten(), PJmodel.w_ep.function_space())
        comms = PJmodel.mesh.mpi_comm()

    def UpdatePJandEP(self):

        probesPJ = self.probesPJ
        PJmodel = self.PJmodel
        EPmodel = self.EPmodel
        state_obj = self.state_obj
        pj_t_nodes = self.pj_t_nodes

        # Update PJ activation time:
        probesPJ(PJmodel.getphivar())
        Nevals = probesPJ.number_of_evaluations()
        probes_val = probesPJ.array()

        # broadcast from proc 0 to other processes
        comms = EPmodel.mesh.mpi_comm()
        rank = MPI.rank(comms)
        probes_val_bcast = probesPJ.array(N=Nevals-1) ## probe will only send to rank =0
        if(not rank == 0):
            probes_val_bcast = np.empty(len(pj_t_nodes))
        comms.Bcast(probes_val_bcast, root=0)

        for p in range(0, len(pj_t_nodes)):
            phi_pj_val = probes_val_bcast[p]
            if(phi_pj_val > 0.9 and self.tstart_arr[p] < 1.0):
                if(self.tstart_arr[p] < 0):
                    self.tstart_arr[p] = 0
                    if("fstim_array" in dir(EPmodel)):
                        EPmodel.fstim_array[p].iStim = self.pj_intensity
                    else:
                        EPmodel.fstim_val_array[p] = self.pj_intensity
                else:
                    self.tstart_arr[p] += state_obj.dt.dt
            else:
                if("fstim_array" in dir(EPmodel)):
                    EPmodel.fstim_array[p].iStim = 0.0
                else:
                    EPmodel.fstim_val_array[p] = 0.0

        if("UpdateActivation" in dir(EPmodel)):
            EPmodel.UpdateActivation()

    def reset(self):

        self.tstart_arr = [-10]*len(self.pj_t_nodes)

        return
  

