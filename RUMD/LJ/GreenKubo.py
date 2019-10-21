"""
Carries out a calculation for the shear viscosity near
the triple point for Lennard-Jones fluid with RUMD
with the use of the Green-Kubo approach 

The value according to Meier et al. (doi:10.1063/1.1770695) at T^*=0.722, rho^*=0.8442 
should be circa \eta^*=3.258.
"""
from __future__ import print_function, division

import subprocess
import gzip
import io

from rumd import *
from rumd.Simulation import Simulation
from rumd.Autotune import Autotune
import rumd.Tools

import numpy as np
import pandas

# Generate the starting state
subprocess.check_call('rumd_init_conf --num_par=1000 --cells=10,10,10 --rho=0.8442', shell=True)

# create simulation object
sim = Simulation("start.xyz.gz")
sim.SetOutputScheduling("trajectory", "logarithmic")
sim.SetOutputScheduling("energies", "linear", interval=1024)

sim.SetOutputMetaData("energies",stress_xy=True,stress_xz=True,stress_yz=True,kineticEnergy=False,potentialEnergy=False,temperature=False,totalEnergy=False,virial=False,pressure=False)

# create potential object.
pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
pot.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=2.5);
sim.SetPotential(pot)

# create integrator object
Tstar = 0.722
itg = IntegratorNVT(timeStep=0.0025, targetTemperature=Tstar)
sim.SetIntegrator(itg)

at = Autotune()
at.Tune(sim)

# Equilibration for 600k steps
sim.Run(300000, suppressAllOutput=True)
# Production for 20 million steps
sim.Run(20*10**6)

# End of run.
sim.sample.WriteConf("end.xyz.gz")
sim.sample.TerminateOutputManagers()

####### Analysis #################

# Do the autocorrelation
subprocess.check_call('rumd_autocorrelations -w 1', shell=True)

# Post-process the autocorrelation
with gzip.open('autocorrelations.dat.gz') as fp:
    contents = fp.read().replace('#  ','')
    df = pandas.read_csv(io.StringIO(unicode(contents)), sep=r'\s+', engine='python')

    V = sim.GetVolume()
    for key in ['sxy', 'sxz', 'syz']:
        print('eta^* from ', key,':', np.trapz(df[key], df['time'])*V/Tstar, "(probably incorrect)")