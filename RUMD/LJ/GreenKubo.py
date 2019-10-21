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

BlockSize = 131072
NoBlocks = 100

# Generate the starting state
subprocess.check_call('rumd_init_conf --num_par=1000 --cells=10,10,10 --rho=0.8442', shell=True)

# create simulation object
sim = Simulation("start.xyz.gz")

sim.SetBlockSize(BlockSize)
sim.SetOutputScheduling("trajectory", "logarithmic")
sim.SetOutputScheduling("energies", "linear", interval=2048)

sim.SetOutputMetaData("energies",stress_xy=True,stress_xz=True,stress_yz=True)

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

# Equilibration for approximately 300k steps
sim.Run(int(300000//BlockSize)*BlockSize, suppressAllOutput=True)
# Production
sim.Run(BlockSize*NoBlocks)

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