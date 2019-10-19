"""
Carries out a calculation for the shear viscosity at 
the triple point for Lennard-Jones fluid with RUMD
with the use of the SLLOD 

The value according to Meier et al. (doi:10.1063/1.1770695) at T^*=0.722, rho^*=0.8442 
should be circa \eta^*=3.258.
"""

from rumd import *
from rumd.Simulation import Simulation
from rumd.Autotune import Autotune
import rumd.Tools

import numpy as np, StringIO, math, itertools, cmath, sys
import sys, os

BlockSize = 131072
NoBlocks = 100

# Generate the starting state
subprocess.check_call('rumd_init_conf --num_par=1000 --cells=10,10,10 --rho=1.0', shell=True)

sim = Simulation("start.xyz.gz")
le = LeesEdwardsSimulationBox(sim.sample.GetSimulationBox())
sim.SetSimulationBox(le)

def SetDensity(sim, new_rho):
    # scale to get right density
    nParticles = sim.GetNumberOfParticles()
    vol = sim.GetVolume()
    currentDensity = nParticles/vol
    scaleFactor = pow(new_rho/currentDensity, -1./3)
    sim.ScaleSystem(scaleFactor)
SetDensity(sim, 0.8442)

sim.SetBlockSize(BlockSize)
sim.SetOutputScheduling("trajectory", "logarithmic")
sim.SetOutputScheduling("energies", "linear", interval=128)

sim.SetOutputMetaData("energies", stress_xy=True,density=True)
sim.SetOutputMetaData("trajectory", precision=10)
sim.SetOutputMetaData("energies", precision=10)

# create potential object.
pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
pot.SetParams(i=0, j=0, Sigma=1.000000, Epsilon=1.000000, Rcut=2.5);
sim.SetPotential(pot)

NumParticles = sim.GetNumberOfParticles()
T = 0.722

# create integrator object
strainRate = 0.07
itg = IntegratorSLLOD(timeStep=0.0025, strainRate=strainRate)
sim.SetIntegrator(itg)
itg.SetKineticEnergy(T*(NumParticles-1)*3./2.)

class ScaleKinEnergy:
    def __init__(self, itg, Ttarget, NumParticles):
        self.itg = itg
        self.Ttarget = Ttarget
        self.NumParticles = NumParticles
    def Scale(self):
        self.itg.SetKineticEnergy(self.Ttarget*(self.NumParticles-1)*3./2.)

# Reset kinetic energy every 100 steps due to numerical drift.
scale_interval = 100
kinEnergy = ScaleKinEnergy(itg, T, NumParticles)
sim.SetRuntimeAction("scale", kinEnergy.Scale, scale_interval)

# Steady state
sim.Run(BlockSize*10, suppressAllOutput=True)
# Production
Nsteps = BlockSize*NoBlocks
print('Production:', Nsteps)
sim.Run(Nsteps)

# End of run.
sim.sample.WriteConf("end.xyz.gz")
sim.sample.TerminateOutputManagers()

# create a rumd_stats object
rs = Tools.rumd_stats()
rs.ComputeStats()
rhostar = sim.GetNumberOfParticles()/sim.GetVolume()
print('rho^*:', rhostar)
print('eta^*:', rs.GetMeanVals()['sxy']/strainRate)
variance_sxy = rs.GetMeanSqVals()['sxy'] - rs.GetMeanVals()['sxy']**2
stddev_sxy = variance_sxy**0.5
print('u(sigma_xy)_{k=2}/sigma_xy', stddev_sxy*2/rs.GetMeanVals()['sxy'])
rs.PrintStats()
rs.WriteStats()