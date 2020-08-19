import subprocess

import LennardJones126
import ChebTools

import numpy as np

T_integration = 2 # Isotherm along which integration is carried out
T_target = 0.7 # Targeted end of integration
rho_target = 0.7 # 

# Get some constants 
rhoc = LennardJones126.rhocstar
Tcstar = LennardJones126.Tcstar
tau = Tcstar/T_integration
tau_integration = tau

# Convenience functions that would be replaced by outputs from simulation

"""
RUMD outputs W/N and U/N

Virial energy
-------------
p - N*k_B*T/V = W/V,  # See below Fig 3 in Pedersen, PRL, 2008
multiply through by V/(N*k_B*T)

Z-1 = (W/N)/(k_B*T)

and our definition is 

Z-1 = rho*dalphar/drho 

and thus

rho*dalphar/drho = (W/N)/(k_B*T)

Internal energy
---------------

Similarly for internal energy

ur/(N*kB*T) = -T*dalphar/dT

the U/N of RUMD is the same thing as ur/N from EOS, so

(U/N)/(kB*T) = -T*dalphar/dT

"""

# # Calculate values from the EOS, for checking of method
# def get_dalphardrho(T, rho):
#     return LennardJones126.get_alphar_deriv(Tcstar/T, rho/rhoc, 0, 1)/rho
# def get_dalphardT(T, rho):
#     return -LennardJones126.get_alphar_deriv(Tcstar/T, rho/rhoc, 1, 0)/T

class RUMDSimulation():
    def __init__(self, *, Tstar, rhostar):
        import rumd
        from rumd.Simulation import Simulation

        BlockSize = 524288//8
        NumBlocks = 51

        # Generate the starting state
        subprocess.check_call('rumd_init_conf --num_par=1372 --cells=15,15,15 --rho='+str(rhostar), shell=True)

        # Create simulation object
        sim = Simulation("start.xyz.gz", pb=16, tp=8)

        sim.SetBlockSize(BlockSize)
        sim.SetOutputScheduling("trajectory", "logarithmic")
        sim.SetOutputScheduling("energies", "linear", interval=8)

        sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=False)

        # Create potential object. 
        pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        pot.SetParams(i=0, j=0, Sigma=1.0, Epsilon=1.0, Rcut=6.50)
        sim.SetPotential(pot)

        # Create integrator object
        itg = rumd.IntegratorNVT(timeStep=0.0025, targetTemperature=Tstar)
        itg.SetRelaxationTime(0.2)
        sim.SetIntegrator(itg)

        # Do the integration
        sim.Run(BlockSize*NumBlocks)

        # Create a rumd_stats object
        rs = rumd.Tools.rumd_stats()
        rs.ComputeStats()
        meanVals = rs.GetMeanVals()

        # Store the outputs
        self.U_over_N = meanVals['pe']
        self.W_over_N = meanVals['W']

def get_dalphardrho(T, rho):
    # Do a simulation, get dalphar/drho|T = (W/N)/(k_B*T)/rho; RUMD returns W/N
    sim = RUMDSimulation(Tstar=T, rhostar=rho)
    val = sim.W_over_N/T/rho
    print('T,rho:', T, rho, 'dalphar/drho values', val, LennardJones126.get_alphar_deriv(Tcstar/T, rho/rhoc, 0, 1)/rho)
    print('p^*:', LennardJones126.LJ_p(T, rho))
    return val

def get_dalphardT(T, rho):
    # Do a simulation, get dalphar/dT|rho = -(U/N)/(kB*T^2); RUMD returns U/N
    sim = RUMDSimulation(Tstar=T, rhostar=rho)
    return -sim.U_over_N/T**2

# --------------------
#     ISOTHERM 
# --------------------
rhomin = 0.01
# Chebyshev expansion in dalphar/drho along the isotherm; alphar=ar/T
ce_isoT = ChebTools.generate_Chebyshev_expansion(10, lambda rho: get_dalphardrho(T_integration, rho), rhomin, rho_target)
ce_anti_isoT = ce_isoT.integrate(1) # Anti-derivative of dalphar/drho
# Correct for the difference in residual Helmholtz energy between zero density and rhomin
alphar_correction = rhomin*ce_isoT.y(rhomin) # deltarho*(dalphar/drho)|rhomin, where deltarho = rhomin-0
alphar_int = ce_anti_isoT.y(rho_target) - ce_anti_isoT.y(rhomin) + alphar_correction

print(alphar_int, LennardJones126.get_alphar_deriv(tau_integration, rho_target/rhoc,0,0) )

# --------------------
#     ISOCHORE
# --------------------
# Chebyshev expansion in dalphar/dT along the isochore; alphar-alphar(Tintegration,rho) = \int_{Tintegration}^{Ttarget} (dalphar/dT|rho) * dT
ce_isoD = ChebTools.generate_Chebyshev_expansion(10, lambda T: get_dalphardT(T, rho_target), T_integration, T_target)
ce_anti_isoD = ce_isoD.integrate(1) # Anti-derivative of dalphar/dT w.r.t. T gives alphar along the isochore
alphar = alphar_int + ce_anti_isoD.y(T_target)-ce_anti_isoD.y(T_integration)
# dalphar_dT = ce_isoD.y(T_target)
# print(alphar,'\n', LennardJones126.get_alphar_deriv(Tcstar/T_target, rho_target/rhoc,0,0))

sex_kB = (-alphar-T_target*ce_isoD.y(T_target))

print('my -sex/kB: ', sex_kB)
print('Thol -sex/kB: ', LennardJones126.LJ_sr_over_R(T_target, rho_target))