import subprocess
import json
import os

import LennardJones126
import ChebTools

import numpy as np

T_integration = 3 # Isotherm along which integration is carried out
T_target = 0.7 # Targeted end of integration
rho_target = 0.85 # 

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
    def __init__(self, *, Tstar, rhostar, Rcut=10):
        import rumd
        from rumd.Simulation import Simulation

        # Generate the starting state in the file start.xyz.gz
        subprocess.check_call('rumd_init_conf -q --num_par=1024 --cells=15,15,15 --rho='+str(rhostar), shell=True)

        # Create simulation object
        sim = Simulation("start.xyz.gz", pb=16, tp=8)

        # Be quiet...
        sim.SetVerbose(False)

        sim.SetOutputScheduling("trajectory", "none")
        sim.SetOutputScheduling("energies", "linear", interval=8)

        sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=False)

        # Create potential object. 
        pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        pot.SetParams(i=0, j=0, Sigma=1.0, Epsilon=1.0, Rcut=Rcut)
        sim.SetPotential(pot)

        # Create integrator object
        itg = rumd.IntegratorNVT(timeStep=0.0025, targetTemperature=Tstar)
        itg.SetRelaxationTime(0.2)
        sim.SetIntegrator(itg)

        n_equil_steps = 100000
        n_run_steps = 100000
        sim.Run(n_equil_steps, suppressAllOutput=True)
        sim.Run(n_run_steps)

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
    print('T,rho:', T, rho)
    return -sim.U_over_N/T**2

force_build = True

# --------------------
#     ISOTHERM 
# --------------------
rhomin = 0.01
isoT_cachefile = 'ce_isoT.json'
# Load expansion or build it
if not os.path.exists(isoT_cachefile) or force_build:
    # Chebyshev expansion in dalphar/drho along the isotherm; alphar=ar/T
    ce_isoT = ChebTools.generate_Chebyshev_expansion(10, lambda rho: get_dalphardrho(T_integration, rho), rhomin, rho_target)
    # Store as JSON
    with open('ce_isoT.json','w') as fp:
        fp.write(json.dumps({
            'coef': ce_isoT.coef().tolist(),
            'xmin': rhomin,
            'xmax': rho_target,
            'T_integration': T_integration        
            }))
# (Re)load from cache file
j = json.load(open(isoT_cachefile))
ce_isoT = ChebTools.ChebyshevExpansion(j['coef'], j['xmin'], j['xmax'])
T_integration = j['T_integration']

ce_anti_isoT = ce_isoT.integrate(1) # Anti-derivative of dalphar/drho
# Correct for the difference in residual Helmholtz energy between zero density and rhomin
alphar_correction = rhomin*ce_isoT.y(rhomin) # deltarho*(dalphar/drho)|rhomin, where deltarho = rhomin-0
alphar_int = ce_anti_isoT.y(rho_target) - ce_anti_isoT.y(rhomin) + alphar_correction
print(alphar_int, LennardJones126.get_alphar_deriv(tau_integration, rho_target/rhoc,0,0) )

# --------------------
#     ISOCHORE
# --------------------
isoD_cachefile = 'ce_isoD.json'
# Load expansion or build it
if not os.path.exists(isoD_cachefile) or force_build:
    # Chebyshev expansion in dalphar/dT along the isochore; alphar-alphar(Tintegration,rho) = \int_{Tintegration}^{Ttarget} (dalphar/dT|rho) * dT
    ce_isoD = ChebTools.generate_Chebyshev_expansion(10, lambda T: get_dalphardT(T, rho_target), T_integration, T_target)
    with open('ce_isoD.json','w') as fp:
        fp.write(json.dumps({
            'coef': ce_isoD.coef().tolist(),
            'xmin': T_integration,
            'xmax': T_target,
            'rho_integration': rho_target
            }))
# (Re)load from cache file
j = json.load(open(isoD_cachefile))
ce_isoD = ChebTools.ChebyshevExpansion(j['coef'], j['xmin'], j['xmax'])
rho_target = j['rho_integration']

ce_anti_isoD = ce_isoD.integrate(1) # Anti-derivative of dalphar/dT w.r.t. T gives alphar along the isochore
# Array of temperatures to evaluate the properties
Ts = np.linspace(T_integration, T_target)
alphar = alphar_int + ce_anti_isoD.y(Ts)-ce_anti_isoD.y(T_integration)
# dalphar_dT = ce_isoD.y(T_target)
# print(alphar,'\n', LennardJones126.get_alphar_deriv(Tcstar/T_target, rho_target/rhoc,0,0))
sex_kB = (-alphar-Ts*ce_isoD.y(Ts))
print(Ts, sex_kB)

print('my -sex/kB: ', sex_kB[-1])
print('Thol -sex/kB: ', LennardJones126.LJ_sr_over_R(T_target, rho_target))