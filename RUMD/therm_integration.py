import subprocess
import json
import os
import gzip

import pandas
import numpy as np

import LennardJones126
import ChebTools
import scipy.integrate

# Convenience functions that would be replaced by outputs from simulation

"""

Definitions
-----------

alphar = Ar/(Nmolecule*kB*T)

which is identical to 

alphar = Ar/(Nseg*kB*T)

for molecular fluids

RUMD outputs W/N and U/N

Virial energy
-------------
p - N*k_B*T/V = W/V,  # See below Fig 3 in Pedersen, PRL, 2008
multiply through by V/(N*k_B*T)

Z-1 = (W/N)/(k_B*T)

and our definition is 

Z-1 = rho*dalphar/drho 

and thus

(Nmol/V)*dalphar/drho = (W/Nmol)/(k_B*T)

Internal energy
---------------

Similarly for internal energy

Ur/(Nmol*kB*T) = -T*dalphar/dT

the U/Nseg of RUMD is the same thing as ur/N from EOS, so

(U/Nmol)/(kB*T) = -T*dalphar/dT

It is also the case that the excess entropy is defined by:

Ar = Ur - TSr

or 

Ar/(Nchain*kB*T) = Ur/(Nchain*kB*T) - T*Sr/(Nchain*kB*T)

or

s^+ = - T*Sr/(Nchain*kB*T) = Ar/(Nchain*kB*T) - Ur/(Nchain*kB*T)

so if both residual internal energy (from U/N of simulation) and residual 
Helmholtz energy (from integration along isotherm) are known, then the residual
entropy is known.

"""

# Calculate values from the EOS, for checking of method
def get_dalphardrho_TholLJ(T, rho):
    # Ar01 = delta*(dalphar_ddelta) = rho*(dalphar/drho)
    Ar01 = LennardJones126.get_alphar_deriv(LennardJones126.Tcstar/T, rho/LennardJones126.rhocstar, 0, 1)
    return Ar01/rho
def get_dalphardT_TholLJ(T, rho):
    return -LennardJones126.get_alphar_deriv(LennardJones126.Tcstar/T, rho/LennardJones126.rhocstar, 1, 0)/T

def calc_s2kB(folder):
    """ Calculate the two-body excess entropy from integration of radial distribution function """

    Ncol = len(open(folder + '/rdf.dat').readlines()[6].split(' '))
    if Ncol == 6:
        names = ['r','gAA','gAB','gBA','gBB','dummy']
    elif Ncol == 3:
        names = ['r','gAA','dummy']
    else:
        raise ValueError(folder)

    rdf = pandas.read_csv(folder + '/rdf.dat', comment='#', names = names, sep=' ')
    # rdf.info()

    dr = rdf.r.iloc[1] - rdf.r.iloc[0]

    Nstr, info = gzip.open('start.xyz.gz').readlines()[0:2]
    N = int(Nstr)
    parts = info.decode('utf-8').strip().split(' ')
    meta = {}
    for part in parts:
        k, v = part.split('=', 1)
        meta[k] = v
    print(meta)
    # assert(int(meta['numTypes'])==1)
    L, W, H = [float(_) for _ in meta['sim_box'].split(',')[1:4]]
    V = L*W*H
    rho = N/V

    molefrac = {'A': 1.0}

    # For multicomponent mixtures, add back determination of mole fractions of each component
    if int(meta['numTypes']) > 1:
        for component, n in zip(['A','B'], meta['num_par']):
            molefrac[component] = n/N

    rdf = rdf[rdf.r < L/2.0]

    if 'gAB' in rdf:
        particle_pairs = [('A','A'), ('A','B'), ('B','A'), ('B','B')]
    else:
        particle_pairs = [('A','A')]

    summer = 0.0
    for particle_pair in particle_pairs:
        p0, p1 = particle_pair
        x0, x1 = molefrac[p0], molefrac[p1]
        k = 'g' + ''.join(particle_pair)
        integrand = np.array((rdf[k]*np.log(rdf[k]) - (rdf[k]-1))*rdf.r**2)
        # Fix points with gxy == 0; limit of x*ln(x) as x --> 0 is zero
        integrand[rdf[k] == 0] = np.array(rdf.r)[rdf[k] == 0]**2
        summer += x0*x1*scipy.integrate.trapz(x=rdf.r, y=integrand)
    return -2*np.pi*rho*summer

class RUMDLJmixSimulation():
    def __init__(self, *, Tstar, rhostar, Rcut=5, numA, numB, massA, massB, epsilonA, epsilonB, sigmaA, sigmaB):
        import rumd
        from rumd.Simulation import Simulation

        masses = [massA,massB]
        num_par = [numA,numB]
        mass_str = ' --mass={mass:s}'.format(mass=','.join([str(m) for m in masses]))
        num_par_str = ','.join([str(n) for n in num_par])

        # Generate the starting state in the file start.xyz.gz
        subprocess.check_call('rumd_init_conf -q --num_par='+num_par_str+' --cells=15,15,15 --rho='+str(rhostar) + mass_str, shell=True)

        # Create simulation object
        sim = Simulation("start.xyz.gz", pb=16, tp=8, verbose=False)

        # Be quiet...
        sim.SetVerbose(False)

        sim.SetOutputScheduling("trajectory", "logarithmic")
        sim.SetOutputScheduling("energies", "linear", interval=8)

        sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=False)

        # Create potential object. 
        pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
        pot.SetParams(i=0, j=0, Sigma=sigmaA, Epsilon=epsilonA, Rcut=Rcut)
        pot.SetParams(i=0, j=1, Sigma=(sigmaA+sigmaB)/2, Epsilon=(epsilonA*epsilonB)**0.5, Rcut=Rcut)
        pot.SetParams(i=1, j=1, Sigma=sigmaB, Epsilon=epsilonB, Rcut=Rcut)
        sim.SetPotential(pot)

        # Create integrator object
        itg = rumd.IntegratorNVT(timeStep=0.00025, targetTemperature=Tstar)
        itg.SetRelaxationTime(0.2)
        sim.SetIntegrator(itg)

        n_equil_steps = 100000
        n_run_steps = 10000
        sim.Run(n_equil_steps, suppressAllOutput=True)
        sim.Run(n_run_steps)

        # Create a rumd_stats object
        rs = rumd.Tools.rumd_stats()
        rs.ComputeStats()
        meanVals = rs.GetMeanVals()
        # print(meanVals)
        # rs.PrintStats()

        sim.sample.TerminateOutputManagers()

        # Store the outputs
        self.U_over_N = meanVals['pe']
        self.W_over_N = meanVals['W']

        # Calculate s_2
        rdf_obj = rumd.Tools.rumd_rdf()
        # Constructor arguments: number of bins and minimum time
        rdf_obj.ComputeAll(1000, 1.0)
        # Include the state point information in the rdf file name
        rdf_obj.WriteRDF("rdf.dat")

        # s^+_2 = -s_2/k_B
        # self.splus2 = -calc_s2kB(os.path.abspath(os.path.dirname(__file__)))
        # print('s^+_2', self.splus2)

class RUMDMonomerSimulation():
    def __init__(self, *, potential, pot_params, Tstar, rhostar, Rcut=10):
        import rumd
        from rumd.Simulation import Simulation

        # Generate the starting state in the file start.xyz.gz
        subprocess.check_call('rumd_init_conf -q --num_par=1024 --cells=15,15,15 --rho='+str(rhostar), shell=True)

        # Create simulation object
        sim = Simulation("start.xyz.gz", pb=16, tp=8, verbose=False)

        # Be quiet...
        sim.SetVerbose(False)

        sim.SetOutputScheduling("trajectory", "logarithmic")
        sim.SetOutputScheduling("energies", "linear", interval=8)

        sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=False)

        # Create potential object. 
        if potential == 'LJ12-6':
            pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedPotential)
            pot.SetParams(i=0, j=0, Sigma=1.0, Epsilon=1.0, Rcut=Rcut)
            sim.SetPotential(pot)
        elif potential == 'EXP6':
            pot = rumd.Pot_Buckingham(cutoff_method=rumd.ShiftedPotential)
            pot.SetParams(i=0, j=0, **pot_params)
            sim.SetPotential(pot)
        else:
            raise KeyError("invalid potential name")

        # Create integrator object
        itg = rumd.IntegratorNVT(timeStep=0.00025, targetTemperature=Tstar)
        itg.SetRelaxationTime(0.2)
        sim.SetIntegrator(itg)

        n_equil_steps = 100000
        n_run_steps = 10000
        sim.Run(n_equil_steps, suppressAllOutput=True)
        sim.Run(n_run_steps)

        # Create a rumd_stats object
        rs = rumd.Tools.rumd_stats()
        rs.ComputeStats()
        meanVals = rs.GetMeanVals()
        print(meanVals)
        # rs.PrintStats()

        sim.sample.TerminateOutputManagers()

        # Store the outputs
        self.U_over_N = meanVals['pe']
        self.W_over_N = meanVals['W']

        # Calculate s_2
        rdf_obj = rumd.Tools.rumd_rdf()
        # Constructor arguments: number of bins and minimum time
        rdf_obj.ComputeAll(1000, 1.0)
        # Include the state point information in the rdf file name
        rdf_obj.WriteRDF("rdf.dat")

        # s^+_2 = -s_2/k_B
        self.splus2 = -calc_s2kB(os.path.abspath(os.path.dirname(__file__)))
        print('s^+_2', self.splus2)

class RUMDCacher():
    def __init__(self, *, sim_class, sim_kwargs = {}):
        self.isoTvals = []
        self.isoDvals = []
        self.sim_class = sim_class
        self.sim_kwargs = sim_kwargs

    def get_dalphardrho_RUMD(self, T, rho):
        # Do a simulation, get dalphar/drho|T = (W/N)/(k_B*T)/rho; RUMD returns W/N
        sim = self.sim_class(Tstar=T, rhostar=rho, **self.sim_kwargs)
        val = sim.W_over_N/T/rho
        urNkBT = sim.U_over_N/T
        vals = {
            'T': T,
            'rho': rho,
            'U/N': sim.U_over_N,
            'W/N': sim.W_over_N,
            # 's^+_2': sim.splus2,
            'dalphar/dT|rho': -sim.U_over_N/T**2,
            'dalphar/drho|T': sim.W_over_N/T/rho,
        }    
        print(vals)
        self.isoTvals.append(vals)
        return val
    
    def get_dalphardT_RUMD(self, T, rho):
        # Do a simulation, get dalphar/dT|rho = -(U/N)/(kB*T^2); RUMD returns U/N
        sim = self.sim_class(Tstar=T, rhostar=rho, **self.sim_kwargs)
        val = -sim.U_over_N/T**2
        vals = {
            'T': T, 
            'rho': rho,
            'U/N': sim.U_over_N,
            'W/N': sim.W_over_N,
            # 's^+_2': sim.splus2,
            'dalphar/dT|rho': -sim.U_over_N/T**2,
            'dalphar/drho|T': sim.W_over_N/T/rho
        }
        self.isoDvals.append(vals)
        return val

    def dump_isoT(self, T_integration, *, rho_min, rho_target, prefix=''):

        # Chebyshev expansion in dalphar/drho along the isotherm; alphar=ar/T
        ce_isoT = ChebTools.generate_Chebyshev_expansion(40, lambda rho: self.get_dalphardrho_RUMD(T_integration, rho), rho_min, rho_target)

        ce_anti_isoT = ce_isoT.integrate(1) # Anti-derivative of dalphar/drho
        # Correct for the difference in residual Helmholtz energy between zero density and rho_min
        alphar_correction = rho_min*ce_isoT.y(rho_min) # deltarho*(dalphar/drho)|rho_min, where deltarho = rho_min-0

        dff = pandas.DataFrame(self.isoTvals)
        dff['alphar'] = ce_anti_isoT.y(dff['rho']) - ce_anti_isoT.y(rho_min) + alphar_correction
        dff['s^+'] = dff['alphar'] + T_integration*dff['dalphar/dT|rho'] # s^+ = alphar - tau*(dalphar/dtau)_delta
        dff.to_csv(f'{prefix:s}T_{T_integration}_LJisotherm.csv', index=False)

force_build = True

def one_integration(T_integration, T_target, rho_target, *, get_dalphardrho, get_dalphardT, prefix=None):

    if prefix is None:
        prefix=f'T_{T_target:0.6f}_D_{rho_target:0.6f}_'

    # Get some constants 
    rhoc = LennardJones126.rhocstar
    Tcstar = LennardJones126.Tcstar
    tau = Tcstar/T_integration
    tau_integration = tau

    # --------------------
    #     ISOTHERM 
    # --------------------
    rho_min = 0.01
    isoT_cachefile = prefix+'ce_isoT.json'
    # Load expansion or build it
    if not os.path.exists(isoT_cachefile) or force_build:
        # Chebyshev expansion in dalphar/drho along the isotherm; alphar=ar/T
        ce_isoT = ChebTools.generate_Chebyshev_expansion(20, lambda rho: get_dalphardrho(T_integration, rho), rho_min, rho_target)
        # Store as JSON
        with open(isoT_cachefile,'w') as fp:
            fp.write(json.dumps({
                'coef': ce_isoT.coef().tolist(),
                'xmin': rho_min,
                'xmax': rho_target,
                'T_integration': T_integration
                }))
    # (Re)load from cache file
    j = json.load(open(isoT_cachefile))
    ce_isoT = ChebTools.ChebyshevExpansion(j['coef'], j['xmin'], j['xmax'])
    rho_min = j['xmin']
    rho_target = j['xmax']
    T_integration = j['T_integration']

    ce_anti_isoT = ce_isoT.integrate(1) # Anti-derivative of dalphar/drho
    # Correct for the difference in residual Helmholtz energy between zero density and rho_min
    alphar_correction = rho_min*ce_isoT.y(rho_min) # deltarho*(dalphar/drho)|rho_min, where deltarho = rho_min-0
    alphar_int = ce_anti_isoT.y(rho_target) - ce_anti_isoT.y(rho_min) + alphar_correction

    # Calculate residual entropy at intersection of isotherm and isochore
    # -T*Sr = Ar - Ur
    # so
    # s^+ = -Sr/(Nmol*k) = Ar/(Nmol*k*T) - Ur/(Nmol*k*T)
    Ar_per_NmolkBT = alphar_int
    Ur_per_NmolkBT = -T_integration*get_dalphardT(T_integration, rho_target) # alphar is per molecule
    splus = Ar_per_NmolkBT - Ur_per_NmolkBT
    print('='*30)
    print('At intersection of T and rho, s^+:', splus, 's^+ for L-J from Thol', -LennardJones126.LJ_sr_over_R(T_integration, rho_target))
    print('Ur/Nmol', Ur_per_NmolkBT*T_integration)
    print('Ur/(Nmol*kB*T)', Ur_per_NmolkBT, LennardJones126.LJ_ur_over_kBT(T_integration, rho_target))
    print('Ar/Nmol:', Ar_per_NmolkBT*T_integration)
    print('Ar/(Nmol*kB*T):', Ar_per_NmolkBT, LennardJones126.LJ_ar_over_kBT(T_integration, rho_target))
    print('='*30)
    
    return {'s^+': splus, 'T^*': T_integration, 'rho': rho_target}

    # --------------------
    #     ISOCHORE
    # --------------------
    isoD_cachefile = prefix+'ce_isoD.json'
    # Load expansion or build it
    if not os.path.exists(isoD_cachefile) or force_build:
        # Chebyshev expansion in dalphar/dT along the isochore; alphar-alphar(Tintegration,rho) = \int_{Tintegration}^{Ttarget} (dalphar/dT|rho) * dT
        ce_isoD = ChebTools.generate_Chebyshev_expansion(50, lambda T: get_dalphardT(T, rho_target), T_integration, T_target)
        with open(isoD_cachefile,'w') as fp:
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

    print('my s^+: ', -sex_kB[-1])
    print('Thol s^+: ', -LennardJones126.LJ_sr_over_R(T_target, rho_target))

    df = pandas.DataFrame('')
#    df = pandas.DataFrame(s2vals)
    if not df.empty:
        Ts = np.array(df['T'])
        alphar = alphar_int + ce_anti_isoD.y(Ts)-ce_anti_isoD.y(T_integration)
        sex_kB = (-alphar-Ts*ce_isoD.y(Ts))
        df['s^+'] = -sex_kB
        df.to_csv(prefix+'s2vals.csv', index=False)

        import matplotlib.pyplot as plt
        plt.plot(df['T'], df['s^+'], label=r'$s^+_{\rm total}$')
        plt.plot(df['T'], df['s^+_2'], label=r'$s^+_2$')
        plt.legend(loc='best')
        plt.savefig(prefix+'splus_terms.pdf')
        plt.close()

import uuid

def do_LJmix():
    assert(os.path.exists('LJmix'))
    massA = 1
    epsilonA = 1.0
    sigmaA = 1.0
    for massB in [1]:
        for epsilonB in [1,2,4]:
            for sigmaB in [1,2,4]:
                for numA in [1, 256, 512, 512+256, 1023]:
                    numB = 1024 - numA
                    sim_kwargs = dict(
                        massA=massA, massB=massB, 
                        epsilonA=epsilonA, epsilonB=epsilonB, 
                        numA=numA, numB=numB,
                        sigmaA=sigmaA, sigmaB=sigmaB
                    )
                    uid = str(uuid.uuid1())
                    with open('LJmix/'+uid+'.json','w') as fp:
                        fp.write(json.dumps(sim_kwargs))
                    cacher = RUMDCacher(sim_class=RUMDLJmixSimulation, sim_kwargs=sim_kwargs)
                    cacher.dump_isoT(10, rho_min = 0.01, rho_target=1, prefix='LJmix/'+uid+'_')

if __name__ == '__main__':

    pot_params = {'Epsilon': 1}
    cacher = RUMDCacher(sim_class=RUMDMonomerSimulation, sim_kwargs={'potential': 'LJ12-6',"pot_params": pot_params})
    cacher.dump_isoT(3, rho_min = 0.01, rho_target=1, prefix='LJ126')

    pot_params = dict(Alpha=14, rm=1.00, Epsilon=1.00, Rcut=5.5)
    cacher = RUMDCacher(sim_class=RUMDMonomerSimulation, sim_kwargs={'potential': 'EXP6',"pot_params": pot_params})
    cacher.dump_isoT(3, rho_min = 0.01, rho_target=1, prefix='EXP6')
    cacher.dump_isoT(4, rho_min = 0.01, rho_target=1, prefix='EXP6')
    cacher.dump_isoT(8, rho_min = 0.01, rho_target=1, prefix='EXP6')
