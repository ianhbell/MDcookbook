import os, subprocess, gzip, glob
import numpy as np
import pandas

####### SETUP OF TOPOLOGY AND MOLECULE #######

# Columns are molecule number, x, y, z
xyzheader = """{chain_length:d}
numTypes=1 mass=1.0
"""

# Topology of the molecule (how the segments are connected)
# Columns are molecule index, atom index #1, atom index #2, bond type
topo_header = """[ bonds ]
; Single FENE chain
"""

def build_topo(*, chain_length):
    o = topo_header
    for i in range(chain_length-1):
        left = i 
        right = i+1
        o += f'0 {left:d} {right:d} 0\n'
    return o

def build_xyz(*, chain_length):
    o = xyzheader.format(chain_length=chain_length)
    for i in range(chain_length):
        z = i
        o += f'0 0 0 {z:d}\n'
    return o

def generate_files(*, chain_length):

    with open('single_fene.top','w') as fp:
        fp.write(build_topo(chain_length=chain_length))

    with open('single_fene.xyz','w') as fp:
        fp.write(build_xyz(chain_length=chain_length))

    with open('rumd_init_conf_mol.stdout','w') as fp:
        subprocess.check_call('rumd_init_conf_mol single_fene.xyz single_fene.top 500', shell=True, stdout=fp)

####### RUN SCRIPT ######

import rumd
from rumd.Simulation import Simulation
from rumd.RunCompress import RunCompress

def collate_energies(memory_max_Mb, **kwargs):
    """
    Collect the energies terms
    """

    def get_meta():
        path = 'TrajectoryFiles/energies0000.dat.gz'
        with gzip.open(path, 'rb') as fp:
            line0 = fp.readlines()[0].decode('ascii')
            meta = line0[1::].strip().split(' ')
            o = {}
            for thing in meta:
                k, v = thing.split('=')
                o[k] = v
            return o

    def get_one(filename, meta, coldrop=None, colkeep=None):
        df = pandas.read_csv(filename, names = meta['columns']+['dummy'], comment='#', sep=r'\s+')
        if coldrop is not None and colkeep is None:
            df.drop(columns=coldrop, inplace=True)
        elif coldrop is None and colkeep is not None:
            df = df.filter(items=colkeep)        
        blocktime = len(df)*meta['Dt']
        I = int(os.path.split(filename)[1].split('.')[0].replace('energies',''))
        df['t'] = meta['Dt']*np.arange(0, len(df)) + I*blocktime
        return df

    files = glob.glob('TrajectoryFiles/energies*.dat.gz')
    meta = get_meta()
    meta['columns'] = meta['columns'].split(',')
    meta['Dt'] = float(meta['Dt'])
    memory_Mb = 0
    dfs = []
    for i, filename in enumerate(files):
        df = get_one(filename, meta, **kwargs)
        memory_Mb += df.memory_usage(index=True).sum()/1024**2
        if memory_Mb > memory_max_Mb:
            raise MemoryError('Dataframe too large')
        if i % 20 == 0:
            print(i, '/', len(files), 'memory:', memory_Mb, 'Mb')
        dfs.append(df)
    df = pandas.concat(dfs,sort=False).sort_values(by='t')
    return df  

def run_one(*, bond_method, chain_length, Tstar, segment_density):

    generate_files(chain_length=chain_length)

    # create simulation object
    sim = Simulation("start.xyz.gz", verbose=False)

    BlockSize = 131072//2
    NoBlocks = 10
    sim.SetVerbose(False)

    sim.SetBlockSize(BlockSize)
    sim.SetOutputScheduling("trajectory", "logarithmic")
    sim.SetOutputScheduling("energies", "linear", interval=4)

    sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=True,
        #potentialVirial=True, constraintVirial=True
    )

    # Read topology file
    sim.ReadMoleculeData("start.top")

    # Create integrator object
    itg = rumd.IntegratorNVT(targetTemperature=Tstar, timeStep=0.00025)
    sim.SetIntegrator(itg)

    # Create potential object; potential is applied initially to all pairs of sites, both in the same chain and cross
    potential = rumd.Pot_LJ_12_6(cutoff_method=rumd.NoShift)
    potential.SetParams(i=0, j=0, Epsilon=1.0, Sigma=1.0, Rcut=3)
    sim.AddPotential(potential)

    cons_pot = None

    if bond_method == 'Constrained':
        # The distance between all bonds of type = 0 is length/sigma = 1.0
        cons_pot = rumd.ConstraintPotential()
        cons_pot.SetParams(bond_type=0, bond_length=1.0) # Does not have an exclude option, so does it exclude or not?
        sim.AddPotential(cons_pot)
        # Iterate the linear set of equations
        cons_pot.SetNumberOfConstraintIterations(5)
    elif bond_method == 'Harmonic':
        # Harmonic bond
        pot_harm = rumd.BondHarmonic()
        # N.B. 1: exclude=True means that the potential defined above will not be used for the bonded interaction, and only the harmonic bit will be
        # N.B. 2: The difference between exclude=True and exclude=False should be very small because the center of the bond is at 
        #         sigma=1, at which separation the potential has a value of zero
        stiffness = 3000.0
        pot_harm.SetParams(bond_type=0, stiffness=stiffness, bond_length=1.0, exclude=True)
        sim.AddPotential(pot_harm)
    elif bond_method == 'FENE':
        # Finite Extensible Nonlinear Elastic (FENE) potential between segments of chain
        pot_FENE = rumd.BondFENE()
        # N.B.: exclude=False means that the potential defined above will be used *in concert* with the FENE potential for the bond interaction
        pot_FENE.SetParams(bond_type=0, bond_length=0.75, stiffness=30.0, exclude=False) 
        sim.AddPotential(pot_FENE)
    else:
        raise KeyError(f"Bad bond method: {bond_method:s}")

    RunCompress(sim, final_density=segment_density) # final_density is the number of segments per volume

    # Run simulation
    sim.Run(BlockSize,suppressAllOutput=True)
    sim.Run(BlockSize*NoBlocks)

    if cons_pot is not None:
        # Check the bond standard deviation
        cons_pot.WritePotential(sim.sample.GetSimulationBox())

    sim.WriteConf("end.xyz.gz")
    sim.sample.TerminateOutputManagers()

    # Create a rumd_stats object
    rs = rumd.Tools.rumd_stats()
    rs.ComputeStats()
    meanVals = rs.GetMeanVals()
    # rs.WriteStats()
    rs.PrintStats()

    # Store the outputs
    U_per_particle = meanVals['pe']
    W_per_particle = meanVals['W']
    p = meanVals['p']
    
    Z = p/((segment_density/chain_length)*Tstar)
    W_per_NmolkBT = Z-1
    W_per_chain = W_per_NmolkBT*Tstar

    # Calculate an approximate value for the potential energy coming from the bonds
    Ubonds_per_chain = 0.0
    if bond_method == 'Harmonic':
        subprocess.check_call('rumd_bonds -n 1000 -t start.top', shell=True)
        df = pandas.read_csv('bonds.dat', sep=' ', comment='#', names = ['length','count','dummy'])
        del df['dummy']
        # df.info()
        # Calculate the average energy per bond based on bond length distribution
        Ubonds_per_bond = stiffness/2.0*((df['length']-1.0)**2*df['count']).sum()/df['count'].sum()
        # Then calculate the bonding energy per chain
        Nbonds = len([line for line in open('start.top').readlines() if line.strip()]) -2 # Total number of bonds (-2 for two lines of header; each non-empty line is a bond)
        Natoms = sim.GetNumberOfParticles() # Total number of atoms
        Ubonds_per_chain = Ubonds_per_bond*Nbonds/Natoms*chain_length

    ########################  FLUCTUATION PROPERTIES ############################
    # We (inefficiently) reload the data because for very, very long simulations 
    # we don't have enough memory to load everything at once
    Mb_max = 6000 # Max amount of memory usage allowed, in megabytes
    df = collate_energies(Mb_max, colkeep=['pe', 'W'])
    DeltaU = df.pe - df.pe.mean()
    DeltaW = df.W  - df.W.mean()
    R_Roskilde = np.mean(DeltaW*DeltaU)/(np.mean(DeltaU**2)*np.mean(DeltaW**2))**0.5
    gamma_IPL = np.mean(DeltaW*DeltaU)/np.mean(DeltaU**2)
    Nparticles = sim.GetNumberOfParticles()
    cvexstar = np.mean(DeltaU**2)/Tstar**2*Nparticles # cv_ex^* = c_{v,ex}/k_B

    o = {
        'chain_length': chain_length,
        'Tstar': Tstar,
        'segment_density': segment_density,
        'U/chain': U_per_particle*chain_length,
        'Ur/chain': U_per_particle*chain_length - Ubonds_per_chain,
        'W/chain': W_per_chain,
        'W/chain #2': (p/(segment_density*Tstar)-1)*Tstar,
        'P': meanVals['p'],
        'gamma_IPL': gamma_IPL,
        'R_Roskilde': R_Roskilde,
        'cv_ex': cvexstar
    }
    print(o)

    print('U/chain:', o['U/chain'])
    print('Ur/chain:', o['Ur/chain'])
    print('W/chain:', W_per_chain, "(Note: seeming bug in RUMD for calculation of W for chains)")
    print('dalphar/drho_chain:', (o['W/chain']/o['segment_density']*chain_length*Tstar))
    print('P', o['P'])
    return o

if __name__ == '__main__':
    # for bond_method in ['Harmonic']:#,'Constrained']:
    #     print('\n***Bond method:', bond_method)
    #     run_one(bond_method=bond_method, chain_length=12, Tstar=4, segment_density=0.5)

    from therm_integration import one_integration
    segment_density = 0.2901
    Tstar = 8
    chain_length = 4
    chain_density = segment_density/chain_length

    # (W/Nmol)/(kB*T) = rho_chain*dalphar/drho_chain
    get_dalphardrho = lambda T, rho_chain: run_one(bond_method='Harmonic', chain_length=chain_length, Tstar=T, segment_density=rho_chain*chain_length)['W/chain']/(rho_chain*T)
    # (U/Nmol)/(kB*T) = -T*dalphar/dT
    get_dalphardT = lambda T, rho_chain: run_one(bond_method='Harmonic', chain_length=chain_length, Tstar=T, segment_density=rho_chain*chain_length)['Ur/chain']/(-T**2)
    one_integration(T_integration=Tstar, T_target=Tstar, rho_target=chain_density, get_dalphardrho=get_dalphardrho, get_dalphardT=get_dalphardT)

    o = []
    for segment_density in [0.3]:
        for Tstar in np.geomspace(3, 50, 30):
            o.append(run_one(bond_method='Harmonic', chain_length=4, Tstar=Tstar, segment_density=segment_density))
    pandas.DataFrame(o).to_csv('gamma_chains.csv', index=False)
    quit()

    # Johnson 12-mer results
    o = []
    for Tstar, segment_density in [
        (4.0,0.9),(4.0,0.8),(4.0,0.7),(4.0,0.5),(4.0,0.3),(4.0,0.1),
        (3.0,0.9),(3.0,0.8),(3.0,0.7),(3.0,0.5),(3.0,0.3),(3.0,0.1),
        (2.0,0.9),(2.0,0.8),(2.0,0.7)]:
        o.append(run_one(bond_method='Harmonic', chain_length=12, Tstar=Tstar, segment_density=segment_density))
    pandas.DataFrame(o).to_csv('Johnson12mer.csv', index=False)
