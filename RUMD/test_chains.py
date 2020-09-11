import sys, os, subprocess
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
        subprocess.check_call('rumd_init_conf_mol single_fene.xyz single_fene.top 100', shell=True, stdout=fp)

####### RUN SCRIPT ######

import rumd
from rumd.Simulation import Simulation
from rumd.RunCompress import RunCompress

def run_one(*, bond_method, chain_length, Tstar, segment_density):

    generate_files(chain_length=chain_length)

    # create simulation object
    sim = Simulation("start.xyz.gz", verbose=False)

    BlockSize = 131072
    NoBlocks = 10
    sim.SetVerbose(False)

    sim.SetBlockSize(BlockSize)
    sim.SetOutputScheduling("trajectory", "logarithmic")
    sim.SetOutputScheduling("energies", "linear", interval=128)

    sim.SetOutputMetaData(
        "energies",
        stress_xy=False, stress_xz=False, stress_yz=False,
        kineticEnergy=False, potentialEnergy=True, temperature=True,
        totalEnergy=False, virial=True, pressure=True, volume=True
    )

    # Read topology file
    sim.ReadMoleculeData("start.top")

    # Create integrator object
    itg = rumd.IntegratorNVT(targetTemperature=Tstar, timeStep=0.0025)
    sim.SetIntegrator(itg)

    # Create potential object; potential is applied initially to all pairs of sites, both in the same chain and cross
    potential = rumd.Pot_LJ_12_6(cutoff_method=rumd.NoShift)
    potential.SetParams(i=0, j=0, Epsilon=1.0, Sigma=1.0, Rcut=20)
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
        # N.B.: exclude=True means that the potential defined above will not be used for the bonded interaction, and only the harmonic bit will be
        stiffness = 3000.0
        pot_harm.SetParams(bond_type=0, stiffness=stiffness, bond_length=1.0, exclude=True) # exclude seems to have no impact here...
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

    # Store the outputs
    U_per_particle = meanVals['pe']
    W_per_particle = meanVals['W']

    # Calculate an approximate value for the potential energy coming from the bonds
    Ubonds_per_chain = 0.0
    if bond_method == 'Harmonic':
        subprocess.check_call('rumd_bonds -n 10000 -t start.top', shell=True)
        df = pandas.read_csv('bonds.dat', sep=' ', comment='#', names = ['length','count','dummy'])
        del df['dummy']
        # df.info()
        # Calculate the average energy per bond based on bond length distribution
        Ubonds_per_bond = stiffness/2.0*((df['length']-1.0)**2*df['count']).sum()/df['count'].sum()
        # Then calculate the bonding energy per chain
        Nbonds = len([line for line in open('start.top').readlines() if line.strip()]) -2 # Total number of bonds (-2 for two lines of header; each non-empty line is a bond)
        Natoms = sim.GetNumberOfParticles() # Total number of atoms
        Ubonds_per_chain = Ubonds_per_bond*Nbonds/Natoms*chain_length

    o = {
        'chain_length': chain_length,
        'Tstar': Tstar,
        'segment_density': segment_density,
        'U/chain': U_per_particle*chain_length,
        'Ur/chain': U_per_particle*chain_length - Ubonds_per_chain,
        'W/chain': W_per_particle*chain_length,
        'P': meanVals['p']
    }
    print(o)

    print('U/chain:', o['U/chain'])
    print('Ur/chain:', o['Ur/chain'])
    print('W/chain:', o['W/chain'])
    print('P', o['P'])
    return o

if __name__ == '__main__':
    # for bond_method in ['Harmonic']:#,'Constrained']:
    #     print('\n***Bond method:', bond_method)
    #     run_one(bond_method=bond_method, chain_length=12, Tstar=4, segment_density=0.5)

    # Johnson 12-mer results
    o = []
    for Tstar, segment_density in [
        (4.0,0.9),(4.0,0.8),(4.0,0.7),(4.0,0.5),(4.0,0.3),(4.0,0.1),
        (3.0,0.9),(3.0,0.8),(3.0,0.7),(3.0,0.5),(3.0,0.3),(3.0,0.1),
        (2.0,0.9),(2.0,0.8),(2.0,0.7)]:
        o.append(run_one(bond_method='Harmonic', chain_length=12, Tstar=Tstar, segment_density=segment_density))
    pandas.DataFrame(o).to_csv('Johnson12mer.csv', index=False)
