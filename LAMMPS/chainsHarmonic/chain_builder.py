template = r"""Polymer chain definition

{segment_density:g}          rhostar
592984          random # seed (8 digits or less)
1               # of sets of chains (blank line + 6 values for each set)
0               molecule tag rule: 0 = by mol, 1 = from 1 end, 2 = from 2 ends

{Nchains:d}     number of chains
{chain_length:d}             monomers/chain
1               type of monomers (for output into LAMMPS file)
1               type of bonds (for output into LAMMPS file)
1.0             distance between monomers (in reduced units)
1.05            no distance less than this from site i-1 to i+1 (reduced unit)"""

import subprocess, io

def build_chain(*, segment_density, Nchains, chain_length, ofname, verbose=False):
    definition = template.format(Nchains=Nchains, chain_length=chain_length, segment_density=segment_density).encode('ascii')
    if verbose:
        print(definition.decode())
    
    from subprocess import Popen, PIPE, STDOUT
    p = Popen(['/chain'], stdout=PIPE, stdin=PIPE, stderr=STDOUT, shell=True)
    chain_stdout = p.communicate(input=definition)[0]
    with open(ofname,'w') as fp:
        fp.write(chain_stdout.decode())

if __name__ == '__main__':
    build_chain(segment_density=0.8442, Nchains=100, chain_length=4, ofname='chain.fene')