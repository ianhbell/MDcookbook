import subprocess, shutil, io, os
import pandas

# Build the chain
import chain_builder as cb; cb.build_chain(segment_density=0.9, Nchains=320, chain_length=8, ofname='data.fene', verbose=True)
# print(open('data.fene').read())

# Relax the chain to remove overlapping segments
subprocess.check_call('mpirun -np 4 --allow-run-as-root lammps -sf opt -in do_chain.lammps',shell=True)
# print(open('chain_equil.data').read())

# Do an equilibrium run from the de-overlapped configuration
subprocess.check_call('mpirun -np 4 --allow-run-as-root lammps -sf opt -in in.fene',shell=True)
print(open('out.dump').read())

if os.path.exists('/out'):
    shutil.copy2('out.dump','/out')

contents = open('out.dump').read().replace('# ','')
df = pandas.read_csv(io.StringIO(contents), sep=r'\s+',skiprows=1)
print('means:::::::::::::::::::::::::::')
for col in df.keys():
    print(col, df[col].mean())