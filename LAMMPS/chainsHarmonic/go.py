import subprocess
import chain_builder as cb; cb.build_chain(segment_density=0.8442, Nchains=320, chain_length=100, ofname='data.fene')
# print(open('data.fene').read())
subprocess.check_call('mpirun -np 1 --allow-run-as-root lammps -sf opt -in in.fene',shell=True)
print(open('out.dump').read())