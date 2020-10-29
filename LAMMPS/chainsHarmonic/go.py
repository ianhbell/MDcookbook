import subprocess, shutil, io, os
import pandas

def run_one(*, Tstar, segment_density, chain_length):

    if not isinstance(Tstar, float):
        Tstar = float(Tstar)

    # Build the chain
    import chain_builder as cb; cb.build_chain(segment_density=segment_density, Nchains=320, chain_length=chain_length, ofname='data.fene', verbose=True)
    # print(open('data.fene').read())

    Nproc = 12

    # Relax the chain to remove overlapping segments
    subprocess.check_call(f'mpirun -np {Nproc} --allow-run-as-root lammps -sf opt -in do_chain.lammps -var Tstar {Tstar}',shell=True)
    # print(open('chain_equil.data').read())

    # Do an equilibrium run from the de-overlapped configuration
    subprocess.check_call(f'mpirun -np {Nproc} --allow-run-as-root lammps -sf opt -in do_run.lammps -var segment_density {segment_density} -var Tstar {Tstar}',shell=True)
    # print(open('out.dump').read())

    # Copy it into the output folder if that folder exists
    if os.path.exists('/out'):
        shutil.copy2('out.dump','/out')

    contents = open('out.dump').read().replace('# ','')
    df = pandas.read_csv(io.StringIO(contents), sep=r'\s+',skiprows=1)
    # print('means:::::::::::::::::::::::::::')
    # for col in df.keys():
    #     print(col, df[col].mean())

    return {
        'chain_length': chain_length,
        'T': df['v_Temp'].mean(),
        'p': df['v_Press'].mean(),
        'rho_seg': df['v_Natoms'].mean()/df['v_Volume'].mean(),
        'Unonbonded/chain': df['v_vdWEnergy'].mean()*chain_length, # vdWEnergy is per atom
        'Uintra/chain': df['v_vdWIntra'].mean()/df['v_Natoms'].mean()*chain_length, # outputs of the compute pe/mol/all are extensive
        'Uinter/chain': df['v_vdWInter'].mean()/df['v_Natoms'].mean()*chain_length, # outputs of the compute pe/mol/all are extensive
    }

if __name__ == '__main__':
    assert(os.path.exists('/out'))
    # with open('/out/something','w') as fp:
    #     fp.write('hihi')

    outputs = []
    for segment_density in [0.001]:
        for Tstar in [2.05,3.00,4.00,6.00,8.00,10.00,50.00,100.00]:
            try:
                outputs.append(run_one(Tstar=Tstar, chain_length=3, segment_density=segment_density))
            except:
                pass
    pandas.DataFrame(outputs).to_csv('/out/chains3_isolated.csv', index=False)

#    outputs = []
#    for segment_density in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
#        try:
#            outputs.append(run_one(Tstar=10.0, chain_length=3, segment_density=segment_density))
#        except:
#            pass
#    pandas.DataFrame(outputs).to_csv('/out/chains3_T4.csv', index=False)
