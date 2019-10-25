"""
Carries out a calculation for the shear viscosity near
the triple point for Lennard-Jones fluid with RUMD
with the use of the Green-Kubo approach 

The value according to Meier et al. (doi:10.1063/1.1770695) 
at T^*=0.722, rho^*=0.8442 should be circa \eta^*=3.258.
"""
from __future__ import print_function, division

import timeit
import subprocess
import gzip
import io

from rumd import *
from rumd.Simulation import Simulation
from rumd.Autotune import Autotune
import rumd.analyze_energies as analyze
import rumd.Tools

import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import scipy.integrate, matplotlib.pyplot as plt, scipy.signal

def do_run(Tstar, rhostar):

    # Generate the starting state
    subprocess.check_call('rumd_init_conf --num_par=1000 --cells=10,10,10 --rho='+str(rhostar), shell=True)

    # Create simulation object
    sim = Simulation("start.xyz.gz")
    sim.SetOutputScheduling("trajectory", "logarithmic")
    sim.SetOutputScheduling("energies", "linear", interval=1)
    block_size = 16384*2
    sim.SetBlockSize(block_size)

    sim.SetOutputMetaData("energies",
                          stress_xy=True,stress_xz=True,stress_yz=True,
                          kineticEnergy=False,potentialEnergy=False,temperature=False,
                          totalEnergy=False,virial=False,pressure=False)

    # Create potential object.
    pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
    pot.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=2.5);
    sim.SetPotential(pot)

    # Create integrator object
    itg = IntegratorNVT(timeStep=0.0025, targetTemperature=Tstar)
    sim.SetIntegrator(itg)

    # Equilibration for 300k steps
    sim.Run(300000, suppressAllOutput=True)
    # Production 
    prod_steps = block_size*400
    print(prod_steps, 'production time steps')
    sim.Run(prod_steps)

    # End of run.
    sim.sample.WriteConf("end.xyz.gz")
    sim.sample.TerminateOutputManagers()

def ACF_FFT(v, Norigins):
    """ 
    See https://github.com/Allen-Tildesley/examples/blob/master/python_examples/corfun.py
    """
    nstep = len(v)
    n = np.linspace(nstep,nstep-Norigins,Norigins+1,dtype=np.float_)
    assert np.all(n>0.5), 'Normalization array error' # Should never happen

    # Data analysis (FFT method)
    fft_len = 2*nstep # Actual length of FFT data

    # Prepare data for FFT
    fft_inp = np.zeros(fft_len,dtype=np.complex_) # Fill input array with zeros
    fft_inp[0:nstep] = v                          # Put data into first part (real only)
    fft_out = np.fft.fft(fft_inp) # Forward FFT
    fft_inp = np.fft.ifft(fft_out * np.conj ( fft_out )) # Backward FFT of the square modulus (the factor of 1/fft_len is built in)
    return fft_inp[0:Norigins+1].real / n

def post_process():

    V = 1184.55 # LJ units
    Tstar = 0.722 # LJ units

    f_autocorrelation = ACF_FFT

    # Create AnalyzeEnergies object, read relevant columns from energy files
    nrgs = analyze.AnalyzeEnergies()
    nrgs.read_energies(['sxy', 'syz', 'sxz'])
    sxy = nrgs.energies['sxy']
    time = nrgs.metadata['interval']*np.arange(len(sxy)) # "Real" time associated with each store

    # Green-Kubo analysis
    SACF = sum([f_autocorrelation(nrgs.energies[k], Norigins=len(sxy)-2) for k in ['sxy', 'syz', 'sxz']])/3.0
    time_ACF = nrgs.metadata['interval']*np.arange(0, len(SACF)) # interval is total time between dumps
    int_SACF = V/Tstar*scipy.integrate.cumtrapz(SACF, time_ACF, initial=0)
    # The autocorrelation function depends on the number of time origin points taken, 
    # but the time origin curves are coincident, and overlap perfectly, so the first local
    # maximum of the SACF is the value to consider when using N-2 time origins
    i1stmaxima = scipy.signal.argrelmax(int_SACF)[0][0]
    print('G-K eta^*:', int_SACF[i1stmaxima])

    def diagnostic_plots():

        # # Einstein analysis
        # Einstein_integrand = scipy.integrate.cumtrapz(sxy, time, initial=0)**2
        # AC_sxy = f_autocorrelation(Einstein_integrand, Norigins=len(sxy)-1)
        # x, y = [], AC_sxy*V/(Tstar*2)
        # plt.plot(y)
        # # print('best fit second half', np.polyfit(x[len(x)//2::], y[len(x)//2::], 1)) # decreasing order
        # plt.xlabel(r'$t^*$')
        # plt.ylabel(r'$\left\langle \int_0^t \tau_{\alpha\beta}(x) {\rm d} x \right\rangle$')
        # plt.tight_layout(pad=0.2)
        # plt.savefig('Einstein_term.pdf')
        # plt.close()

        for Norigins in [10, 100, 1000, 10000, 100000]:
            SACF = sum([f_autocorrelation(nrgs.energies[k], Norigins=Norigins) for k in ['sxy', 'syz', 'sxz']])/3.0
            time_ACF = nrgs.metadata['interval']*np.arange(0, len(SACF)) # interval is total simulation time interval between dumps
            int_SACF = V/Tstar*scipy.integrate.cumtrapz(SACF, time_ACF, initial=0)
            print(Norigins, 'G-K eta^*:', int_SACF[-1])
            plt.plot(int_SACF)
            print(int_SACF)
            plt.plot(len(int_SACF), int_SACF[-1], 'd')
        plt.xscale('log')
        plt.savefig('selection_of_Norigins.pdf')
        plt.close()

        plt.plot(time_ACF, SACF)
        plt.xlabel(r'$t^*$')
        plt.ylabel(r'$\left\langle \tau_{\alpha\beta}(x) \right\rangle$')
        plt.tight_layout(pad=0.2)
        plt.savefig('SACF_GK.pdf')
        plt.close()

        plt.plot(time_ACF, int_SACF)
        plt.xlabel(r'$t^*$')
        plt.ylabel(r'$\int_0^{t^*} \left\langle \tau_{\alpha\beta}(0)\tau_{\alpha\beta}(x) \right\rangle {\rm d} x$')
        plt.tight_layout(pad=0.2)
        plt.savefig('GK_runningintegral.pdf')
        plt.close()
    # diagnostic_plots()

if __name__ == '__main__':
    do_run(rhostar=0.8442, Tstar=0.722)
    post_process()