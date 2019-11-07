"""
Carries out a calculation for the shear viscosity near
the triple point for Lennard-Jones fluid with RUMD
with the use of the Green-Kubo approach 

The value according to Meier et al. (doi:10.1063/1.1770695) 
at T^*=0.722, rho^*=0.8442 should be circa \eta^*=3.258.

For self diffusion at the same state point, the value should be
circa D^*=0.0325 (doi: 10.1063/1.1786579) for 1372 particles, but
this is not nearly enough to capture the infinite size limit
"""
from __future__ import print_function, division

import timeit
import subprocess
import gzip
import io
import json

from rumd import *
from rumd.Simulation import Simulation
from rumd.Autotune import Autotune
import rumd.analyze_energies as analyze
import rumd.Tools

import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import scipy.integrate, matplotlib.pyplot as plt, scipy.signal, scipy.optimize

def do_run(Tstar, rhostar):

    # Generate the starting state
    subprocess.check_call('rumd_init_conf --num_par=1331 --cells=11,11,11 --rho='+str(rhostar), shell=True)

    # Create simulation object
    sim = Simulation("start.xyz.gz")
    sim.SetOutputScheduling("trajectory", "logarithmic")
    sim.SetOutputScheduling("energies", "linear", interval=1)
    block_size = 16384//2
    sim.SetBlockSize(block_size)

    sim.SetOutputMetaData("energies",
                          stress_xy=True,stress_xz=True,stress_yz=True,
                          kineticEnergy=False,potentialEnergy=False,temperature=True,
                          totalEnergy=False,virial=False,pressure=False,volume=True)

    # Create potential object.
    pot = Pot_LJ_12_6(cutoff_method=ShiftedPotential)
    pot.SetParams(i=0, j=0, Sigma=1.00, Epsilon=1.00, Rcut=6.5);
    sim.SetPotential(pot)

    # Create integrator object
    itg = IntegratorNVT(timeStep=0.0025, targetTemperature=Tstar)
    sim.SetIntegrator(itg)

    # Autotune
    at = Autotune()
    at.Tune(sim)

    # Equilibration for 300k steps
    sim.Run(300*10**3, suppressAllOutput=True)
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

    f_autocorrelation = ACF_FFT

    def diagnostic_plots():

        # Create AnalyzeEnergies object, read relevant columns from energy files
        nrgs = analyze.AnalyzeEnergies()
        nrgs.read_energies(['sxy', 'syz', 'sxz', 'T', 'V'])
        sxy = nrgs.energies['sxy']
        Tstar = np.mean(nrgs.energies['T'])
        V = np.mean(nrgs.energies['V'])
        time = nrgs.metadata['interval']*np.arange(len(sxy)) # "Real" time associated with each store

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
    #diagnostic_plots()

    def shear_viscosity_GreenKubo():

        # Create AnalyzeEnergies object, read relevant columns from energy files
        nrgs = analyze.AnalyzeEnergies()
        nrgs.read_energies(['sxy', 'syz', 'sxz','T', 'V'])
        sxy = nrgs.energies['sxy']
        Tstar = np.mean(nrgs.energies['T'])
        V = np.mean(nrgs.energies['V'])
        Nparticles = nrgs.metadata['num_part']
        time = nrgs.metadata['interval']*np.arange(len(sxy)) # "Real" time associated with each store
        rhostar = Nparticles/V

        # Green-Kubo analysis
        SACF = sum([f_autocorrelation(nrgs.energies[k], Norigins=len(sxy)-2) for k in ['sxy', 'syz', 'sxz']])/3.0
        time_ACF = nrgs.metadata['interval']*np.arange(0, len(SACF)) # interval is total time between dumps
        int_SACF = V/Tstar*scipy.integrate.cumtrapz(SACF, time_ACF, initial=0)
        # The autocorrelation function depends on the number of time origin points taken, 
        # but the time origin curves are coincident, and overlap perfectly, so the first local
        # maximum of the SACF is the value to consider when using N-2 time origins
        i1stmaxima = scipy.signal.argrelmax(int_SACF)[0][0]
        print('G-K eta^*:', int_SACF[i1stmaxima])

        # Fit a double-exponential function to the data up to the first local maximum
        def func(t, coeffs):
            A, alpha, tau1, tau2 = coeffs
            return A*alpha*tau1*(1-np.exp(-t/tau1)) + A*(1-alpha)*tau2*(1-np.exp(-t/tau2))

        def objective(coeffs, t, yinput):
            yfit = func(t, coeffs)
            return ((yfit-yinput)**2).sum()

        res = scipy.optimize.differential_evolution(
            objective,
            bounds = [(0.0001,100),(-1000, 1000),(-1000,1000),(-100,100)],
            disp = True,
            args = (df['v_Time'].iloc[0:i1stmaxima], ys[0:i1stmaxima])
        )
        coeffs = res.x 
        print(coeffs)
        A, alpha, tau1, tau2 = coeffs
        etastar_func = (A*alpha*tau1 + A*(1-alpha)*tau2)/3
        print('G-K eta^* from double-exponential function:', etastar_func)

        return Tstar, rhostar, etastar_func, int_SACF[i1stmaxima]

    Tstar, rhostar, etastar, etastar_1stmax = shear_viscosity_GreenKubo()

    def self_diffusion_Einstein():

        rdf_obj = rumd.Tools.rumd_rdf()
        # Constructor arguments: number of bins and minimum time
        rdf_obj.ComputeAll(1000, 1.0)
        # Include the state point information in the rdf file name
        rdf_obj.WriteRDF("rdf.dat")

        # Run rumd_sq to find qmax for particle type 0 (only one particle type)
        subprocess.check_call('rumd_sq 0.1 100 0.85', shell=True) # args are qmin, qmax, density (not clear what density is)
        qmax = float(open('qmax.dat').read())

        # Write MSD data to a file
        msd_obj = rumd.Tools.rumd_msd()
        msd_obj.SetQValues([qmax])
        msd_obj.ComputeAll()
        msd_obj.WriteMSD("msd.dat")

        df = pandas.read_csv("msd.dat", sep=' ', names=['t*', 'MSD'])
        plt.plot(df['t*'], df['MSD'], 'o')
        dfend = df.tail(5)
        m,b = np.polyfit(dfend['t*'], dfend['MSD'], 1)
        print('Einstein D*:', m/6)
        plt.xlabel(r'$t^*$')
        plt.ylabel(r'MSD / ??')
        plt.savefig('MSD.pdf')
        plt.close()
        return m/6
    Dstar = self_diffusion_Einstein()

    with open('results.json', 'w') as fp:
        fp.write(json.dumps(
            {
                'T^*': Tstar,
                'rho^*': rhostar,
                'eta^* (1st maximum)':etastar_1stmax,
                'eta^*':etastar_func,
                'D^*':Dstar
            }
        ))

if __name__ == '__main__':
    do_run(Tstar=0.722, rhostar=0.8442)
    post_process()
