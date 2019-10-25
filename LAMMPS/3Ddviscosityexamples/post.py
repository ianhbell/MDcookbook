import pandas, io, numpy as np, matplotlib.pyplot as plt, scipy.integrate

def ACF_FFT(v, Norigins):
    """ 
    See https://github.com/Allen-Tildesley/examples/blob/master/python_examples/corfun.py
    """
    nstep = len(v)
    nt = Norigins
    n = np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_)
    assert np.all(n>0.5), 'Normalization array error' # Should never happen

    # Data analysis (FFT method)
    fft_len = 2*nstep # Actual length of FFT data

    # Prepare data for FFT
    fft_inp = np.zeros(fft_len,dtype=np.complex_) # Fill input array with zeros
    fft_inp[0:nstep] = v                          # Put data into first part (real only)
    fft_out = np.fft.fft(fft_inp) # Forward FFT
    fft_inp = np.fft.ifft(fft_out * np.conj ( fft_out )) # Backward FFT of the square modulus (the factor of 1/fft_len is built in)
    return fft_inp[0:nt+1].real / n

def ACF_numpy(v, Norigins):
    """ 
    See https://github.com/Allen-Tildesley/examples/blob/master/python_examples/corfun.py
    """
    nstep = len(v)
    nt = Norigins
    n = np.linspace(nstep,nstep-nt,nt+1,dtype=np.float_)
    assert np.all(n>0.5), 'Normalization array error' # Should never happen
    c_full = np.correlate(v, v, mode='full')
    mid = c_full.size//2
    return c_full[mid:mid+nt+1]/n

def load_dump(path):
    with open(path, 'r') as fp:
        contents = fp.read()
        header_row = contents.split('\n')[1].strip()
        df = pandas.read_csv(io.StringIO(contents), sep=' ', comment='#', names=header_row[2::].split(' '))
        df['time'] = df['TimeStep']*0.0025
        return df

def calc_avg_autocorr(data):
    """
    data is a numpy array
    Thanks to Alta
    """
    # Some parameters for averaging over time origins
    n_window = 100       # Length of window
    n_istart_freq = 10   # Lag between windows

    # Specify time origins
    i_starts = np.arange(0, len(data) - n_window + 1, n_istart_freq)

    # Compute auto-correlation function, averaged over time origins.
    ACF = np.zeros(n_window)
    for i_start in i_starts:
        ACF += data[i_start]*data[i_start:i_start + n_window]
    ACF = ACF/float(len(i_starts))

    return ACF

def GreenKubo(df):
    V = 1200/0.8442 # LJ units
    Tstar = np.mean(df['v_Temp'])
    ys = 0
    for key in ['v_pxy', 'v_pxz', 'v_pyz']:
        ACF = ACF_FFT(np.array(df[key]), Norigins = 100)
        y = scipy.integrate.cumtrapz(ACF, initial=0)*V/Tstar*0.0025*10
        ys += y
        plt.plot(y, lw=0.4)
    print('eta^*:', ys[-1]/3)
    plt.plot(ys/3, lw = 3)
    plt.show()

def Einstein(df):
    V = 1200/0.8442 # LJ units
    Tstar = np.mean(df['v_Temp'])
    sumy = 0
    for key in ['v_pxy','v_pxz', 'v_pyz']:
        integrated = scipy.integrate.cumtrapz(np.array(df[key]), df['time'], initial=0)
        ACF_ab = ACF_FFT(integrated**2, Norigins = 1000)
        y = ACF_ab*V/Tstar/2*0.0025
        # plt.plot(y, lw=0.75)
        sumy += y
    plt.plot(sumy/3, lw = 3)
    plt.show()

if __name__ == '__main__':
    df = load_dump('out.stressdump')
    GreenKubo(df)
    Einstein(df)