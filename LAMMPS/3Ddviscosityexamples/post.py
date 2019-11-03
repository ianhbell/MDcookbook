import pandas, io, numpy as np, matplotlib.pyplot as plt, scipy.integrate, scipy.signal

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

def GreenKubo(df, V):
    Tstar = np.mean(df['v_Temp'])
    ys = 0
    df['v_Time'] -= df['v_Time'].iloc[0]
    print(df.head(5))
    for key in ['v_pxy', 'v_pxz', 'v_pyz']:
        ACF = ACF_FFT(np.array(df[key]), Norigins = len(df)-1)
        y = scipy.integrate.cumtrapz(ACF, df['v_Time'], initial=0)*V/Tstar
        ys += y
        plt.plot(df['v_Time'], y, lw=0.4)
    # The autocorrelation function depends on the number of time origin points taken, 
    # but the time origin curves are coincident, and overlap perfectly, so the first local
    # maximum of the SACF is the value to consider when using N-2 time origins
    i1stmaxima = scipy.signal.argrelmax(ys)[0][0]
    print('G-K eta^*:', ys[i1stmaxima]/3)
    plt.plot(df['v_Time'].iloc[i1stmaxima], ys[i1stmaxima]/3, 'd')
    plt.plot(df['v_Time'], ys/3, lw = 3)

    # Fit function up to the first maximum
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
    print((A*alpha*tau1 + A*(1-alpha)*tau2)/3)
    t = np.linspace(0, df.v_Time.iloc[i1stmaxima]*100, 10000)
    y = func(t, coeffs)
    plt.plot(t, y/3)
    plt.xlim(0, df['v_Time'].iloc[i1stmaxima*20])
    plt.ylim(0, df['v_Time'].iloc[i1stmaxima*20])

    plt.xlabel('ACF time points')
    plt.ylabel(r'$\int_0^{t^*} \left\langle \tau_{\alpha\beta}(0)\tau_{\alpha\beta}(x) \right\rangle {\rm d} x$')
    plt.savefig('GreenKubo.pdf')
    plt.show()

def Einstein(df, V):
    Tstar = np.mean(df['v_Temp'])
    sumy = 0
    for key in ['v_pxy','v_pyz','v_pxz']:
        integrated = scipy.integrate.cumtrapz(y=np.array(df[key]), x=df['time'], initial=0)
        ACF_ab = ACF_FFT(integrated**2, Norigins = len(df)-2)
        y = ACF_ab*V/(2*Tstar)*0.003
        sumy += y
    plt.plot(sumy/3, lw = 3)
    plt.show()

if __name__ == '__main__':
    df = load_dump('out.stressdump')
    V = 1200/0.8442 # LJ units
    GreenKubo(df, V)
    Einstein(df, V)