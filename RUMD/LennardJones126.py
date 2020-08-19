from __future__ import division
from numpy import exp, log, pi, sqrt
import numpy as np
import scipy.optimize

kB = 1.38064852e-23 
N_A = 6.022140857e23

Tcstar = 1.32
rhocstar = 0.31
Ttstar = 0.661

# import pymcx 

def term_library(term):
    if term == 'powerish':
        # Power and exponential
        N = [0.52080730e-2, 0.21862520e+1, -0.21610160e+1, 0.14527000e+1, -0.20417920e+1, 0.18695286e+0, -0.62086250e+0, -0.56883900e+0, -0.80055922e+0, 0.10901431e+0, -0.49745610e+0, -0.90988445e-1]
        t = [1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294]
        d = [4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5]
        p = [0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1]
        return N, t, d, p

    elif term == 'gaussian':
        # Gaussian
        N = [-0.14667177e+1, 0.18914690e+1, -0.13837010e+0, -0.38696450e+0, 0.12657020e+0, 0.60578100e+0, 0.11791890e+1, -0.47732679e+0, -0.99218575e-1, -0.57479320e+0, 0.37729230e-2]
        t = [2.830, 2.548, 4.650, 1.385, 1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970]
        d = [1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1]
        eta = [2.067, 1.522, 8.82, 1.722, 0.679, 1.883, 3.925, 2.461, 28.2, 0.753, 0.82]
        beta = [0.625, 0.638, 3.91, 0.156, 0.157, 0.153, 1.16, 1.73, 383, 0.112, 0.119]
        gamma = [0.71, 0.86, 1.94, 1.48, 1.49, 1.945, 3.02, 1.11, 1.17, 1.33, 0.24]
        epsilon = [ 0.2053, 0.409, 0.6, 1.203, 1.829, 1.397, 1.39, 0.539, 0.934, 2.369, 2.43]
        return N,t,d,eta,beta,gamma,epsilon
    else:
        raise ValueError(term)

def alphar_residual(tau, delta):
    # Initialize variables
    A00, A10, A01, A20, A11, A02, A30, A21, A12, A03, A22 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    N,t,d,p = term_library('powerish')
    for i in range(0, 6):
        _s = N[i]*delta**d[i]*tau**t[i]
        A00 += _s
        A10 += t[i]*_s
        A01 += d[i]*_s
        A20 += t[i]*(t[i]-1)*_s
        A11 += d[i]*t[i]*_s
        A02 += d[i]*(d[i]-1)*_s
        A30 += t[i]*(t[i]-1)*(t[i]-2)*_s
        A21 += t[i]*(t[i]-1)*d[i]*_s
        A12 += d[i]*(d[i]-1)*t[i]*_s
        A03 += d[i]*(d[i]-1)*(d[i]-2)*_s
        A22 += d[i]*(d[i]-1)*t[i]*(t[i]-1)*_s

    for i in range(6, 12):
        _s = N[i]*delta**d[i]*tau**t[i]*exp(-delta**p[i])
        A00 += _s
        A10 += _s*t[i]
        A01 += _s*(d[i]-p[i]*delta**p[i])
        
        A02 += _s*((d[i]-p[i]*delta**p[i])*(d[i]-1-p[i]*delta**p[i])-p[i]**2*delta**p[i])
        A11 += _s*t[i]*(d[i]-p[i]*delta**p[i])
        A20 += _s*t[i]*(t[i]-1)
        
        A30 += _s*t[i]*(t[i]-1)*(t[i]-2)
        A21 += _s*t[i]*(t[i]-1)*(d[i]-p[i]*delta**p[i])
        A12 += _s*t[i]*((d[i]-p[i]*delta**p[i])*(d[i]-1-p[i]*delta**p[i])-p[i]**2*delta**p[i])
        A03 += _s*(d[i]**3 - 3*d[i]**2*delta**p[i]*p[i] - 3*d[i]**2 + 3*d[i]*delta**(2*p[i])*p[i]**2 - 3*d[i]*delta**p[i]*p[i]**2 + 6*d[i]*delta**p[i]*p[i] + 2*d[i] - delta**(3*p[i])*p[i]**3 + 3*delta**(2*p[i])*p[i]**3 - 3*delta**(2*p[i])*p[i]**2 - delta**p[i]*p[i]**3 + 3*delta**p[i]*p[i]**2 - 2*delta**p[i]*p[i])
        A22 += _s*t[i]*(t[i]-1)*((d[i]-p[i]*delta**p[i])*(d[i]-1-p[i]*delta**p[i])-p[i]**2*delta**p[i])

    N,t,d,eta,beta,gamma,epsilon = term_library('gaussian')
    for i in range(11):
        _s = N[i]*delta**d[i]*tau**t[i]*exp(-eta[i]*(delta-epsilon[i])**2-beta[i]*(tau-gamma[i])**2)
        A00 += _s
        A10 += _s*(t[i] - 2*beta[i]*tau*(tau - gamma[i]))
        A01 += _s*(d[i] - 2*eta[i]*delta*(delta - epsilon[i]))
        A20 += _s*((t[i]-2*beta[i]*tau*(tau-gamma[i]))**2-t[i]-2*beta[i]*tau**2)
        A11 += _s*(t[i]-2*beta[i]*tau*(tau-gamma[i]))*(d[i]-2*eta[i]*delta*(delta-epsilon[i]))
        A02 += _s*((d[i]-2*eta[i]*delta*(delta-epsilon[i]))**2-d[i]-2*eta[i]*delta**2)
        A30 += _s*(beta[i]**2*tau**3*(8*beta[i]*(gamma[i] - tau)**3 - 12*gamma[i] + 12*tau) + 6*beta[i]*t[i]*tau**2*(2*beta[i]*(gamma[i] - tau)**2 - 1) + 6*beta[i]*t[i]*tau*(gamma[i] - tau)*(t[i] - 1) + t[i]*(t[i]**2 - 3*t[i] + 2))
        A12 += _s*(t[i] - 2*beta[i]*tau*(tau - gamma[i]))*((d[i]-2*eta[i]*delta*(delta-epsilon[i]))**2-d[i]-2*eta[i]*delta**2)
        A21 += _s*((t[i]-2*beta[i]*tau*(tau-gamma[i]))**2-t[i]-2*beta[i]*tau**2)*(d[i] - 2*eta[i]*delta*(delta - epsilon[i]))
        A03 += _s*(6*d[i]*delta**2*eta[i]*(2*eta[i]*(delta - epsilon[i])**2 - 1) + 6*d[i]*delta*eta[i]*(-d[i] + 1)*(delta - epsilon[i]) + d[i]*(d[i]**2 - 3*d[i] + 2) + 4*delta**3*eta[i]**2*(delta - epsilon[i])*(-2*eta[i]*(delta - epsilon[i])**2 + 3))

    return A00, A01, A10, A02, A11, A20, A03, A12, A21, A30, A22

def get_alphar_deriv(tau,delta,itau,idelta):
    As = alphar_residual(tau, delta)
    indices = [(0,0),(0,1),(1,0),(0,2),(1,1),(2,0),(0,3),(1,2),(2,1),(3,0),(2,2)]
    return As[indices.index((itau,idelta))]

def alphar_ideal(tau, delta):
    a1 = 0.4
    a2 = 0.4
    A00 = (0 if delta==0 else log(delta)) + a1 + a2*tau + 1.5*log(tau)
    A01 = 1
    A02 = -1
    A03 = 2
    A11 = 0
    A10 = a2*tau + 1.5
    A20 = -1.5
    A30 = 3.0
    A21 = 0
    A12 = 0
    A22 = 0
    return A00, A01, A10, A02, A11, A20, A03, A12, A21, A30, A22

def get_alpha0_deriv(tau,delta,itau,idelta):
    As = alphar_ideal(tau, delta)
    indices = [(0,0),(0,1),(1,0),(0,2),(1,1),(2,0),(0,3),(1,2),(2,1),(3,0),(2,2)]
    return As[indices.index((itau,idelta))]

def check_Aij(tau, delta, itau, idelta, getter):
    exact = getter(tau,delta,itau,idelta)
    im = 0+1j; h = 1e-100
    if itau-1 >= 0:
        numer = (getter(tau+im*h, delta,itau-1,idelta)/h).imag*tau - (itau-1)*getter(tau,delta,itau-1,idelta)
        # print('tau',exact, numer, numer-exact)
        assert(abs(numer-exact) < 1e-12)
        
    if idelta-1 >= 0:
        numer = (getter(tau, delta+im*h,itau,idelta-1)/h).imag*delta - (idelta-1)*getter(tau,delta,itau,idelta-1)
        # print('delta',exact, numer, numer-exact)
        assert(abs(numer-exact) < 1e-12)

def validate_derivatives():
    tau, delta = 0.7, 3
    for getter in [get_alphar_deriv, get_alpha0_deriv]:
        for itau in range(0,4):
            for idelta in range(0,4):
                if (itau + idelta) > 3: continue
                check_Aij(tau, delta, itau, idelta, getter)
validate_derivatives()

def validate_Thol_checks():
    data = [
        [0.8,0.005,3.8430053e-3,-5.4597389e-2,5.5672903e-2,1.1324263e0,2.7768170e-1],
        [0.8,0.8,1.5894013e-2,-5.7174120e0,9.5995160e-1,5.0522400e0,1.1838093e0],
        [1.0,0.02,1.7886470e-2,-1.8772644e-1,1.3016045e-1,1.2290934e0,1.8318141e0],
        [1.0,0.71,7.5247483e-2,-4.9564222e0,6.8903536e-1,4.1644650e0,2.9792860e0],
        [2.0,0.5,1.0751638e0,-3.1525021e0,3.1068090e-1,3.5186329e0,9.5274193e0],
        [5.0,0.6,6.9432008e0,-2.6956781e0, 3.1772707e-1, 6.8375197e0, 2.6122755e1],
        [7.0,1.0,4.1531352e1,-6.2393078e-1, 7.3348579e-1, 1.4201978e1, 4.8074394e-1],
    ]
    for row in data:
        Tstar, rhostar = row[0:2]
        tau = Tcstar/Tstar
        delta = rhostar/rhocstar
        p = rhostar*Tstar*(1+alphar_residual(tau,delta)[1])
        tau = Tcstar/Tstar
        delta = rhostar/rhocstar
        ur = get_alphar_deriv(tau,delta,1,0)*Tstar
        cvr = -get_alphar_deriv(tau,delta,2,0)
        ar = get_alphar_deriv(tau,delta,0,0)
        Ar01 = get_alphar_deriv(tau,delta,0,1)
        Ar11 = get_alphar_deriv(tau,delta,1,1)
        Ar02 = get_alphar_deriv(tau,delta,0,2)
        Ar20 = get_alphar_deriv(tau,delta,2,0)
        Ao20 = get_alpha0_deriv(tau,delta,2,0)
        w = (Tstar*(1+2*Ar01+Ar02-(1+Ar01-Ar11)**2/(Ao20+Ar20)))**0.5
        err = np.abs(np.array([p,ur,cvr,w])/np.array(row[2:6])-1)
        assert(np.max(err < 1e-7))

validate_Thol_checks()

def LJ_p(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    try:
        A00, A01, A10, A02, A11, A20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    except OverflowError:
        return 1e10
    #print 'p', rhostar*Tstar*(1 + A10)
    return (1+A01)*rhostar*Tstar

pcstar = LJ_p(Tcstar, rhocstar)

def cvrstar(Tstar, rhostar):
    tau = Tcstar/Tstar; delta = rhostar/rhocstar
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    return -(Ar20)

def cvstar(Tstar, rhostar):
    tau = Tcstar/Tstar; delta = rhostar/rhocstar
    Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    return -(Ao20+Ar20)

def cpstar(Tstar, rhostar):
    tau = Tcstar/Tstar; delta = rhostar/rhocstar
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    return cvstar(Tstar, rhostar) + (1+Ar01-Ar11)**2/(1+2*Ar01+Ar02)

def dpdrho(Tstar, rhostar):
    tau = Tcstar/Tstar; delta = rhostar/rhocstar
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    return Tstar*(1+2*Ar01+Ar02)

def psat_anc(Tstar):
    summer = 0
    for n,t in [(-0.54000e+1,1.00),( 0.44704e0,1.50),(-0.18530e+1,4.70),( 0.19890e0,2.50),(-0.11250e+1,21.4)]:
        summer += n*(1-Tstar/Tcstar)**t
    return pcstar*np.exp(Tcstar/Tstar*summer)

def rholsat_anc(Tstar):
    coeffs_rholiq = [(0.1362e1,0.313),(0.2093e1,0.940),(-0.2110e1,1.630),(0.3290e0,17.00),(0.1410e1,2.400)]
    summer = 0
    for n,t in coeffs_rholiq:
        summer += n*(1-Tstar/Tcstar)**t
    rholstar = rhocstar*(1+summer)
    return rholstar

def rhovsat_anc(Tstar):
    coeffs_rhovap = [(-0.69655e1,1.320),(-0.10331e3,19.24),(-0.20325e1,0.360),(-0.44481e2,8.780),(-0.18463e2,4.040),(-0.26070e3,41.60)]
    summer = 0
    for n,t in coeffs_rhovap:
        summer += n*(1-Tstar/Tcstar)**t
    rhovstar = rhocstar*np.exp(summer)
    return rhovstar

def get_Boyle():
    return scipy.optimize.newton(lambda Tstar: get_B2_LJ(Tstar), 2)

def get_acentric_factor():
    psat_07Tr = psat_anc(0.7*Tcstar)
    return -np.log10(psat_07Tr/pcstar)-1

def LJ_sr_over_R(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    try:
        A00, A01, A10, A02, A11, A20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    except OverflowError:
        return 1e10
    #print 'p', rhostar*Tstar*(1 + A10)
    return A10-A00

def LJ_ur_over_kBT(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    return get_alphar_deriv(tau, delta, 1, 0)

def LJ_gr_over_kBT(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    return get_alphar_deriv(tau, delta, 0, 1) + get_alphar_deriv(tau, delta, 0, 0)

def LJ_pint_over_rhoNkBT(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    return -get_alphar_deriv(tau, delta, 1, 1)

def LJ_Zminus1(Tstar, rhostar):
    tau = Tcstar/Tstar
    delta = rhostar/rhocstar
    der = get_alphar_deriv(tau, delta, 0, 1)
    return der

def RiemFluid_TP(T, P, *, fluid):
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    PATH = r'D:\REFPROP'
    RP = REFPROPFunctionLibrary(PATH)
    MOLAR_BASE_SI = RP.GETENUMdll(0, 'MOLAR BASE SI').iEnum
    RP.SETPATHdll(PATH)
    RP.SETFLUIDSdll(fluid)
    rhomolar, Riem = RP.REFPROPdll('','TP','D;RIEM',MOLAR_BASE_SI,0,0,T,P,[1.0]).Output[0:2]
    rhoN = rhomolar*N_A

    z = [1.0] + [0.0]*19
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30 = RP.REFPROPdll('','TD','PHIR00;PHIR01;PHIR10;PHIR02;PHIR11;PHIR20;PHIR03;PHIR12;PHIR21;PHIR30',MOLAR_BASE_SI,0,0,T,rhomolar,z).Output[0:10]
    Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30 = RP.REFPROPdll('','TD','PHIG00;PHIG01;PHIG10;PHIG02;PHIG11;PHIG20;PHIG03;PHIG12;PHIG21;PHIG30',MOLAR_BASE_SI,0,0,T,rhomolar,z).Output[0:10]
    Bstar = 1+2*Ar01+Ar02
    cvstar = -Ao20-Ar20

    # First derivatives
    TdBdT = -(2*Ar11+Ar12)
    TdcvdT = ((2*Ao20+Ao30)+(2*Ar20+Ar30))
    rhodBdrho = (2*(Ar01+Ar02)+(2*Ar02+Ar03))
    rhodcvdrho = -Ar21

    # Combinations of first derivatives
    rhodlnBcvdrho = 1/Bstar*rhodBdrho + 1/cvstar*rhodcvdrho
    TdlnBcvdT = 1/Bstar*TdBdT + 1/cvstar*TdcvdT
    rhodlnBovercvdrho = 1/Bstar*rhodBdrho - 1/cvstar*rhodcvdrho

    # Second derivatives, same treatment as first derivatives,
    # Fourth derivatives cancel conveniently
    T2dBdT2_plus_rho2dcvdrho2 = (4*Ar11+2*Ar12+2*Ar21)

    Rstar = 1/(Bstar*cvstar)*(
        TdBdT*(1-0.5*TdlnBcvdT)
        + rhodcvdrho*(1-0.5*rhodlnBcvdrho)
        -0.5*cvstar*rhodlnBovercvdrho
        +T2dBdT2_plus_rho2dcvdrho2
        )
    here = Rstar/rhoN # [m^3]
    return here*1e27, Riem # [nm^3]

def RiemMe(Tstar, rhostar):
    
    e_over_kB = 1
    sigma = 1
    tau=Tcstar/Tstar
    delta=rhostar/rhocstar
    T = Tstar*e_over_kB
    rhoN = rhostar/sigma**3
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    Bstar = 1+2*Ar01+Ar02
    cvstar = -Ao20-Ar20
    gouter = Bstar*cvstar/T**2

    im = 0+1j
    h = 1e-100

    def bracket_1(T, rhoN):
        Tstar = T/e_over_kB
        rhostar = rhoN*sigma**3
        tau=Tcstar/Tstar
        delta=rhostar/rhocstar
        Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
        Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
        dgrhodT = -1/(rhoN*T)*(2*(Ar11+Ao11)+Ar12+Ao12)
        Bstar = 1+2*Ar01+Ar02
        cvstar = -Ao20-Ar20
        g = Bstar*cvstar/T**2
        return 1/g**0.5*dgrhodT

    def bracket_2(T, rhoN):
        Tstar = T/e_over_kB
        rhostar = rhoN*sigma**3        
        tau=Tcstar/Tstar
        delta=rhostar/rhocstar
        Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
        Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
        dgTdrhoN = -1/(T**2)*((Ar20+Ao20)+(Ar21+Ao21))
        Bstar = 1+2*Ar01+Ar02
        cvstar = -Ao20-Ar20
        g = Bstar*cvstar/T**2
        return 1/g**0.5*dgTdrhoN

    return 1/gouter**0.5*((bracket_1(T+im*h, rhoN)/h).imag + (bracket_2(T, rhoN+im*h)/h).imag)

def Riem(Tstar, rhostar):
    tau=Tcstar/Tstar
    delta=rhostar/rhocstar
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    Bstar = 1+2*Ar01+Ar02
    cvstar = -Ao20-Ar20

    # def f_cv(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     return -Ao20-Ar20

    # def f_B(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     return 1+2*Ar01+Ar02

    # def f_lnBC(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     B = 1+2*Ar01+Ar02
    #     C = -Ao20-Ar20
    #     return log(B*C)

    # def f_lnBC2(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ao00, Ao01, Ao10, Ao02, Ao11, Ao20, Ao03, Ao12, Ao21, Ao30, Ao22 = alphar_ideal(tau, delta)
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     B = 1+2*Ar01+Ar02
    #     C = -Ao20-Ar20
    #     return log(B/C)

    # def f_rdcdr(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     return -Ar21
    # def f_TdBdT(Tstar, rhostar):
    #     tau=Tcstar/Tstar
    #     delta=rhostar/rhocstar
    #     Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    #     return -(2*Ar11+Ar12)

    # im = 0+1j
    # h = 1e-100
    
    # First derivatives, all are starred quantities,
    # stars dropped for concision
    TdBdT = -(2*Ar11+Ar12)
    TdcvdT = ((2*Ao20+Ao30)+(2*Ar20+Ar30))
    rhodBdrho = (2*(Ar01+Ar02)+(2*Ar02+Ar03))
    rhodcvdrho = -Ar21
    # print((f_B(Tstar+im*h, rhostar)/h).imag-TdBdT/Tstar)
    # print((f_cv(Tstar+im*h, rhostar)/h).imag-TdcvdT/Tstar)
    # print((f_B(Tstar, rhostar+im*h)/h).imag-rhodBdrho/rhostar)
    # print((f_cv(Tstar, rhostar+im*h)/h).imag-rhodcvdrho/rhostar)

    # Combinations of first derivatives
    rhodlnBcvdrho = 1/Bstar*rhodBdrho + 1/cvstar*rhodcvdrho
    TdlnBcvdT = 1/Bstar*TdBdT + 1/cvstar*TdcvdT
    rhodlnBovercvdrho = 1/Bstar*rhodBdrho - 1/cvstar*rhodcvdrho
    # print((f_lnBC(Tstar, rhostar+im*h)/h).imag-rhodlnBcvdrho/rhostar)
    # print((f_lnBC(Tstar+im*h, rhostar)/h).imag-TdlnBcvdT/Tstar)
    # print((f_lnBC2(Tstar, rhostar+im*h)/h).imag-rhodlnBovercvdrho/rhostar)

    # Second derivatives, same treatment as first derivatives,
    # Fourth derivatives cancel conveniently
    T2dBdT2_plus_rho2dcvdrho2 = (4*Ar11+2*Ar12+2*Ar21)

    # print( ((f_TdBdT(Tstar+im*h, rhostar)/h).imag*Tstar-TdBdT) + ((f_rdcdr(Tstar, rhostar+im*h)/h).imag*rhostar-rhodcvdrho))
    # print(T2dBdT2_plus_rho2dcvdrho2,'2')

    Rstar = 1/(Bstar*cvstar)*(
        TdBdT*(1-0.5*TdlnBcvdT)
        + rhodcvdrho*(1-0.5*rhodlnBcvdrho)
        -0.5*cvstar*rhodlnBovercvdrho
        +T2dBdT2_plus_rho2dcvdrho2
        )
    # print(TdBdT*(1-0.5*TdlnBcvdT),
    #     + rhodcvdrho*(1-0.5*rhodlnBcvdrho),
    #     -0.5*cvstar*rhodlnBovercvdrho,
    #     +T2dBdT2_plus_rho2dcvdrho2)
    return Rstar/rhostar



# pc = CP.PropsSI("pcrit","Argon")
# Tcrit = CP.PropsSI("Tcrit","Argon")
# xx,yy = [],[]
# for pr in np.logspace(-2, 3):
#     xx.append(pr)
#     yy.append(RiemFluid_TP(3*Tcrit, pr*pc,fluid='Argon')[0]/0.33952**3)
# plt.plot(xx, yy,'g')
# # print(xx)

# Tr = 3
# xx,yy=[],[]
# for rhostar in np.logspace(-2, np.log10(1.6)):
#     Tstar = Tr*Tcstar
#     pr = LJ_p(Tstar, rhostar)/pcstar
#     xx.append(pr)
#     R1 = Riem(Tstar, rhostar)
#     R2 = RiemMe(Tstar, rhostar)
#     print(R1, R2)
#     yy.append(R1)
# plt.plot(xx,yy,'r')
# plt.show()

# rhostar = 0.316
# for Tstar in np.arange(1.4, 50, 0.1):
#     print(Tstar, LJ_p(Tstar, rhostar)/pcstar, RiemMe(Tstar, rhostar))

# sigma = 3.4e-10
# Tr = 3
# for rhostar in np.logspace(-2, np.log10(5)):
#     Tstar = Tr*Tcstar
#     print(LJ_p(Tstar, rhostar)/pcstar)
#     print(RiemMe(Tstar, rhostar)*sigma**3*1e27, 'me')
#     # print(Riem(Tstar, rhostar)*sigma**3, 'Branka')

sr_over_R = LJ_sr_over_R

def Gamma_IPL(Tstar, rhostar):
    tau = Tcstar/Tstar; delta = rhostar/rhocstar
    Ar00, Ar01, Ar10, Ar02, Ar11, Ar20, Ar03, Ar12, Ar21, Ar30, Ar22 = alphar_residual(tau, delta)
    rho_dnsrR_drho__T = Ar01-Ar11
    T_dnsrR_dT__rho = Ar20
    return -rho_dnsrR_drho__T/T_dnsrR_dT__rho

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
#                                                  TRANSPORT                       
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def get_Omega_coeffs(l,s):
    return {
    (1,1): [-1.1036729 ,  2.6431984  ,  1.6690746    ,0.0060432255  ,-6.9145890e-1    ,-1.5158773e-1   , 1.5502132e-1    , 0.54237938e-1   ,-2.0642189e-2    ,-0.90468682e-2   , 1.5402077e-3   , 0.61742007e-3   ,-0.49729535e-4],
    (1,2): [ 1.3555554 , -0.44668594 , -0.47499422   ,0.42734391    , 1.4482036e-1    ,-1.6036459e-1   ,-0.32158368e-1   , 0.31461648e-1   , 0.44357933e-2   ,-0.32587575e-2   ,-0.34138118e-3  , 0.13860257e-3   , 0.11259742e-4],
    (1,3): [ 1.0677115 , -0.13945390 , -0.25258689   ,0.17696362    , 0.59709197e-1   ,-0.26252211e-1  ,-0.13332695e-1   ,-0.043814141e-1  , 0.19619285e-2   , 0.16752100e-2   ,-0.16063076e-3  ,-0.14382801e-3   , 0.055804557e-4],
    (1,4): [ 0.80959899,  0.12938170 , -0.045055948  ,0.059760309   ,-0.22642753e-1   , 0.071109469e-1 , 0.056672308e-1  ,-0.063851124e-1  ,-0.065708760e-2  , 0.10498938e-2   , 0.040733113e-3 ,-0.058149257e-3  ,-0.010820157e-4],
    (1,5): [ 0.74128322,  0.17788850 ,  0.0013668724 ,0.027398438   ,-0.41730962e-1  , 0.076254248e-1 , 0.10378923e-1   ,-0.031650182e-1  ,-0.13492954e-2   , 0.032786518e-2  , 0.096963599e-3 ,-0.0092890016e-3 ,-0.030307552e-4],
    (1,6): [ 0.80998324,  0.073071217, -0.071180849  ,0.034607908   ,-0.12738119e-1   ,-0.011457199e-1 , 0.038582834e-1  , 0.0028198596e-1 ,-0.047060425e-2  ,-0.020060540e-2  , 0.030466929e-3 , 0.021446483e-3  ,-0.0085305576e-4],
    (1,7): [ 0.81808091,  0.044232851, -0.089417548  ,0.029750283   ,-0.051856424e-1  ,-0.022011682e-1 , 0.021882143e-1  , 0.0063264120e-1 ,-0.024874471e-2  ,-0.017555530e-2  , 0.013745859e-3 , 0.014255704e-3  ,-0.0030285365e-4],
    (2,2): [-0.92032979,  2.3508044  ,  1.6330213    ,0.50110649    ,-6.9795156e-1    ,-4.7193769e-1   , 1.6096572e-1    , 1.5806367e-1    ,-2.2109440e-2    ,-2.6367184e-2    , 1.7031434e-3   , 1.8120118e-3    ,-0.56699986e-4],
    (2,3): [ 2.5955799 , -1.8569443  , -1.4586197    ,0.96985775    , 5.2947262e-1    ,-3.9888526e-1   ,-1.1946363e-1    , 0.90063692e-1   , 1.6264589e-2    ,-1.0918991e-2    ,-1.2354315e-3   , 0.56646797e-3   , 0.40366357e-4],
    (2,4): [ 1.6042745 , -0.67406115 , -0.62774499   ,0.42671907    , 2.0700644e-1    ,-1.0177069e-1   ,-0.47601690e-1   , 0.0061857136e-1 , 0.67153792e-2   , 0.31225358e-2   ,-0.52706167e-3  ,-0.35206051e-3   , 0.17705708e-4],
    (2,5): [ 0.82064641,  0.23195128 ,  0.039184885  ,0.12233793    ,-0.57316906e-1   , 0.13891578e-1  , 0.12794497e-1   ,-0.20903423e-1   ,-0.15336449e-2   , 0.46715462e-2   , 0.10241454e-3  ,-0.35204303e-3   ,-0.029975563e-4],
    (2,6): [ 0.79413652,  0.23766123 ,  0.050470266  ,0.077125802   ,-0.62621672e-1   , 0.13060901e-1  , 0.14326724e-1   ,-0.10982362e-1   ,-0.17806541e-2   , 0.18034505e-2   , 0.12353365e-3  ,-0.095982571e-3  ,-0.037501381e-4],
    (3,3): [ 1.2630491 , -0.36104243 , -0.33227158   ,0.68116214    , 0.79723851e-1   ,-3.6401583e-1   ,-0.15470355e-1   , 1.0500196e-1    , 0.18686705e-2   ,-1.6400134e-2    ,-0.12179945e-3  , 1.0880886e-3    , 0.032594587e-4],
    (3,4): [ 2.2114636 , -1.4743107  , -1.1942554    ,0.64918549    , 4.3000688e-1    ,-2.4075196e-1   ,-0.97525871e-1   , 0.51820149e-1   , 1.3399366e-2    ,-0.60565396e-2   ,-1.0283777e-3   , 0.29812326e-3   , 0.33956674e-4],
    (3,5): [ 1.5049809 , -0.64335529 , -0.60014514   ,0.3261704     , 1.9764859e-1    ,-0.82126072e-1  ,-0.45212434e-1   , 0.059682011e-1  , 0.63650284e-2   , 0.10269488e-2   ,-0.49991689e-3  ,-0.15957252e-3   , 0.16833944e-4],
    (4,4): [ 2.6222393 , -1.9158462  , -1.4676253    ,1.0166380     , 5.3048161e-1    ,-4.3355278e-1   ,-1.1909781e-1    , 1.0496591e-1    , 1.6123847e-2    ,-1.3951104e-2    ,-1.2174905e-3   , 0.80048534e-3   , 0.39545100e-4]
    }[(l,s)]

def get_Omegals(Tstar,l,s):
    # Collision integral from Kim & Monroe, JCP, http://dx.doi.org/10.1016/j.jcp.2014.05.018
    coeffs = get_Omega_coeffs(l,s)
    CI = coeffs[0]
    for k in range(1,7):
        i = 1 + 2*(k-1)
        B,C = coeffs[i], coeffs[i+1]
        CI += B/(Tstar**k)+C*np.log(Tstar)**k
    return CI

def get_Omega22(Tstar): 
    return get_Omegals(Tstar,2,2)

def check_Omegals():
    """
    Kim and Monroe Table 4
    """
    Tstar = 0.3
    for (l,s), Omegals in [((1,1),2.6500024),((1,2),2.2568342),((1,3),1.9665277),((2,2),2.8436719),((2,3),2.5806610),((2,4),2.3622719),((2,5),2.1704207),((2,6),2.0010465),((4,4),2.5710549)]:
        calc = get_Omegals(Tstar,l,s)
        assert((get_Omegals(Tstar,l,s) - Omegals) < 1e-4)
check_Omegals()

def f3_eta(Tstar):
    """ The third-order correction multiplier for viscosity """
    Omega22 = get_Omegals(Tstar,2,2)
    Omega23 = get_Omegals(Tstar,2,3)
    Omega24 = get_Omegals(Tstar,2,4)
    Omega25 = get_Omegals(Tstar,2,5)
    Omega26 = get_Omegals(Tstar,2,6)
    Omega44 = get_Omegals(Tstar,4,4)

    b11 = 4*Omega22
    b12 = 7*Omega22-8*Omega23
    b22 = 301/12*Omega22-28*Omega23+20*Omega24
    b13 = 63/8*Omega22-18*Omega23+10*Omega24
    b23 = 1365/32*Omega22-321/4*Omega23+125/2*Omega24-30*Omega25
    b33 = 25137/256*Omega22-1755/8*Omega23+1905/8*Omega24-135*Omega25+105/2*Omega26+12*Omega44
    b = [[b11,b12,b13],[b12,b22,b23],[b13,b23,b33]]
    return 1+b12**2/(b11*b22-b12**2)+(b11*(b12*b23-b22*b13)**2)/((b11*b22-b12**2)*np.linalg.det(b))

def LJ_eta0(Tstar, order):
    etastar_1 = 5.0*Tstar**0.5/(16.0*pi**0.5*get_Omega22(Tstar))
    if order == 1:
        return etastar_1
    elif order == 3:
        return etastar_1*f3_eta(Tstar)
    else:
        raise ValueError('order is invalid:'+str(order))

def f3_lambda(Tstar):
    """ The third-order correction multiplier for thermal conductivity """

    Omega22 = get_Omegals(Tstar,2,2)
    Omega23 = get_Omegals(Tstar,2,3)
    Omega24 = get_Omegals(Tstar,2,4)
    Omega25 = get_Omegals(Tstar,2,5)
    Omega26 = get_Omegals(Tstar,2,6)
    Omega44 = get_Omegals(Tstar,4,4)

    a11 = 4*Omega22
    a12 = 7*Omega22-8*Omega23
    a13 = 63/8*Omega22-18*Omega23+10*Omega24
    a22 = 77/4*Omega22-28*Omega23+20*Omega24
    a23 = 945/32*Omega22-261/4*Omega23+125/2*Omega24-30*Omega25
    a33 = 14553/256*Omega22-1215/8*Omega23+1565/8*Omega24-135*Omega25+105/2*Omega26+4*Omega44
    a = [[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]]
    return 1+a12**2/(a11*a22-a12**2)+(a11*(a12*a23-a22*a13)**2)/((a11*a22-a12**2)*np.linalg.det(a))

def LJ_lambdastar0(Tstar, *, order):
    # Also equal to eta0*5/2*cv, where cv/particle = 3/2 kB
    # See also Mulero
    lambdastar_1 = 75*Tstar**0.5/(64*pi**0.5*get_Omega22(Tstar))
    if order == 1:
        return lambdastar_1
    elif order == 3:
        return lambdastar_1*f3_lambda(Tstar)
    else:
        raise ValueError('order is invalid:'+str(order))

def get_f2_D(Tstar):
    Omega22 = get_Omegals(Tstar,2,2)
    Omega11 = get_Omegals(Tstar,1,1)
    Omega12 = get_Omegals(Tstar,1,2)
    Omega13 = get_Omegals(Tstar,1,3)
    A = Omega22/Omega11
    B = (5*Omega12-4*Omega13)/Omega11
    C = Omega12/Omega11
    f2_recip = 1-(6*C-5)**2/(55-12*B+16*A)
    return 1/f2_recip

def LJ_rhostarDstar0(Tstar, *, order):
    rhoDstar_1 = 3*Tstar**(0.5)/(8*pi**0.5*get_Omegals(Tstar,l=1,s=1))
    if order == 1:
        return rhoDstar_1
    elif order == 2:
        return rhoDstar_1*get_f2_D(Tstar)
    else:
        raise ValueError('order is invalid:'+str(order))

def nsrR_crit():
    return -LJ_sr_over_R(Tcstar, rhocstar)

def get_B2_LJ(Tstar):
    tau = Tcstar/Tstar
    # All the terms where d_k = 1, remove all contributions for which delta=0 is not meaningful
    o = 0 # Collector for dalphar/ddelta|tau at zero delta
    for n, t, d, l in zip(*term_library('powerish')):
        if d == 1:
            o += n*d*tau**t
    for n, t, d, eta, beta, gamma, epsilon in zip(*term_library('gaussian')):
        if d == 1:
            o += n*d*tau**t*exp(-eta*(-epsilon)**2-beta*(tau-gamma)**2)
    return o/rhocstar

def get_B3_LJ(Tstar):
    tau = Tcstar/Tstar
    o = 0 # Collector for dalphar^2/ddelta^2|tau at zero delta
    for n, t, d, l in zip(*term_library('powerish')):
        if d <= 2:
            if l == 0:
                if d == 2:
                    # d2termdDelta2 = pymcx.diff_mcx1(lambda delta: n*delta**d*tau**t, 0, 2)[1]; o += d2termdDelta2
                    o += n*d*(d-1)*tau**t
            else:
                val = 0
                # d2termdDelta2 = pymcx.diff_mcx1(lambda delta: n*delta**d*tau**t*exp(-delta**l), 0, 2)[1]; o += d2termdDelta2
                if d == 1:
                    val = n*tau**t*(-l-1) if l == 1 else 0
                else:
                    val = n*tau**t*2
                o += val
    for n, t, d, eta, beta, gamma, epsilon in zip(*term_library('gaussian')):
        if d <= 2:
            if d == 1:
                val = n*tau**t*(4*eta*epsilon)*exp(-eta*(-epsilon)**2-beta*(tau-gamma)**2)
            else:
                val = n*tau**t*2*exp(-eta*(-epsilon)**2-beta*(tau-gamma)**2)
            # d2termdDelta2 = pymcx.diff_mcx1(lambda delta: n*delta**d*tau**t*exp(-eta*(delta-epsilon)**2-beta*(tau-gamma)**2), 0, 2)[1]; o += d2termdDelta2
            o += val
            
    return o/rhocstar**2

def get_B2_LJ_mcx(Tstar):
    tau = Tcstar/Tstar
    dalphardDelta = pymcx.diff_mcx1(lambda delta: alphar_residual(tau, delta)[0], 0, 1)[0]
    return dalphardDelta/0.31

def get_B2_LJ_num(Tstar):
    delta_almost_zero = 1e-10
    tau = Tcstar/Tstar
    Ar01 = get_alphar_deriv(tau,delta_almost_zero,0,1)
    return Ar01/delta_almost_zero/0.31

def get_B3_LJ_mcx(Tstar):
    tau = Tcstar/Tstar
    d2alphardDelta2 = pymcx.diff_mcx1(lambda delta: alphar_residual(tau, delta)[0], 0, 2)[1]
    return d2alphardDelta2/0.31**2

def get_B3_LJ_num(Tstar):
    delta_almost_zero = 1e-7
    tau = Tcstar/Tstar
    Ar02 = get_alphar_deriv(tau,delta_almost_zero,0,2)
    return Ar02/delta_almost_zero**2/0.31**2

# B21 = get_B2_LJ_num(1.32)
# B22 = get_B2_LJ(1.32)
# B23 = get_B2_LJ_mcx(1.32)

# B31 = get_B3_LJ_num(1.32)
# B32 = get_B3_LJ(1.32)
# B33 = get_B3_LJ_mcx(1.32)

# print(B31-B32, B31, B32, B33, B22-B23)
# quit()

im = 0+1j
def magic1(f, x, h):
    """ 
    "Magical" first derivative 
    See https://sinews.siam.org/Details-Page/differentiation-without-a-difference
    """
    return (f(x+im*h)/h).imag

def get_dB2dTstar_LJ_num(Tstar):
    dTstar = Tstar*1e-6
    return (get_B2_LJ(Tstar+dTstar)-get_B2_LJ(Tstar-dTstar))/(2*dTstar)

def get_dB2dTstar_LJ(Tstar):
    return magic1(get_B2_LJ,Tstar,1e-100)

def get_frakB2(Tstar):
    return get_B2_LJ(Tstar) + Tstar*get_dB2dTstar_LJ(Tstar)

def get_frakB3(Tstar):
    return get_B3_LJ(Tstar) + Tstar*magic1(get_B3_LJ_num, Tstar, 1e-100)

# import pymcx
# B2star_HS = 2*np.pi/3
# print(B2star_HS)
# Tstar = 4; print(get_B2_LJ(Tstar) + Tstar*get_dB2dTstar_LJ(Tstar)); quit() #print(Tstar*pymcx.diff_mcx1(get_B2_LJ, Tstar, 1)); 

def get_d2B2dTstar2_LJ(Tstar):
    dTstar = Tstar*1e-4
    return (get_dB2dTstar_LJ_num(Tstar+dTstar)-get_dB2dTstar_LJ_num(Tstar-dTstar))/(2*dTstar)

def get_d2B2dTstar2_LJ_exact_but_wrong(Tstar):
    return magic1(get_dB2dTstar_LJ,Tstar,1e-100)

def get_etaplus_dilute(Tstar,*,order):
    return LJ_eta0(Tstar,order=order)/np.sqrt(Tstar)*(get_B2_LJ(Tstar)+Tstar*get_dB2dTstar_LJ(Tstar))**(2/3)

def get_lambdaplus_dilute(Tstar, *, order):
    return LJ_lambdastar0(Tstar, order=order)/np.sqrt(Tstar)*(get_B2_LJ(Tstar)+Tstar*get_dB2dTstar_LJ(Tstar))**(2/3)

def get_Dplus_dilute(Tstar, *, order):
    return LJ_rhostarDstar0(Tstar, order=order)/np.sqrt(Tstar)*(get_B2_LJ(Tstar)+Tstar*get_dB2dTstar_LJ(Tstar))**(2/3)

ffff = lambda x: get_B2_LJ(x) + x*get_dB2dTstar_LJ_num(x)

if __name__ == '__main__':
    Tstar = 10
    rhostar = 0.85
    print(cvrstar(Tstar, rhostar))
    print(Gamma_IPL(Tstar, rhostar))
    quit()

    T = 3.17
    rhostar = 1.090
    print(T, LJ_ur_over_kBT(T, rhostar)*T, LJ_Zminus1(T, rhostar)*T)
    print('omega', get_acentric_factor())
    print('pc^*:', pcstar)
    print(get_B2_LJ(0.9), get_B2_LJ_num(0.9))
    print(get_dB2dTstar_LJ(0.9), get_dB2dTstar_LJ_num(0.9))
    print(get_d2B2dTstar2_LJ_exact_but_wrong(0.9), get_d2B2dTstar2_LJ(0.9))