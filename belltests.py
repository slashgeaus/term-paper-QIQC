# Functions for Bell's inequality testing using quadratures (homodyne
# measurements)
#
# Robert Johansson and Neill Lambert
#
from scipy import *
from scipy.misc import factorial
from scipy.special import hermite

from pylab import *

import qutip as q
from qutip.states import state_number_index, state_index_number

M1 = None
M2 = None


def P_x1x2(x1_vec, x2_vec, theta1, theta2, psi):
    """
    calculate probability distribution for quadrature measurement
    outcomes given a two-mode wavefunction or density matrix
    """
    
    if q.isket(psi):
        return P_x1x2_psi(x1_vec, x2_vec, theta1, theta2, psi)
    else:
        return P_x1x2_rho(x1_vec, x2_vec, theta1, theta2, psi)
        

def P_x1x2_psi(x1_vec, x2_vec, theta1, theta2, psi):
    """
    calculate probability distribution for quadrature measurement
    outcomes given a two-mode wavefunction
    """
    
    X1, X2 = meshgrid(x1_vec, x2_vec)
    
    p = zeros((len(x1_vec),len(x2_vec)), dtype=complex)
    N = psi.dims[0][0]
    
    for n1 in range(N):
        kn1 = exp(-1j * theta1 * n1) / sqrt(sqrt(pi) * 2**n1 * factorial(n1)) * exp(-X1**2/2.0) * polyval(hermite(n1), X1)
        for n2 in range(N):
            kn2 = exp(-1j * theta2 * n2) / sqrt(sqrt(pi) * 2**n2 * factorial(n2)) * exp(-X2**2/2.0) * polyval(hermite(n2), X2)
            i = state_number_index([N, N], [n1, n2])
            p += kn1 * kn2 * psi.data[i,0]

    return abs(p)**2


def P_x1x2_rho(x1_vec, x2_vec, theta1, theta2, rho, clear_cache=False):
    """
    calculate probability distribution for quadrature measurement
    outcomes given a two-mode density matrix
    """
    global M1, M2
    
    if clear_cache:
        M1 = M2 = None
    
    p = np.zeros((len(x1_vec), len(x2_vec)), dtype=complex)
    N = rho.dims[0][0]
    
    if M1 is None or M2 is None:
        print("computing M1 and M2")
        X1, X2 = meshgrid(x1_vec, x2_vec)
        
        M1 = np.zeros((N, N, len(x1_vec), len(x2_vec)), dtype=complex)
        M2 = np.zeros((N, N, len(x1_vec), len(x2_vec)), dtype=complex)
    
        for m in range(N):
            for n in range(N):
                M1[m,n] = 1.0 / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X1 ** 2) * np.polyval(hermite(m), X1) * np.polyval(hermite(n), X1)
                M2[m,n] = 1.0 / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X2 ** 2) * np.polyval(hermite(m), X2) * np.polyval(hermite(n), X2)

    for i in range(N**2):
        n1, n2 = state_index_number([N, N], i)
        for j in range(N**2):
            if abs(rho.data[i, j]) > 1e-8:
                p1, p2 = state_index_number([N, N], j)
                p += exp(-1j * theta1 * (n1 - p1)) * M1[n1, p1] * \
                     exp(-1j * theta2 * (n2 - p2)) * M2[n2, p2] * \
                     rho.data[i, j]

    return real(p)


def P_x_rho(x1_vec, x2_vec, theta, rho, clear_cache=False):
    """
    calculate probability distribution for quadrature measurement
    outcomes given a two-mode density matrix
    """
    global M1, M2
    
    if clear_cache:
        M1 = M2 = None
    
    p = np.zeros((len(x1_vec), len(x2_vec)), dtype=complex)
    N = rho.dims[0][0]
    
    if M1 is None or M2 is None:
        print("computing M1 and M2")
        X1, X2 = meshgrid(x1_vec, x2_vec)
        
        M1 = np.zeros((N, N, len(x1_vec), len(x2_vec)), dtype=complex)
        M2 = np.zeros((N, N, len(x1_vec), len(x2_vec)), dtype=complex)
    
        for m in range(N):
            for n in range(N):
                M1[m,n] = 1.0 / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X1 ** 2) * np.polyval(hermite(m), X1) * np.polyval(hermite(n), X1)
                M2[m,n] = 1.0 / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X2 ** 2) * np.polyval(hermite(m), X2) * np.polyval(hermite(n), X2)
                        
    for i in range(N**2):
        n1, n2 = state_index_number([N, N], i)
        for j in range(N**2):
            if abs(rho.data[i, j]) > 1e-8:
                p1, p2 = state_index_number([N, N], j)
                p += exp(-1j * theta1 * (n1 - p1)) * M1[n1, p1] * \
                     exp(-1j * theta2 * (n2 - p2)) * M2[n2, p2] * \
                     rho.data[i, j]

    return real(p)


def quadrature_binning(x1_vec, x2_vec, P):
    """
    Quadrature binning strategy: If X >= 0 is measured, map to P_11 = 1,
    and if X < 0 is measured, map to P_11 = 0.
    """
    
    X1,X2 = meshgrid(x1_vec, x2_vec)
    dx1dx2 = (x1_vec[1]-x1_vec[0]) * (x2_vec[1]-x2_vec[0])
    
    p11 = sum(P * (X1>=0) * (X2>=0)) * dx1dx2
    p00 = sum(P * (X1<0)  * (X2<0))  * dx1dx2
    p01 = sum(P * (X1<0)  * (X2>=0)) * dx1dx2
    p10 = sum(P * (X1>=0) * (X2<0))  * dx1dx2

    p1a = sum(P * (X1>=0)) * dx1dx2
    p1b = sum(P * (X2>=0)) * dx1dx2
    
    return p1a, p1b, p00, p01, p10, p11


def bell_quadrature_ch(x1_vec, x2_vec, theta1, phi1, theta2, phi2, psi):
    """
    Calculate the CH Bell test.

    LHV limit: 1.0
    """
    
    P_theta1_phi1 = P_x1x2(x1_vec, x2_vec, theta1, phi1, psi)
    P_theta2_phi1 = P_x1x2(x1_vec, x2_vec, theta2, phi1, psi)
    P_theta1_phi2 = P_x1x2(x1_vec, x2_vec, theta1, phi2, psi)
    P_theta2_phi2 = P_x1x2(x1_vec, x2_vec, theta2, phi2, psi)
    
    p1_theta1, p1_phi1, p00, p01, p10, p11_theta1_phi1 = quadrature_binning(x1_vec, x2_vec, P_theta1_phi1)
    p1_theta2, p1_phi1, p00, p01, p10, p11_theta2_phi1 = quadrature_binning(x1_vec, x2_vec, P_theta2_phi1)
    p1_theta1, p1_phi2, p00, p01, p10, p11_theta1_phi2 = quadrature_binning(x1_vec, x2_vec, P_theta1_phi2)
    p1_theta2, p1_phi2, p00, p01, p10, p11_theta2_phi2 = quadrature_binning(x1_vec, x2_vec, P_theta2_phi2)
    
    # munro
    b_ch = (p11_theta1_phi1 - p11_theta2_phi1 + p11_theta1_phi2 + p11_theta2_phi2) / (p1_theta2 + p1_phi1)

    # gilchrist
    #b_ch = (p11_theta1_phi1 + p11_theta2_phi1 - p11_theta1_phi2 + p11_theta2_phi2) / (p1_theta2 + p1_phi1)

    return b_ch


def bell_quadrature_ch_simplified(x1_vec, x2_vec, chi, psi):
    """
    Calculate the simplified CH Bell test.

    Note that this simplified version of the inequality is only
    valid when the probabilities satisfies:

    P_11(theta, phi)  = P_11(theta + phi) 
    P_11(theta + phi) = P_11(-theta - phi) 
    
    Whether P_11 satisfies these relations of not depends on the 
    type of state psi.

    LHV limit: 1.0
    """
    
    P_1chi = P_x1x2(x1_vec, x2_vec, 0.0,   chi, psi)
    P_3chi = P_x1x2(x1_vec, x2_vec, 0.0, 3*chi, psi)

    p1a1, p1b1, p00, p01, p10, p11_1chi = quadrature_binning(x1_vec, x2_vec, P_1chi)
    p1a2, p1b2, p00, p01, p10, p11_3chi = quadrature_binning(x1_vec, x2_vec, P_3chi)
    
    b_ch = (3*p11_1chi - p11_3chi) / (2*p1a1)    
    
    return b_ch


def bell_quadrature_chsh(x1_vec, x2_vec, theta1, phi1, theta2, phi2, psi):
    """
    Calculate the CHSH Bell test.

    LHV limit: 2.0
    """
    
    P = P_x1x2(x1_vec, x2_vec, theta1, phi1, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E1 = p11 + p00 - p10 - p01
    
    P = P_x1x2(x1_vec, x2_vec, theta2, phi1, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E2 = p11 + p00 - p10 - p01

    P = P_x1x2(x1_vec, x2_vec, theta1, phi2, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E3 = p11 + p00 - p10 - p01

    P = P_x1x2(x1_vec, x2_vec, theta2, phi2, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E4 = p11 + p00 - p10 - p01
    
    b_chsh = abs(E1 - E2 + E3 + E4)
    
    return b_chsh


def bell_quadrature_chsh_simplified(x1_vec, x2_vec, chi, psi):
    """
    Calculate the CHSH Bell test.

    Note that this simplified version of the inequality is only
    valid when the probabilities satisfies:

    P_11(theta, phi)  = P_11(theta + phi) 
    P_11(theta + phi) = P_11(-theta - phi) 
    
    Whether P_11 satisfies these relations of not depends on the 
    type of state psi.

    LHV limit: 2.0
    """
    
    P = P_x1x2(x1_vec, x2_vec, 0.0, chi, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E_1chi = p11 + p00 - p10 - p01
    
    P = P_x1x2(x1_vec, x2_vec, 0.0, 3*chi, psi)
    p1a, p1b, p00, p01, p10, p11 = quadrature_binning(x1_vec, x2_vec, P)
    E_3chi = p11 + p00 - p10 - p01

    b_chsh = abs(3*E_1chi - E_3chi)
    
    return b_chsh

