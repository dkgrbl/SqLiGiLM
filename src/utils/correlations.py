#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####


import numpy as np
import scipy as sp
import scipy.linalg as lg

from src.utils.decompositions import *



def transfrom_correlators_U(C_adag_a, C_a_a, U):
    # the transformation A = U a,  
    # input matricies are for a-operators,
    # returned matricies are for A-operators,
    # rows of matrix U define broadband modes for A 
    return np.conj(U) @ C_adag_a @ U.T, U @ C_a_a @ U.T

def transfrom_covariance_U(cov_matrix, U_matrix):
    N = int(cov_matrix.shape[0]/2)
    M = U_matrix.shape[0]
    
    if N!=U_matrix.shape[1]: 
        raise ValueError("Covariance_matrix is not compatible with unitary transformation!")        

    O = np.zeros((2*M, 2*N))
    O[:M,:N] = np.real(U_matrix)
    O[:M,N:] = -np.imag(U_matrix)
    O[M:,:N] = np.imag(U_matrix)
    O[M:,N:] = np.real(U_matrix)

    return O @ cov_matrix @ O.T

def check_covariance_matrix(cov_matrix, tol=1e-12):
    if not np.allclose(cov_matrix, cov_matrix.T): 
        raise ValueError("Covariance matrix is not symetric!")        
    
    M = int(cov_matrix.shape[0]/2)
    Omega = np.zeros((2*M,2*M))
    Omega[:M,M:] = np.eye(M)
    Omega[M:,:M] = -np.eye(M)
    eigenValues, _ =  make_eigendecomp_h(cov_matrix + 1.j*Omega, order="increasing")

    if (np.min(eigenValues)+tol) < 0: 
        raise ValueError("Covariance matrix is unphysical: uncertanty principle broken with {}".format(np.min(eigenValues)))        


    print("Covariance matrix is correct")


def XX_matrix_f(C_adag_a, C_a_a):
    In = np.identity(C_adag_a.shape[0])
    return In + C_adag_a + C_adag_a.T + C_a_a + np.conj(C_a_a)

def PP_matrix_f(C_adag_a, C_a_a):
    In = np.identity(C_adag_a.shape[0])
    return In + C_adag_a + C_adag_a.T - C_a_a - np.conj(C_a_a)

def PX_matrix_f(C_adag_a, C_a_a):
    In = np.identity(C_adag_a.shape[0])
    return 1.j * (-In + C_adag_a - C_adag_a.T - C_a_a + np.conj(C_a_a)  )

def XP_matrix_f(C_adag_a, C_a_a):
    In = np.identity(C_adag_a.shape[0])
    return 1.j * (In - C_adag_a + C_adag_a.T - C_a_a + np.conj(C_a_a)  )


def covariance_xxpp_f(C_adag_a, C_a_a):
    N = C_adag_a.shape[0]
        
    XX = XX_matrix_f(C_adag_a, C_a_a)
    PP = PP_matrix_f(C_adag_a, C_a_a)
    PX = PX_matrix_f(C_adag_a, C_a_a)
    XP = XP_matrix_f(C_adag_a, C_a_a)
    
    PXXP = 0.5 * (PX + XP.T)
    XPPX = 0.5 * (XP + PX.T)
    maximag = [np.max(np.imag(XX)), np.max(np.imag(PP)), np.max(np.imag(XPPX)), np.max(np.imag(PXXP))]
    if np.max(maximag) > 1e-7: 
        raise ValueError("The covariance valies are imaginary!")        
    
    Covariance_matrix = np.zeros((2*N,2*N))
    Covariance_matrix[0:N,0:N]      = np.real_if_close(XX,tol=10000)
    Covariance_matrix[0:N,N:2*N]    = np.real_if_close(XPPX,tol=10000)
    Covariance_matrix[N:2*N,0:N]    = np.real_if_close(PXXP,tol=10000)
    Covariance_matrix[N:2*N,N:2*N]  = np.real_if_close(PP,tol=10000)
    
    if not np.allclose(Covariance_matrix, Covariance_matrix.T): 
        raise ValueError("Covariance_matrix is NOT symmetric!")        

    return Covariance_matrix


def calc_number_of_modes(C_adag_a):
    spectrum = np.real_if_close(np.diag(C_adag_a))
    alpha = np.sum(spectrum)
    normalized_spectrum = spectrum / alpha
    k_schmidt = 1. / np.sum(normalized_spectrum**2.)
    return normalized_spectrum, k_schmidt


def calc_fidelity_with_vac(Covariance_matrix):
    # see 2013 J. Phys. A: Math. Theor. 46 025304
    N = int(Covariance_matrix.shape[0]/2)
    Covariance_matrix_vac = np.eye(2*N)
    determ = np.sqrt(lg.det(Covariance_matrix+Covariance_matrix_vac))
    F = 2**int(N)/(determ)
    return F

 