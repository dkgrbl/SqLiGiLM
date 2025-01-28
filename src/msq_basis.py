#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####

import numpy as np
import scipy.linalg as lg

from src.utils.correlations import *
from src.utils.decompositions import *

##########
########## Additionals
##########

def make_eigendecomp_h(matrix, order="decreasing"):
    eigenValues, eigenVectors = np.linalg.eigh(matrix)
    if order=="decreasing":
        idx = np.real(eigenValues).argsort()[::-1] 
    elif order=="increasing":
        idx = np.real(eigenValues).argsort() 
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors


##########
########## Basis recunstruction
##########
 

def build_MSq_basis(cov_matrix, N=20):
    '''   
        For a given cov_matrix, the function computes the first N-modes of the MSq-basis

        input:  cov_matrix -- M x M matrices; N-number of computed modes
        output: U_matrix with the shape N x M; the rows of matrix U_matrix correspond to the Msq-modes 
    '''

    M = int(np.shape(cov_matrix)[0]/2)
    if M < N:
        N = M

    U_matrix = np.zeros((N, M), dtype=complex)
    buf_U_matrix   = np.identity(M, dtype=complex)

    for i in range(N):
        print("Itteration:", i)
        buf_cov_matrix = transfrom_covariance_U(cov_matrix, buf_U_matrix) 

        eigen_vals, eigenVec_COV = make_eigendecomp_h(buf_cov_matrix, order="increasing")

        Leig_T = eigenVec_COV.T[0:(M-i),0:(M-i)]
        Reig_T = eigenVec_COV.T[0:(M-i),(M-i):]

        psevdo_U_matrix = Reig_T + 1.j * Leig_T 
        Q, _ = lg.qr(psevdo_U_matrix.T)

        buf_U_matrix = Q.T @ buf_U_matrix
        U_matrix[i, :] = buf_U_matrix[0, :]
        buf_U_matrix = buf_U_matrix[1:, :]
    

    if not np.allclose( U_matrix @ np.conj(U_matrix).T, np.identity(N)):
        raise ValueError("MSq-modes are not orthogonal!")        

    return U_matrix


def build_MSq_basis_swap(cov_matrix, N=20, swap_cutoff=1e-5):
    '''   
        For a given cov_matrix, the function computes the first N-modes of the MSq-basis
        For the itteration, when (1.-eigen_vals[0]) < swap_cutoff, the sorting of eigenvalues is changed to get occupied unsqueezed modes.

        input:  cov_matrix -- M x M matrices; N-number of computed modes
        output: U_matrix with the shape N x M; the rows of matrix U_matrix correspond to the Msq-modes 
    '''
    ORDER = "increasing"

    M = int(np.shape(cov_matrix)[0]/2)
    if M < N:
        N = M

    U_matrix = np.zeros((N, M), dtype=complex)
    buf_U_matrix   = np.identity(M, dtype=complex)

    for i in range(N):
        print("Itteration:", i)
        buf_cov_matrix = transfrom_covariance_U(cov_matrix, buf_U_matrix) 

        eigen_vals, eigenVec_COV = make_eigendecomp_h(buf_cov_matrix, order=ORDER)
        print(eigen_vals[0])

        if ( (1.-eigen_vals[0] ) < swap_cutoff and ORDER == "increasing" ):
            print("SWAP itteration:", i+1)
            ORDER = "decreasing"

        Leig_T = eigenVec_COV.T[0:(M-i),0:(M-i)]
        Reig_T = eigenVec_COV.T[0:(M-i),(M-i):]

        psevdo_U_matrix = Reig_T + 1.j * Leig_T 
        Q, _ = lg.qr(psevdo_U_matrix.T)

        buf_U_matrix = Q.T @ buf_U_matrix
        U_matrix[i, :] = buf_U_matrix[0, :]
        buf_U_matrix = buf_U_matrix[1:, :]
    

    if not np.allclose( U_matrix @ np.conj(U_matrix).T, np.identity(N)):
        raise ValueError("MSq-modes are not orthogonal!")        

    return U_matrix