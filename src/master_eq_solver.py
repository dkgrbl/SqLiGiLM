#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp


def solve_slow_master_eqs(length, Gamma, F_matrix, Deltas, alphas_matrix=0, rtol=1e-7, atol=1e-12):
    ''' 
         Solution of slow master equations from initial vacuum state
    '''

    N = F_matrix.shape[0]
    NN  = N*N

    adag_a_range = np.array([0, NN])
    a_a_range    = np.array([NN, 2*NN])

    indent_N = np.diag(np.ones(N, dtype="complex"))  

    def right_part_diff_eq_deg_corr(z, corr_vector): 

        C_adag_a = corr_vector[adag_a_range[0]:adag_a_range[1]] 
        C_a_a    = corr_vector[a_a_range[0]:a_a_range[1]] 
        C_adag_a.shape   = (N,N)
        C_a_a.shape      = (N,N)

        F_a = F_matrix * np.exp(1.j * Deltas * z)
        Bff = F_a @ C_adag_a

        C_adag_a_new = - 0.5 * alphas_matrix * C_adag_a +  1j * Gamma * ( np.conj(C_a_a) @ F_a.T - (np.conj(F_a) @ C_a_a)  ) 
        C_a_a_new    = - 0.5 * alphas_matrix * C_a_a + 1j * Gamma * ( Bff + Bff.T )+ 1j*Gamma*F_a
        
        C_adag_a_new.shape = (NN)
        C_a_a_new.shape = (NN)
         
        buff_corr_vector = np.zeros( 2*NN, dtype="complex")
        buff_corr_vector[adag_a_range[0]:adag_a_range[1]] = C_adag_a_new
        buff_corr_vector[a_a_range[0]:a_a_range[1]]       = C_a_a_new

        return buff_corr_vector

    
    num_of_z_points = 2 
    initial_vector = np.zeros( 2*N*N , dtype="complex")
    z_array_ideq_solution = np.linspace(0, length, num_of_z_points)

    Soution = solve_ivp(right_part_diff_eq_deg_corr, [0, length], 
                    initial_vector, t_eval=z_array_ideq_solution, rtol=rtol, atol=atol)
    
    solution_transposed = Soution.y.T
    C_adag_a_tensor = solution_transposed[:,adag_a_range[0]:adag_a_range[1]]
    C_a_a_tensor    = solution_transposed[:,a_a_range[0]:a_a_range[1]]
    
    C_adag_a_tensor.shape = (num_of_z_points,N,N)    
    C_a_a_tensor.shape = (num_of_z_points,N,N)
            
    C_adag_a = C_adag_a_tensor[-1,...]
    C_a_a    = C_a_a_tensor[-1,...]

    return C_adag_a, C_a_a
