#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import src.pdc_type_I as pdc
from src.msq_basis import *
from src.utils.routines import *
from src.utils.correlations import *

import src.utils.plot_functions as plotutils

####
#### Crystal dispersions
####

def no(wave_um):
    no = np.sqrt(2.7359 + 0.01878 / (wave_um**2. - 0.01822) - 0.01354 * wave_um**2.)
    return no

def ne(wave_um, theta = np.pi/2.):
    ne_90 = np.sqrt(2.3753 + 0.01224 / (wave_um**2. - 0.01667) - 0.01516 * wave_um**2.)
    buf1 = np.sin(theta)**2. / (ne_90**2.)
    buf2 = np.cos(theta)**2. / (no(wave_um)**2.)
    BUF = 1. / np.sqrt(buf1 + buf2)
    return BUF




####
#### EXAMPLE 1
####


def run_example_1():

    ########
    ######## init parameters
    ########

    NUMPOINTS = 63

    tau = 0.05           #ps
    length = 1.          #cm
    wave_pump = 800.e-7  #cm
    dnu_step = 2.      

    profile_pump_func = lambda x: F_pump_gaussian_func(x, tau_fwhm=tau)

    n_pump_func     = lambda x: ne(x, theta=19.866 * np.pi/180.)
    n_pdc_func      = lambda x: no(x)
    alpha_db_pdc_func  = lambda x: 6.*np.heaviside(1.6-x, 0.5)   # stepwise losses


    calc_parameters = {
        "NUMPOINTS": NUMPOINTS,  
        "dnu_step": dnu_step,
        "length": length,
        "wave_pump": wave_pump,
        "profile_pump_func": profile_pump_func,
        "n_pump_func": n_pump_func,
        "n_pdc_func":  n_pdc_func,
        "alpha_db_pdc_func": alpha_db_pdc_func,
    }




    ########
    ######## init comptation
    ########

    pdc_example = pdc.PDC_type_I(calc_parameters)
    
    ########
    ######## solve master eqs
    ########

    pdc_example.solve_master_eq(2.)


    
    ########
    ######## Plot solutions
    ########

    plotutils.plot_2d(pdc_example.pdc_grid.dnu, pdc_example.pdc_grid.dnu, np.abs(pdc_example.tpa), title="TPA")

    plotutils.plot_matrix(pdc_example.C_adag_a, title="C_adag_a")
    plotutils.plot_matrix(pdc_example.C_a_a, title="C_a_a")

    
    ########
    ######## Plot cov matrix
    ########
    cov_matrix = covariance_xxpp_f(pdc_example.C_adag_a, pdc_example.C_a_a)
    check_covariance_matrix(cov_matrix, tol=2e-11)


    plotutils.plot_covariance_matrix_xxpp_log(cov_matrix, title = "Initial covariance")
 

    U_basis = build_MSq_basis_swap(cov_matrix)
    cov_mat_msq = transfrom_covariance_U(cov_matrix, U_basis)
    plotutils.plot_covariance_matrix_xxpp_log(cov_mat_msq, title = "Covariance, MSq basis")

    ########
    ######## Squeezing MSq basis
    ########
    plotutils.plot_squeezing(cov_mat_msq)

    ########
    ######## Plot modes MSq basis
    ########
    plotutils.plot_mode_profile(U_basis, n=0)
    plotutils.plot_mode_profile(U_basis, n=1)


    ########
    ######## Savefile
    ########
    pdc_example.save_to_npz("output/example_1")

if __name__ == '__main__':

    run_example_1()    

    plt.show()
