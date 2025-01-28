#####
#####   Author: Denis Kopylov 
#####   License: CC-BY-4.0 
#####   Source code: https://github.com/dkgrbl/SqLiGiLM
#####


import numpy as np
import scipy as sp

from src.utils.routines import *
from src.utils.correlations import *
from src.master_eq_solver import *

''' Main class ''' 

class PDC_type_I():
    """
        Class for prepare the grid for type-I PDC
    """
    def __init__(self, calc_parameters):
        self.NUMPOINTS          = calc_parameters["NUMPOINTS"] 
        self.length             = calc_parameters["length"] 

        #####
        ##### Grids
        #####  
        wave_pump       = calc_parameters["wave_pump"]
        self.dnu_step   = calc_parameters["dnu_step"]
        
        self.pdc_grid           = c_grid.init_with_step(self.NUMPOINTS, wave_pump*2., self.dnu_step)
        self.nu_pump_central    = LIGHTSPEED_CM_THZ/wave_pump
        self.nu_pdc_central     = 0.5*self.nu_pump_central


        #####
        ##### Dispersions
        #####  
        self.n_pump_func          = calc_parameters["n_pump_func"]
        self.n_pdc_func           = calc_parameters["n_pdc_func"]
        self.alpha_db_mu_pdc_func = calc_parameters["alpha_db_pdc_func"]

        self.profile_pump_func    = calc_parameters["profile_pump_func"]
        
        self.QPM_vector = 0

        if "QPM_vector" in calc_parameters.keys():
            self.QPM_vector = calc_parameters["QPM_vector"]

        self.k_pdc_grid = np.empty(0)
        
        self.v_group_pump_central = np.nan
        self.output_phases = np.empty(0)  

        #####
        ##### Two-photon amplitudes
        #####  

        self.tpa            = np.empty(0)
        self.Delta_k        = np.empty(0)
        self.F_spectrum     = np.empty(0)

        ##### 
        ##### Master eqs. solution 
        #####

        self.C_adag_a  = np.empty(0)
        self.C_a_a     = np.empty(0)
        
        ##### 
        ##### Prepare matrices 
        #####
        
        self.calc_Delta_and_F_mesh()
        self.apply_filter()
        self.calc_TPA()
        

    def save_to_npz(self, FilePath):

        result_dict =  {
                            "type": "type_I",
                            "NUMPOINTS": self.NUMPOINTS,
                            "length": self.length,

                            "nu_pump_central": self.nu_pump_central,
                            "nu_pdc_central": self.nu_pdc_central,
                            
                            "dnu_pdc_grid": self.pdc_grid.dnu,
                            
                            "C_adag_a": self.C_adag_a,
                            "C_a_a": self.C_a_a,
                        }

        np.savez(FilePath, **result_dict)

    def calc_Delta_and_F_mesh(self):
        
        def kpump(nu):
            return 2. * np.pi * self.n_pump_func(nu_to_um(nu)) * nu / LIGHTSPEED_CM_THZ

        def kpdc(nu):
            return 2. * np.pi * self.n_pdc_func(nu_to_um(nu)) * nu / LIGHTSPEED_CM_THZ

        def delta_k_func(dnu_s, dnu_i, nu_p):
            k_p = kpump(nu_p + dnu_s + dnu_i)
            k_s = kpdc(0.5 * nu_p + dnu_s)
            k_i = kpdc(0.5 * nu_p + dnu_i)
            DelK = k_p - k_s - k_i
            return DelK

        self.k_pdc_grid     = kpdc(self.nu_pdc_central + self.pdc_grid.dnu) 
        
        dnu_mesh_sig, dnu_mesh_idl  = np.meshgrid(self.pdc_grid.dnu, self.pdc_grid.dnu)
        self.F_spectrum     = self.profile_pump_func(dnu_mesh_sig+dnu_mesh_idl)
        self.Delta_k        = delta_k_func(dnu_mesh_sig, dnu_mesh_idl, self.nu_pump_central)

        if self.QPM_vector:
            if self.QPM_vector=="auto":
                self.QPM_vector = delta_k_func(0, 0, self.nu_pump_central)
                self.Delta_k    = self.Delta_k - self.QPM_vector
            else:
                self.Delta_k = self.Delta_k - self.QPM_vector

        #####
        ##### calculate self.output_phases
        #####                   

        del_nu  = self.nu_pdc_central * 1e-5
        d_kpump = kpump(self.nu_pump_central + del_nu) - kpump(self.nu_pump_central) 
        v_group_pump_central = 2. * np.pi * del_nu / d_kpump
        T       = self.length / v_group_pump_central       ## central time for the new time-frame
        phi_1   = self.k_pdc_grid * self.length        ## compensate slow phases 
        phi_2   = 2. * np.pi * self.pdc_grid.dnu * T   ## change time-frame 
        self.output_phases  = np.exp(1.j* phi_1) * np.exp(-1.j* phi_2)
   
    def apply_filter(self, ratio=0.94, power=15):
        sig_dnu = self.pdc_grid.dnu
        self.filter_1d = np.exp(- (sig_dnu)**(2*power)/(ratio*sig_dnu[-1])**(2*power) )
        self.filter_2d = np.outer(self.filter_1d, self.filter_1d)
        self.F_spectrum = self.F_spectrum*self.filter_2d

    def solve_master_eq(self, Gain):
        length = self.length
        
        ####
        #### normalize Gamma
        ####
        _, SVals, _ = sp.linalg.svd(self.tpa)
        Gamma       = Gain / (SVals[0] * length)
        
        ####
        #### prepare alpha_matrix  
        ####

        def alpha_dB_func(nu):
            return self.alpha_db_mu_pdc_func(nu_to_um(nu))

        self.alphas_dB_array = alpha_dB_func(self.nu_pdc_central + self.pdc_grid.dnu)
        alphas_array = self.alphas_dB_array / (2*2.1714724095162565) 
         
        alphas_i, alphas_j = np.meshgrid(alphas_array, alphas_array)
        alphas_matrix = alphas_i + alphas_j

        ####
        #### prepare Delta_k, F_spectrum
        ####

        Delta_matrix =  self.Delta_k
        F_matrix     =  self.F_spectrum

        ####
        #### solve slow equations and change time frame
        ####

        C_adag_a_slow, C_a_a_slow = solve_slow_master_eqs(length, Gamma, F_matrix, Delta_matrix, alphas_matrix, rtol=1e-7, atol=1e-12)
        self.C_adag_a, self.C_a_a = transfrom_correlators_U(C_adag_a_slow, C_a_a_slow, np.diag(self.output_phases))
        return self.C_adag_a, self.C_a_a

    def calc_TPA(self):
        B = self.F_spectrum
        C = phys_sinc(self.Delta_k * self.length / 2.)
        D = np.exp(1.0j * self.Delta_k * self.length / 2.)
        self.tpa = -1.0j * B * C * D
        return self.tpa 
 